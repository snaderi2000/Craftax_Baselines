"""Track natural achievement-transition representations over PPO training.

For every fixed first-completion pair (s_before, s_after), this script extracts
the 512-dimensional post-activation output of critic Dense_6, immediately before
the scalar Dense_7 value head.  For achievement c and checkpoint t it forms

    delta_h_i = h_t(s_after_i) - h_t(s_before_i)
    M_c(t) = delta_H.T @ delta_H / N_c

and records the leading eigenvalue, its fraction of the trace, the leading
eigenvector, directional-consistency diagnostics, and scalar value changes.
Natural transitions are observational: they include every state change caused
by the achievement-producing environment step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)
    jax.tree = _JaxTreeCompat()
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from concept_mapping.analyze_achievement_stabilized_effect_over_checkpoints import selected_steps
from concept_mapping.analyze_diamond_representation_coherence import _coherence_statistics
from concept_mapping.critic_representation import critic_pre_value_representation, validate_critic_representation
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state
from craftax.craftax_env import make_craftax_env_from_name


MASTER = Path("concept_mapping/runs/master_symbolic_1p5b_1000episodes")
DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_BANK = MASTER / "achievement_transition_fixed_bank.npz"
DEFAULT_PLACE_STONE_BANK = MASTER / "place_stone_transition_fixed_bank.npz"
DEFAULT_OUT = MASTER / "achievement_transition_coherence_stride3"
DISPLAY_ORDER = (
    "collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
    "place_stone", "place_furnace", "make_wood_sword", "make_stone_pickaxe",
    "collect_coal", "make_stone_sword", "collect_iron", "make_iron_sword",
    "make_iron_pickaxe", "collect_diamond",
)


@dataclass(frozen=True)
class TransitionBank:
    names: tuple[str, ...]
    before: np.ndarray        # [N_total, 1345]
    after: np.ndarray         # [N_total, 1345]
    concept_index: np.ndarray # [N_total]
    counts: np.ndarray        # [C]


def load_transition_bank(main_path: Path, place_stone_path: Path) -> TransitionBank:
    with np.load(main_path) as data:
        names = [str(x) for x in data["concept_names"]]
        before = np.asarray(data["before"], dtype=np.float32)
        after = np.asarray(data["after"], dtype=np.float32)
        concept_index = np.asarray(data["concept_index"], dtype=np.int16)
    # Place Stone was added after the original 13-achievement bank was frozen.
    # Append it without changing any of the original transition identities.
    with np.load(place_stone_path) as data:
        ps_before = np.asarray(data["before"], dtype=np.float32)
        ps_after = np.asarray(data["after"], dtype=np.float32)
    place_index = len(names)
    names.append("place_stone")
    before = np.concatenate([before, ps_before])
    after = np.concatenate([after, ps_after])
    concept_index = np.concatenate([
        concept_index, np.full(len(ps_before), place_index, dtype=np.int16)
    ])
    counts = np.bincount(concept_index, minlength=len(names))
    if before.ndim != 2 or before.shape != after.shape or before.shape[1] != 1345:
        raise ValueError(f"Malformed transition bank: before={before.shape}, after={after.shape}")
    return TransitionBank(tuple(names), before, after, concept_index, counts)


def checkpoint_path(out_dir: Path, concept: str, step: int) -> Path:
    return out_dir / concept / "per_checkpoint" / f"checkpoint_{step}.npz"


def apply_hidden_and_value(params, observations: np.ndarray, activation: str, batch_size: int):
    """Return critic h [N,512] and V [N] in deterministic inference batches."""
    @jax.jit
    def apply_batch(parameters, batch):
        return critic_pre_value_representation(parameters, batch, activation)

    hidden_chunks, value_chunks = [], []
    for start in range(0, len(observations), batch_size):
        # batch: [B,1345]; hidden: [B,512]; values: [B].
        hidden, values = apply_batch(
            params, jnp.asarray(observations[start:start + batch_size])
        )
        hidden_chunks.append(np.asarray(hidden))
        value_chunks.append(np.asarray(values))
    return np.concatenate(hidden_chunks), np.concatenate(value_chunks)


def run_worker(args, steps: list[int], bank: TransitionBank) -> None:
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    activation = str(config.get("ACTIVATION", "tanh"))
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    packed = np.concatenate([bank.before, bank.after]) # [2*N_total,1345]
    n_pairs = len(bank.before)
    worker_steps = [s for i, s in enumerate(steps) if i % args.num_workers == args.worker_index]
    print(f"worker {args.worker_index}/{args.num_workers}: {len(worker_steps)} checkpoints", flush=True)
    validated = False

    for step in worker_steps:
        destinations = [checkpoint_path(Path(args.out_dir), name, step) for name in bank.names]
        if all(path.exists() for path in destinations):
            continue
        print(f"worker {args.worker_index}: evaluating {step:,}", flush=True)
        train_state, restored = _restore_train_state(
            args.run_path, config, network, env, env_params, step
        )
        if not validated:
            check = validate_critic_representation(
                network, train_state.params, jnp.asarray(packed[:32]), activation
            )
            if check["representation_shape"][-1] != 512 or check["value_reconstruction_max_abs_error"] > 1e-5:
                raise RuntimeError(f"Critic representation validation failed: {check}")
            print(f"worker {args.worker_index}: representation validated {check}", flush=True)
            validated = True

        hidden, values = apply_hidden_and_value(
            train_state.params, packed, activation, args.batch_size
        )
        h_before, h_after = np.split(hidden, [n_pairs])
        v_before, v_after = np.split(values, [n_pairs])

        for concept, name in enumerate(bank.names):
            destination = destinations[concept]
            if destination.exists():
                continue
            mask = bank.concept_index == concept
            delta_h = h_after[mask] - h_before[mask] # [N_c,512]
            delta_v = v_after[mask] - v_before[mask] # [N_c]
            stats, eigenvalues, top_vector, mean_delta, delta_norms = _coherence_statistics(delta_h)

            # Because v1 is sign-anchored to mean(delta_h), these projections
            # distinguish a common signed transition from a merely axial one.
            projections = np.asarray(delta_h, np.float64) @ top_vector # [N_c]
            safe_norms = np.maximum(delta_norms, np.finfo(np.float64).eps)
            cosines = projections / safe_norms
            payload = {
                "checkpoint": np.int64(restored),
                "concept": np.asarray(name),
                "n_transitions": np.int32(mask.sum()),
                "eigenvalues": eigenvalues,
                "lambda_1": np.float64(stats["lambda_1"]),
                "trace_M": np.float64(stats["total_eigenvalue_variance"]),
                "top_eigenvalue_fraction": np.float64(stats["top_eigenvalue_fraction"]),
                "top_eigenvector": top_vector,
                "mean_delta_h": mean_delta,
                "delta_h_norms": delta_norms,
                "mean_delta_h_norm": np.float64(stats["mean_delta_l2_norm"]),
                "median_delta_h_norm": np.float64(stats["delta_norm_median"]),
                "mean_cosine_to_v1": np.float64(np.mean(cosines)),
                "median_cosine_to_v1": np.float64(np.median(cosines)),
                "fraction_projection_positive": np.float64(np.mean(projections > 0)),
                "mean_delta_v": np.float64(np.mean(delta_v)),
                "median_delta_v": np.float64(np.median(delta_v)),
                "fraction_delta_v_negative": np.float64(np.mean(delta_v < 0)),
                "delta_v": delta_v.astype(np.float32),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            temporary.replace(destination)


def read_rows(out_dir: Path, concept: str, steps: list[int]) -> list[dict]:
    rows = []
    fields = (
        "n_transitions", "lambda_1", "trace_M", "top_eigenvalue_fraction",
        "mean_delta_h_norm", "median_delta_h_norm", "mean_cosine_to_v1",
        "median_cosine_to_v1", "fraction_projection_positive", "mean_delta_v",
        "median_delta_v", "fraction_delta_v_negative",
    )
    for step in steps:
        path = checkpoint_path(out_dir, concept, step)
        if not path.exists():
            continue
        with np.load(path) as data:
            row = {"checkpoint": int(data["checkpoint"])}
            row.update({field: float(data[field]) for field in fields})
            row["top_eigenvector"] = np.asarray(data["top_eigenvector"], dtype=np.float64)
        rows.append(row)
    return sorted(rows, key=lambda x: x["checkpoint"])


def plot_small_multiples(out_dir: Path, rows_by_name: dict, names: list[str], field: str,
                         ylabel: str, title: str, stem: str, percent: bool = False) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        rows = rows_by_name[name]
        x = np.asarray([row["checkpoint"] for row in rows], dtype=float)
        y = np.asarray([row[field] for row in rows], dtype=float)
        if percent:
            y *= 100.0
        ax.plot(x, y, color="#2563eb", linewidth=1.65)
        ax.set_xscale("log")
        ax.set_title(name.replace("_", " ").title(), fontsize=11, fontweight="semibold")
        ax.grid(True, which="both", alpha=.18)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.supxlabel("Training environment steps")
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=17)
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def aggregate(args, steps: list[int], bank: TransitionBank) -> None:
    out_dir = Path(args.out_dir)
    rows_by_name = {name: read_rows(out_dir, name, steps) for name in bank.names}
    missing = {name: len(steps) - len(rows) for name, rows in rows_by_name.items()}
    if any(missing.values()):
        raise RuntimeError(f"Sweep incomplete: {missing}")

    for name, rows in rows_by_name.items():
        final_vector = rows[-1]["top_eigenvector"]
        final_vector /= np.linalg.norm(final_vector)
        for row in rows:
            vector = row.pop("top_eigenvector")
            vector /= np.linalg.norm(vector)
            row["alignment_to_final"] = float(abs(np.dot(vector, final_vector)))
        fields = list(rows[0].keys())
        path = out_dir / name / "metrics.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)

    names = [name for name in DISPLAY_ORDER if name in rows_by_name]
    names.extend(name for name in bank.names if name not in names)
    plot_small_multiples(out_dir, rows_by_name, names, "top_eigenvalue_fraction",
                         r"Coherence, $r_c(t)=\lambda_1/\mathrm{tr}(M_c)$",
                         "Achievement-transition representation coherence",
                         "achievement_transition_coherence", percent=False)
    plot_small_multiples(out_dir, rows_by_name, names, "lambda_1",
                         r"Leading eigenvalue, $\lambda_1(M_c(t))$",
                         "Magnitude of dominant achievement-transition direction",
                         "achievement_transition_lambda1", percent=False)
    plot_small_multiples(out_dir, rows_by_name, names, "alignment_to_final",
                         r"Alignment, $|v_1(c,t)^\top v_1(c,T)|$",
                         "Alignment to the final achievement-transition direction",
                         "achievement_transition_alignment", percent=False)
    plot_small_multiples(out_dir, rows_by_name, names, "median_delta_v",
                         r"Median $V(s_{after})-V(s_{before})$",
                         "Scalar critic-value signature of achievement completion",
                         "achievement_transition_median_delta_v", percent=False)

    metadata = {
        "run_path": args.run_path,
        "checkpoint_stride": args.checkpoint_stride,
        "n_checkpoints": len(steps),
        "concepts": names,
        "n_transitions": {name: int(count) for name, count in zip(names, bank.counts)},
        "representation": "post-activation Dense_6 critic output immediately before scalar Dense_7",
        "representation_dim": 512,
        "coherence": "lambda_1((DeltaH^T DeltaH)/N) / trace((DeltaH^T DeltaH)/N)",
        "directional_diagnostics": ["mean_cosine_to_v1", "fraction_projection_positive"],
        "warning": "Natural consecutive transitions are observational and include action, map, inventory, and status changes; coherence is not by itself proof of achievement identity.",
    }
    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved aggregate outputs to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--place_stone_bank", type=Path, default=DEFAULT_PLACE_STONE_BANK)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint_stride", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_stride <= 0 or args.batch_size <= 0 or args.num_workers <= 0:
        raise ValueError("stride, batch size, and worker count must be positive")
    steps = selected_steps(args.run_path, args.checkpoint_stride)
    bank = load_transition_bank(args.bank, args.place_stone_bank)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args, steps, bank)
    else:
        run_worker(args, steps, bank)


if __name__ == "__main__":
    main()

"""Sweep frozen hierarchy counterfactual banks across all saved PPO checkpoints.

For concept c and checkpoint t, the saved positive/negative observations define
Delta h_i = h(s_i+) - h(s_i-) at the post-activation output of critic Dense_6
(the 512-dimensional representation immediately before the scalar value head).
Each worker writes an atomic per-checkpoint NPZ.  A separate aggregate pass
computes alignment to the final vector and produces the cross-concept figures.
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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.analyze_diamond_representation_coherence import _coherence_statistics
from concept_mapping.critic_representation import critic_pre_value_representation, validate_critic_representation
from concept_mapping.diamond_affordance_over_time import _make_counterfactuals
from concept_mapping.plot_diamond_ordering_over_checkpoints import _checkpoint_steps
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


DEFAULT_RUN_PATH = "wandb/run-20260830_231509-616wuss8"
DEFAULT_BANK_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks"
DEFAULT_OUT_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/checkpoint_sweep"
CONCEPT_ORDER = ["wood", "table", "wood_pickaxe", "stone", "furnace", "coal", "iron", "iron_pickaxe", "diamond"]


@dataclass(frozen=True)
class Bank:
    name: str
    plus: np.ndarray   # [N, 1345]
    minus: np.ndarray  # [N, 1345]
    n_states: int
    source: str


def _load_banks(bank_dir: Path, dataset_dir: Path) -> list[Bank]:
    banks: list[Bank] = []
    for concept in CONCEPT_ORDER:
        if concept == "diamond":
            path = dataset_dir / "concept_index" / "diamond_collectible_state_bank.npz"
            with np.load(path) as data:
                plus = np.asarray(data["base_obs"], dtype=np.float32)
            minus = np.stack([
                _make_counterfactuals(obs)["C_diamond_no_pickaxe"] for obs in plus
            ]).astype(np.float32)
        else:
            path = bank_dir / f"{concept}_bank.npz"
            with np.load(path) as data:
                plus = np.asarray(data["base_obs"], dtype=np.float32)
                minus = np.asarray(data["counterfactual_obs"], dtype=np.float32)
        if plus.ndim != 2 or minus.shape != plus.shape:
            raise ValueError(f"Malformed {concept} bank: plus={plus.shape}, minus={minus.shape}")
        banks.append(Bank(concept, plus, minus, len(plus), str(path)))
    return banks


def _concatenate_banks(banks: list[Bank]) -> tuple[np.ndarray, np.ndarray, dict[str, slice]]:
    """Pack each condition once, minimizing GPU inference calls per checkpoint."""
    offsets: dict[str, slice] = {}
    cursor = 0
    for bank in banks:
        offsets[bank.name] = slice(cursor, cursor + bank.n_states)
        cursor += bank.n_states
    return (
        np.concatenate([bank.plus for bank in banks]),
        np.concatenate([bank.minus for bank in banks]),
        offsets,
    )


def _apply_all(params, observations: np.ndarray, activation: str, batch_size: int):
    """Return h [N,512] and V [N] deterministically for one packed condition."""
    @jax.jit
    def apply_batch(parameters, batch):
        return critic_pre_value_representation(parameters, batch, activation)

    h_chunks, v_chunks = [], []
    for start in range(0, len(observations), batch_size):
        h, values = apply_batch(params, jnp.asarray(observations[start:start + batch_size]))
        h_chunks.append(np.asarray(h))
        v_chunks.append(np.asarray(values))
    return np.concatenate(h_chunks), np.concatenate(v_chunks)


def _checkpoint_npz(out_dir: Path, concept: str, step: int) -> Path:
    return out_dir / concept / "per_checkpoint" / f"checkpoint_{step}.npz"


def _evaluate_worker(args, steps: list[int], banks: list[Bank], packed_plus, packed_minus, offsets, activation: str):
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    validation_done = False
    for step in steps:
        output_paths = [_checkpoint_npz(Path(args.out_dir), bank.name, step) for bank in banks]
        if all(path.exists() for path in output_paths):
            print(f"worker {args.worker_index}: checkpoint {step:,} already complete", flush=True)
            continue
        print(f"worker {args.worker_index}: evaluating checkpoint {step:,}", flush=True)
        train_state, restored_step = _restore_train_state(args.run_path, config, network, env, env_params, step)
        if not validation_done:
            validation = validate_critic_representation(
                network, train_state.params, jnp.asarray(packed_plus[:32]), activation
            )
            if validation["value_reconstruction_max_abs_error"] > 1e-5:
                raise RuntimeError(f"Critic representation reconstruction failed: {validation}")
            print(f"worker {args.worker_index}: representation validated {validation}", flush=True)
            validation_done = True
        h_plus, v_plus = _apply_all(train_state.params, packed_plus, activation, args.batch_size)
        h_minus, v_minus = _apply_all(train_state.params, packed_minus, activation, args.batch_size)
        for bank, output_path in zip(banks, output_paths):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sl = offsets[bank.name]
            delta_h = h_plus[sl] - h_minus[sl]  # [N_c, 512]
            delta_v = v_plus[sl] - v_minus[sl]  # [N_c]
            stats, eigenvalues, top_vector, mean_delta, delta_norms = _coherence_statistics(delta_h)
            payload = {
                "checkpoint": np.int64(restored_step), "concept": np.asarray(bank.name),
                "n_states": np.int32(bank.n_states), "eigenvalues": eigenvalues,
                "lambda_1": np.float64(stats["lambda_1"]),
                "total_eigenvalue_variance": np.float64(stats["total_eigenvalue_variance"]),
                "top_eigenvalue_fraction": np.float64(stats["top_eigenvalue_fraction"]),
                "top_eigenvector": top_vector, "mean_delta_h": mean_delta,
                "delta_norms": delta_norms,
                "delta_norm_mean": np.float64(stats["delta_norm_mean"]),
                "delta_norm_std": np.float64(stats["delta_norm_std"]),
                "delta_norm_median": np.float64(stats["delta_norm_median"]),
                "delta_norm_q05": np.float64(stats["delta_norm_q05"]),
                "delta_norm_q95": np.float64(stats["delta_norm_q95"]),
                "mean_delta_h_l2_norm": np.float64(stats["mean_delta_l2_norm"]),
                "mean_delta_v": np.float64(delta_v.mean()),
                "median_delta_v": np.float64(np.median(delta_v)),
                "fraction_delta_v_positive": np.float64((delta_v > 0).mean()),
            }
            temporary = output_path.with_name(f".{output_path.name}.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            temporary.replace(output_path)


def _read_rows(out_dir: Path, concept: str, steps: list[int]) -> list[dict]:
    rows = []
    for step in steps:
        path = _checkpoint_npz(out_dir, concept, step)
        if not path.exists():
            continue
        with np.load(path) as data:
            rows.append({
                "checkpoint": int(data["checkpoint"]), "n_states": int(data["n_states"]),
                "lambda_1": float(data["lambda_1"]),
                "total_eigenvalue_variance": float(data["total_eigenvalue_variance"]),
                "top_eigenvalue_fraction": float(data["top_eigenvalue_fraction"]),
                "top_eigenvector": np.asarray(data["top_eigenvector"], dtype=np.float64),
                "mean_delta_v": float(data["mean_delta_v"]),
                "median_delta_v": float(data["median_delta_v"]),
                "fraction_delta_v_positive": float(data["fraction_delta_v_positive"]),
            })
    return sorted(rows, key=lambda row: row["checkpoint"])


def _plot_metric(out_dir: Path, rows_by_concept: dict[str, list[dict]], field: str, ylabel: str, filename: str, percentage=False):
    fig, ax = plt.subplots(figsize=(10.0, 5.7), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(CONCEPT_ORDER)))
    for color, concept in zip(colors, CONCEPT_ORDER):
        rows = rows_by_concept[concept]
        if not rows:
            continue
        x = np.asarray([row["checkpoint"] for row in rows], dtype=float)
        y = np.asarray([row[field] for row in rows], dtype=float)
        if percentage:
            y *= 100
        ax.plot(x, y, linewidth=1.65, color=color, label=concept.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    fig.savefig(out_dir / f"{filename}.png", dpi=300)
    fig.savefig(out_dir / f"{filename}.pdf")
    plt.close(fig)


def _aggregate(args, steps: list[int], banks: list[Bank]):
    out_dir = Path(args.out_dir)
    rows_by_concept: dict[str, list[dict]] = {}
    missing = {}
    for bank in banks:
        rows = _read_rows(out_dir, bank.name, steps)
        rows_by_concept[bank.name] = rows
        missing[bank.name] = len(steps) - len(rows)
        if not rows:
            continue
        final = max(rows, key=lambda row: row["checkpoint"])["top_eigenvector"]
        final /= np.linalg.norm(final)
        for row in rows:
            current = row["top_eigenvector"]
            current /= np.linalg.norm(current)
            row["alignment_to_final"] = float(abs(np.dot(current, final)))
        metrics_path = out_dir / bank.name / "metrics.csv"
        with metrics_path.open("w", newline="") as handle:
            fields = ["checkpoint", "n_states", "lambda_1", "total_eigenvalue_variance", "top_eigenvalue_fraction", "alignment_to_final", "mean_delta_v", "median_delta_v", "fraction_delta_v_positive"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{key: row[key] for key in fields} for row in rows])
    if any(missing.values()):
        raise RuntimeError(f"Sweep is incomplete; missing checkpoints: {missing}")
    _plot_metric(out_dir, rows_by_concept, "lambda_1", r"$\lambda_1(M_c(t))$", "lambda1_all_concepts")
    _plot_metric(out_dir, rows_by_concept, "alignment_to_final", r"$|v_1(c,t)^\top v_1(c,T)|$", "alignment_all_concepts")
    _plot_metric(out_dir, rows_by_concept, "fraction_delta_v_positive", r"Counterfactual success: $\Delta V>0$ (%)", "counterfactual_success_all_concepts", percentage=True)
    summary = {
        "run_path": args.run_path, "n_checkpoints": len(steps),
        "concept_order": CONCEPT_ORDER,
        "n_states_per_concept": {bank.name: bank.n_states for bank in banks},
        "alignment_reference": "largest saved checkpoint per concept", "missing": missing,
        "representation": "post-activation critic Dense_6 output immediately before Dense_7",
    }
    (out_dir / "sweep_metadata.json").write_text(json.dumps(summary, indent=2))
    print("Aggregate complete; saved per-concept metrics and multi-concept figures.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN_PATH)
    parser.add_argument("--bank_dir", default=DEFAULT_BANK_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--worker_index", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--checkpoints", default="", help="Optional comma-separated checkpoint steps for a pilot.")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers <= 0:
        raise ValueError("batch_size and num_workers must be positive")
    steps = [int(x) for x in args.checkpoints.split(",") if x.strip()] if args.checkpoints else _checkpoint_steps(args.run_path)
    bank_dir = Path(args.bank_dir)
    dataset_dir = bank_dir.parent
    banks = _load_banks(bank_dir, dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(args, steps, banks)
        return
    worker_index = 0 if args.worker_index is None else args.worker_index
    if not 0 <= worker_index < args.num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    worker_steps = [step for index, step in enumerate(steps) if index % args.num_workers == worker_index]
    packed_plus, packed_minus, offsets = _concatenate_banks(banks)
    print(f"worker {worker_index}/{args.num_workers}: {len(worker_steps)} checkpoints, {len(packed_plus)} states per condition", flush=True)
    config = _load_wandb_config(args.run_path)
    _evaluate_worker(args, worker_steps, banks, packed_plus, packed_minus, offsets, str(config.get("ACTIVATION", "tanh")))


if __name__ == "__main__":
    main()

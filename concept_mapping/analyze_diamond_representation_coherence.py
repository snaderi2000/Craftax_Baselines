"""Measure coherent critic directions for diamond--pickaxe counterfactuals.

For every saved PPO checkpoint, this script evaluates the same 5,000 matched
diamond states in two conditions: D+, P+ and D+, P-.  It extracts the critic
representation immediately before its scalar value head and eigendecomposes
the second-moment matrix of within-state representation differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
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

from concept_mapping.critic_representation import (
    critic_pre_value_representation,
    validate_critic_representation,
)
from concept_mapping.diamond_affordance_over_time import _make_counterfactuals
from concept_mapping.plot_diamond_ordering_over_checkpoints import _checkpoint_steps
from concept_mapping.render_ppo_episodes import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from craftax.craftax_env import make_craftax_env_from_name


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _evaluate_pair(
    params,
    d_plus_p_plus: np.ndarray,
    d_plus_p_minus: np.ndarray,
    activation: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate both counterfactual conditions in deterministic minibatches.

    Returns arrays with shapes:
      h_plus, h_minus: [N=5000, d=512]
      value_plus, value_minus: [N=5000]
    """

    @jax.jit
    def apply_batch(batch_observations):
        return critic_pre_value_representation(params, batch_observations, activation)

    outputs = []
    for observations in (d_plus_p_plus, d_plus_p_minus):
        representation_chunks, value_chunks = [], []
        for start in range(0, len(observations), batch_size):
            batch = jnp.asarray(observations[start : start + batch_size])  # [B, 1345]
            h, values = apply_batch(batch)  # [B, 512], [B]
            representation_chunks.append(np.asarray(h))
            value_chunks.append(np.asarray(values))
        outputs.extend((np.concatenate(representation_chunks), np.concatenate(value_chunks)))
    return outputs[0], outputs[2], outputs[1], outputs[3]


def _coherence_statistics(
    delta_h: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute second-moment eigenstatistics for ``delta_h`` of shape [N, d]."""
    # delta_h: [N=5000, d=512]. Float64 gives a stable symmetric eigendecomposition.
    delta_h64 = np.asarray(delta_h, dtype=np.float64)
    n_states, representation_dim = delta_h64.shape
    matrix = (delta_h64.T @ delta_h64) / n_states  # M_t: [d=512, d=512]
    matrix = 0.5 * (matrix + matrix.T)  # remove roundoff-level asymmetry
    eigenvalues_ascending, eigenvectors_ascending = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues_ascending)[::-1]
    eigenvalues = eigenvalues_ascending[order]  # [512], descending
    eigenvectors = eigenvectors_ascending[:, order]  # [512, 512], columns are eigenvectors
    top_eigenvector = eigenvectors[:, 0].copy()  # v_1: [512]

    # The sign of an eigenvector is arbitrary. Anchor it to the mean Delta h
    # so that saved top directions have a consistent, interpretable orientation.
    mean_delta = delta_h64.mean(axis=0)  # [512]
    if float(np.dot(top_eigenvector, mean_delta)) < 0:
        top_eigenvector *= -1.0

    # Tiny negative values can occur only through numerical roundoff. Preserve raw
    # eigenvalues in the NPZ, but use non-negative variance for the ratio.
    nonnegative_eigenvalues = np.clip(eigenvalues, 0.0, None)
    total_variance = float(nonnegative_eigenvalues.sum())
    top_fraction = (
        float(nonnegative_eigenvalues[0] / total_variance)
        if total_variance > np.finfo(np.float64).eps
        else float("nan")
    )
    delta_norms = np.linalg.norm(delta_h64, axis=1)  # one norm for each of N states: [5000]
    stats = {
        "n_states": int(n_states),
        "representation_dim": int(representation_dim),
        "lambda_1": float(eigenvalues[0]),
        "total_eigenvalue_variance": total_variance,
        "top_eigenvalue_fraction": top_fraction,
        "delta_norm_mean": float(delta_norms.mean()),
        "delta_norm_std": float(delta_norms.std(ddof=1)),
        "delta_norm_median": float(np.median(delta_norms)),
        "delta_norm_q05": float(np.quantile(delta_norms, 0.05)),
        "delta_norm_q95": float(np.quantile(delta_norms, 0.95)),
        "mean_delta_l2_norm": float(np.linalg.norm(mean_delta)),
    }
    return stats, eigenvalues, top_eigenvector, mean_delta, delta_norms


def _plot_single_metric(rows: list[dict], field: str, ylabel: str, title: str, path: Path):
    steps = np.asarray([row["checkpoint"] for row in rows], dtype=float)
    values = np.asarray([row[field] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    ax.plot(steps, values, color="#7c3aed", marker="o", markersize=3.7, linewidth=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_spectrum(rows: list[dict], top_k: int, path: Path):
    steps = np.asarray([row["checkpoint"] for row in rows], dtype=float)
    eigenspectra = np.stack([row["eigenvalues"][:top_k] for row in rows])  # [T, top_k]
    fig, ax = plt.subplots(figsize=(7.7, 4.7), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.90, top_k))
    for index in range(top_k):
        ax.plot(steps, eigenspectra[:, index], color=colors[index], linewidth=2.0,
                label=rf"$\lambda_{{{index + 1}}}$")
    ax.set_xscale("log")
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(r"Eigenvalue of $M_t$")
    ax.set_title("Leading representation-change eigenvalues")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_alignment(rows: list[dict], ordering_csv: Path, success_csv: Path, path: Path):
    """Plot coherence, existing ordering success, and collection success on one % scale."""
    fig, ax = plt.subplots(figsize=(8.1, 4.8), constrained_layout=True)
    steps = np.asarray([row["checkpoint"] for row in rows], dtype=float)
    coherence = 100 * np.asarray([row["top_eigenvalue_fraction"] for row in rows], dtype=float)
    ax.plot(steps, coherence, color="#7c3aed", linewidth=2.2, marker="o", markersize=3.5,
            label=r"Representation coherence ($100r_t$)")

    ordering = _csv_rows(ordering_csv)
    if ordering:
        x = np.asarray([float(row["checkpoint"]) for row in ordering])
        y = 100 * np.asarray([float(row["full_ordering_satisfied"]) for row in ordering])
        ax.plot(x, y, color="#dc2626", linewidth=2.0, marker="o", markersize=3.8,
                label="Diamond-pickaxe ordering")

    success = _csv_rows(success_csv)
    if success:
        x = np.asarray([float(row["environment_steps"]) for row in success])
        y = np.asarray([float(row["diamond_success_smoothed_pct"]) for row in success])
        ax.plot(x, y, color="#2563eb", linewidth=2.0,
                label="Diamond collection success (smoothed)")

    ax.set_xscale("log")
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Coherence / success (%)")
    ax.set_title("Representation coherence and diamond concept emergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument(
        "--states_npz",
        default=("concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
                 "merged/clean_diamond_tminus_states.npz"),
    )
    parser.add_argument(
        "--out_dir",
        default=("concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
                 "merged/representation_coherence_over_checkpoints"),
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument(
        "--checkpoints", default="",
        help="Optional comma-separated checkpoint steps; default evaluates every saved checkpoint.",
    )
    parser.add_argument(
        "--ordering_csv",
        default=("concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
                 "merged/ordering_over_checkpoints/ordering_over_checkpoints.csv"),
    )
    parser.add_argument(
        "--success_csv",
        default=("concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
                 "merged/diamond_success_curve/diamond_success_smoothed.csv"),
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "per_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    states_path = Path(args.states_npz)
    fixed_states = np.load(states_path)
    base_obs = np.asarray(fixed_states["base_obs"], dtype=np.float32)  # [N=5000, 1345]
    if base_obs.ndim != 2 or base_obs.shape[0] != 5000:
        raise ValueError(f"Expected fixed states [5000, observation_dim], received {base_obs.shape}")
    counterfactuals = [_make_counterfactuals(obs) for obs in base_obs]
    d_plus_p_plus = np.stack([cf["D_diamond_pickaxe"] for cf in counterfactuals])  # [5000, 1345]
    d_plus_p_minus = np.stack([cf["C_diamond_no_pickaxe"] for cf in counterfactuals])  # [5000, 1345]

    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    activation = str(config.get("ACTIVATION", "tanh"))
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    steps = (
        [int(value) for value in args.checkpoints.split(",") if value.strip()]
        if args.checkpoints else _checkpoint_steps(args.run_path)
    )
    if not steps:
        raise ValueError("No checkpoints selected")

    metadata = {
        "run_path": args.run_path,
        "states_npz": str(states_path),
        "states_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "n_fixed_base_states": int(len(base_obs)),
        "observation_shape": list(base_obs.shape),
        "conditions": {
            "plus": "D_diamond_pickaxe (D+, P+)",
            "minus": "C_diamond_no_pickaxe (D+, P-)",
        },
        "representation": "post-activation critic Dense_6 output, immediately before Dense_7 value head",
        "activation": activation,
        "selected_checkpoint_steps": steps,
    }

    rows = []
    validation = None
    for requested_step in steps:
        print(f"Evaluating representation coherence at checkpoint {requested_step:,}", flush=True)
        train_state, restored_step = _restore_train_state(
            args.run_path, config, network, env, env_params, requested_step
        )
        if validation is None:
            validation = validate_critic_representation(
                network, train_state.params, jnp.asarray(d_plus_p_plus[: min(args.batch_size, 32)]), activation
            )
            if validation["value_reconstruction_max_abs_error"] > 1e-5:
                raise RuntimeError(f"Critic representation validation failed: {validation}")
            print(f"Validated critic representation: {validation}", flush=True)

        h_plus, h_minus, value_plus, value_minus = _evaluate_pair(
            train_state.params, d_plus_p_plus, d_plus_p_minus, activation, args.batch_size
        )
        # D_t = h(D+, P+) - h(D+, P-), with shape [N=5000, d=512].
        delta_h = h_plus - h_minus
        stats, eigenvalues, top_eigenvector, mean_delta, delta_norms = _coherence_statistics(delta_h)
        ordering_success = float(np.mean(value_plus > value_minus))
        row = {"checkpoint": int(restored_step), **stats, "D_plus_pickaxe_gt_no_pickaxe": ordering_success,
               "eigenvalues": eigenvalues}
        rows.append(row)
        np.savez_compressed(
            checkpoint_dir / f"checkpoint_{restored_step}.npz",
            checkpoint=np.int64(restored_step),
            eigenvalues=eigenvalues,  # [512], descending
            lambda_1=np.float64(stats["lambda_1"]),
            top_eigenvalue_fraction=np.float64(stats["top_eigenvalue_fraction"]),
            top_eigenvector=top_eigenvector,  # [512]
            mean_delta=mean_delta,  # [512]
            delta_norms=delta_norms,  # [5000]
            value_plus=value_plus,  # [5000]
            value_minus=value_minus,  # [5000]
        )

    csv_fields = [key for key in rows[0] if key != "eigenvalues"]
    with (out_dir / "representation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in csv_fields} for row in rows])
    metadata["validation"] = validation
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    _plot_single_metric(rows, "lambda_1", r"$\lambda_1(M_t)$",
                        "Magnitude of the leading collectability direction",
                        out_dir / "top_eigenvalue_vs_steps")
    _plot_single_metric(rows, "top_eigenvalue_fraction", r"$r_t = \lambda_1 / \sum_j \lambda_j$",
                        "Coherence of the diamond--pickaxe representation change",
                        out_dir / "top_eigenvalue_fraction_vs_steps")
    _plot_spectrum(rows, min(args.top_k, base_obs.shape[1]), out_dir / "top_eigenvalues_vs_steps")
    _plot_alignment(rows, Path(args.ordering_csv), Path(args.success_csv),
                    out_dir / "coherence_ordering_collection_overlay")
    print(f"Saved representation-coherence analysis to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

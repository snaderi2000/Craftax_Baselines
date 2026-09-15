"""Measure emergence of an inventory-progression heuristic across checkpoints.

The primary metric is the fraction of frozen pre-stone states whose critic
value has a negative within-state OLS slope along the cumulative empirical
inventory ladder:

wood pickaxe -> stone -> wood sword -> stone pickaxe -> coal -> stone sword -> iron.

Only every other saved checkpoint is evaluated by default; the final checkpoint
is always retained.  The same observations are used at every checkpoint.
"""

from __future__ import annotations

import argparse
import csv
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.plot_diamond_ordering_over_checkpoints import _checkpoint_steps
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_BANK = (
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "progression_vs_pickaxe_value/progression_bank.npz"
)
DEFAULT_OUT = (
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "progression_vs_pickaxe_value/progression_checkpoint_sweep_stride2"
)
FUNCTIONAL_DIR = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "wood_pickaxe_functional/checkpoint_sweep"
)
PRIMARY_LEVEL_START = 3  # wood pickaxe has just been added
PRIMARY_LEVEL_STOP = 10  # exclusive; level 9 adds iron


def _selected_steps(run_path: str, stride: int, explicit: str) -> list[int]:
    if explicit:
        return sorted({int(x) for x in explicit.split(",") if x.strip()})
    all_steps = _checkpoint_steps(run_path)
    selected = all_steps[::stride]
    if selected[-1] != all_steps[-1]:
        selected.append(all_steps[-1])
    return selected


def _load_bank(path: Path) -> dict:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _output_path(out_dir: Path, step: int) -> Path:
    return out_dir / "per_checkpoint" / f"checkpoint_{step}.npz"


def _run_worker(args, steps: list[int], bank: dict) -> None:
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    # [7, N, 1345]: the fixed post-pickaxe early/middle progression ladder.
    primary = bank["levels"][PRIMARY_LEVEL_START:PRIMARY_LEVEL_STOP].astype(np.float32)
    n_levels, n_states, obs_dim = primary.shape
    packed = primary.reshape(n_levels * n_states, obs_dim)

    @jax.jit
    def value_batch(params, observations):
        _policy, value = network.apply(params, observations)
        return value

    worker_steps = [step for index, step in enumerate(steps)
                    if index % args.num_workers == args.worker_index]
    print(f"worker {args.worker_index}/{args.num_workers}: {len(worker_steps)} checkpoints", flush=True)
    for step in worker_steps:
        output = _output_path(Path(args.out_dir), step)
        if output.exists():
            continue
        print(f"worker {args.worker_index}: checkpoint {step:,}", flush=True)
        state, restored = _restore_train_state(args.run_path, config, network, env, env_params, step)
        chunks = []
        for start in range(0, len(packed), args.batch_size):
            chunks.append(np.asarray(value_batch(state.params, jnp.asarray(packed[start:start + args.batch_size]))))
        # [7, N]. x is centered, so each state gets an independent OLS slope.
        values = np.concatenate(chunks).reshape(n_levels, n_states)
        x = np.arange(n_levels, dtype=np.float64)
        centered_x = x - x.mean()
        slopes = (centered_x[:, None] * values).sum(axis=0) / np.square(centered_x).sum()
        payload = {
            "checkpoint": np.int64(restored), "n_states": np.int32(n_states),
            "level_values": values.astype(np.float32), "per_state_slopes": slopes.astype(np.float32),
            "success_rate": np.float64(np.mean(slopes < 0)),
            "mean_slope": np.float64(slopes.mean()), "median_slope": np.float64(np.median(slopes)),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(output)


def _functional_success_by_step() -> dict[int, float]:
    result = {}
    with (FUNCTIONAL_DIR / "value_policy_metrics.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["group"] == "stone":
                result[int(row["checkpoint"])] = float(row["fraction_value_interaction_positive"])
    return result


def _first_sustained(steps: np.ndarray, success: np.ndarray, threshold=.5, persistence=5):
    for index in range(len(success) - persistence + 1):
        if np.all(success[index:index + persistence] > threshold):
            return int(steps[index])
    return None


def _aggregate(args, steps: list[int], bank: dict) -> None:
    out_dir = Path(args.out_dir)
    missing = [step for step in steps if not _output_path(out_dir, step).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} checkpoints; first={missing[:5]}")
    functional = _functional_success_by_step()
    item_names = bank["level_added_item"].astype(str)[PRIMARY_LEVEL_START:PRIMARY_LEVEL_STOP]
    rows = []
    for step in steps:
        with np.load(_output_path(out_dir, step)) as data:
            level_values = np.asarray(data["level_values"])
            row = {
                "checkpoint": step, "n_states": int(data["n_states"]),
                "progression_slope_negative_rate": float(data["success_rate"]),
                "mean_per_state_slope": float(data["mean_slope"]),
                "median_per_state_slope": float(data["median_slope"]),
                "wood_pickaxe_value_interaction_success": functional.get(step, float("nan")),
            }
            for level_index, item in enumerate(item_names):
                row[f"mean_value_{item}"] = float(level_values[level_index].mean())
            rows.append(row)
    with (out_dir / "metrics_over_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    x = np.asarray([row["checkpoint"] for row in rows], dtype=float)
    y = np.asarray([row["progression_slope_negative_rate"] for row in rows], dtype=float)
    functional_y = np.asarray([row["wood_pickaxe_value_interaction_success"] for row in rows], dtype=float)
    emergence = _first_sustained(x, y)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(x, 100 * y, color="#b91c1c", linewidth=1.8, label="Negative progression slope")
    ax.axhline(50, color="#6b7280", linestyle="--", linewidth=1, label="50% reference")
    if emergence is not None:
        ax.axvline(emergence, color="#111827", linestyle=":", linewidth=1.2,
                   label=f"First sustained >50%: {emergence:,}")
    ax.set_xscale("log"); ax.set_ylim(0, 100)
    ax.set_xlabel("Training environment steps"); ax.set_ylabel("States with negative slope (%)")
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False)
    fig.savefig(out_dir / "inventory_heuristic_success.png", dpi=300)
    fig.savefig(out_dir / "inventory_heuristic_success.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(x, 100 * y, color="#b91c1c", linewidth=1.8, label="Inventory-progression heuristic")
    ax.plot(x, 100 * functional_y, color="#2563eb", linewidth=1.8,
            label="Wood-pickaxe value interaction")
    ax.axhline(50, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xscale("log"); ax.set_ylim(0, 100)
    ax.set_xlabel("Training environment steps"); ax.set_ylabel("Success rate (%)")
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False)
    fig.savefig(out_dir / "inventory_vs_pickaxe_success.png", dpi=300)
    fig.savefig(out_dir / "inventory_vs_pickaxe_success.pdf"); plt.close(fig)

    summary = {
        "n_states": int(rows[0]["n_states"]), "n_checkpoints": len(steps),
        "checkpoint_stride": args.checkpoint_stride,
        "final_checkpoint_forced_inclusion": True,
        "primary_levels": item_names.tolist(),
        "metric": "P(per-state OLS slope across primary cumulative inventory levels < 0)",
        "emergence_rule": "first of five consecutive evaluated checkpoints above 50%",
        "emergence_checkpoint": emergence,
        "final_success_rate": float(y[-1]), "peak_success_rate": float(y.max()),
        "peak_checkpoint": int(x[np.argmax(y)]),
    }
    (out_dir / "temporal_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "sweep_metadata.json").write_text(json.dumps({
        "run_path": args.run_path, "bank": args.bank, "selected_checkpoints": steps,
        "same_frozen_states_at_all_checkpoints": True,
    }, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN); parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--out_dir", default=DEFAULT_OUT); parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--checkpoint_stride", type=int, default=2)
    parser.add_argument("--worker_index", type=int, default=0); parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--checkpoints", default=""); parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    steps = _selected_steps(args.run_path, args.checkpoint_stride, args.checkpoints)
    bank = _load_bank(Path(args.bank)); Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(args, steps, bank)
    else:
        _run_worker(args, steps, bank)


if __name__ == "__main__":
    main()

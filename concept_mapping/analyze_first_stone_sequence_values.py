"""Evaluate natural and no-wood-pickaxe values around first stone collection."""

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
from concept_mapping.render_ppo_episodes import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_DIM


DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_CHECKPOINT = 1_499_987_968
DEFAULT_SEQUENCE_DIR = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "first_collect_stone_preceding_states_10episodes"
)
WOOD_PICKAXE_OFFSET = MAP_DIM + INV_NAMES.index("wood_pickaxe")


def load_selection(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty selection file: {path}")
    return rows


def load_windows(selection: list[dict]):
    observations, records = [], []
    for selected in selection:
        episode_id = int(selected["episode_id"])
        event_step = int(selected["event_step_t"])
        with np.load(selected["source_npz"]) as episode:
            episode_observations = np.asarray(episode["observations"], dtype=np.float32)
            terminal = np.asarray(episode["terminal_observation"], dtype=np.float32)
        for absolute_step in range(event_step - 10, event_step + 2):
            if absolute_step < len(episode_observations):
                observation = episode_observations[absolute_step]
            elif absolute_step == len(episode_observations):
                observation = terminal
            else:
                raise RuntimeError(f"Episode {episode_id} lacks step {absolute_step}")
            observations.append(observation)
            records.append({
                "episode_id": episode_id,
                "event_step_t": event_step,
                "absolute_step": absolute_step,
                "relative_step": absolute_step - event_step,
                "wood_pickaxe_count": float(observation[WOOD_PICKAXE_OFFSET] * 10.0),
            })
    return np.asarray(observations, dtype=np.float32), records


def evaluate(params, network, observations: np.ndarray, batch_size: int) -> np.ndarray:
    @jax.jit
    def apply_batch(parameters, batch):
        _policy, value = network.apply(parameters, batch)
        return value
    chunks = []
    for start in range(0, len(observations), batch_size):
        chunks.append(np.asarray(apply_batch(
            params, jnp.asarray(observations[start:start + batch_size])
        )))
    return np.concatenate(chunks).reshape(-1)


def write_values(path: Path, records: list[dict], natural: np.ndarray,
                 counterfactual: np.ndarray) -> None:
    fields = [
        "episode_id", "event_step_t", "absolute_step", "relative_step",
        "wood_pickaxe_count", "V_natural", "V_no_wood_pickaxe",
        "wood_pickaxe_value_effect",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            has_pickaxe = record["wood_pickaxe_count"] > 0
            writer.writerow({
                **record,
                "V_natural": float(natural[index]),
                "V_no_wood_pickaxe": float(counterfactual[index]) if has_pickaxe else "",
                "wood_pickaxe_value_effect": (
                    float(natural[index] - counterfactual[index]) if has_pickaxe else ""
                ),
            })
    temporary.replace(path)


def plot_values(out_dir: Path, records: list[dict], natural: np.ndarray,
                counterfactual: np.ndarray) -> None:
    episodes = list(dict.fromkeys(record["episode_id"] for record in records))
    fig, axes = plt.subplots(5, 2, figsize=(13, 17), sharex=True,
                             constrained_layout=True)
    axes = axes.ravel()
    for ax, episode_id in zip(axes, episodes):
        indices = np.asarray([
            index for index, record in enumerate(records)
            if record["episode_id"] == episode_id
        ])
        x = np.asarray([records[index]["relative_step"] for index in indices])
        has_pickaxe = np.asarray([
            records[index]["wood_pickaxe_count"] > 0 for index in indices
        ])
        natural_values = natural[indices]
        no_pickaxe_values = np.full(len(indices), np.nan)
        no_pickaxe_values[has_pickaxe] = counterfactual[indices][has_pickaxe]
        ax.plot(x, natural_values, marker="o", color="#2563eb", linewidth=2,
                label="Natural state")
        ax.plot(x, no_pickaxe_values, marker="o", color="#d97706", linewidth=2,
                linestyle="--", label="Wood pickaxe removed")
        if np.any(has_pickaxe):
            first = x[np.flatnonzero(has_pickaxe)[0]]
            ax.axvline(first, color="#7c3aed", linestyle=":", linewidth=1.2,
                       label="Pickaxe first present")
        ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.2,
                   label="First stone action")
        ax.axvline(1, color="#059669", linestyle="--", linewidth=1.2,
                   label="Post-collection")
        ax.set_title(f"Episode {episode_id}")
        ax.set_ylabel("Critic value")
        ax.grid(alpha=.22)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    for ax in axes[-2:]:
        ax.set_xlabel("Step relative to first Collect Stone action")
        ax.set_xticks(np.arange(-10, 2, 2))
    fig.suptitle(
        "Critic values around first stone collection\n"
        "Counterfactual evaluated only when the natural state contains a wood pickaxe",
        fontsize=16,
    )
    fig.savefig(out_dir / "ten_first_stone_value_trajectories.png", dpi=300)
    fig.savefig(out_dir / "ten_first_stone_value_trajectories.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(13, 16), sharex=True,
                             constrained_layout=True)
    axes = axes.ravel()
    for ax, episode_id in zip(axes, episodes):
        indices = np.asarray([
            index for index, record in enumerate(records)
            if record["episode_id"] == episode_id
        ])
        x = np.asarray([records[index]["relative_step"] for index in indices])
        has_pickaxe = np.asarray([
            records[index]["wood_pickaxe_count"] > 0 for index in indices
        ])
        effect = natural[indices] - counterfactual[indices]
        ax.plot(x[has_pickaxe], effect[has_pickaxe], marker="o", color="#7c3aed",
                linewidth=2)
        ax.axhline(0, color="#475569", linestyle="--", linewidth=1)
        ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.2)
        ax.axvline(1, color="#059669", linestyle="--", linewidth=1.2)
        ax.set_title(f"Episode {episode_id}")
        ax.set_ylabel(r"$V(s)-V(s_{-pickaxe})$")
        ax.grid(alpha=.22)
    for ax in axes[-2:]:
        ax.set_xlabel("Step relative to first Collect Stone action")
        ax.set_xticks(np.arange(-10, 2, 2))
    fig.suptitle("Wood-pickaxe counterfactual value effect across the ten sequences",
                 fontsize=16)
    fig.savefig(out_dir / "ten_first_stone_pickaxe_effects.png", dpi=300)
    fig.savefig(out_dir / "ten_first_stone_pickaxe_effects.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence_dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selection = load_selection(args.sequence_dir / "selection.csv")
    observations, records = load_windows(selection)

    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, restored = _restore_train_state(
        args.run_path, config, network, env, env_params, args.checkpoint
    )
    if restored != args.checkpoint:
        raise RuntimeError(f"Requested checkpoint {args.checkpoint}, restored {restored}")

    natural_values = evaluate(train_state.params, network, observations, args.batch_size)
    no_pickaxe_observations = observations.copy()
    no_pickaxe_observations[:, WOOD_PICKAXE_OFFSET] = 0.0
    no_pickaxe_values = evaluate(
        train_state.params, network, no_pickaxe_observations, args.batch_size
    )
    # Where no pickaxe was naturally present, the edit is the identity and must
    # reproduce the natural value exactly (up to float32 batching arithmetic).
    absent = np.asarray([record["wood_pickaxe_count"] <= 0 for record in records])
    identity_error = float(np.max(np.abs(
        natural_values[absent] - no_pickaxe_values[absent]
    ))) if np.any(absent) else 0.0
    if identity_error > 1e-6:
        raise RuntimeError(f"Identity intervention value mismatch: {identity_error}")

    output_csv = args.sequence_dir / "natural_and_no_pickaxe_values.csv"
    write_values(output_csv, records, natural_values, no_pickaxe_values)
    plot_values(args.sequence_dir, records, natural_values, no_pickaxe_values)
    metadata = {
        "run_path": args.run_path, "checkpoint": int(restored),
        "selection_csv": str(args.sequence_dir / "selection.csv"),
        "n_episodes": len(selection), "states_per_episode": 12,
        "relative_steps": list(range(-10, 2)),
        "counterfactual": "Set only wood_pickaxe observation coordinate to zero",
        "counterfactual_reported_only_if_naturally_present": True,
        "max_identity_intervention_value_error": identity_error,
    }
    (args.sequence_dir / "value_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Evaluated {len(records)} states at checkpoint {restored:,}")
    print(f"Natural states with wood pickaxe: {int((~absent).sum())}/{len(records)}")
    print(output_csv)


if __name__ == "__main__":
    main()

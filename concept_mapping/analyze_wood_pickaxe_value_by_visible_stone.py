"""Test whether visible stone moderates the critic value of a wood pickaxe.

From each frozen-policy episode, select the natural observation immediately
after the first successful Make Wood Pickaxe action, requiring that Collect
Stone is still incomplete.  Toggle only the wood-pickaxe inventory coordinate:

    s_plus  = natural post-crafting state with its wood pickaxe
    s_minus = the same observation with wood_pickaxe set to zero
    delta_V = V(s_plus) - V(s_minus)

Then compare delta_V between states with versus without a visible STONE tile in
the 7x9 symbolic observation.  Only the requested frozen checkpoint is evaluated.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from craftax.craftax_classic.constants import Achievement, Action, BlockType, OBS_DIM
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_CHANNELS, MAP_DIM


DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_CHECKPOINT = 1_499_987_968
DEFAULT_EPISODES = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/episodes"
)
DEFAULT_OUT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "wood_pickaxe_value_by_visible_stone_final_1p5b"
)
WOOD_PICKAXE_OFFSET = MAP_DIM + INV_NAMES.index("wood_pickaxe")


def extract_states(episodes_dir: Path) -> dict[str, np.ndarray]:
    records = []
    for path in sorted(episodes_dir.glob("episode_*.npz")):
        with np.load(path) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            achievements = np.asarray(data["achievements"])
            actions = np.asarray(data["actions"])
            terminal_observation = np.asarray(data["terminal_observation"], dtype=np.float32)
            terminal_achievements = np.asarray(data["terminal_achievements"])
            episode_id = int(data["episode_id"])
        post_achievements = np.concatenate(
            [achievements[1:], terminal_achievements[None]], axis=0
        )
        hits = np.flatnonzero(
            (actions == Action.MAKE_WOOD_PICKAXE.value)
            & (achievements[:, Achievement.MAKE_WOOD_PICKAXE.value] == 0)
            & (post_achievements[:, Achievement.MAKE_WOOD_PICKAXE.value] > 0)
        )
        if not len(hits):
            continue
        action_step = int(hits[0])
        if post_achievements[action_step, Achievement.COLLECT_STONE.value] > 0:
            continue
        if action_step + 1 < len(observations):
            natural = observations[action_step + 1]
            state_step = action_step + 1
        else:
            natural = terminal_observation
            state_step = len(observations)
        blocks = np.argmax(
            natural[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)[..., :len(BlockType)],
            axis=-1,
        )
        stone_positions = np.argwhere(blocks == BlockType.STONE.value)
        center = np.asarray(OBS_DIM) // 2
        distances = (
            np.abs(stone_positions - center).sum(axis=1)
            if len(stone_positions) else np.asarray([], dtype=int)
        )
        direction = int(np.argmax(natural[MAP_DIM + 16:MAP_DIM + 20]))
        direction_offsets = ((0, -1), (0, 1), (-1, 0), (1, 0))
        dr, dc = direction_offsets[direction]
        front = (int(center[0] + dr), int(center[1] + dc))
        if natural[WOOD_PICKAXE_OFFSET] <= 0:
            raise RuntimeError(f"Episode {episode_id} post-crafting state lacks a wood pickaxe")
        no_pickaxe = natural.copy()
        no_pickaxe[WOOD_PICKAXE_OFFSET] = 0.0
        changed = np.flatnonzero(natural != no_pickaxe)
        if changed.tolist() != [WOOD_PICKAXE_OFFSET]:
            raise RuntimeError(f"Episode {episode_id} counterfactual is not a one-slot edit")
        records.append({
            "episode_id": episode_id,
            "action_step": action_step,
            "state_step": state_step,
            "source_npz": str(path),
            "s_plus": natural,
            "s_minus": no_pickaxe,
            "stone_visible": bool(len(stone_positions)),
            "n_visible_stone": int(len(stone_positions)),
            "nearest_stone_manhattan": int(distances.min()) if len(distances) else -1,
            "stone_directly_actionable": bool(blocks[front] == BlockType.STONE.value),
            "natural_wood_pickaxe_observation_value": float(natural[WOOD_PICKAXE_OFFSET]),
        })
    if not records:
        raise RuntimeError(f"No eligible post-crafting states found in {episodes_dir}")
    keys = records[0].keys()
    return {key: np.asarray([record[key] for record in records]) for key in keys}


def evaluate_values(params, network, observations: np.ndarray, batch_size: int) -> np.ndarray:
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


def group_stats(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean_delta_V": float(values.mean()),
        "median_delta_V": float(np.median(values)),
        "std_delta_V": float(values.std(ddof=1)),
        "fraction_delta_V_positive": float(np.mean(values > 0)),
    }


def bootstrap_contrast(delta: np.ndarray, visible: np.ndarray, draws: int, seed: int) -> dict:
    with_stone = delta[visible]
    without_stone = delta[~visible]
    rng = np.random.default_rng(seed)
    contrasts = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        visible_draw = rng.choice(with_stone, size=len(with_stone), replace=True)
        absent_draw = rng.choice(without_stone, size=len(without_stone), replace=True)
        contrasts[draw] = visible_draw.mean() - absent_draw.mean()
    observed = float(with_stone.mean() - without_stone.mean())
    return {
        "estimand": "E[delta_V | stone visible] - E[delta_V | no stone visible]",
        "difference_in_mean_delta_V": observed,
        "bootstrap_95ci": [float(x) for x in np.quantile(contrasts, [.025, .975])],
        "bootstrap_probability_difference_positive": float(np.mean(contrasts > 0)),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def write_per_state(path: Path, bank: dict, value_plus, value_minus, delta) -> None:
    fields = [
        "episode_id", "action_step", "state_step", "source_npz", "stone_visible",
        "n_visible_stone", "nearest_stone_manhattan", "stone_directly_actionable",
        "natural_wood_pickaxe_observation_value", "V_s_plus", "V_s_minus", "delta_V",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(delta)):
            writer.writerow({
                **{field: bank[field][index] for field in fields[:-3]},
                "V_s_plus": float(value_plus[index]),
                "V_s_minus": float(value_minus[index]),
                "delta_V": float(delta[index]),
            })
    temporary.replace(path)


def plot_results(out_dir: Path, delta: np.ndarray, visible: np.ndarray,
                 summary: dict) -> None:
    with_stone = delta[visible]
    without_stone = delta[~visible]
    colors = ["#64748b", "#d97706"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    ax = axes[0]
    parts = ax.violinplot([without_stone, with_stone], positions=[0, 1],
                          showmeans=True, showmedians=True, widths=.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color); body.set_alpha(.65)
    ax.axhline(0, color="#475569", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], [f"No stone visible\n(n={len(without_stone)})",
                           f"Stone visible\n(n={len(with_stone)})"])
    ax.set_ylabel(r"$\Delta V=V(s^+)-V(s^-)$")
    ax.set_title("Wood-pickaxe value effect by visual context")
    ax.grid(axis="y", alpha=.22)

    ax = axes[1]
    means = [without_stone.mean(), with_stone.mean()]
    ses = [without_stone.std(ddof=1) / np.sqrt(len(without_stone)),
           with_stone.std(ddof=1) / np.sqrt(len(with_stone))]
    ax.bar([0, 1], means, yerr=1.96 * np.asarray(ses), color=colors, capsize=4)
    ax.axhline(0, color="#475569", linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], ["No stone visible", "Stone visible"])
    ax.set_ylabel(r"Mean $\Delta V$ (95% normal CI)")
    contrast = summary["contextual_contrast"]
    low, high = contrast["bootstrap_95ci"]
    ax.set_title(
        f"Visible − not visible = {contrast['difference_in_mean_delta_V']:+.3f}\n"
        f"episode bootstrap 95% CI [{low:+.3f}, {high:+.3f}]"
    )
    ax.grid(axis="y", alpha=.22)
    fig.suptitle("Does the critic value a new wood pickaxe more when stone is visible?",
                 fontsize=15)
    fig.savefig(out_dir / "wood_pickaxe_delta_value_by_visible_stone.png", dpi=300)
    fig.savefig(out_dir / "wood_pickaxe_delta_value_by_visible_stone.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes_dir", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.bootstrap_draws <= 0:
        raise ValueError("batch_size and bootstrap_draws must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank = extract_states(args.episodes_dir)

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
    n = len(bank["s_plus"])
    packed = np.concatenate([bank["s_plus"], bank["s_minus"]]).astype(np.float32)
    evaluated = evaluate_values(train_state.params, network, packed, args.batch_size)
    value_plus, value_minus = np.split(evaluated, [n])
    delta = value_plus - value_minus
    visible = bank["stone_visible"].astype(bool)

    summary = {
        "checkpoint": int(restored),
        "eligible_states": int(n),
        "groups": {
            "stone_visible": group_stats(delta[visible]),
            "no_stone_visible": group_stats(delta[~visible]),
        },
        "contextual_contrast": bootstrap_contrast(
            delta, visible, args.bootstrap_draws, args.bootstrap_seed
        ),
        "directly_actionable_stone_diagnostic": {
            "n": int(bank["stone_directly_actionable"].sum()),
            "definition": "STONE occupies the single agent-facing action cell",
        },
    }
    per_state_path = args.out_dir / "wood_pickaxe_value_by_visible_stone_per_state.csv"
    write_per_state(per_state_path, bank, value_plus, value_minus, delta)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata = {
        "run_path": args.run_path,
        "environment": config["ENV_NAME"],
        "checkpoint": int(restored),
        "source_episodes_dir": str(args.episodes_dir),
        "n_episode_files": len(list(args.episodes_dir.glob("episode_*.npz"))),
        "cohort": (
            "Observation immediately after first successful Make Wood Pickaxe, "
            "with Collect Stone still incomplete"
        ),
        "stone_visible": "At least one BlockType.STONE tile anywhere in the 7x9 observation",
        "intervention": "Set only the wood_pickaxe observation coordinate to zero",
        "delta_V": "V(natural state with wood pickaxe) - V(same state with wood pickaxe zero)",
        "primary_contrast": "mean delta_V(stone visible) - mean delta_V(no stone visible)",
        "source_file_listing_sha256": hashlib.sha256("\n".join(
            str(path) for path in sorted(args.episodes_dir.glob("episode_*.npz"))
        ).encode()).hexdigest(),
        "caveat": (
            "The inventory edit is controlled within state, but visible-stone group membership "
            "is naturally occurring rather than randomized; the two groups are also imbalanced."
        ),
    }
    (args.out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    plot_results(args.out_dir, delta, visible, summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(per_state_path)


if __name__ == "__main__":
    main()

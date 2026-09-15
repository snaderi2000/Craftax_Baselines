"""Track stabilized achievement-completion value effects over PPO training.

For every third checkpoint by default, this script evaluates the same natural
before/after first-achievement transition pairs from the frozen 1,000-episode
master dataset.  It saves one resumable NPZ per checkpoint and aggregates the
results into a compact small-multiples figure.

The metric for achievement c at checkpoint t is

  R_c(t) = median_i[(V_after - V_before) / (1 + abs(V_before))].

Natural before/after transitions are observational: the transition includes
all consequences of the achievement-producing action, not only inventory.
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

from craftax.craftax_classic.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.plot_diamond_ordering_over_checkpoints import _checkpoint_steps
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


MASTER = Path("concept_mapping/runs/master_symbolic_1p5b_1000episodes")
DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_OUT = MASTER / "achievement_stabilized_effect_over_checkpoints_stride3"
DEFAULT_BANK = MASTER / "achievement_transition_fixed_bank.npz"
SELECTED = (
    "collect_wood", "place_table", "make_wood_pickaxe", "make_wood_sword",
    "collect_stone", "make_stone_sword", "make_stone_pickaxe", "place_furnace",
    "collect_coal", "collect_iron", "make_iron_pickaxe", "make_iron_sword",
    "collect_diamond",
)


def selected_steps(run_path: str, stride: int) -> list[int]:
    all_steps = _checkpoint_steps(run_path)
    steps = all_steps[::stride]
    if steps[-1] != all_steps[-1]:
        steps.append(all_steps[-1])
    return steps


def build_or_load_bank(path: Path) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path) as data:
            return {key: np.asarray(data[key]) for key in data.files}

    achievement_names = [item.name.lower() for item in Achievement]
    selected_indices = [achievement_names.index(name) for name in SELECTED]
    before, after, concept_index, episode_ids, completion_steps = [], [], [], [], []
    for episode_path in sorted((MASTER / "episodes").glob("episode_*.npz")):
        with np.load(episode_path) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            achievements = np.concatenate([
                np.asarray(data["achievements"]),
                np.asarray(data["terminal_achievements"])[None],
            ])
            terminal_observation = np.asarray(data["terminal_observation"], dtype=np.float32)
            episode_id = int(data["episode_id"])
        for concept, achievement_index in enumerate(selected_indices):
            hits = np.flatnonzero(achievements[:, achievement_index] > 0)
            if not len(hits) or hits[0] == 0:
                continue
            after_index = int(hits[0])
            before.append(observations[after_index - 1])
            after.append(observations[after_index] if after_index < len(observations)
                         else terminal_observation)
            concept_index.append(concept)
            episode_ids.append(episode_id)
            completion_steps.append(after_index)

    payload = {
        "before": np.asarray(before, dtype=np.float32),
        "after": np.asarray(after, dtype=np.float32),
        "concept_index": np.asarray(concept_index, dtype=np.int16),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "completion_step": np.asarray(completion_steps, dtype=np.int32),
        "concept_names": np.asarray(SELECTED),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)
    return payload


def output_path(out_dir: Path, checkpoint: int) -> Path:
    return out_dir / "per_checkpoint" / f"checkpoint_{checkpoint}.npz"


def run_worker(args, steps: list[int], bank: dict[str, np.ndarray]) -> None:
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    packed = np.concatenate([bank["before"], bank["after"]]).astype(np.float32)
    n_pairs = len(bank["before"])

    @jax.jit
    def value_batch(params, observations):
        _policy, value = network.apply(params, observations)
        return value

    worker_steps = [step for index, step in enumerate(steps)
                    if index % args.num_workers == args.worker_index]
    print(f"worker {args.worker_index}/{args.num_workers}: {len(worker_steps)} checkpoints", flush=True)
    for step in worker_steps:
        destination = output_path(Path(args.out_dir), step)
        if destination.exists():
            continue
        print(f"worker {args.worker_index}: evaluating {step:,}", flush=True)
        train_state, restored = _restore_train_state(
            args.run_path, config, network, env, env_params, step
        )
        chunks = []
        for start in range(0, len(packed), args.batch_size):
            chunks.append(np.asarray(value_batch(
                train_state.params, jnp.asarray(packed[start:start + args.batch_size])
            )))
        before_values, after_values = np.split(np.concatenate(chunks), [n_pairs])
        effects = (after_values - before_values) / (1.0 + np.abs(before_values))
        medians = np.asarray([
            np.median(effects[bank["concept_index"] == concept])
            for concept in range(len(SELECTED))
        ], dtype=np.float64)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, checkpoint=np.int64(restored), median_effect=medians,
                effects=effects.astype(np.float32),
                value_before=before_values.astype(np.float32),
                value_after=after_values.astype(np.float32),
            )
        temporary.replace(destination)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    radius = window // 2
    for index in range(radius, len(values) - radius):
        result[index] = np.median(values[index - radius:index + radius + 1])
    return result


def aggregate(args, steps: list[int], bank: dict[str, np.ndarray]) -> None:
    out_dir = Path(args.out_dir)
    missing = [step for step in steps if not output_path(out_dir, step).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} checkpoint outputs; first: {missing[:5]}")

    order_csv = MASTER / "achievement_transition_relative_value_1p5b/achievement_summary_and_order.csv"
    with order_csv.open(newline="") as handle:
        empirical_names = [row["achievement"] for row in csv.DictReader(handle)]
    concept_lookup = {name: index for index, name in enumerate(SELECTED)}
    order = np.asarray([concept_lookup[name] for name in empirical_names])

    metrics = []
    for step in steps:
        with np.load(output_path(out_dir, step)) as data:
            medians = np.asarray(data["median_effect"])
        row = {"checkpoint": step}
        row.update({f"R_{name}": float(medians[index]) for index, name in enumerate(SELECTED)})
        metrics.append(row)
    with (out_dir / "achievement_stabilized_effect_over_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader(); writer.writerows(metrics)

    x = np.asarray(steps, dtype=float)
    matrix = np.asarray([[row[f"R_{name}"] for name in SELECTED] for row in metrics]).T
    panels = matrix[order] * 100.0
    ordered_names = [SELECTED[index] for index in order]

    # A compact, centered 5+4+4 layout uses a shared scale so relative effect
    # sizes remain visually comparable and the figure fits a landscape slide.
    finite = panels[np.isfinite(panels)]
    bound = max(10.0, float(np.ceil(np.quantile(np.abs(finite), .995) / 10.0) * 10.0))
    fig = plt.figure(figsize=(16.0, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 10)
    slots = (
        [(0, column) for column in (0, 2, 4, 6, 8)]
        + [(1, column) for column in (1, 3, 5, 7)]
        + [(2, column) for column in (1, 3, 5, 7)]
    )
    axes = []
    for row, column in slots:
        axes.append(fig.add_subplot(
            grid[row, column:column + 2],
            sharex=axes[0] if axes else None,
            sharey=axes[0] if axes else None,
        ))
    for panel, (ax, name) in enumerate(zip(axes, ordered_names)):
        raw = panels[panel]
        smooth = rolling_median(raw, args.smooth_window)
        ax.plot(x, raw, color="#93c5fd", linewidth=.75, alpha=.55)
        ax.plot(x, smooth, color="#1d4ed8", linewidth=1.8)
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=.75)
        ax.set_xscale("log")
        ax.set_ylim(-bound, bound)
        ax.set_title(name.replace("_", " ").title(), fontsize=11, fontweight="semibold")
        ax.grid(True, which="both", alpha=.18)
    fig.supxlabel("Training environment steps", fontsize=13)
    fig.supylabel("Median stabilized relative value change, $R_c(t)$ (%)", fontsize=13)
    fig.suptitle("Achievement-completion value effects over training", fontsize=17)
    fig.savefig(out_dir / "achievement_stabilized_effect_small_multiples.png", dpi=300)
    fig.savefig(out_dir / "achievement_stabilized_effect_small_multiples.pdf")
    plt.close(fig)

    # A heatmap is useful when the individual curves overlap or fluctuate.
    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)
    image = ax.imshow(panels, aspect="auto", interpolation="nearest", cmap="RdBu_r",
                      vmin=-bound, vmax=bound,
                      extent=[np.log10(x[0]), np.log10(x[-1]), len(panels)-.5, -.5])
    tick_powers = np.arange(np.ceil(np.log10(x[0])), np.floor(np.log10(x[-1])) + 1)
    ax.set_xticks(tick_powers, [rf"$10^{{{int(power)}}}$" for power in tick_powers])
    ax.set_yticks(np.arange(len(ordered_names)),
                  [name.replace("_", " ").title() for name in ordered_names])
    ax.set_xlabel("Training environment steps")
    ax.set_title("Achievement-completion value effects over training")
    colorbar = fig.colorbar(image, ax=ax, pad=.015)
    colorbar.set_label(r"$R_c(t)$ (%)")
    fig.savefig(out_dir / "achievement_stabilized_effect_heatmap.png", dpi=300)
    fig.savefig(out_dir / "achievement_stabilized_effect_heatmap.pdf")
    plt.close(fig)

    (out_dir / "sweep_metadata.json").write_text(json.dumps({
        "run_path": args.run_path,
        "n_checkpoints": len(steps),
        "checkpoint_stride": args.checkpoint_stride,
        "final_checkpoint_forced_inclusion": True,
        "n_fixed_transitions": int(len(bank["before"])),
        "metric": "median[(V_after - V_before) / (1 + abs(V_before))]",
        "smooth_curve": f"centered rolling median over {args.smooth_window} evaluated checkpoints",
        "raw_curve_also_shown": True,
        "achievement_order": ordered_names,
        "interpretation_warning": "Natural before/after transitions are observational, not isolated counterfactuals.",
    }, indent=2) + "\n")
    print(f"Saved aggregate results for {len(steps)} checkpoints to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint_stride", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    steps = selected_steps(args.run_path, args.checkpoint_stride)
    bank = build_or_load_bank(args.bank)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args, steps, bank)
    else:
        run_worker(args, steps, bank)


if __name__ == "__main__":
    main()

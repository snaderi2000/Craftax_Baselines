"""Final-checkpoint 2x2 value interaction for stone and a wood pickaxe.

Uses the fixed, validated natural stone bank derived from the 1,000 frozen-policy
episodes.  For every natural state immediately before a successful stone
collection, the four conditions are

    s11: stone directly in front, wood pickaxe present (natural state)
    s01: stone directly in front, wood pickaxe absent
    s10: front stone replaced by PATH, wood pickaxe present
    s00: front stone replaced by PATH, wood pickaxe absent

and the per-state interaction is

    I = (V(s11) - V(s01)) - (V(s10) - V(s00)).

Only the requested frozen checkpoint is evaluated; no training occurs.
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

from craftax.craftax_classic.constants import BlockType, OBS_DIM
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_CHANNELS, MAP_DIM


DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_CHECKPOINT = 1_499_987_968
DEFAULT_BANK = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "hierarchy_concept_banks/wood_pickaxe_functional/wood_pickaxe_functional_bank.npz"
)
DEFAULT_OUT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "stone_wood_pickaxe_interaction_final_1p5b"
)
REFERENCE_RESULT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "hierarchy_concept_banks/wood_pickaxe_functional/checkpoint_sweep/"
    f"per_checkpoint/checkpoint_{DEFAULT_CHECKPOINT}.npz"
)
WOOD_PICKAXE_OFFSET = MAP_DIM + INV_NAMES.index("wood_pickaxe")


def load_and_validate_stone_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        bank = {key: np.asarray(handle[key]) for key in handle.files}
    mask = bank["resource"].astype(str) == "stone"
    result = {key: value[mask] for key, value in bank.items()}
    if not len(result["A"]):
        raise RuntimeError(f"No stone states in {path}")

    # Existing A/B/C/D notation maps to s00/s10/s01/s11, respectively.
    s00, s10, s01, s11 = (result[key].astype(np.float32) for key in "ABCD")
    n = len(s11)
    rows = result["target_row"].astype(int)
    cols = result["target_col"].astype(int)
    blocks_11 = np.argmax(
        s11[:, :MAP_DIM].reshape(n, *OBS_DIM, MAP_CHANNELS)[..., :len(BlockType)],
        axis=-1,
    )
    blocks_10 = np.argmax(
        s10[:, :MAP_DIM].reshape(n, *OBS_DIM, MAP_CHANNELS)[..., :len(BlockType)],
        axis=-1,
    )
    index = np.arange(n)
    if not np.all(blocks_11[index, rows, cols] == BlockType.STONE.value):
        raise RuntimeError("Some s11 states do not have stone in the action cell")
    if not np.all(blocks_10[index, rows, cols] == BlockType.PATH.value):
        raise RuntimeError("Some s10 states do not replace the action-cell stone with PATH")
    if not np.all(s11[:, WOOD_PICKAXE_OFFSET] > 0):
        raise RuntimeError("Some natural s11 states lack a wood pickaxe")
    if not np.all(s10[:, WOOD_PICKAXE_OFFSET] > 0):
        raise RuntimeError("Some s10 controls lack a wood pickaxe")
    if not np.all(s01[:, WOOD_PICKAXE_OFFSET] == 0):
        raise RuntimeError("Some s01 controls retain a wood pickaxe")
    if not np.all(s00[:, WOOD_PICKAXE_OFFSET] == 0):
        raise RuntimeError("Some s00 controls retain a wood pickaxe")

    def changed_coordinates(left, right):
        return np.any(left != right, axis=0)

    # The pickaxe edit changes exactly one observation coordinate.
    expected_pickaxe = np.zeros(s11.shape[1], dtype=bool)
    expected_pickaxe[WOOD_PICKAXE_OFFSET] = True
    if not np.array_equal(changed_coordinates(s11, s01), expected_pickaxe):
        raise RuntimeError("s11 -> s01 is not a pure wood-pickaxe-slot intervention")
    if not np.array_equal(changed_coordinates(s10, s00), expected_pickaxe):
        raise RuntimeError("s10 -> s00 is not a pure wood-pickaxe-slot intervention")
    # The resource intervention must be identical with and without the tool.
    if not np.array_equal(s11 - s10, s01 - s00):
        raise RuntimeError("Stone intervention differs across pickaxe conditions")

    result.update({"s00": s00, "s10": s10, "s01": s01, "s11": s11})
    return result


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


def bootstrap_summary(interaction: np.ndarray, episode_id: np.ndarray,
                      draws: int, seed: int) -> dict:
    unique = np.unique(episode_id)
    groups = {episode: interaction[episode_id == episode] for episode in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    fractions = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([groups[episode] for episode in sampled])
        means[draw] = values.mean()
        fractions[draw] = np.mean(values > 0)
    return {
        "n_states": int(len(interaction)),
        "n_episodes": int(len(unique)),
        "mean_I": float(interaction.mean()),
        "median_I": float(np.median(interaction)),
        "std_I": float(interaction.std(ddof=1)),
        "fraction_I_positive": float(np.mean(interaction > 0)),
        "cluster_bootstrap_95ci_mean_I": [
            float(x) for x in np.quantile(means, [.025, .975])
        ],
        "cluster_bootstrap_95ci_fraction_I_positive": [
            float(x) for x in np.quantile(fractions, [.025, .975])
        ],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def write_csv(path: Path, bank: dict, values: dict[str, np.ndarray],
              tool_effect_stone, tool_effect_no_stone, interaction) -> None:
    fields = [
        "sample_index", "episode_id", "timestep", "source_npz", "target_row",
        "target_col", "V_s11", "V_s01", "V_s10", "V_s00",
        "wood_pickaxe_effect_stone", "wood_pickaxe_effect_no_stone", "I",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(interaction)):
            writer.writerow({
                "sample_index": index,
                "episode_id": int(bank["episode_id"][index]),
                "timestep": int(bank["timestep"][index]),
                "source_npz": str(bank["source_npz"][index]),
                "target_row": int(bank["target_row"][index]),
                "target_col": int(bank["target_col"][index]),
                "V_s11": float(values["s11"][index]),
                "V_s01": float(values["s01"][index]),
                "V_s10": float(values["s10"][index]),
                "V_s00": float(values["s00"][index]),
                "wood_pickaxe_effect_stone": float(tool_effect_stone[index]),
                "wood_pickaxe_effect_no_stone": float(tool_effect_no_stone[index]),
                "I": float(interaction[index]),
            })
    temporary.replace(path)


def plot_results(out_dir: Path, values: dict[str, np.ndarray], interaction: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    ax = axes[0]
    labels = ["s00\nno stone\nno pickaxe", "s10\nno stone\npickaxe",
              "s01\nstone\nno pickaxe", "s11\nstone\npickaxe"]
    keys = ["s00", "s10", "s01", "s11"]
    means = np.asarray([values[key].mean() for key in keys])
    standard_errors = np.asarray([
        values[key].std(ddof=1) / np.sqrt(len(values[key])) for key in keys
    ])
    ax.bar(np.arange(4), means, yerr=1.96 * standard_errors, color=[
        "#cbd5e1", "#93c5fd", "#fbbf24", "#34d399"
    ], capsize=3)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Mean critic value (95% normal CI)")
    ax.set_title("Four counterfactual conditions")
    ax.grid(axis="y", alpha=.22)

    ax = axes[1]
    ax.hist(interaction, bins=50, color="#7c3aed", alpha=.82, edgecolor="white")
    ax.axvline(0, color="#475569", linestyle="--", linewidth=1.2, label="No interaction")
    ax.axvline(interaction.mean(), color="#dc2626", linewidth=2.0,
               label=f"Mean I = {interaction.mean():.3f}")
    ax.set_xlabel(r"$I=[V(s_{11})-V(s_{01})]-[V(s_{10})-V(s_{00})]$")
    ax.set_ylabel("Natural states")
    ax.set_title("State-level interaction distribution")
    ax.legend(frameon=False)
    fig.suptitle("Stone × wood-pickaxe value interaction at the final checkpoint", fontsize=15)
    fig.savefig(out_dir / "stone_wood_pickaxe_interaction.png", dpi=300)
    fig.savefig(out_dir / "stone_wood_pickaxe_interaction.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", type=int, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--bootstrap_draws", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.bootstrap_draws <= 0:
        raise ValueError("batch_size and bootstrap_draws must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank = load_and_validate_stone_bank(args.bank)

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

    order = ("s11", "s01", "s10", "s00")
    packed = np.concatenate([bank[key] for key in order])
    evaluated = evaluate_values(train_state.params, network, packed, args.batch_size)
    values = dict(zip(order, np.split(evaluated, len(order))))
    tool_effect_stone = values["s11"] - values["s01"]
    tool_effect_no_stone = values["s10"] - values["s00"]
    interaction = tool_effect_stone - tool_effect_no_stone

    reference_error = None
    reference = REFERENCE_RESULT.with_name(f"checkpoint_{restored}.npz")
    if reference.exists():
        with np.load(args.bank) as source, np.load(reference) as saved:
            # The focused bank has already been filtered, while the historical
            # checkpoint output contains both stone and coal.
            stone_mask = np.asarray(source["resource"]).astype(str) == "stone"
            expected = np.asarray(saved["interaction_value"])[stone_mask]
        reference_error = float(np.max(np.abs(interaction - expected)))
        # CPU and GPU XLA reductions can differ by a few float32 ULPs.
        if reference_error > 5e-5:
            raise RuntimeError(f"Existing checkpoint result differs by {reference_error}")

    summary = bootstrap_summary(
        interaction, bank["episode_id"], args.bootstrap_draws, args.bootstrap_seed
    )
    summary.update({
        "checkpoint": int(restored),
        "mean_V_s11": float(values["s11"].mean()),
        "mean_V_s01": float(values["s01"].mean()),
        "mean_V_s10": float(values["s10"].mean()),
        "mean_V_s00": float(values["s00"].mean()),
        "mean_wood_pickaxe_effect_stone": float(tool_effect_stone.mean()),
        "mean_wood_pickaxe_effect_no_stone": float(tool_effect_no_stone.mean()),
        "max_abs_error_vs_existing_sweep_result": reference_error,
    })
    csv_path = args.out_dir / "stone_wood_pickaxe_interaction_per_state.csv"
    write_csv(csv_path, bank, values, tool_effect_stone, tool_effect_no_stone, interaction)
    npz_path = args.out_dir / "stone_wood_pickaxe_interaction_values.npz"
    temporary = npz_path.with_name(f".{npz_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, checkpoint=np.int64(restored), episode_id=bank["episode_id"],
            timestep=bank["timestep"], **{f"V_{key}": value for key, value in values.items()},
            wood_pickaxe_effect_stone=tool_effect_stone,
            wood_pickaxe_effect_no_stone=tool_effect_no_stone, interaction=interaction,
        )
    temporary.replace(npz_path)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata = {
        "run_path": args.run_path,
        "environment": config["ENV_NAME"],
        "checkpoint": int(restored),
        "source_bank": str(args.bank),
        "source_bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        "source_dataset": "1,000 complete frozen-policy episodes",
        "selection": "natural state immediately before first successful stone collection",
        "conditions": {
            "s11": "stone directly in front + natural wood pickaxe count",
            "s01": "stone directly in front + wood pickaxe set to zero",
            "s10": "front stone replaced by PATH + natural wood pickaxe count",
            "s00": "front stone replaced by PATH + wood pickaxe set to zero",
        },
        "estimand": "I = [V(s11)-V(s01)] - [V(s10)-V(s00)]",
        "note": (
            "Actionable means the block in the agent-facing action cell. Other state "
            "features are fixed within each four-condition counterfactual quartet."
        ),
    }
    (args.out_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    plot_results(args.out_dir, values, interaction)
    print(json.dumps(summary, indent=2), flush=True)
    print(csv_path)
    print(npz_path)


if __name__ == "__main__":
    main()

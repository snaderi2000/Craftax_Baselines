"""Analyze relative critic-value changes at first achievement completion.

Uses natural before/after transitions from the 1,000-episode master bank and
the final 1.5B critic. These are observational transition effects, not
single-variable counterfactual effects.
"""

from __future__ import annotations

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
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


MASTER = Path("concept_mapping/runs/master_symbolic_1p5b_1000episodes")
OUT = MASTER / "achievement_transition_relative_value_1p5b"
RUN = "wandb/run-20260830_231509-616wuss8"
CHECKPOINT = 1_499_987_968
SELECTED = (
    "collect_wood", "place_table", "make_wood_pickaxe", "make_wood_sword",
    "collect_stone", "make_stone_sword", "make_stone_pickaxe", "place_furnace",
    "collect_coal", "collect_iron", "make_iron_pickaxe", "make_iron_sword",
    "collect_diamond",
)


def _bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, draws=5000):
    medians = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        count = min(250, draws - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        medians[start:start + count] = np.median(values[indices], axis=1)
    return np.quantile(medians, [0.025, 0.975])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    achievement_names = [x.name.lower() for x in Achievement]
    selected_indices = [achievement_names.index(name) for name in SELECTED]
    episode_files = sorted((MASTER / "episodes").glob("episode_*.npz"))
    first_steps = np.full((len(episode_files), len(SELECTED)), -1, dtype=np.int32)
    records, before_observations, after_observations = [], [], []

    for episode_row, path in enumerate(episode_files):
        with np.load(path) as data:
            episode_id = int(data["episode_id"])
            observations = np.asarray(data["observations"], dtype=np.float32)
            achievements = np.concatenate(
                [np.asarray(data["achievements"]), np.asarray(data["terminal_achievements"])[None]], axis=0
            )
            terminal_observation = np.asarray(data["terminal_observation"], dtype=np.float32)
        for selected_column, (name, achievement_index) in enumerate(zip(SELECTED, selected_indices)):
            hits = np.flatnonzero(achievements[:, achievement_index] > 0)
            if not len(hits) or hits[0] == 0:
                continue
            after_index = int(hits[0])
            before_index = after_index - 1
            before = observations[before_index]
            after = observations[after_index] if after_index < len(observations) else terminal_observation
            first_steps[episode_row, selected_column] = after_index
            records.append({
                "episode_id": episode_id, "achievement": name,
                "completion_step": after_index, "before_state_step": before_index,
                "terminal_transition": int(after_index == len(observations)),
            })
            before_observations.append(before); after_observations.append(after)

    before_observations = np.asarray(before_observations, dtype=np.float32)
    after_observations = np.asarray(after_observations, dtype=np.float32)
    config = _load_wandb_config(RUN); config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False); env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, restored = _restore_train_state(RUN, config, network, env, env_params, CHECKPOINT)

    @jax.jit
    def value_batch(params, observations):
        _policy, value = network.apply(params, observations)
        return value

    packed = np.concatenate([before_observations, after_observations])
    chunks = []
    for start in range(0, len(packed), 2048):
        chunks.append(np.asarray(value_batch(train_state.params, jnp.asarray(packed[start:start + 2048]))))
    before_values, after_values = np.split(np.concatenate(chunks), 2)
    delta = after_values - before_values
    denominator = np.abs(after_values) + np.abs(before_values)
    relative = np.divide(2 * delta, denominator, out=np.zeros_like(delta), where=denominator > 1e-8)
    for row, before, after, change, relative_change in zip(
        records, before_values, after_values, delta, relative
    ):
        row.update({"value_before": float(before), "value_after": float(after),
                    "delta_value": float(change), "symmetric_relative_change": float(relative_change),
                    "symmetric_relative_change_percent": float(100 * relative_change)})
    with (OUT / "achievement_transition_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    rng = np.random.default_rng(50)
    summary_rows = []
    for selected_column, name in enumerate(SELECTED):
        selected_records = [row for row in records if row["achievement"] == name]
        relative_values = np.asarray([row["symmetric_relative_change_percent"] for row in selected_records])
        steps = first_steps[:, selected_column]; observed = steps >= 0
        low, high = _bootstrap_median_ci(relative_values, rng)
        summary_rows.append({
            "achievement": name, "episodes_completed": int(observed.sum()),
            "completion_rate": float(observed.mean()),
            "median_first_completion_step": float(np.median(steps[observed])),
            "mean_first_completion_step": float(steps[observed].mean()),
            "median_relative_value_change_percent": float(np.median(relative_values)),
            "bootstrap_95ci_low": float(low), "bootstrap_95ci_high": float(high),
            "mean_relative_value_change_percent": float(relative_values.mean()),
            "median_raw_delta_value": float(np.median([row["delta_value"] for row in selected_records])),
        })
    summary_rows.sort(key=lambda row: (row["median_first_completion_step"], row["achievement"]))
    for rank, row in enumerate(summary_rows, start=1):
        row["empirical_order_rank"] = rank
    fields = ["empirical_order_rank", *[key for key in summary_rows[0] if key != "empirical_order_rank"]]
    with (OUT / "achievement_summary_and_order.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary_rows)

    # Conditional pairwise precedence counts make the empirical order reusable
    # without forcing a total order in episodes where an achievement is absent.
    with (OUT / "achievement_pairwise_precedence.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["achievement_a", "achievement_b", "episodes_with_both", "a_before_b",
                         "b_before_a", "ties", "fraction_a_before_b_given_both"])
        for a, name_a in enumerate(SELECTED):
            for b, name_b in enumerate(SELECTED):
                if a == b:
                    continue
                both = (first_steps[:, a] >= 0) & (first_steps[:, b] >= 0)
                av, bv = first_steps[both, a], first_steps[both, b]
                writer.writerow([name_a, name_b, int(both.sum()), int(np.sum(av < bv)),
                                 int(np.sum(bv < av)), int(np.sum(av == bv)),
                                 float(np.mean(av < bv)) if both.any() else ""])

    labels = [row["achievement"].replace("_", " ").title() for row in summary_rows]
    medians = np.asarray([row["median_relative_value_change_percent"] for row in summary_rows])
    low = np.asarray([row["bootstrap_95ci_low"] for row in summary_rows])
    high = np.asarray([row["bootstrap_95ci_high"] for row in summary_rows])
    y = np.arange(len(summary_rows))
    colors = np.where(medians < 0, "#c52420", "#198754")
    fig, ax = plt.subplots(figsize=(9.4, 7.0), constrained_layout=True)
    for index in range(len(y)):
        ax.errorbar(medians[index], y[index],
                    xerr=[[medians[index] - low[index]], [high[index] - medians[index]]],
                    fmt="o", color=colors[index], ecolor="#334155", markersize=8,
                    elinewidth=1.4, capsize=3)
    ax.axvline(0, color="#475569", linewidth=1.1)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    ax.set_xlabel("Median symmetric relative critic-value change (%)")
    ax.set_title("Relative critic-value change at first achievement completion")
    fig.savefig(OUT / "achievement_relative_value_effects.png", dpi=300)
    fig.savefig(OUT / "achievement_relative_value_effects.pdf"); plt.close(fig)

    metadata = {
        "run_path": RUN, "checkpoint": int(restored), "n_episodes": len(episode_files),
        "selected_achievements": list(SELECTED),
        "empirical_order": [row["achievement"] for row in summary_rows],
        "relative_effect": "2*(V_after-V_before)/(abs(V_after)+abs(V_before))",
        "aggregation": "median across first-completion transitions; episode bootstrap 95% CI",
        "interpretation_warning": (
            "Natural one-step before/after transitions change whatever the achievement action changes; "
            "these are not single-variable counterfactual effects."
        ),
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

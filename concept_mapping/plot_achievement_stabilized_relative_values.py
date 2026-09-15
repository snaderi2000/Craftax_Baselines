"""Plot stabilized relative critic changes at first achievement completion.

This is a lightweight re-analysis of the saved 1.5B before/after values, so it
does not load the PPO checkpoint or reevaluate observations.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN_DIR = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "achievement_transition_relative_value_1p5b"
)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 5000):
    medians = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        count = min(250, draws - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        medians[start:start + count] = np.median(values[indices], axis=1)
    return np.quantile(medians, [0.025, 0.975])


def main() -> None:
    with (RUN_DIR / "achievement_transition_values.csv").open(newline="") as handle:
        transitions = list(csv.DictReader(handle))
    with (RUN_DIR / "achievement_summary_and_order.csv").open(newline="") as handle:
        ordering = list(csv.DictReader(handle))

    # The +1 is a fixed stabilizer in critic-value units. Absolute value keeps
    # the direction meaningful when V_before is negative.
    for row in transitions:
        before = float(row["value_before"])
        after = float(row["value_after"])
        effect = (after - before) / (1.0 + abs(before))
        row["stabilized_relative_change"] = effect
        row["stabilized_relative_change_percent"] = 100.0 * effect

    transition_fields = list(transitions[0].keys())
    with (RUN_DIR / "achievement_transition_values_stabilized.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=transition_fields)
        writer.writeheader()
        writer.writerows(transitions)

    rng = np.random.default_rng(50)
    summary = []
    for order_row in ordering:
        name = order_row["achievement"]
        values = np.asarray([
            float(row["stabilized_relative_change_percent"])
            for row in transitions if row["achievement"] == name
        ])
        low, high = bootstrap_median_ci(values, rng)
        summary.append({
            "empirical_order_rank": int(order_row["empirical_order_rank"]),
            "achievement": name,
            "episodes_completed": int(order_row["episodes_completed"]),
            "median_first_completion_step": float(order_row["median_first_completion_step"]),
            "median_stabilized_relative_change_percent": float(np.median(values)),
            "bootstrap_95ci_low": float(low),
            "bootstrap_95ci_high": float(high),
            "mean_stabilized_relative_change_percent": float(np.mean(values)),
        })

    with (RUN_DIR / "achievement_stabilized_relative_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    labels = [row["achievement"].replace("_", " ").title() for row in summary]
    medians = np.asarray([row["median_stabilized_relative_change_percent"] for row in summary])
    low = np.asarray([row["bootstrap_95ci_low"] for row in summary])
    high = np.asarray([row["bootstrap_95ci_high"] for row in summary])
    y = np.arange(len(summary))
    colors = np.where(medians < 0, "#c52420", "#198754")

    fig, ax = plt.subplots(figsize=(9.4, 7.0), constrained_layout=True)
    for index in range(len(y)):
        ax.errorbar(
            medians[index], y[index],
            xerr=[[medians[index] - low[index]], [high[index] - medians[index]]],
            fmt="o", color=colors[index], ecolor="#334155", markersize=8,
            elinewidth=1.4, capsize=3,
        )
    ax.axvline(0, color="#475569", linewidth=1.1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("Median stabilized relative critic-value change (%)")
    ax.set_title("Relative critic-value change at first achievement completion")
    fig.savefig(RUN_DIR / "achievement_stabilized_relative_value_effects.png", dpi=300)
    fig.savefig(RUN_DIR / "achievement_stabilized_relative_value_effects.pdf")
    plt.close(fig)

    metadata = {
        "source": "achievement_transition_values.csv",
        "checkpoint": 1499987968,
        "metric": "(V_after - V_before) / (1 + abs(V_before))",
        "aggregation": "median across first-completion transitions; episode bootstrap 95% CI",
        "stabilizer": 1.0,
    }
    (RUN_DIR / "stabilized_relative_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    for row in summary:
        print(
            f"{row['empirical_order_rank']:2d} {row['achievement']:22s} "
            f"{row['median_stabilized_relative_change_percent']:8.3f}% "
            f"[{row['bootstrap_95ci_low']:8.3f}, {row['bootstrap_95ci_high']:8.3f}]"
        )


if __name__ == "__main__":
    main()

"""Replot empirical inventory acquisition order with bootstrap median CIs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INVENTORY = (
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    """Percentile 95% CI for the median conditional on acquisition."""
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.percentile(medians, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="concept_mapping/runs/empirical_inventory_order_50episodes_m0mw4end")
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    first_step = np.load(run_dir / "first_acquisition_steps.npz")["first_step"]
    rng = np.random.default_rng(args.seed)
    rows = []
    for item_index, item in enumerate(INVENTORY):
        steps = first_step[:, item_index]
        observed = steps[steps >= 0]
        if len(observed):
            median = float(np.median(observed))
            ci_low, ci_high = bootstrap_median_ci(observed, rng, args.bootstrap_samples)
        else:
            median = ci_low = ci_high = float("nan")
        rows.append({
            "item": item,
            "episodes_acquired": int(len(observed)),
            "acquisition_rate": float(len(observed) / len(steps)),
            "median_first_step": median,
            "median_ci_low_95": ci_low,
            "median_ci_high_95": ci_high,
        })
    rows.sort(key=lambda row: (row["median_first_step"], row["item"]))
    with (run_dir / "acquisition_summary_with_ci.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    medians = np.array([row["median_first_step"] for row in rows])
    lower = medians - np.array([row["median_ci_low_95"] for row in rows])
    upper = np.array([row["median_ci_high_95"] for row in rows]) - medians
    rates = np.array([100 * row["acquisition_rate"] for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.1, 5.6), constrained_layout=True)
    ax.errorbar(medians, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#1d4ed8", capsize=3, lw=1.5, zorder=1)
    ax.scatter(medians, y, s=rates * 2.2 + 15, color="#2563eb", edgecolor="white", linewidth=.7, zorder=2)
    ax.set_yticks(y, [row["item"].replace("_", " ").title() for row in rows])
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    ax.set_xlabel("Median first-acquisition step (bootstrap 95% CI)")
    ax.set_title("Empirical frozen-policy inventory acquisition order")
    for y_value, median, rate in zip(y, medians, rates):
        ax.annotate(f"{rate:.0f}%", (median, y_value), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
    fig.savefig(run_dir / "empirical_acquisition_order.png", dpi=300)
    fig.savefig(run_dir / "empirical_acquisition_order.pdf")
    print(f"Updated plot and saved bootstrap intervals to {run_dir}")


if __name__ == "__main__":
    main()

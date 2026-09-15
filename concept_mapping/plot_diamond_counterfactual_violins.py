"""Plot the four diamond counterfactual critic-value distributions."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Ordered from the expected highest-value condition to the expected lowest.
# The two middle conditions do not have a strict theoretical ordering.
CONDITIONS = (
    ("Diamond +\nPickaxe", "D_diamond_pickaxe"),
    ("Diamond +\nNo Pickaxe", "C_diamond_no_pickaxe"),
    ("No Diamond +\nNo Pickaxe", "A_no_diamond_no_pickaxe"),
    ("No Diamond +\nPickaxe", "B_no_diamond_pickaxe"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--values_csv",
        default=(
            "concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
            "merged/clean_counterfactual_values.csv"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default=(
            "concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
            "merged/violin_plot"
        ),
    )
    args = parser.parse_args()

    with Path(args.values_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No counterfactual values found in {args.values_csv}")

    values = [np.asarray([float(row[column]) for row in rows]) for _, column in CONDITIONS]
    means = np.asarray([condition_values.mean() for condition_values in values])
    medians = np.asarray([np.median(condition_values) for condition_values in values])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diamond_counterfactual_violin_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "condition", "source_column", "n", "mean", "median"])
        for index, ((label, column), condition_values, mean, median) in enumerate(
            zip(CONDITIONS, values, means, medians), start=1
        ):
            writer.writerow([
                index,
                label.replace("\n", " "),
                column,
                len(condition_values),
                mean,
                median,
            ])

    positions = np.arange(1, len(CONDITIONS) + 1)
    colors = ("#1d4ed8", "#60a5fa", "#f59e0b", "#dc2626")
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#334155")
        body.set_linewidth(0.8)
        body.set_alpha(0.78)

    # A compact box plot makes the median and interquartile range explicit.
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.14,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#1e293b", "linewidth": 1.0},
        medianprops={"color": "#1e293b", "linewidth": 1.5},
        whiskerprops={"color": "#475569", "linewidth": 0.9},
        capprops={"color": "#475569", "linewidth": 0.9},
    )
    for element in ("boxes", "medians", "whiskers", "caps"):
        for artist in boxes[element]:
            artist.set_zorder(3)

    ax.plot(
        positions,
        means,
        color="#111827",
        linewidth=2.0,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.6,
        label="Mean",
        zorder=4,
    )
    value_span = max(array.max() for array in values) - min(array.min() for array in values)
    label_offset = max(0.025 * value_span, 0.015)
    for x, mean in zip(positions, means):
        ax.annotate(
            f"{mean:.3f}",
            (x, mean),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            zorder=5,
        )

    ax.set_xticks(positions, [label for label, _ in CONDITIONS])
    ax.set_ylabel(r"Frozen critic value $\hat{V}(s)$")
    ax.set_xlabel("Counterfactual condition (expected high to low)")
    ax.set_title(f"Diamond collectability counterfactuals ($n={len(rows):,}$ matched states)")
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.08)

    fig.savefig(out_dir / "diamond_counterfactual_violins.png", dpi=300)
    fig.savefig(out_dir / "diamond_counterfactual_violins.pdf")
    print(f"Saved violin plot and summary for {len(rows):,} matched states to {out_dir}")


if __name__ == "__main__":
    main()

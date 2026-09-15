"""Plot publication-ready violin distributions for water and sword probes."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WATER_CONDITIONS = (
    ("Empty +\nWater", "empty_water"),
    ("Empty +\nNo Water", "empty_no_water"),
    ("Full +\nWater", "full_water"),
    ("Full +\nNo Water", "full_no_water"),
)

# The primary theoretical ordering is iron > stone > wood. The auxiliary
# factual and no-sword conditions are intentionally excluded.
SWORD_CONDITIONS = (
    ("Iron Sword", "iron_only"),
    ("Stone Sword", "stone_only"),
    ("Wood Sword", "wood_only"),
)


def read_values(csv_path: Path, conditions: tuple[tuple[str, str], ...]):
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No counterfactual values found in {csv_path}")
    return rows, [np.asarray([float(row[column]) for row in rows]) for _, column in conditions]


def save_summary(
    out_path: Path,
    conditions: tuple[tuple[str, str], ...],
    values: list[np.ndarray],
) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "condition", "source_column", "n", "mean", "median"])
        for index, ((label, column), condition_values) in enumerate(zip(conditions, values), start=1):
            writer.writerow([
                index,
                label.replace("\n", " "),
                column,
                len(condition_values),
                condition_values.mean(),
                np.median(condition_values),
            ])


def plot_violins(
    values: list[np.ndarray],
    conditions: tuple[tuple[str, str], ...],
    colors: tuple[str, ...],
    title: str,
    xlabel: str,
    output_stem: Path,
    pair_break: int | None = None,
) -> None:
    positions = np.arange(1, len(conditions) + 1, dtype=float)
    if pair_break is not None:
        positions[pair_break:] += 0.35
    means = np.asarray([condition_values.mean() for condition_values in values])

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

    if pair_break is None:
        segments = [(positions, means)]
    else:
        segments = [
            (positions[:pair_break], means[:pair_break]),
            (positions[pair_break:], means[pair_break:]),
        ]
    for segment_index, (segment_x, segment_y) in enumerate(segments):
        ax.plot(
            segment_x,
            segment_y,
            color="#111827",
            linewidth=2.0,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.6,
            label="Mean" if segment_index == 0 else None,
            zorder=4,
        )
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

    ax.set_xticks(positions, [label for label, _ in conditions])
    ax.set_ylabel(r"Frozen critic value $\hat{V}(s)$")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.08)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged",
    )
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "violin_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    water_rows, water_values = read_values(run_dir / "water_counterfactual_values.csv", WATER_CONDITIONS)
    save_summary(out_dir / "water_counterfactual_violin_summary.csv", WATER_CONDITIONS, water_values)
    plot_violins(
        water_values,
        WATER_CONDITIONS,
        ("#0ea5e9", "#94a3b8", "#2563eb", "#64748b"),
        f"Context-dependent value of water ($n={len(water_rows):,}$ matched states)",
        "Counterfactual condition (paired by drink level)",
        out_dir / "water_counterfactual_violins",
        pair_break=2,
    )

    sword_rows, sword_values = read_values(run_dir / "sword_counterfactual_values.csv", SWORD_CONDITIONS)
    save_summary(out_dir / "sword_counterfactual_violin_summary.csv", SWORD_CONDITIONS, sword_values)
    plot_violins(
        sword_values,
        SWORD_CONDITIONS,
        ("#7c3aed", "#64748b", "#b45309"),
        f"Sword-type counterfactuals ($n={len(sword_rows):,}$ matched states)",
        "Sword condition (expected high to low)",
        out_dir / "sword_counterfactual_violins",
    )
    print(f"Saved water and sword violin plots and summaries to {out_dir}")


if __name__ == "__main__":
    main()

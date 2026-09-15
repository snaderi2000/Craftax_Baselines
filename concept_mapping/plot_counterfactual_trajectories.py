"""Plot matched-state counterfactual trajectories for diamond, water, and swords.

The visual design follows the inventory-progression plot: a readable sample of
faint base-state trajectories, the mean across all states, and its 95% CI.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIAMOND_CONDITIONS = (
    ("Diamond +\nPickaxe", "D_diamond_pickaxe"),
    ("Diamond +\nNo Pickaxe", "C_diamond_no_pickaxe"),
    ("No Diamond +\nNo Pickaxe", "A_no_diamond_no_pickaxe"),
    ("No Diamond +\nPickaxe", "B_no_diamond_pickaxe"),
)

# Water has two paired predictions rather than a strict four-way ordering.
WATER_CONDITIONS = (
    ("Empty +\nWater", "empty_water"),
    ("Empty +\nNo Water", "empty_no_water"),
    ("Full +\nWater", "full_water"),
    ("Full +\nNo Water", "full_no_water"),
)

# Expected theoretical ordering; factual and no-sword variants are excluded.
SWORD_CONDITIONS = (
    ("Iron Sword", "iron_only"),
    ("Stone Sword", "stone_only"),
    ("Wood Sword", "wood_only"),
)


def read_matrix(csv_path: Path, conditions: tuple[tuple[str, str], ...]) -> np.ndarray:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No counterfactual values found in {csv_path}")
    return np.asarray([[float(row[column]) for _, column in conditions] for row in rows])


def plot_trajectories(
    matrix: np.ndarray,
    conditions: tuple[tuple[str, str], ...],
    title: str,
    xlabel: str,
    output_stem: Path,
    seed: int,
    max_trajectories: int,
) -> None:
    x = np.arange(matrix.shape[1])
    rng = np.random.default_rng(seed)
    selected = rng.choice(
        matrix.shape[0],
        size=min(max_trajectories, matrix.shape[0]),
        replace=False,
    )
    means = matrix.mean(axis=0)
    sem = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
    ci_low, ci_high = means - 1.96 * sem, means + 1.96 * sem

    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    for state_index in selected:
        ax.plot(x, matrix[state_index], color="#94a3b8", alpha=0.10, linewidth=0.7)
    ax.plot(
        x,
        means,
        color="#b91c1c",
        linewidth=2.7,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="#b91c1c",
        markeredgewidth=1.8,
        label="Mean across base states",
        zorder=3,
    )
    ax.fill_between(x, ci_low, ci_high, color="#b91c1c", alpha=0.18, label="95% CI", zorder=2)
    for condition_x, mean in zip(x, means):
        ax.annotate(
            f"{mean:.3f}",
            (condition_x, mean),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#991b1b",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#fecaca",
                "alpha": 0.9,
            },
            zorder=4,
        )

    ax.set_xticks(x, [label for label, _ in conditions])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Frozen critic value $\hat{V}(s)$")
    ax.set_title(title)
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, loc="upper right")
    ax.margins(x=0.05, y=0.08)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

    with output_stem.with_name(output_stem.name + "_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "order", "condition", "source_column", "n", "mean", "standard_error",
            "ci_low_95", "ci_high_95",
        ])
        for index, ((label, column), mean, standard_error, low, high) in enumerate(
            zip(conditions, means, sem, ci_low, ci_high), start=1
        ):
            writer.writerow([
                index, label.replace("\n", " "), column, matrix.shape[0], mean,
                standard_error, low, high,
            ])


def plot_water_by_health(
    matrix: np.ndarray,
    health: np.ndarray,
    output_stem: Path,
    seed: int,
    max_trajectories: int,
) -> None:
    rounded_health = np.rint(health).astype(int)
    groups = (
        ("Low health ($\\leq 3$)", rounded_health <= 3),
        ("High health ($\\geq 4$)", rounded_health >= 4),
    )
    if any(not np.any(mask) for _, mask in groups):
        raise ValueError("Both low- and high-health water groups must contain states")

    x = np.arange(matrix.shape[1])
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True, constrained_layout=True)
    summary_rows = []
    for panel_index, (ax, (group_label, mask)) in enumerate(zip(axes, groups)):
        group_matrix = matrix[mask]
        selected = rng.choice(
            group_matrix.shape[0],
            size=min(max_trajectories, group_matrix.shape[0]),
            replace=False,
        )
        for state_index in selected:
            ax.plot(x, group_matrix[state_index], color="#94a3b8", alpha=0.10, linewidth=0.7)

        means = group_matrix.mean(axis=0)
        sem = group_matrix.std(axis=0, ddof=1) / np.sqrt(group_matrix.shape[0])
        ci_low, ci_high = means - 1.96 * sem, means + 1.96 * sem
        ax.plot(
            x,
            means,
            color="#b91c1c",
            linewidth=2.7,
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="#b91c1c",
            markeredgewidth=1.8,
            label="Mean across base states",
            zorder=3,
        )
        ax.fill_between(
            x,
            ci_low,
            ci_high,
            color="#b91c1c",
            alpha=0.18,
            label="95% CI",
            zorder=2,
        )
        for condition_x, mean in zip(x, means):
            ax.annotate(
                f"{mean:.3f}",
                (condition_x, mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#991b1b",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#fecaca",
                    "alpha": 0.9,
                },
                zorder=4,
            )
        ax.set_xticks(x, [label for label, _ in WATER_CONDITIONS])
        ax.set_title(f"{group_label}; $n={len(group_matrix):,}$")
        ax.set_xlabel("Water counterfactual condition")
        ax.grid(alpha=0.24)
        ax.margins(x=0.06, y=0.10)
        if panel_index == 1:
            ax.legend(frameon=False, loc="upper right")

        for order, ((condition, column), mean, standard_error, low, high) in enumerate(
            zip(WATER_CONDITIONS, means, sem, ci_low, ci_high), start=1
        ):
            summary_rows.append([
                group_label.replace("$", ""),
                len(group_matrix),
                order,
                condition.replace("\n", " "),
                column,
                mean,
                standard_error,
                low,
                high,
            ])

    axes[0].set_ylabel(r"Frozen critic value $\hat{V}(s)$")
    fig.suptitle("Health moderates value across water counterfactuals")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)

    with output_stem.with_name(output_stem.name + "_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "health_group", "n", "order", "condition", "source_column", "mean",
            "standard_error", "ci_low_95", "ci_high_95",
        ])
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diamond_csv",
        default=(
            "concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/"
            "merged/clean_counterfactual_values.csv"
        ),
    )
    parser.add_argument(
        "--water_sword_dir",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged",
    )
    parser.add_argument("--water_states_npz", default=None)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max_trajectories", type=int, default=160)
    args = parser.parse_args()

    diamond_csv = Path(args.diamond_csv)
    diamond_matrix = read_matrix(diamond_csv, DIAMOND_CONDITIONS)
    diamond_output = diamond_csv.parent / "trajectory_plot" / "diamond_counterfactual_trajectories"
    plot_trajectories(
        diamond_matrix,
        DIAMOND_CONDITIONS,
        f"Critic value across diamond counterfactuals ($n={len(diamond_matrix):,}$ matched states)",
        "Counterfactual condition (expected high to low)",
        diamond_output,
        args.seed,
        args.max_trajectories,
    )
    water_sword_dir = Path(args.water_sword_dir)
    water_matrix = read_matrix(water_sword_dir / "water_counterfactual_values.csv", WATER_CONDITIONS)
    water_output = water_sword_dir / "trajectory_plots" / "water_counterfactual_trajectories"
    plot_trajectories(
        water_matrix,
        WATER_CONDITIONS,
        f"Critic value across water counterfactuals ($n={len(water_matrix):,}$ matched states)",
        "Counterfactual condition (paired by drink level)",
        water_output,
        args.seed,
        args.max_trajectories,
    )
    water_states_path = (
        Path(args.water_states_npz)
        if args.water_states_npz
        else water_sword_dir / "water_base_states.npz"
    )
    water_obs = np.load(water_states_path)["base_obs"]
    if len(water_obs) != len(water_matrix):
        raise ValueError("Water base-state and counterfactual-value counts do not match")
    map_dim = water_obs.shape[1] - 22
    health = water_obs[:, map_dim + 12] * 10.0
    plot_water_by_health(
        water_matrix,
        health,
        water_sword_dir / "trajectory_plots" / "water_counterfactual_trajectories_by_health",
        args.seed,
        args.max_trajectories,
    )

    sword_matrix = read_matrix(water_sword_dir / "sword_counterfactual_values.csv", SWORD_CONDITIONS)
    sword_output = water_sword_dir / "trajectory_plots" / "sword_counterfactual_trajectories"
    plot_trajectories(
        sword_matrix,
        SWORD_CONDITIONS,
        f"Critic value across sword counterfactuals ($n={len(sword_matrix):,}$ matched states)",
        "Sword condition (expected high to low)",
        sword_output,
        args.seed,
        args.max_trajectories,
    )
    print(f"Saved diamond trajectories to {diamond_output.parent}")
    print(f"Saved water and sword trajectories to {water_output.parent}")


if __name__ == "__main__":
    main()

"""Relate empirical inventory acquisition timing to counterfactual value effects."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _correlations(names: list[str], values: dict[str, np.ndarray]) -> dict:
    """Pairwise Pearson and Spearman coefficients, with two-sided p-values."""
    result = {"variables": names, "pearson_r": {}, "pearson_p_value": {},
              "spearman_rho": {}, "spearman_p_value": {}}
    for name_i in names:
        result["pearson_r"][name_i] = {}
        result["pearson_p_value"][name_i] = {}
        result["spearman_rho"][name_i] = {}
        result["spearman_p_value"][name_i] = {}
        for name_j in names:
            pearson = pearsonr(values[name_i], values[name_j])
            spearman = spearmanr(values[name_i], values[name_j])
            result["pearson_r"][name_i][name_j] = float(pearson.statistic)
            result["pearson_p_value"][name_i][name_j] = float(pearson.pvalue)
            result["spearman_rho"][name_i][name_j] = float(spearman.statistic)
            result["spearman_p_value"][name_i][name_j] = float(spearman.pvalue)
    return result


def _heatmap(matrix: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.7), constrained_layout=True)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                    color="white" if abs(matrix[row, column]) > .55 else "black", fontsize=10)
    fig.colorbar(image, ax=ax, label="Correlation")
    ax.set_title(title)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acquisition_csv",
        default="concept_mapping/runs/empirical_inventory_order_50episodes_m0mw4end/acquisition_summary_with_ci.csv",
    )
    parser.add_argument(
        "--effects_csv",
        default="concept_mapping/runs/inventory_progress_empirical_order_3gpu_m0mw4end/merged/item_addition_effects.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="concept_mapping/runs/inventory_progress_empirical_order_3gpu_m0mw4end/merged",
    )
    args = parser.parse_args()
    acquisition = {row["item"]: row for row in _read_csv(Path(args.acquisition_csv))}
    effects = {row["item_added"]: row for row in _read_csv(Path(args.effects_csv))}
    items = sorted(set(acquisition) & set(effects), key=lambda item: float(acquisition[item]["median_first_step"]))
    if len(items) < 3:
        raise ValueError("Need at least three overlapping inventory items.")
    rows = []
    for rank, item in enumerate(items, start=1):
        acq, effect = acquisition[item], effects[item]
        rows.append({
            "item": item,
            "acquisition_rank": rank,
            "episodes_acquired": int(acq["episodes_acquired"]),
            "acquisition_rate": float(acq["acquisition_rate"]),
            "median_first_acquisition_step": float(acq["median_first_step"]),
            "median_first_acquisition_step_ci_low_95": float(acq["median_ci_low_95"]),
            "median_first_acquisition_step_ci_high_95": float(acq["median_ci_high_95"]),
            "mean_delta_value": float(effect["mean_delta_value"]),
            "mean_delta_value_ci_low_95": float(effect["mean_delta_value_95_ci_low"]),
            "mean_delta_value_ci_high_95": float(effect["mean_delta_value_95_ci_high"]),
            "probability_delta_value_negative": float(effect["probability_delta_value_negative"]),
        })
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "acquisition_timing_value_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    names = ["Acquisition rank", "Median first-acquisition step", "Mean counterfactual $\\Delta V$"]
    values = {
        names[0]: np.asarray([row["acquisition_rank"] for row in rows], dtype=float),
        names[1]: np.asarray([row["median_first_acquisition_step"] for row in rows], dtype=float),
        names[2]: np.asarray([row["mean_delta_value"] for row in rows], dtype=float),
    }
    stats = _correlations(names, values)
    (out_dir / "acquisition_timing_value_correlations.json").write_text(json.dumps(stats, indent=2))
    pearson_matrix = np.array([[stats["pearson_r"][left][right] for right in names] for left in names])
    spearman_matrix = np.array([[stats["spearman_rho"][left][right] for right in names] for left in names])
    for matrix, filename in ((pearson_matrix, "acquisition_timing_value_pearson.csv"),
                             (spearman_matrix, "acquisition_timing_value_spearman.csv")):
        with (out_dir / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["variable", *names])
            writer.writerows([[name, *matrix[index]] for index, name in enumerate(names)])
    _heatmap(pearson_matrix, names, "Pearson correlations across inventory measures", out_dir / "acquisition_timing_value_pearson")
    _heatmap(spearman_matrix, names, "Spearman correlations across inventory measures", out_dir / "acquisition_timing_value_spearman")

    x, y = values[names[1]], values[names[2]]
    rho = stats["spearman_rho"][names[1]][names[2]]
    rho_p = stats["spearman_p_value"][names[1]][names[2]]
    fig, ax = plt.subplots(figsize=(7.4, 5.3), constrained_layout=True)
    ax.scatter(x, y, color="#2563eb", s=65, zorder=2)
    for row in rows:
        ax.annotate(row["item"].replace("_", " ").title(),
                    (row["median_first_acquisition_step"], row["mean_delta_value"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8.5)
    fit = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, np.polyval(fit, x_line), color="#dc2626", linewidth=1.8, label="Linear fit")
    ax.axhline(0, color="#64748b", linewidth=1)
    ax.set_xlabel("Median first-acquisition step")
    ax.set_ylabel("Mean counterfactual change in critic value ($\\Delta V$)")
    ax.set_title("Later-acquired inventory items have more negative value effects")
    ax.grid(alpha=.25); ax.legend(frameon=False, loc="upper right")
    ax.text(.03, .04, f"Spearman $\\rho$ = {rho:.2f}\n$p$ = {rho_p:.3g}\n$n = {len(rows)}$ items",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            bbox={"boxstyle": "round,pad=.3", "facecolor": "white", "edgecolor": "#cbd5e1"})
    fig.savefig(out_dir / "acquisition_timing_vs_value_effect.png", dpi=300)
    fig.savefig(out_dir / "acquisition_timing_vs_value_effect.pdf")
    print(json.dumps({
        "n_items": len(rows),
        "median_step_vs_mean_delta_value": {
            "pearson_r": stats["pearson_r"][names[1]][names[2]],
            "pearson_p": stats["pearson_p_value"][names[1]][names[2]],
            "spearman_rho": rho,
            "spearman_p": rho_p,
        },
        "rank_vs_mean_delta_value": {
            "pearson_r": stats["pearson_r"][names[0]][names[2]],
            "pearson_p": stats["pearson_p_value"][names[0]][names[2]],
            "spearman_rho": stats["spearman_rho"][names[0]][names[2]],
            "spearman_p": stats["spearman_p_value"][names[0]][names[2]],
        },
    }, indent=2))


if __name__ == "__main__":
    main()

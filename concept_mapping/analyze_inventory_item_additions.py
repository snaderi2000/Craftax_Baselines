"""Identify which progression components drive inventory-induced value changes."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 10_000):
    """Bootstrap CI over base states, preserving all tied-order transitions per state."""
    # Input has shape (n_base_states, n_observations_per_state_for_item).
    base_means = values.mean(axis=1)
    indices = rng.integers(0, len(base_means), size=(draws, len(base_means)))
    means = base_means[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--values_csv",
        default="concept_mapping/runs/inventory_progress_hierarchy_3gpu_m0mw4end/merged_raw_counterfactual_values.csv",
    )
    parser.add_argument(
        "--shard_root",
        default="concept_mapping/runs/inventory_progress_hierarchy_3gpu_m0mw4end",
        help="Used when --values_csv does not exist; reads each raw shard CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="concept_mapping/runs/inventory_progress_hierarchy_3gpu_m0mw4end/merged",
    )
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    source = Path(args.values_csv)
    rows = []
    if source.exists():
        with source.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    else:
        base_offset = 0
        for shard in sorted(path for path in Path(args.shard_root).glob("shard_*") if path.is_dir()):
            with (shard / "counterfactual_values.csv").open(newline="") as handle:
                shard_rows = list(csv.DictReader(handle))
            for row in shard_rows:
                row["base_index"] = int(row["base_index"]) + base_offset
                rows.append(row)
            base_offset += max(int(row["base_index"]) for row in shard_rows) + 1
    # Values are one row per (base state, valid ordering, inventory level).
    by_path = {(int(row["base_index"]), int(row["ordering_index"]), int(row["inventory_level"])): float(row["value"]) for row in rows}
    item_changes = defaultdict(lambda: defaultdict(list))
    for row in rows:
        level = int(row["inventory_level"])
        if level == 0:
            continue
        base, order = int(row["base_index"]), int(row["ordering_index"])
        item = row["item_added"]
        delta = float(row["value"]) - by_path[(base, order, level - 1)]
        item_changes[item][base].append(delta)
    rng = np.random.default_rng(args.seed)
    summary = []
    for item, by_base in item_changes.items():
        base_values = np.asarray([values for _, values in sorted(by_base.items())])
        flat = base_values.ravel()
        ci_low, ci_high = _bootstrap_ci(base_values, rng)
        summary.append({
            "item_added": item,
            "n_base_states": len(base_values),
            "n_transitions": len(flat),
            "mean_delta_value": float(flat.mean()),
            "median_delta_value": float(np.median(flat)),
            "mean_delta_value_95_ci_low": float(ci_low),
            "mean_delta_value_95_ci_high": float(ci_high),
            "probability_delta_value_negative": float((flat < 0).mean()),
        })
    summary.sort(key=lambda row: row["mean_delta_value"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "item_addition_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    # Publication figure: means/cluster-bootstrap CIs, ordered from most negative.
    labels = [row["item_added"].replace("_", " ").title() for row in summary]
    means = np.asarray([row["mean_delta_value"] for row in summary])
    lower = means - np.asarray([row["mean_delta_value_95_ci_low"] for row in summary])
    upper = np.asarray([row["mean_delta_value_95_ci_high"] for row in summary]) - means
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    colors = np.where(means < 0, "#b91c1c", "#15803d")
    y = np.arange(len(summary))
    ax.errorbar(means, y, xerr=np.stack([lower, upper]), fmt="none", ecolor="#334155", capsize=3, linewidth=1.2, zorder=1)
    ax.scatter(means, y, s=55, c=colors, zorder=2)
    ax.axvline(0, color="#475569", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in frozen critic value after adding one component ($\\Delta V$)")
    ax.set_title("Item-specific effects along the inventory progression")
    ax.grid(axis="x", alpha=.25)
    fig.savefig(out_dir / "item_addition_effects.png", dpi=300)
    fig.savefig(out_dir / "item_addition_effects.pdf")
    lines = [
        "\\begin{table}[H]", "\\centering",
        "\\caption{Item-specific critic-value changes along the progression-ordered inventory counterfactual. Values are mean within-state changes after adding one component; confidence intervals are bootstrapped over base states.}",
        "\\label{tab:inventory_item_additions}",
        "\\begin{tabular}{lrr}", "\\toprule",
        "\\textbf{Item added} & \\textbf{Mean $\\Delta V$} & \\textbf{$P(\\Delta V < 0)$} \\\\", "\\midrule",
    ]
    for row in summary:
        item = row["item_added"].replace("_", " ").title()
        lines.append(f"{item} & {row['mean_delta_value']:.3f} & {100 * row['probability_delta_value_negative']:.1f}\\% \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (out_dir / "item_addition_effects.tex").write_text("\n".join(lines))
    print(f"Saved {len(summary)} item effects to {out_dir}")
    for row in summary:
        print(f"{row['item_added']}: mean={row['mean_delta_value']:.4f}, P(negative)={100 * row['probability_delta_value_negative']:.1f}%")


if __name__ == "__main__":
    main()

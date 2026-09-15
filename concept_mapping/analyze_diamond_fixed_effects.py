"""Fit the primary state-fixed-effects diamond--pickaxe factorial model.

The four counterfactual critic values for each base state form a 2x2 design:
diamond absent/present x iron pickaxe absent/present. State fixed effects absorb
all non-manipulated features shared by the four variants.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


CONDITIONS = (
    ("no diamond, no pickaxe", "A_no_diamond_no_pickaxe", 0, 0),
    ("no diamond, pickaxe", "B_no_diamond_pickaxe", 0, 1),
    ("diamond, no pickaxe", "C_diamond_no_pickaxe", 1, 0),
    ("diamond, pickaxe", "D_diamond_pickaxe", 1, 1),
)


def _estimate_fixed_effects(y: np.ndarray, x: np.ndarray, groups: np.ndarray):
    """Within estimator with a CR1 covariance matrix clustered by base state."""
    xd, yd = x.copy(), y.copy()
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        xd[mask] -= xd[mask].mean(axis=0)
        yd[mask] -= yd[mask].mean()
    bread = np.linalg.inv(xd.T @ xd)
    beta = bread @ xd.T @ yd
    residual = yd - xd @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in unique_groups:
        mask = groups == group
        score = xd[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, g = len(y), x.shape[1], len(unique_groups)
    covariance = bread @ meat @ bread * (g / (g - 1)) * ((n - 1) / (n - k))
    return beta, covariance, xd, yd, residual


def _effect_row(name: str, contrast: np.ndarray, beta: np.ndarray, covariance: np.ndarray) -> dict:
    estimate = float(contrast @ beta)
    se = float(np.sqrt(contrast @ covariance @ contrast))
    z = estimate / se
    p = float(2 * norm.sf(abs(z)))
    return {
        "effect": name,
        "estimate": estimate,
        "standard_error_clustered_by_state": se,
        "ci_low_95": estimate - 1.96 * se,
        "ci_high_95": estimate + 1.96 * se,
        "z": z,
        "p_value_two_sided": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--values_csv",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/clean_counterfactual_values.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/fixed_effects",
    )
    parser.add_argument("--bootstrap_draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.values_csv).open(newline="") as handle:
        source = list(csv.DictReader(handle))

    long_rows, interactions = [], []
    for base_index, row in enumerate(source):
        values = {}
        for condition, column, diamond, pickaxe in CONDITIONS:
            value = float(row[column]); values[column] = value
            long_rows.append({
                "base_state": base_index,
                "source_merged_index": row.get("merged_index", base_index),
                "condition": condition,
                "diamond_present": diamond,
                "iron_pickaxe_present": pickaxe,
                "diamond_x_pickaxe": diamond * pickaxe,
                "critic_value": value,
            })
        interactions.append(
            values["D_diamond_pickaxe"] - values["C_diamond_no_pickaxe"]
            - values["B_no_diamond_pickaxe"] + values["A_no_diamond_no_pickaxe"]
        )
    fields = list(long_rows[0])
    with (out_dir / "diamond_fixed_effects_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(long_rows)

    groups = np.asarray([row["base_state"] for row in long_rows])
    y = np.asarray([row["critic_value"] for row in long_rows])
    x = np.asarray([[row["diamond_present"], row["iron_pickaxe_present"], row["diamond_x_pickaxe"]] for row in long_rows], dtype=float)
    beta, covariance, xd, yd, residual = _estimate_fixed_effects(y, x, groups)
    coefficient_rows = [
        _effect_row("Diamond present (without pickaxe)", np.array([1., 0., 0.]), beta, covariance),
        _effect_row("Iron pickaxe present (without diamond)", np.array([0., 1., 0.]), beta, covariance),
        _effect_row("Diamond x iron-pickaxe interaction", np.array([0., 0., 1.]), beta, covariance),
    ]
    simple_effect_rows = [
        _effect_row("Diamond effect without pickaxe: C - A", np.array([1., 0., 0.]), beta, covariance),
        _effect_row("Iron-pickaxe effect without diamond: B - A", np.array([0., 1., 0.]), beta, covariance),
        _effect_row("Diamond effect with pickaxe: D - B", np.array([1., 0., 1.]), beta, covariance),
        _effect_row("Iron-pickaxe effect with diamond: D - C", np.array([0., 1., 1.]), beta, covariance),
    ]
    for filename, rows in (("diamond_fixed_effects_coefficients.csv", coefficient_rows),
                           ("diamond_simple_effects.csv", simple_effect_rows)):
        with (out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    interaction = np.asarray(interactions)
    rng = np.random.default_rng(args.seed)
    bootstrap = interaction[rng.integers(0, len(interaction), size=(args.bootstrap_draws, len(interaction)))].mean(axis=1)
    interaction_summary = {
        "n_base_states": len(source),
        "n_counterfactual_values": len(long_rows),
        "model": "V_ic = alpha_i + beta_D Diamond_ic + beta_P Pickaxe_ic + beta_DP (Diamond_ic x Pickaxe_ic) + epsilon_ic",
        "fixed_effect": "base_state",
        "standard_errors": "CR1 clustered by base_state",
        "within_r_squared": float(1 - np.sum(residual ** 2) / np.sum(yd ** 2)),
        "mean_state_level_interaction": float(interaction.mean()),
        "median_state_level_interaction": float(np.median(interaction)),
        "share_state_level_interactions_positive": float(np.mean(interaction > 0)),
        "interaction_bootstrap_ci_low_95": float(np.quantile(bootstrap, .025)),
        "interaction_bootstrap_ci_high_95": float(np.quantile(bootstrap, .975)),
        "coefficients": coefficient_rows,
        "simple_effects": simple_effect_rows,
    }
    (out_dir / "diamond_fixed_effects_summary.json").write_text(json.dumps(interaction_summary, indent=2))

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    plot_rows = simple_effect_rows
    estimates = np.asarray([row["estimate"] for row in plot_rows])
    low = estimates - np.asarray([row["ci_low_95"] for row in plot_rows])
    high = np.asarray([row["ci_high_95"] for row in plot_rows]) - estimates
    y_pos = np.arange(len(plot_rows))
    colors = ["#2563eb", "#b91c1c", "#2563eb", "#b91c1c"]
    ax.errorbar(estimates, y_pos, xerr=np.vstack([low, high]), fmt="none", color="#334155", capsize=3, zorder=1)
    ax.scatter(estimates, y_pos, color=colors, s=65, zorder=2)
    ax.axvline(0, color="#64748b", linewidth=1)
    ax.set_yticks(y_pos, [row["effect"].split(":")[0] for row in plot_rows])
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    ax.set_xlabel("Change in frozen critic value")
    ax.set_title("Diamond and pickaxe effects depend on each other")
    fig.savefig(out_dir / "diamond_simple_effects_forest.png", dpi=300)
    fig.savefig(out_dir / "diamond_simple_effects_forest.pdf")

    fig, ax = plt.subplots(figsize=(7.1, 4.3), constrained_layout=True)
    ax.hist(interaction, bins=50, color="#93c5fd", edgecolor="white")
    ax.axvline(0, color="#475569", linewidth=1, label="No interaction")
    ax.axvline(interaction.mean(), color="#b91c1c", linewidth=2.3, label=f"Mean = {interaction.mean():.3f}")
    ax.set_xlabel("State-level interaction: (D - C) - (B - A)")
    ax.set_ylabel("Number of base states")
    ax.set_title("Diamond--pickaxe complementarity across counterfactual states")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.savefig(out_dir / "diamond_interaction_distribution.png", dpi=300)
    fig.savefig(out_dir / "diamond_interaction_distribution.pdf")

    latex = [
        "\\begin{table}[H]", "\\centering",
        "\\caption{State-fixed-effects model of diamond collectability. The interaction tests whether an iron pickaxe has a greater effect on critic value when a diamond is present. Standard errors are clustered by base state.}",
        "\\label{tab:diamond_fixed_effects}", "\\begin{tabular}{lrrr}", "\\toprule",
        "\\textbf{Term} & \\textbf{Estimate} & \\textbf{95\\% CI} & \\textbf{$p$-value} \\\\", "\\midrule",
    ]
    for row in coefficient_rows:
        p = row["p_value_two_sided"]
        p_text = "$< 10^{-300}$" if p == 0 else f"{p:.2g}"
        latex.append(f"{row['effect']} & {row['estimate']:.3f} & [{row['ci_low_95']:.3f}, {row['ci_high_95']:.3f}] & {p_text} \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (out_dir / "diamond_fixed_effects_table.tex").write_text("\n".join(latex))
    print(json.dumps(interaction_summary, indent=2))


if __name__ == "__main__":
    main()

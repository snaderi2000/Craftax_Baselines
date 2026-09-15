"""Fit the primary state-fixed-effects empty-drink x visible-water model."""

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
    ("full drink, no water", "full_no_water", 0, 0),
    ("full drink, water", "full_water", 0, 1),
    ("empty drink, no water", "empty_no_water", 1, 0),
    ("empty drink, water", "empty_water", 1, 1),
)


def _fixed_effects(y: np.ndarray, x: np.ndarray, groups: np.ndarray):
    xd, yd = x.copy(), y.copy()
    unique = np.unique(groups)
    for group in unique:
        mask = groups == group
        xd[mask] -= xd[mask].mean(axis=0)
        yd[mask] -= yd[mask].mean()
    bread = np.linalg.inv(xd.T @ xd)
    beta = bread @ xd.T @ yd
    residual = yd - xd @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in unique:
        mask = groups == group
        score = xd[mask].T @ residual[mask]
        meat += np.outer(score, score)
    n, k, g = len(y), x.shape[1], len(unique)
    covariance = bread @ meat @ bread * (g / (g - 1)) * ((n - 1) / (n - k))
    return beta, covariance, yd, residual


def _effect(name: str, contrast: np.ndarray, beta: np.ndarray, covariance: np.ndarray) -> dict:
    estimate = float(contrast @ beta)
    se = float(np.sqrt(contrast @ covariance @ contrast))
    z = estimate / se
    return {
        "effect": name,
        "estimate": estimate,
        "standard_error_clustered_by_state": se,
        "ci_low_95": estimate - 1.96 * se,
        "ci_high_95": estimate + 1.96 * se,
        "z": z,
        "p_value_two_sided": float(2 * norm.sf(abs(z))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--values_csv",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/water_counterfactual_values.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/water_fixed_effects",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.values_csv).open(newline="") as handle:
        source = list(csv.DictReader(handle))
    long_rows, interactions = [], []
    for state, row in enumerate(source):
        values = {}
        for condition, column, empty, water in CONDITIONS:
            value = float(row[column]); values[column] = value
            long_rows.append({
                "base_state": state, "source_merged_index": row.get("merged_index", state),
                "condition": condition, "empty_drink": empty, "visible_water": water,
                "empty_x_water": empty * water, "critic_value": value,
            })
        interactions.append((values["empty_water"] - values["empty_no_water"])
                            - (values["full_water"] - values["full_no_water"]))
    with (out_dir / "water_fixed_effects_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0])); writer.writeheader(); writer.writerows(long_rows)
    groups = np.asarray([row["base_state"] for row in long_rows])
    y = np.asarray([row["critic_value"] for row in long_rows])
    x = np.asarray([[row["empty_drink"], row["visible_water"], row["empty_x_water"]] for row in long_rows], dtype=float)
    beta, covariance, yd, residual = _fixed_effects(y, x, groups)
    coefficients = [
        _effect("Empty drink (without visible water)", np.array([1., 0., 0.]), beta, covariance),
        _effect("Visible water (when drink is full)", np.array([0., 1., 0.]), beta, covariance),
        _effect("Empty-drink x visible-water interaction", np.array([0., 0., 1.]), beta, covariance),
    ]
    simple_effects = [
        _effect("Visible-water effect when drink is full", np.array([0., 1., 0.]), beta, covariance),
        _effect("Visible-water effect when drink is empty", np.array([0., 1., 1.]), beta, covariance),
        _effect("Empty-drink effect without visible water", np.array([1., 0., 0.]), beta, covariance),
        _effect("Empty-drink effect with visible water", np.array([1., 0., 1.]), beta, covariance),
    ]
    for filename, rows in (("water_fixed_effects_coefficients.csv", coefficients),
                           ("water_simple_effects.csv", simple_effects)):
        with (out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    interaction = np.asarray(interactions)
    summary = {
        "n_base_states": len(source), "n_counterfactual_values": len(long_rows),
        "model": "V_ic = alpha_i + beta_E Empty_ic + beta_W Water_ic + beta_EW (Empty_ic x Water_ic) + epsilon_ic",
        "fixed_effect": "base_state", "standard_errors": "CR1 clustered by base_state",
        "within_r_squared": float(1 - np.sum(residual ** 2) / np.sum(yd ** 2)),
        "mean_state_level_interaction": float(interaction.mean()),
        "median_state_level_interaction": float(np.median(interaction)),
        "share_state_level_interactions_positive": float(np.mean(interaction > 0)),
        "coefficients": coefficients, "simple_effects": simple_effects,
    }
    (out_dir / "water_fixed_effects_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    estimates = np.asarray([row["estimate"] for row in simple_effects])
    low = estimates - np.asarray([row["ci_low_95"] for row in simple_effects])
    high = np.asarray([row["ci_high_95"] for row in simple_effects]) - estimates
    positions = np.arange(len(simple_effects))
    colors = ["#2563eb", "#2563eb", "#b91c1c", "#b91c1c"]
    ax.errorbar(estimates, positions, xerr=np.vstack([low, high]), fmt="none", color="#334155", capsize=3, zorder=1)
    ax.scatter(estimates, positions, color=colors, s=65, zorder=2)
    ax.axvline(0, color="#64748b", linewidth=1)
    ax.set_yticks(positions, [row["effect"] for row in simple_effects])
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    ax.set_xlabel("Change in frozen critic value")
    ax.set_title("Visible water is valuable specifically when drink is empty")
    fig.savefig(out_dir / "water_simple_effects_forest.png", dpi=300)
    fig.savefig(out_dir / "water_simple_effects_forest.pdf")

    latex = [
        "\\begin{table}[H]", "\\centering",
        "\\caption{State-fixed-effects model of water value. The interaction tests whether visible water has a greater value effect when drink is empty. Standard errors are clustered by base state.}",
        "\\label{tab:water_fixed_effects}", "\\begin{tabular}{lrrr}", "\\toprule",
        "\\textbf{Term} & \\textbf{Estimate} & \\textbf{95\\% CI} & \\textbf{$p$-value} \\\\", "\\midrule",
    ]
    for row in coefficients:
        p = row["p_value_two_sided"]
        p_text = "$< 10^{-200}$" if p < 1e-200 else f"{p:.2g}"
        latex.append(f"{row['effect']} & {row['estimate']:.3f} & [{row['ci_low_95']:.3f}, {row['ci_high_95']:.3f}] & {p_text} \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (out_dir / "water_fixed_effects_table.tex").write_text("\n".join(latex))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

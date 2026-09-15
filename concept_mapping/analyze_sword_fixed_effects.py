"""Fit a state-fixed-effects model for counterfactual sword type values.

Wood is the reference weapon, so the two coefficients quantify the critic's
within-scene preference for stone and iron respectively over wood.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def fixed_effects(y: np.ndarray, x: np.ndarray, groups: np.ndarray):
    """Within estimator with CR1 standard errors clustered by base state."""
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
    return beta, covariance, yd, residual


def result(name, contrast, beta, covariance):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--values_csv",
        default=("concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/"
                 "merged/sword_counterfactual_values.csv"),
    )
    parser.add_argument(
        "--out_dir",
        default=("concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/"
                 "merged/sword_fixed_effects"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.values_csv).open(newline="") as handle:
        source = list(csv.DictReader(handle))

    long_rows = []
    pairwise = {"stone_minus_wood": [], "iron_minus_wood": [], "iron_minus_stone": []}
    for state, row in enumerate(source):
        values = {key: float(row[key]) for key in ("wood_only", "stone_only", "iron_only")}
        pairwise["stone_minus_wood"].append(values["stone_only"] - values["wood_only"])
        pairwise["iron_minus_wood"].append(values["iron_only"] - values["wood_only"])
        pairwise["iron_minus_stone"].append(values["iron_only"] - values["stone_only"])
        for sword, column, stone, iron in (
            ("wood", "wood_only", 0, 0),
            ("stone", "stone_only", 1, 0),
            ("iron", "iron_only", 0, 1),
        ):
            long_rows.append({
                "base_state": state,
                "source_merged_index": row.get("merged_index", state),
                "sword": sword,
                "stone_sword": stone,
                "iron_sword": iron,
                "critic_value": values[column],
            })

    with (out_dir / "sword_fixed_effects_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader(); writer.writerows(long_rows)

    groups = np.asarray([row["base_state"] for row in long_rows])
    y = np.asarray([row["critic_value"] for row in long_rows])
    x = np.asarray([[row["stone_sword"], row["iron_sword"]] for row in long_rows], dtype=float)
    beta, covariance, yd, residual = fixed_effects(y, x, groups)
    effects = [
        result("Stone sword minus wood sword", np.array([1.0, 0.0]), beta, covariance),
        result("Iron sword minus wood sword", np.array([0.0, 1.0]), beta, covariance),
        result("Iron sword minus stone sword", np.array([-1.0, 1.0]), beta, covariance),
    ]
    with (out_dir / "sword_fixed_effects_coefficients.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader(); writer.writerows(effects)

    summary = {
        "n_base_states": len(source),
        "n_counterfactual_values": len(long_rows),
        "model": "V_ic = alpha_i + beta_S Stone_ic + beta_I Iron_ic + epsilon_ic (wood reference)",
        "fixed_effect": "base_state",
        "standard_errors": "CR1 clustered by base_state",
        "within_r_squared": float(1 - np.sum(residual ** 2) / np.sum(yd ** 2)),
        "effects": effects,
        "share_states_stone_exceeds_wood": float(np.mean(np.asarray(pairwise["stone_minus_wood"]) > 0)),
        "share_states_iron_exceeds_wood": float(np.mean(np.asarray(pairwise["iron_minus_wood"]) > 0)),
        "share_states_iron_exceeds_stone": float(np.mean(np.asarray(pairwise["iron_minus_stone"]) > 0)),
    }
    (out_dir / "sword_fixed_effects_summary.json").write_text(json.dumps(summary, indent=2))

    latex = [
        "\\begin{table}[H]", "\\centering",
        "\\caption{State-fixed-effects estimates of the critic's relative value for sword types in hostile-visible states. Wood is the reference condition; standard errors are clustered by base state.}",
        "\\label{tab:sword_fixed_effects}", "\\begin{tabular}{lrrr}", "\\toprule",
        "\\textbf{Contrast} & \\textbf{Estimate} & \\textbf{95\\% CI} & \\textbf{$p$-value} \\\\", "\\midrule",
    ]
    for row in effects:
        p = row["p_value_two_sided"]
        p_text = "$< 10^{-200}$" if p < 1e-200 else f"{p:.2g}"
        latex.append(
            f"{row['effect']} & {row['estimate']:.3f} & "
            f"[{row['ci_low_95']:.3f}, {row['ci_high_95']:.3f}] & {p_text} \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (out_dir / "sword_fixed_effects_table.tex").write_text("\n".join(latex))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Exploratory health-moderated state-fixed-effects model of water value."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)
    jax.tree = _JaxTreeCompat()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from concept_mapping.visualize_symbolic_trajectory import MAP_DIM


CONDITIONS = (
    ("full drink, no water", "full_no_water", 0, 0),
    ("full drink, water", "full_water", 0, 1),
    ("empty drink, no water", "empty_no_water", 1, 0),
    ("empty drink, water", "empty_water", 1, 1),
)
TERM_NAMES = (
    "Empty drink", "Visible water", "Empty drink x visible water",
    "Empty drink x health", "Visible water x health",
    "Empty drink x visible water x health",
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
        "effect": name, "estimate": estimate, "standard_error_clustered_by_state": se,
        "ci_low_95": estimate - 1.96 * se, "ci_high_95": estimate + 1.96 * se,
        "z": z, "p_value_two_sided": float(2 * norm.sf(abs(z))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values_csv", default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/water_counterfactual_values.csv")
    parser.add_argument("--states_npz", default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/water_base_states.npz")
    parser.add_argument("--out_dir", default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/water_health_fixed_effects")
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.values_csv).open(newline="") as handle:
        source = list(csv.DictReader(handle))
    base_obs = np.load(args.states_npz)["base_obs"]
    if len(source) != len(base_obs):
        raise ValueError("Water value and base-state counts do not match")
    health = base_obs[:, MAP_DIM + 12] * 10.0
    health_mean = float(health.mean())
    centered_health = health - health_mean

    long_rows = []
    for state, row in enumerate(source):
        for condition, column, empty, water in CONDITIONS:
            long_rows.append({
                "base_state": state, "source_merged_index": row.get("merged_index", state),
                "condition": condition, "health": float(health[state]),
                "centered_health": float(centered_health[state]), "empty_drink": empty,
                "visible_water": water, "critic_value": float(row[column]),
            })
    with (out_dir / "water_health_fixed_effects_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0])); writer.writeheader(); writer.writerows(long_rows)
    groups = np.asarray([row["base_state"] for row in long_rows])
    y = np.asarray([row["critic_value"] for row in long_rows])
    x = []
    for row in long_rows:
        e, w, h = row["empty_drink"], row["visible_water"], row["centered_health"]
        x.append([e, w, e * w, e * h, w * h, e * w * h])
    x = np.asarray(x, dtype=float)
    beta, covariance, yd, residual = _fixed_effects(y, x, groups)
    coefficients = [_effect(name, np.eye(len(beta))[index], beta, covariance) for index, name in enumerate(TERM_NAMES)]
    # Per-health-point slopes. The water effect when empty has both the
    # water-by-health and three-way interaction contributions.
    health_slopes = [
        _effect("Health slope of water effect when drink is full", np.array([0, 0, 0, 0, 1, 0.]), beta, covariance),
        _effect("Health slope of water effect when drink is empty", np.array([0, 0, 0, 0, 1, 1.]), beta, covariance),
        _effect("Health slope of empty-drink effect without water", np.array([0, 0, 0, 1, 0, 0.]), beta, covariance),
        _effect("Health slope of empty-drink effect with water", np.array([0, 0, 0, 1, 0, 1.]), beta, covariance),
    ]
    for filename, rows in (("water_health_fixed_effects_coefficients.csv", coefficients),
                           ("water_health_simple_slopes.csv", health_slopes)):
        with (out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    health_grid = np.arange(int(np.rint(health.min())), int(np.rint(health.max())) + 1)
    effects = {"full": [], "empty": []}
    for value in health_grid:
        h = value - health_mean
        full = np.array([0, 1, 0, 0, h, 0.])
        empty = np.array([0, 1, 1, 0, h, h])
        for name, contrast in (("full", full), ("empty", empty)):
            result = _effect("", contrast, beta, covariance)
            effects[name].append(result)
    fig, ax = plt.subplots(figsize=(7.1, 4.6), constrained_layout=True)
    colors = {"full": "#64748b", "empty": "#2563eb"}
    for name, label in (("full", "Full drink"), ("empty", "Empty drink")):
        estimates = np.asarray([row["estimate"] for row in effects[name]])
        low = np.asarray([row["ci_low_95"] for row in effects[name]])
        high = np.asarray([row["ci_high_95"] for row in effects[name]])
        ax.plot(health_grid, estimates, color=colors[name], linewidth=2.5, label=label)
        ax.fill_between(health_grid, low, high, color=colors[name], alpha=.18)
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_xlabel("Original health in base state")
    ax.set_ylabel("Estimated effect of visible water on critic value")
    ax.set_title("Health moderation of the visible-water value effect")
    ax.legend(frameon=False); ax.grid(alpha=.24)
    fig.savefig(out_dir / "water_effect_by_health.png", dpi=300)
    fig.savefig(out_dir / "water_effect_by_health.pdf")

    health_counts = {str(int(level)): int(np.sum(np.rint(health) == level)) for level in health_grid}
    summary = {
        "analysis_status": "exploratory; health distribution is strongly concentrated at 9",
        "n_base_states": len(source), "n_counterfactual_values": len(long_rows),
        "model": "V_ic = alpha_i + E + W + E:W + E:H_i + W:H_i + E:W:H_i + epsilon_ic",
        "health_center": health_mean, "health_counts_rounded": health_counts,
        "fixed_effect": "base_state", "standard_errors": "CR1 clustered by base_state",
        "within_r_squared": float(1 - np.sum(residual ** 2) / np.sum(yd ** 2)),
        "coefficients": coefficients, "health_simple_slopes": health_slopes,
        "water_effect_by_health": {key: values for key, values in effects.items()},
    }
    (out_dir / "water_health_fixed_effects_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

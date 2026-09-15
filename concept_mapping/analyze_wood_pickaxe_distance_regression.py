"""Regress the wood-pickaxe value effect on visible-stone distance.

This analysis is restricted to the post-Make-Wood-Pickaxe states with at least
one visible stone from ``analyze_wood_pickaxe_value_by_visible_stone.py``.
The primary OLS model controls for episode step, achievement count, and every
nonconstant inventory field other than wood_pickaxe.  It reports HC3 robust
standard errors and an episode-resampling bootstrap for the distance slope.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_EXPERIMENT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "wood_pickaxe_value_by_visible_stone_final_1p5b"
)
INVENTORY_NAMES = (
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)


def load_visible_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stone_visible"] == "True"]
    if not rows:
        raise RuntimeError(f"No stone-visible rows in {path}")
    return rows


def add_covariates(rows: list[dict]) -> tuple[dict[str, np.ndarray], list[str], dict]:
    columns: dict[str, list[float]] = {
        "delta_V": [], "stone_distance": [], "episode_step": [],
        "log1p_episode_step": [], "achievement_count": [],
        "n_visible_stone": [],
    }
    inventory = {name: [] for name in INVENTORY_NAMES}
    for row in rows:
        with np.load(row["source_npz"]) as episode:
            state_step = int(row["state_step"])
            achievements = np.asarray(episode["achievements"])[state_step]
            items = np.asarray(episode["inventory"])[state_step]
        columns["delta_V"].append(float(row["delta_V"]))
        columns["stone_distance"].append(float(row["nearest_stone_manhattan"]))
        columns["episode_step"].append(float(state_step))
        columns["log1p_episode_step"].append(float(np.log1p(state_step)))
        columns["achievement_count"].append(float(achievements.sum()))
        columns["n_visible_stone"].append(float(row["n_visible_stone"]))
        for index, name in enumerate(INVENTORY_NAMES):
            inventory[name].append(float(items[index]))
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    inventory_controls = []
    inventory_audit = {}
    for name, values in inventory.items():
        values_array = np.asarray(values, dtype=np.float64)
        nonzero = int(np.sum(values_array != 0))
        inventory_audit[name] = {
            "unique_values": np.unique(values_array).tolist(),
            "n_nonzero": nonzero,
        }
        # A singleton sapling or sword indicator yields leverage one and cannot
        # support robust inference. Require at least ten observations on both
        # sides of a varying inventory covariate.
        if (name != "wood_pickaxe" and np.ptp(values_array) > 0
                and min(nonzero, len(values_array) - nonzero) >= 10):
            field = f"inventory_{name}"
            arrays[field] = values_array
            inventory_controls.append(field)
            inventory_audit[name]["included"] = True
        else:
            inventory_audit[name]["included"] = False
    return arrays, inventory_controls, inventory_audit


def fit_ols(columns: dict[str, np.ndarray], predictors: list[str]) -> dict:
    y = columns["delta_V"]
    X = np.column_stack([np.ones(len(y))] + [columns[name] for name in predictors])
    names = ["intercept"] + predictors
    beta, _residuals, rank, _singular = np.linalg.lstsq(X, y, rcond=None)
    if rank < X.shape[1]:
        raise RuntimeError(f"Rank-deficient design for predictors {predictors}")
    residual = y - X @ beta
    xtx_inverse = np.linalg.inv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inverse, X)
    adjusted = residual / np.maximum(1.0 - leverage, np.finfo(float).eps)
    meat = X.T @ (X * adjusted[:, None] ** 2)
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_error = np.sqrt(np.diag(covariance))
    dof = len(y) - X.shape[1]
    critical = stats.t.ppf(.975, dof)
    t_value = beta / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_value), dof)
    total = np.sum((y - y.mean()) ** 2)
    r_squared = 1.0 - np.sum(residual ** 2) / total
    coefficients = {}
    for index, name in enumerate(names):
        coefficients[name] = {
            "estimate": float(beta[index]),
            "hc3_standard_error": float(standard_error[index]),
            "hc3_ci_95": [
                float(beta[index] - critical * standard_error[index]),
                float(beta[index] + critical * standard_error[index]),
            ],
            "hc3_t": float(t_value[index]),
            "hc3_p_two_sided": float(p_value[index]),
        }
    return {
        "n": len(y), "predictors": predictors, "rank": int(rank),
        "residual_dof": int(dof), "r_squared": float(r_squared),
        "coefficients": coefficients, "fitted": X @ beta, "residual": residual,
        "beta": beta, "X": X,
    }


def bootstrap_distance(columns: dict[str, np.ndarray], predictors: list[str],
                       draws: int, seed: int) -> dict:
    y = columns["delta_V"]
    X = np.column_stack([np.ones(len(y))] + [columns[name] for name in predictors])
    distance_index = predictors.index("stone_distance") + 1
    rng = np.random.default_rng(seed)
    slopes = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = rng.integers(0, len(y), size=len(y))
        slopes[draw] = np.linalg.lstsq(X[indices], y[indices], rcond=None)[0][distance_index]
    return {
        "draws": draws, "seed": seed,
        "percentile_ci_95": [float(x) for x in np.quantile(slopes, [.025, .975])],
        "probability_slope_negative": float(np.mean(slopes < 0)),
    }


def serializable_model(model: dict) -> dict:
    return {key: value for key, value in model.items()
            if key not in {"fitted", "residual", "beta", "X"}}


def write_coefficients(path: Path, models: dict[str, dict]) -> None:
    fields = [
        "model", "n", "r_squared", "term", "estimate", "hc3_standard_error",
        "hc3_ci_low_95", "hc3_ci_high_95", "hc3_t", "hc3_p_two_sided",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, model in models.items():
            for term, result in model["coefficients"].items():
                writer.writerow({
                    "model": model_name, "n": model["n"], "r_squared": model["r_squared"],
                    "term": term, "estimate": result["estimate"],
                    "hc3_standard_error": result["hc3_standard_error"],
                    "hc3_ci_low_95": result["hc3_ci_95"][0],
                    "hc3_ci_high_95": result["hc3_ci_95"][1],
                    "hc3_t": result["hc3_t"], "hc3_p_two_sided": result["hc3_p_two_sided"],
                })
    temporary.replace(path)


def plot_results(out_dir: Path, columns: dict[str, np.ndarray], model: dict) -> None:
    distance = columns["stone_distance"]
    delta = columns["delta_V"]
    unique = np.unique(distance)
    group_mean = np.asarray([delta[distance == value].mean() for value in unique])
    group_se = np.asarray([
        delta[distance == value].std(ddof=1) / np.sqrt(np.sum(distance == value))
        if np.sum(distance == value) > 1 else np.nan for value in unique
    ])
    group_n = np.asarray([np.sum(distance == value) for value in unique])

    # Added-variable plot: residualize both delta_V and distance against all
    # controls. Its slope equals the primary model's distance coefficient.
    control_names = [name for name in model["predictors"] if name != "stone_distance"]
    controls = np.column_stack(
        [np.ones(len(delta))] + [columns[name] for name in control_names]
    )
    delta_residual = delta - controls @ np.linalg.lstsq(controls, delta, rcond=None)[0]
    distance_residual = distance - controls @ np.linalg.lstsq(controls, distance, rcond=None)[0]
    slope = model["coefficients"]["stone_distance"]["estimate"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(unique, group_mean, yerr=1.96 * group_se, marker="o", linewidth=2,
                capsize=4, color="#d97706")
    for x, y, n in zip(unique, group_mean, group_n):
        ax.annotate(f"n={n}", (x, y), xytext=(5, 7), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="#475569", linestyle="--", linewidth=1)
    ax.set_xlabel("Nearest visible-stone Manhattan distance")
    ax.set_ylabel(r"Mean $\Delta V$ (95% normal CI)")
    ax.set_title("Unadjusted distance pattern")
    ax.grid(alpha=.22)

    ax = axes[1]
    ax.scatter(distance_residual, delta_residual, s=13, alpha=.25, color="#7c3aed")
    line_x = np.asarray([distance_residual.min(), distance_residual.max()])
    ax.plot(line_x, slope * line_x, color="#dc2626", linewidth=2.2,
            label=f"Adjusted slope = {slope:+.3f} per tile")
    ci = model["coefficients"]["stone_distance"]["hc3_ci_95"]
    ax.set_xlabel("Stone distance residual after controls")
    ax.set_ylabel(r"$\Delta V$ residual after controls")
    ax.set_title(f"Added-variable plot; HC3 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    ax.grid(alpha=.22); ax.legend(frameon=False)
    fig.suptitle("Wood-pickaxe value falls as visible stone becomes more distant", fontsize=15)
    fig.savefig(out_dir / "wood_pickaxe_distance_regression.png", dpi=300)
    fig.savefig(out_dir / "wood_pickaxe_distance_regression.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=50)
    args = parser.parse_args()
    if args.bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    source = args.experiment_dir / "wood_pickaxe_value_by_visible_stone_per_state.csv"
    rows = load_visible_rows(source)
    columns, inventory_controls, inventory_audit = add_covariates(rows)
    primary_predictors = [
        "stone_distance", "episode_step", "achievement_count", *inventory_controls,
    ]
    specifications = {
        "unadjusted": ["stone_distance"],
        "progress_controls": ["stone_distance", "episode_step", "achievement_count"],
        "primary_inventory_adjusted": primary_predictors,
        "log_step_robustness": [
            "stone_distance", "log1p_episode_step", "achievement_count", *inventory_controls,
        ],
        "stone_count_robustness": [*primary_predictors, "n_visible_stone"],
    }
    models = {name: fit_ols(columns, predictors)
              for name, predictors in specifications.items()}
    bootstrap = bootstrap_distance(
        columns, primary_predictors, args.bootstrap_draws, args.bootstrap_seed
    )
    primary = models["primary_inventory_adjusted"]
    distance_result = primary["coefficients"]["stone_distance"]
    summary = {
        "cohort": "stone-visible natural states immediately after Make Wood Pickaxe",
        "n": primary["n"],
        "distance_definition": "Manhattan distance from player to nearest visible STONE tile",
        "primary_controls": ["episode_step", "achievement_count", *inventory_controls],
        "inventory_control_audit": inventory_audit,
        "primary_distance_effect": distance_result,
        "interpretation_per_additional_tile": distance_result["estimate"],
        "episode_bootstrap": bootstrap,
        "models": {name: serializable_model(model) for name, model in models.items()},
        "caveat": (
            "Distance is observational and only five distances occur; HC3 and bootstrap "
            "uncertainty do not eliminate unmeasured scene-composition confounding."
        ),
    }
    out_dir = args.experiment_dir / "distance_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_coefficients(out_dir / "coefficients.csv", models)
    plot_results(out_dir, columns, primary)
    print(json.dumps({
        "n": summary["n"], "inventory_controls": inventory_controls,
        "primary_distance_effect": distance_result, "bootstrap": bootstrap,
        "model_distance_estimates": {
            name: model["coefficients"]["stone_distance"]["estimate"]
            for name, model in models.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

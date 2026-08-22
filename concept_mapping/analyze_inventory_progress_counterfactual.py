"""Merge hierarchy-inventory counterfactual shards and fit the within-state model."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def _fixed_effects(x, y, groups):
    unique = np.unique(groups)
    xd, yd = np.empty_like(x, dtype=float), np.empty_like(y, dtype=float)
    for group in unique:
        mask = groups == group
        xd[mask], yd[mask] = x[mask] - x[mask].mean(), y[mask] - y[mask].mean()
    slope = float(np.dot(xd, yd) / np.dot(xd, xd))
    residual = yd - slope * xd
    scores = np.asarray([np.sum(xd[groups == group] * residual[groups == group]) for group in unique])
    variance = (scores @ scores) / (xd @ xd) ** 2 * len(unique) / (len(unique) - 1)
    se = float(np.sqrt(variance))
    z = slope / se
    log_p = float(np.log(2) + norm.logsf(abs(z)))
    p_value = float(np.exp(log_p)) if log_p > np.log(np.finfo(float).tiny) else 0.0
    return slope, se, p_value, log_p, xd, yd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    root, out_dir = Path(args.shard_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, all_rows = [], []
    n_orderings_per_state = None
    base_offset = 0
    for shard in sorted(path for path in root.glob("shard_*") if path.is_dir()):
        manifest = json.loads((shard / "manifest.json").read_text())
        shard_orderings = int(manifest["n_orderings"])
        if n_orderings_per_state is None:
            n_orderings_per_state = shard_orderings
        elif n_orderings_per_state != shard_orderings:
            raise ValueError("All shards must use the same number of orderings")
        states = np.load(shard / "base_states.npz")
        arrays.append(states["base_obs"])
        with (shard / "counterfactual_values.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["base_index"] = int(row["base_index"]) + base_offset
                row["ordering_index"] = int(row["ordering_index"])
                row["inventory_level"] = int(row["inventory_level"])
                row["value"] = float(row["value"])
                all_rows.append(row)
        base_offset += len(states["base_obs"])
    np.savez_compressed(out_dir / "base_states.npz", base_obs=np.concatenate(arrays))
    # Average valid tie-breaks, making each base state contribute exactly one
    # value at each inventory level. For an empirical ordering this is one path.
    grouped = {}
    for row in all_rows:
        grouped.setdefault((row["base_index"], row["inventory_level"]), []).append(row["value"])
    base, level, value = zip(*[(key[0], key[1], float(np.mean(vals))) for key, vals in sorted(grouped.items())])
    base, level, value = np.asarray(base), np.asarray(level, float), np.asarray(value, float)
    n_levels = int(level.max()) + 1
    slope, se, p_value, log_p, xd, yd = _fixed_effects(level, value, base)
    ci = [slope - 1.96 * se, slope + 1.96 * se]
    # Held-out state-level validation of the same within-state slope.
    rng = np.random.default_rng(args.seed)
    unique = np.unique(base)
    test_groups = set(rng.choice(unique, size=max(1, round(.2 * len(unique))), replace=False))
    train_mask = np.asarray([group not in test_groups for group in base])
    test_mask = ~train_mask
    train_slope, *_ = _fixed_effects(level[train_mask], value[train_mask], base[train_mask])
    test_xd, test_yd = level[test_mask].copy(), value[test_mask].copy()
    for group in np.unique(base[test_mask]):
        mask = base[test_mask] == group
        test_xd[mask] -= test_xd[mask].mean(); test_yd[mask] -= test_yd[mask].mean()
    test_pred = train_slope * test_xd
    heldout_r2 = float(1 - np.sum((test_yd - test_pred) ** 2) / np.sum(test_yd ** 2))
    heldout_rmse = float(np.sqrt(np.mean((test_yd - test_pred) ** 2)))
    consecutive = []
    for group in unique:
        series = value[base == group][np.argsort(level[base == group])]
        consecutive.extend(np.diff(series))
    summary = {
        "n_base_states": int(len(unique)), "n_counterfactual_values_raw": int(len(all_rows)),
        "n_orderings_per_state": n_orderings_per_state, "n_levels": n_levels,
        "slope_fixed_effects": slope, "slope_standard_error_clustered_by_state": se,
        "slope_95_ci": ci, "slope_p_value_two_sided": p_value,
        "slope_log10_p_value_two_sided": float(log_p / np.log(10)),
        "global_intercept_at_level_zero": float(value.mean() - slope * level.mean()),
        "probability_consecutive_inventory_addition_lowers_value": float(np.mean(np.asarray(consecutive) < 0)),
        "heldout_within_state_r2": heldout_r2, "heldout_within_state_rmse": heldout_rmse,
        "heldout_train_slope": train_slope,
    }
    (out_dir / "regression_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (out_dir / "mean_values_by_level.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["inventory_level", "mean_value", "standard_error", "n_states"])
        means = []
        for k in range(n_levels):
            vals = value[level == k]; means.append(vals.mean())
            writer.writerow([k, vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals)), len(vals)])
    fig, ax = plt.subplots(figsize=(7.3, 4.6), constrained_layout=True)
    for group in np.random.default_rng(args.seed).choice(unique, size=min(160, len(unique)), replace=False):
        mask = base == group
        ax.plot(level[mask], value[mask], color="#94a3b8", alpha=.10, linewidth=.7)
    mean_arr = np.asarray(means)
    plot_levels = range(n_levels)
    sem = np.asarray([value[level == k].std(ddof=1) / np.sqrt(np.sum(level == k)) for k in plot_levels])
    ax.plot(plot_levels, mean_arr, color="#b91c1c", linewidth=2.5, label="Mean across base states")
    ax.fill_between(plot_levels, mean_arr - 1.96 * sem, mean_arr + 1.96 * sem, color="#b91c1c", alpha=.18, label="95% CI")
    ax.set_xlabel("Progression-ordered inventory components present")
    ax.set_ylabel("Frozen critic value")
    ax.set_title("Critic value along counterfactual inventory progression")
    ax.grid(alpha=.24); ax.legend(frameon=False)
    fig.savefig(out_dir / "inventory_progress_value.png", dpi=300)
    fig.savefig(out_dir / "inventory_progress_value.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

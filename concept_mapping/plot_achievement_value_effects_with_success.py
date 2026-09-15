"""Overlay W&B achievement success on stabilized critic effects over training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml


DEFAULT_RUN = Path("wandb/run-20260830_231509-616wuss8")
DEFAULT_EFFECT_DIR = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "achievement_stabilized_effect_over_checkpoints_stride3"
)
DEFAULT_WANDB_RUN = "synaderi-uc-davis/Craftax_Baselines/616wuss8"


def config_value(config: dict, key: str):
    value = config[key]
    return value["value"] if isinstance(value, dict) and "value" in value else value


def fetch_success_history(run_ref: str, names: list[str], steps_per_update: int, destination: Path):
    import wandb

    keys = [f"Achievements/{name}" for name in names]
    run = wandb.Api(timeout=120).run(run_ref)
    rows = []
    for history in run.scan_history(keys=["_step", *keys], page_size=10_000):
        if history.get("_step") is None:
            continue
        available = {name: history.get(key) for name, key in zip(names, keys)}
        if not any(value is not None for value in available.values()):
            continue
        rows.append({
            "wandb_update": int(history["_step"]),
            "environment_steps": int(history["_step"] + 1) * steps_per_update,
            **{name: value for name, value in available.items()},
        })
    if not rows:
        raise RuntimeError(f"No achievement history found in {run_ref}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def log_bin_mean(x: np.ndarray, y: np.ndarray, bins: int = 500):
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    edges = np.geomspace(x.min(), x.max() * (1.0 + 1e-12), bins + 1)
    indices = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    counts = np.bincount(indices, minlength=bins)
    sums = np.bincount(indices, weights=y, minlength=bins)
    keep = counts > 0
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers[keep], sums[keep] / counts[keep]


def rolling_median(values: np.ndarray, window: int = 9):
    # Edge-aware smoothing prevents the line from changing color halfway into
    # training and preserves a single continuous primary trajectory.
    radius = window // 2
    return np.asarray([
        np.median(values[max(0, index - radius):min(len(values), index + radius + 1)])
        for index in range(len(values))
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--effect_dir", type=Path, default=DEFAULT_EFFECT_DIR)
    parser.add_argument("--wandb_run", default=DEFAULT_WANDB_RUN)
    parser.add_argument("--refresh_wandb", action="store_true")
    parser.add_argument(
        "--value_metric", choices=("stabilized", "raw"), default="stabilized",
        help="Plot stabilized relative change or median raw V_after - V_before.",
    )
    args = parser.parse_args()

    effects_path = args.effect_dir / "achievement_stabilized_effect_over_checkpoints.csv"
    with effects_path.open(newline="") as handle:
        effect_rows = list(csv.DictReader(handle))
    base_names = [key[2:] for key in effect_rows[0] if key.startswith("R_")]
    names = list(base_names)
    place_stone_path = (
        args.effect_dir / "place_stone_checkpoint_sweep_stride3/"
        "place_stone_effect_over_checkpoints.csv"
    )
    place_stone_rows = None
    if place_stone_path.exists():
        with place_stone_path.open(newline="") as handle:
            place_stone_rows = list(csv.DictReader(handle))
        names.append("place_stone")

    order_path = (
        args.effect_dir.parent /
        "achievement_transition_relative_value_1p5b/achievement_summary_and_order.csv"
    )
    with order_path.open(newline="") as handle:
        ordered_names = [row["achievement"] for row in csv.DictReader(handle)]
    if place_stone_rows is not None:
        # Empirically, Place Stone has median first-completion step 28, directly
        # after Collect Stone (median step 26).
        ordered_names.insert(ordered_names.index("collect_stone") + 1, "place_stone")

    with (args.run_path / "files/config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    steps_per_update = int(config_value(config, "NUM_ENVS")) * int(config_value(config, "NUM_STEPS"))
    success_path = args.effect_dir / "wandb_achievement_success_history.csv"
    cached_names = []
    if success_path.exists():
        with success_path.open(newline="") as handle:
            cached_names = list(csv.DictReader(handle).fieldnames or [])
    if args.refresh_wandb or not success_path.exists() or not all(name in cached_names for name in names):
        fetch_success_history(args.wandb_run, names, steps_per_update, success_path)
    with success_path.open(newline="") as handle:
        success_rows = list(csv.DictReader(handle))

    x_effect = np.asarray([float(row["checkpoint"]) for row in effect_rows])
    x_success_all = np.asarray([float(row["environment_steps"]) for row in success_rows])
    if args.value_metric == "stabilized":
        effect_by_name = {
            name: 100.0 * np.asarray([float(row[f"R_{name}"]) for row in effect_rows])
            for name in base_names
        }
        if place_stone_rows is not None:
            effect_by_name["place_stone"] = 100.0 * np.asarray([
                float(row["median_stabilized_place_stone"]) for row in place_stone_rows
            ])
            combined_csv = (
                args.effect_dir /
                "achievement_stabilized_effect_over_checkpoints_with_place_stone.csv"
            )
            with combined_csv.open("w", newline="") as handle:
                fields = ["checkpoint", *[f"R_{name}" for name in names]]
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                for checkpoint_column, checkpoint in enumerate(x_effect.astype(np.int64)):
                    writer.writerow({
                        "checkpoint": int(checkpoint),
                        **{f"R_{name}": float(effect_by_name[name][checkpoint_column] / 100.0)
                           for name in names},
                    })
        left_label = r"Median stabilized relative value change, $R_c(t)$ (\%)"
        blue_legend = r"Stabilized value change, $R_c(t)$"
        output_stem = "achievement_value_effects_with_wandb_success_4x4"
        value_metric_description = "median[(V_after - V_before)/(1 + abs(V_before))]"
        left_bound = 50.0
    else:
        bank_path = args.effect_dir.parent / "achievement_transition_fixed_bank.npz"
        with np.load(bank_path) as bank:
            concept_index = np.asarray(bank["concept_index"])
            bank_names = np.asarray(bank["concept_names"]).astype(str).tolist()
        raw_matrix = np.empty((len(base_names), len(effect_rows)), dtype=np.float64)
        bank_lookup = {name: index for index, name in enumerate(bank_names)}
        for checkpoint_column, row in enumerate(effect_rows):
            checkpoint = int(row["checkpoint"])
            with np.load(args.effect_dir / "per_checkpoint" / f"checkpoint_{checkpoint}.npz") as data:
                delta = np.asarray(data["value_after"]) - np.asarray(data["value_before"])
            for name_index, name in enumerate(base_names):
                raw_matrix[name_index, checkpoint_column] = np.median(
                    delta[concept_index == bank_lookup[name]]
                )
        effect_by_name = {name: raw_matrix[index] for index, name in enumerate(base_names)}
        if place_stone_rows is not None:
            effect_by_name["place_stone"] = np.asarray([
                float(row["median_raw_delta_place_stone"]) for row in place_stone_rows
            ])
        raw_csv = args.effect_dir / "achievement_raw_value_effect_over_checkpoints.csv"
        with raw_csv.open("w", newline="") as handle:
            fields = ["checkpoint", *[f"median_delta_{name}" for name in names]]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for checkpoint_column, checkpoint in enumerate(x_effect.astype(np.int64)):
                writer.writerow({
                    "checkpoint": int(checkpoint),
                    **{f"median_delta_{name}": float(effect_by_name[name][checkpoint_column])
                       for name in names},
                })
        left_label = r"Median raw critic-value change, $\Delta V_c(t)$"
        blue_legend = r"Raw value change, $\Delta V_c(t)$"
        output_stem = "achievement_raw_value_effects_with_wandb_success_4x4"
        value_metric_description = "median[V_after - V_before]"
        all_raw = np.concatenate(list(effect_by_name.values()))
        robust_maximum = float(np.quantile(np.abs(all_raw[np.isfinite(all_raw)]), .995))
        left_bound = max(.25, np.ceil(1.1 * robust_maximum * 4.0) / 4.0)
    success_by_name = {}
    for name in names:
        x_values, y_values = [], []
        for x_value, row in zip(x_success_all, success_rows):
            raw = row.get(name, "")
            if raw not in ("", None):
                x_values.append(x_value); y_values.append(float(raw))
        success_by_name[name] = log_bin_mean(
            np.asarray(x_values), np.asarray(y_values), bins=500
        )

    fig, left_axes = plt.subplots(
        4, 4, figsize=(16.0, 11.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    left_axes = left_axes.ravel()
    right_axes = []
    for index, (left, name) in enumerate(zip(left_axes, ordered_names)):
        right = left.twinx()
        right_axes.append(right)
        left.plot(
            x_effect, rolling_median(effect_by_name[name]),
            color="#1d4ed8", linewidth=1.8, zorder=3,
        )
        x_success, success = success_by_name[name]
        right.plot(x_success, success, color="#dc2626", linewidth=1.6, alpha=.9, zorder=2)
        left.axhline(0, color="#64748b", linestyle="--", linewidth=.7)
        left.set_xscale("log")
        left.set_xticks(
            [1e5, 1e6, 1e7, 1e8, 1e9],
            [r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$"],
        )
        left.tick_params(axis="x", labelbottom=True)
        left.set_ylim(-left_bound, left_bound)
        right.set_ylim(0, 100)
        left.set_title(name.replace("_", " ").title(), fontsize=11, fontweight="semibold")
        left.grid(True, which="both", alpha=.17)

        row, column = divmod(index, 4)
        if column != 3 and index != len(ordered_names) - 1:
            right.set_yticklabels([])
            right.tick_params(axis="y", length=0)
        else:
            right.tick_params(axis="y", colors="#b91c1c", labelsize=8)
        if column != 0:
            left.tick_params(axis="y", labelleft=False)

    for left in left_axes[len(ordered_names):]:
        left.axis("off")

    fig.supxlabel("Training environment steps", fontsize=13)
    fig.supylabel(left_label, fontsize=13)
    right_axes[7].set_ylabel("Achievement success rate (%)", color="#b91c1c", fontsize=12)
    fig.suptitle("Achievement completion: critic-value effects and behavioral success", fontsize=17)
    # Use an otherwise empty last-row panel for the legend so it cannot cover
    # the subplot titles or data.
    left_axes[14].legend(
        handles=[
            Line2D([0], [0], color="#1d4ed8", linewidth=2.2,
                   label=blue_legend),
            Line2D([0], [0], color="#dc2626", linewidth=2.2,
                   label="W&B achievement success"),
        ],
        loc="center", frameon=False, fontsize=12,
    )
    png = args.effect_dir / f"{output_stem}.png"
    pdf = args.effect_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=300); fig.savefig(pdf); plt.close(fig)

    metadata = {
        "wandb_run": args.wandb_run,
        "value_metric": value_metric_description,
        "value_smoothing": "edge-aware rolling median over 9 evaluated checkpoints",
        "success_metric": "training-time Achievements/<name> logged by W&B",
        "success_smoothing": "mean within 500 logarithmically spaced environment-step bins",
        "layout": f"4x4 equal-size panels for {len(ordered_names)} achievements",
    }
    (args.effect_dir / f"{output_stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved {png}")


if __name__ == "__main__":
    main()

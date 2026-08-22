"""Create a paper-style smoothed diamond-collection learning curve from W&B data."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from google.protobuf.message import DecodeError

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


METRIC = "Achievements/collect_diamond"


def _files_dir(run_path: str) -> Path:
    path = Path(run_path)
    return path if path.name == "files" else path / "files"


def _config_value(config: dict, key: str):
    value = config[key]
    return value["value"] if isinstance(value, dict) and "value" in value else value


def _read_local_history(run_path: str):
    run_dir = _files_dir(run_path).parent
    wandb_files = sorted(run_dir.glob("run-*.wandb"))
    if len(wandb_files) != 1:
        raise FileNotFoundError(f"Expected exactly one run-*.wandb file in {run_dir}")
    store = DataStore()
    store.open_for_scan(str(wandb_files[0]))
    points = []
    while True:
        try:
            scanned = store.scan_record()
        except (IndexError, OSError):
            # Local W&B files may end with an incomplete record if the process
            # was interrupted after all history records had already been saved.
            break
        if scanned is None:
            break
        _, encoded = scanned
        record = wandb_internal_pb2.Record()
        try:
            record.ParseFromString(encoded)
        except DecodeError:
            continue
        if record.WhichOneof("record_type") != "history":
            continue
        values = {
            item.key or "/".join(item.nested_key): json.loads(item.value_json)
            for item in record.history.item
        }
        if METRIC in values and "_step" in values:
            points.append((int(values["_step"]), float(values[METRIC])))
    if not points:
        raise RuntimeError(f"No {METRIC!r} history records found")
    points.sort()
    return np.asarray(points, dtype=np.float64)


def _read_wandb_history(run_ref: str):
    import wandb

    api = wandb.Api(timeout=60)
    run = api.run(run_ref)
    points = []
    for row in run.scan_history(keys=["_step", METRIC], page_size=10_000):
        if row.get("_step") is not None and row.get(METRIC) is not None:
            points.append((int(row["_step"]), float(row[METRIC])))
    if not points:
        raise RuntimeError(f"No {METRIC!r} history records found in W&B run {run_ref}")
    points.sort()
    return np.asarray(points, dtype=np.float64)


def _centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window_updates must be positive")
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument(
        "--out_dir",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/diamond_success_curve",
    )
    parser.add_argument(
        "--window_updates",
        type=int,
        default=1024,
        help="Centered rolling-mean width in PPO updates.",
    )
    parser.add_argument(
        "--history_source",
        choices=("wandb", "local"),
        default="wandb",
        help="Read complete synced history from W&B, or the local .wandb file.",
    )
    parser.add_argument(
        "--wandb_run",
        default="synaderi-uc-davis/Craftax_Baselines/m0mw4end",
        help="entity/project/run-id used when --history_source=wandb.",
    )
    args = parser.parse_args()

    files_dir = _files_dir(args.run_path)
    with (files_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    steps_per_update = int(_config_value(config, "NUM_ENVS")) * int(_config_value(config, "NUM_STEPS"))
    history = (
        _read_wandb_history(args.wandb_run)
        if args.history_source == "wandb"
        else _read_local_history(args.run_path)
    )
    update = history[:, 0]
    raw = history[:, 1]
    environment_steps = (update + 1) * steps_per_update
    smoothed = _centered_rolling_mean(raw, args.window_updates)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "diamond_success_smoothed.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wandb_update", "environment_steps", "diamond_success_raw_pct", "diamond_success_smoothed_pct"])
        writer.writerows(zip(update.astype(int), environment_steps.astype(np.int64), raw, smoothed))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(environment_steps, smoothed, color="#6d28d9", linewidth=2.4)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Diamond collection success (%)")
    ax.set_title("Diamond collection during PPO training")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, environment_steps.max())
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e9:g}B"))
    ax.text(
        0.99,
        0.03,
        f"Centered rolling mean: {args.window_updates:,} updates ({args.window_updates * steps_per_update / 1e6:.1f}M steps)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4b5563",
    )
    fig.savefig(out_dir / "diamond_success_smoothed.png", dpi=300)
    fig.savefig(out_dir / "diamond_success_smoothed.pdf")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

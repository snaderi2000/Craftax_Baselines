"""Plot the direct critic effect of a wood pickaxe beside collectible stone.

Reuses the saved functional-sweep arrays. A state succeeds when
V(stone present, wood pickaxe) > V(stone present, no wood pickaxe).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from concept_mapping.plot_diamond_ordering_over_checkpoints import _checkpoint_steps


DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_BANK = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "wood_pickaxe_functional/wood_pickaxe_functional_bank.npz"
)
DEFAULT_SWEEP = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "wood_pickaxe_functional/checkpoint_sweep"
)
DEFAULT_OUT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "progression_vs_pickaxe_value/direct_pickaxe_effect_stride4"
)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    # Leave the boundary undefined rather than averaging a smaller, asymmetric
    # window that can visually distort the earliest and latest checkpoints.
    result = np.full_like(values, np.nan, dtype=float)
    radius = window // 2
    for index in range(radius, len(values) - radius):
        result[index] = values[index - radius:index + radius + 1].mean()
    return result


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    radius = window // 2
    for index in range(radius, len(values) - radius):
        result[index] = np.median(values[index - radius:index + radius + 1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--sweep_dir", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint_stride", type=int, default=4)
    parser.add_argument("--smooth_window", type=int, default=9)
    args = parser.parse_args()

    all_steps = _checkpoint_steps(args.run_path)
    steps = all_steps[::args.checkpoint_stride]
    if steps[-1] != all_steps[-1]:
        steps.append(all_steps[-1])
    with np.load(args.bank) as bank:
        stone = np.asarray(bank["resource"]).astype(str) == "stone"
    rows = []
    for step in steps:
        path = args.sweep_dir / "per_checkpoint" / f"checkpoint_{step}.npz"
        with np.load(path) as data:
            effect = np.asarray(data["tool_effect_value_resource_present"])[stone]
        rows.append({
            "checkpoint": step, "n_states": int(len(effect)),
            "success_rate": float(np.mean(effect > 0)),
            "mean_direct_effect": float(effect.mean()),
            "median_direct_effect": float(np.median(effect)),
            "standard_error_direct_effect": float(effect.std(ddof=1) / np.sqrt(len(effect))),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "direct_pickaxe_effect_over_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    x = np.asarray([row["checkpoint"] for row in rows], dtype=float)
    success = np.asarray([row["success_rate"] for row in rows], dtype=float)
    smooth = _rolling_mean(success, args.smooth_window)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(x, 100 * success, color="#60a5fa", linewidth=1.0, alpha=.45, label="Raw")
    ax.plot(x, 100 * smooth, color="#1d4ed8", linewidth=2.2,
            label=f"Centered mean ({args.smooth_window} checkpoints)")
    ax.axhline(50, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xscale("log"); ax.set_ylim(0, 100)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(r"States with $V(\mathrm{stone},P^+)>V(\mathrm{stone},P^-)$ (\%)")
    ax.set_title("Direct critic value of the wood pickaxe beside stone")
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False)
    fig.savefig(args.out_dir / "wood_pickaxe_direct_success.png", dpi=300)
    fig.savefig(args.out_dir / "wood_pickaxe_direct_success.pdf"); plt.close(fig)

    mean_effect = np.asarray([row["mean_direct_effect"] for row in rows], dtype=float)
    standard_error = np.asarray([row["standard_error_direct_effect"] for row in rows], dtype=float)
    smooth_effect = _rolling_mean(mean_effect, args.smooth_window)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.fill_between(x, mean_effect - 1.96 * standard_error, mean_effect + 1.96 * standard_error,
                    color="#93c5fd", alpha=.22, linewidth=0, label="95% CI")
    ax.plot(x, mean_effect, color="#60a5fa", linewidth=1.0, alpha=.5, label="Checkpoint mean")
    ax.plot(x, smooth_effect, color="#1d4ed8", linewidth=2.2,
            label=f"Centered mean ({args.smooth_window} checkpoints)")
    ax.axhline(0, color="#4b5563", linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(r"Mean $\Delta V_{\mathrm{pickaxe}}$")
    ax.set_title("Mean critic-value effect of the wood pickaxe beside stone")
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False)
    fig.savefig(args.out_dir / "wood_pickaxe_direct_mean_effect.png", dpi=300)
    fig.savefig(args.out_dir / "wood_pickaxe_direct_mean_effect.pdf"); plt.close(fig)

    median_effect = np.asarray([row["median_direct_effect"] for row in rows], dtype=float)
    smooth_median_effect = _rolling_median(median_effect, args.smooth_window)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(x, median_effect, color="#60a5fa", linewidth=1.0, alpha=.5,
            label="Checkpoint median")
    ax.plot(x, smooth_median_effect, color="#1d4ed8", linewidth=2.2,
            label=f"Centered median ({args.smooth_window} checkpoints)")
    ax.axhline(0, color="#4b5563", linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel(r"Median $[V(\mathrm{stone},P^+)-V(\mathrm{stone},P^-)]$")
    ax.set_title("Median critic-value effect of the wood pickaxe beside stone")
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False)
    fig.savefig(args.out_dir / "wood_pickaxe_direct_median_effect.png", dpi=300)
    fig.savefig(args.out_dir / "wood_pickaxe_direct_median_effect.pdf"); plt.close(fig)

    summary = {
        "definition": "P[V(stone, wood pickaxe) > V(stone, no wood pickaxe)]",
        "n_states": int(rows[0]["n_states"]), "checkpoint_stride": args.checkpoint_stride,
        "n_checkpoints": len(rows), "final_checkpoint_included": int(steps[-1]),
        "first_success_rate": float(success[0]), "final_success_rate": float(success[-1]),
        "peak_raw_success_rate": float(success.max()),
        "peak_raw_checkpoint": int(x[np.argmax(success)]),
        "peak_smoothed_success_rate": float(np.nanmax(smooth)),
        "peak_smoothed_checkpoint": int(x[np.nanargmax(smooth)]),
        "first_mean_direct_effect": float(mean_effect[0]),
        "final_mean_direct_effect": float(mean_effect[-1]),
        "peak_mean_direct_effect": float(mean_effect.max()),
        "peak_mean_effect_checkpoint": int(x[np.argmax(mean_effect)]),
        "first_median_direct_effect": float(median_effect[0]),
        "final_median_direct_effect": float(median_effect[-1]),
        "peak_median_direct_effect": float(median_effect.max()),
        "peak_median_effect_checkpoint": int(x[np.argmax(median_effect)]),
        "peak_smoothed_median_direct_effect": float(np.nanmax(smooth_median_effect)),
        "peak_smoothed_median_effect_checkpoint": int(x[np.nanargmax(smooth_median_effect)]),
    }
    (args.out_dir / "direct_pickaxe_effect_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

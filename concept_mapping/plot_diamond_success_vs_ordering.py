"""Overlay smoothed behavioural diamond success with critic-ordering emergence."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--success_csv",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/diamond_success_curve/diamond_success_smoothed.csv",
    )
    parser.add_argument(
        "--ordering_csv",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/ordering_over_checkpoints/ordering_over_checkpoints.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/diamond_success_vs_ordering",
    )
    parser.add_argument("--xscale", choices=("linear", "log"), default="linear")
    args = parser.parse_args()

    success = _read_csv(Path(args.success_csv))
    ordering = _read_csv(Path(args.ordering_csv))
    success_x = np.asarray([float(row["environment_steps"]) for row in success])
    success_y = np.asarray([float(row["diamond_success_smoothed_pct"]) for row in success])
    ordering_x = np.asarray([float(row["checkpoint"]) for row in ordering])
    ordering_y = 100 * np.asarray([float(row["full_ordering_satisfied"]) for row in ordering])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    ax.plot(
        success_x,
        success_y,
        color="#2563eb",
        linewidth=2.3,
        label="Diamond collection success (smoothed)",
    )
    ax.plot(
        ordering_x,
        ordering_y,
        color="#dc2626",
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        label="Diamond-pickaxe ordering",
    )
    if args.xscale == "log":
        ax.set_xscale("log")
        # Use interpretable decade ticks rather than fractional billions.
        ax.set_xlim(1e5, 1e10)
        ax.xaxis.set_major_locator(FixedLocator([10.0**exponent for exponent in range(5, 11)]))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    else:
        ax.set_xlim(0, success_x.max())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e9:g}B"))
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Behavioural success and critic concept emergence")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    suffix = "_logx" if args.xscale == "log" else ""
    fig.savefig(out_dir / f"diamond_success_vs_ordering{suffix}.png", dpi=300)
    fig.savefig(out_dir / f"diamond_success_vs_ordering{suffix}.pdf")


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_DIR = Path("concept_mapping/runs/diamond_affordance_over_time")
CSV_PATH = RUN_DIR / "partial_order_summary.csv"
OUT_PATH = RUN_DIR / "partial_order_over_time.png"
OUT_CATEGORICAL_PATH = RUN_DIR / "partial_order_over_time_even_spacing.png"


def _label_step(step: int) -> str:
    if step >= 1_000_000_000:
        value = step / 1_000_000_000
        return f"{value:.0f}B" if abs(value - round(value)) < 0.05 else f"{value:.1f}B"
    if step >= 1_000_000:
        value = step / 1_000_000
        return f"{value:.0f}M" if abs(value - round(value)) < 0.05 else f"{value:.1f}M"
    return f"{round(step / 1000):.0f}k"


def main() -> None:
    steps = []
    pct = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["checkpoint"]))
            pct.append(float(row["partial_order_pct"]) * 100.0)
    steps = np.asarray(steps)
    pct = np.asarray(pct)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D7DCE2",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.2), dpi=180)
    ax.plot(steps, pct, color="#175CD3", linewidth=3.0, marker="o", markersize=7)
    ax.fill_between(steps, pct, 0, color="#175CD3", alpha=0.10)

    for x, y in zip(steps, pct):
        ax.text(x, y + 3.0, f"{y:.0f}%", ha="center", va="bottom", fontsize=10, color="#1F2937")

    ax.set_xscale("log")
    ax.set_ylim(-3, 100)
    ax.set_title("Emergence of Diamond-Pickaxe Affordance", loc="left", pad=12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Partial-order correct (%)")
    ax.set_xticks(steps)
    ax.set_xticklabels([_label_step(int(s)) for s in steps], rotation=0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E8ECF2", linewidth=0.9)
    ax.grid(axis="x", color="#F3F5F8", linewidth=0.6)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")

    fig, ax = plt.subplots(figsize=(10.8, 5.2), dpi=180)
    x = np.arange(len(steps))
    ax.plot(x, pct, color="#175CD3", linewidth=3.0, marker="o", markersize=7)
    ax.fill_between(x, pct, 0, color="#175CD3", alpha=0.10)
    for xi, y in zip(x, pct):
        ax.text(xi, y + 3.0, f"{y:.0f}%", ha="center", va="bottom", fontsize=10, color="#1F2937")
    ax.set_ylim(-3, 100)
    ax.set_title("Emergence of Diamond-Pickaxe Affordance", loc="left", pad=12)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Partial-order correct (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([_label_step(int(s)) for s in steps], rotation=0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#E8ECF2", linewidth=0.9)
    ax.grid(axis="x", color="#F3F5F8", linewidth=0.6)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(OUT_CATEGORICAL_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_CATEGORICAL_PATH}")


if __name__ == "__main__":
    main()

"""Plot all natural achievement-transition coherence trajectories together."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "achievement_transition_coherence_stride3"
)

# Empirical order from median first completion in 1,000 frozen-policy episodes.
# Values in parentheses are median within-episode completion steps, not PPO
# environment-training steps on the horizontal axis.
EMPIRICAL_ORDER = (
    ("collect_wood", 10.0),
    ("place_table", 21.0),
    ("make_wood_pickaxe", 24.0),
    ("collect_stone", 26.0),
    ("place_stone", 28.0),
    ("place_furnace", 35.0),
    ("make_wood_sword", 42.0),
    ("make_stone_pickaxe", 77.0),
    ("collect_coal", 90.0),
    ("make_stone_sword", 107.0),
    ("collect_iron", 110.0),
    ("make_iron_sword", 157.0),
    ("make_iron_pickaxe", 223.0),
    ("collect_diamond", 274.5),
)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.2), constrained_layout=True)
    colors = plt.cm.turbo(np.linspace(0.04, 0.96, len(EMPIRICAL_ORDER)))

    for rank, ((name, median_step), color) in enumerate(zip(EMPIRICAL_ORDER, colors), 1):
        path = ROOT / name / "metrics.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        training_steps = np.asarray([float(row["checkpoint"]) for row in rows])
        coherence = np.asarray([float(row["top_eigenvalue_fraction"]) for row in rows])
        label = f"{rank}. {name.replace('_', ' ').title()} ({median_step:g})"
        ax.plot(training_steps, coherence, color=color, linewidth=2.0,
                alpha=0.94, label=label)

    ax.set_xscale("log")
    ax.set_xlim(6.0e4, 1.65e9)
    ax.set_ylim(0.0, 0.95)
    ax.set_xlabel("Training environment steps", fontsize=14)
    ax.set_ylabel(r"Achievement-transition coherence, $r_c(t)$", fontsize=14)
    ax.set_title("Emergence of critic achievement-transition representations", fontsize=18)
    ax.grid(True, which="major", alpha=0.26)
    ax.grid(True, which="minor", axis="x", alpha=0.10)
    ax.tick_params(labelsize=12)
    ax.legend(
        title="Empirical acquisition order (median episode step)",
        frameon=True, framealpha=0.94, fontsize=9.5, title_fontsize=10.5,
        ncol=2, loc="upper left",
    )

    png = ROOT / "achievement_transition_coherence_all_concepts.png"
    pdf = ROOT / "achievement_transition_coherence_all_concepts.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

"""Overlay achievement-transition coherence and behavioral success."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from concept_mapping.plot_achievement_value_effects_with_success import log_bin_mean


ROOT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "achievement_transition_coherence_stride3"
)
SUCCESS_CSV = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "achievement_stabilized_effect_over_checkpoints_stride3/"
    "wandb_achievement_success_history.csv"
)
ORDER = (
    "collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
    "place_stone", "place_furnace", "make_wood_sword", "make_stone_pickaxe",
    "collect_coal", "make_stone_sword", "collect_iron", "make_iron_sword",
    "make_iron_pickaxe", "collect_diamond",
)
LOG_TICKS = (1e5, 1e6, 1e7, 1e8, 1e9)
LOG_LABELS = (r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$")


def main() -> None:
    with SUCCESS_CSV.open(newline="") as handle:
        success_rows = list(csv.DictReader(handle))
    success_steps = np.asarray([float(row["environment_steps"]) for row in success_rows])

    fig, left_axes = plt.subplots(
        4, 4, figsize=(16, 11), sharex=True, sharey=True,
        constrained_layout=True,
    )
    left_axes = left_axes.ravel()
    right_axes = []

    for index, (left, name) in enumerate(zip(left_axes, ORDER)):
        with (ROOT / name / "metrics.csv").open(newline="") as handle:
            coherence_rows = list(csv.DictReader(handle))
        checkpoints = np.asarray([float(row["checkpoint"]) for row in coherence_rows])
        coherence = np.asarray([
            float(row["top_eigenvalue_fraction"]) for row in coherence_rows
        ])

        available_steps, available_success = [], []
        for step, row in zip(success_steps, success_rows):
            value = row.get(name, "")
            if value not in (None, ""):
                available_steps.append(step)
                available_success.append(float(value))
        success_x, success_y = log_bin_mean(
            np.asarray(available_steps), np.asarray(available_success), bins=500
        )

        right = left.twinx()
        right_axes.append(right)
        left.plot(checkpoints, coherence, color="#1d4ed8", linewidth=1.8, zorder=3)
        right.plot(success_x, success_y, color="#dc2626", linewidth=1.55,
                   alpha=.90, zorder=2)

        left.set_xscale("log")
        left.set_xlim(6e4, 1.65e9)
        left.set_ylim(0, 1)
        right.set_ylim(0, 100)
        left.set_xticks(LOG_TICKS, LOG_LABELS)
        # Shared axes normally hide inner tick labels. Re-enable them because
        # every panel should be independently readable when cropped for slides.
        left.tick_params(axis="x", labelbottom=True, labelsize=9)
        left.tick_params(axis="y", labelsize=9)
        left.set_title(name.replace("_", " ").title(), fontsize=11,
                       fontweight="semibold")
        left.grid(True, which="major", alpha=.22)
        left.grid(True, which="minor", axis="x", alpha=.09)

        row, column = divmod(index, 4)
        if column == 3 or index == len(ORDER) - 1:
            right.tick_params(axis="y", colors="#b91c1c", labelsize=8)
        else:
            right.set_yticklabels([])
            right.tick_params(axis="y", length=0)

    for left in left_axes[len(ORDER):]:
        left.axis("off")

    left_axes[14].legend(
        handles=(
            Line2D([0], [0], color="#1d4ed8", linewidth=2.2,
                   label=r"Representation coherence, $r_c(t)$"),
            Line2D([0], [0], color="#dc2626", linewidth=2.2,
                   label="W&B achievement success"),
        ),
        loc="center", frameon=False, fontsize=12,
    )
    fig.supxlabel("Training environment steps", fontsize=13)
    fig.supylabel(r"Achievement-transition coherence, $r_c(t)$", fontsize=13)
    right_axes[7].set_ylabel("Achievement success rate (%)", color="#b91c1c",
                             fontsize=12)
    fig.suptitle(
        "Achievement-transition coherence and behavioral success over training",
        fontsize=17,
    )
    png = ROOT / "achievement_transition_coherence_with_wandb_success_4x4.png"
    pdf = ROOT / "achievement_transition_coherence_with_wandb_success_4x4.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

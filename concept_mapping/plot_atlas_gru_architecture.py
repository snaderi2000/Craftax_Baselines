"""Render the concat GRU policy used in the 10B Craftax field atlas."""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


RUN = Path(__file__).resolve().parent / "runs/lava_only_gru_concat_dp2_10b_seed50/gru"
INK, MUTED, PAPER = "#142c23", "#475e50", "#f7f8ef"
GREEN, BLUE, GOLD = "#337053", "#3b6786", "#a36521"


def draw(target):
    config = json.loads((RUN / "inference_config.json").read_text())
    metadata = json.loads((RUN / "experiment_metadata.json").read_text())
    assert config["GRU_CONCAT_OBSERVATION"] and not config["RESET_GRU_EVERY_STEP"]
    assert config["SYMBOLIC_ENCODER"] == "mlp" and config["MEMORY_TYPE"] == "gru"
    width = config["LAYER_SIZE"]
    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11.4, 8.1), facecolor=PAPER)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(100, 0))
    ax.axis("off")

    def text(x, y, value, size, color=INK, bold=False, align="left", family="DejaVu Sans"):
        ax.text(x, y, value, fontsize=size, color=color, weight="bold" if bold else "normal",
                va="center", ha=align, family=family)

    def node(x, y, w, h, title, detail, tint, border):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.22,rounding_size=1.1",
                                    facecolor=tint, edgecolor=border, linewidth=1.3))
        text(x+w/2, y+h*.35, title, 15, bold=True, align="center")
        text(x+w/2, y+h*.72, detail, 11.4, align="center")

    def arrow(points, color, dashed=False):
        # Stroke all right-angle segments before adding a single terminal arrowhead.
        x, y = zip(*points[:-1])
        ax.plot(x, y, color=color, linewidth=1.8, linestyle="--" if dashed else "-",
                dash_capstyle="round", solid_capstyle="round")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     mutation_scale=13, lw=1.8, color=color,
                                     linestyle="--" if dashed else "-", shrinkA=0, shrinkB=2))

    text(5, 5, "CRAFTAX FIELD ATLAS    /    POLICY ARCHITECTURE", 10, GREEN, bold=True)
    text(5, 11, "Local-view GRU", 28, family="STIXGeneral")
    text(95, 11, f"{metadata['trainable_parameters']:,} trainable parameters",
         11.5, MUTED, align="right")
    node(5, 18, 40, 12, "Local observation", "7 × 9 × 21 tile features + 22 status features", "#fdfdf7", "#a9baa8")
    node(56, 18, 39, 12, "Current embedding", f"1,345 → {width}    ·    Dense + ReLU", "#eff4e7", GREEN)
    arrow([(45.3, 24), (55.8, 24)], GREEN)

    node(12, 44, 34, 12, "Gated recurrent unit", f"Updated hidden state    ·    {width}", "#e5efe7", GREEN)
    arrow([(64, 30.2), (64, 36), (36, 36), (36, 43.8)], GREEN)
    arrow([(11.8, 50), (5.5, 50), (5.5, 38.3), (19.5, 38.3), (19.5, 43.8)], BLUE, dashed=True)
    text(5.4, 35.4, "State across steps", 10.8, BLUE)
    node(35, 62, 32, 12, "Concatenate", f"Current + recurrent    ·    {2*width:,}", "#e6eccf", "#7a9249")
    arrow([(29, 56.2), (29, 68), (34.8, 68)], GREEN)
    arrow([(81, 30.2), (81, 68), (67.2, 68)], GOLD)
    text(79.2, 47, "Current-input", 12, GOLD, align="right")
    text(79.2, 50.3, "bypass", 12, GOLD, align="right")

    node(5, 82, 43, 11.5, "Actor · 17 action logits", f"Dense + ReLU ({width}) twice → linear (17)", "#eff4e7", GREEN)
    node(54, 82, 41, 11.5, "Critic · state value", f"Dense + ReLU ({width}) twice → linear (1)", "#eef2ee", "#779187")
    arrow([(51, 74.2), (51, 78), (26.5, 78), (26.5, 81.8)], GREEN)
    arrow([(51, 78), (74.5, 78), (74.5, 81.8)], GREEN)
    text(5, 97.1, "At episode start the GRU state is zero. Each updated state becomes the carry for the next action.",
         10.8, MUTED)
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / "local-gru.pdf", metadata={"Title": "Local-view GRU | Craftax Field Atlas"})
    fig.savefig(target / "local-gru.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    draw(parser.parse_args().output)

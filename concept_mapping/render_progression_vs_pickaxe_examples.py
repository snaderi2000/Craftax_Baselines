"""Render two audit examples of the cumulative empirical inventory ladder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from craftax.craftax_classic.constants import load_all_textures
from concept_mapping.visualize_symbolic_trajectory import render_local_view


BANK = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "hierarchy_concept_banks/progression_vs_pickaxe_value/progression_bank.npz"
)


def _font(size: int):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if os.path.isfile(path) else ImageFont.load_default()


def main() -> None:
    with np.load(BANK) as data:
        levels = np.asarray(data["levels"])
        labels = data["level_added_item"].astype(str)
        episodes, timesteps = np.asarray(data["episode_id"]), np.asarray(data["timestep"])
    chosen = np.sort(np.random.default_rng(50).choice(levels.shape[1], 2, replace=False))
    block_size, scale = 7, 3
    textures, panel_textures = load_all_textures(block_size), load_all_textures(block_size * scale)
    for example_number, state_index in enumerate(chosen):
        panels = []
        for level, label in enumerate(labels):
            view = render_local_view(levels[level, state_index], block_size=block_size,
                                     include_inventory=True, display_scale=scale,
                                     textures=textures, panel_textures=panel_textures)
            footer_h = 48
            panel = Image.new("RGB", (view.width, view.height + footer_h), "white"); panel.paste(view)
            draw = ImageDraw.Draw(panel)
            title = "Empty inventory" if level == 0 else f"+ {label.replace('_', ' ')}"
            draw.text((4, view.height + 3), title, fill="black", font=_font(14))
            draw.text((4, view.height + 25), f"level {level}", fill="black", font=_font(11))
            panels.append(panel)
        columns, gap, margin = 5, 12, 16
        rows = (len(panels) + columns - 1) // columns
        width = 2 * margin + columns * panels[0].width + (columns - 1) * gap
        height = 52 + 2 * margin + rows * panels[0].height + (rows - 1) * gap
        sheet = Image.new("RGB", (width, height), "white")
        ImageDraw.Draw(sheet).text(
            (margin, 10), f"episode {episodes[state_index]}, step {timesteps[state_index]}: cumulative inventory",
            fill="black", font=_font(18),
        )
        for rank, panel in enumerate(panels):
            x = margin + (rank % columns) * (panel.width + gap)
            y = 52 + margin + (rank // columns) * (panel.height + gap)
            sheet.paste(panel, (x, y))
        output = BANK.parent / f"progression_ladder_example_{example_number:02d}.png"
        sheet.save(output); print(output)


if __name__ == "__main__":
    main()

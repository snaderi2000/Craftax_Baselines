"""Render one sword counterfactual as a clean, paper-ready five-panel figure."""

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from craftax.craftax_classic.constants import load_all_textures
from PIL import Image, ImageDraw, ImageFont

from concept_mapping.collect_water_sword_counterfactuals import _sword_counterfactuals
from concept_mapping.visualize_symbolic_trajectory import render_local_view


PANELS = (
    ("factual", "Factual state"),
    ("no_sword", "No sword"),
    ("wood_only", "Wood sword"),
    ("stone_only", "Stone sword"),
    ("iron_only", "Iron sword"),
)


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for base in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation2"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged")
    parser.add_argument("--merged_index", type=int, default=953)
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--layout", choices=("two_row", "simple_row"), default="two_row")
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    default_name = f"sword_idx_{args.merged_index}_stone_vs_wood.png"
    if args.layout == "simple_row":
        default_name = f"sword_idx_{args.merged_index}_clean_row.png"
    out_path = Path(args.out_path) if args.out_path else run_dir / "paper_figures" / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    states = np.load(run_dir / "sword_base_states.npz")["base_obs"]
    with (run_dir / "sword_counterfactual_values.csv").open(newline="") as handle:
        rows = {int(row["merged_index"]): row for row in csv.DictReader(handle)}
    if args.merged_index not in rows or args.merged_index >= len(states):
        raise ValueError(f"No sword state with merged index {args.merged_index}")
    row, counterfactuals = rows[args.merged_index], _sword_counterfactuals(states[args.merged_index])
    textures = load_all_textures(args.block_size)
    panel_textures = load_all_textures(args.block_size * args.display_scale)
    title_font, value_font = _font(28, bold=True), _font(25)
    if args.layout == "simple_row":
        frames = []
        for key, label in PANELS:
            view = render_local_view(counterfactuals[key], block_size=args.block_size, include_inventory=True, display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
            footer = Image.new("RGB", (view.width, 96), "white")
            draw = ImageDraw.Draw(footer)
            draw.text((4, 8), label, fill="black", font=_font(26, bold=True))
            draw.text((4, 49), f"value: {float(row[key]):.4f}", fill="black", font=_font(25, bold=True))
            frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
            frame.paste(view, (0, 0))
            frame.paste(footer, (0, view.height))
            frames.append(frame)
        panel_w, panel_h = frames[0].size
        margin, gap = 24, 18
        canvas = Image.new("RGB", (5 * panel_w + 4 * gap + 2 * margin, panel_h + 2 * margin), "white")
        for index, frame in enumerate(frames):
            canvas.paste(frame, (margin + index * (panel_w + gap), margin))
        canvas.save(out_path)
        print(f"Saved {out_path}")
        return

    frames = []
    for key, label in PANELS:
        view = render_local_view(counterfactuals[key], block_size=args.block_size, include_inventory=True, display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
        header = Image.new("RGB", (view.width, 84), "white")
        draw = ImageDraw.Draw(header)
        draw.text((10, 8), label, fill="black", font=title_font)
        draw.text((10, 44), f"value: {float(row[key]):.4f}", fill="black", font=value_font)
        frame = Image.new("RGB", (view.width, view.height + header.height), "white")
        frame.paste(header, (0, 0))
        frame.paste(view, (0, header.height))
        frames.append(frame)
    panel_w, panel_h = frames[0].size
    margin, gap, banner_h = 24, 18, 58
    canvas = Image.new("RGB", (3 * panel_w + 2 * gap + 2 * margin, 2 * panel_h + gap + banner_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "Stone-versus-wood ordering failure", fill="black", font=_font(30, bold=True))
    draw.text((margin, margin + 33), f"stone value {float(row['stone_only']):.4f} < wood value {float(row['wood_only']):.4f}", fill="black", font=_font(23))
    y_top, y_bottom = margin + banner_h, margin + banner_h + panel_h + gap
    for index in range(3):
        canvas.paste(frames[index], (margin + index * (panel_w + gap), y_top))
    for index in range(3, 5):
        canvas.paste(frames[index], (margin + (index - 3) * (panel_w + gap), y_bottom))
    canvas.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

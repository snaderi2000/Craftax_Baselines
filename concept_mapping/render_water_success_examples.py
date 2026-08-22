"""Render random successful water-value counterfactuals for qualitative selection."""

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

from concept_mapping.collect_water_sword_counterfactuals import _water_counterfactuals
from concept_mapping.visualize_symbolic_trajectory import render_local_view


PANELS = (
    ("empty_no_water", "Empty drink\nno water"),
    ("empty_water", "Empty drink\nwater"),
    ("full_no_water", "Full drink\nno water"),
    ("full_water", "Full drink\nwater"),
)


def _font(size: int, bold: bool = True):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--minimal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--regular_text", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    states = np.load(run_dir / "water_base_states.npz")["base_obs"]
    with (run_dir / "water_counterfactual_values.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(states):
        raise RuntimeError("Water state/value count mismatch")
    empty_effect = np.asarray([float(row["empty_water"]) - float(row["empty_no_water"]) for row in rows])
    full_effect = np.asarray([float(row["full_water"]) - float(row["full_no_water"]) for row in rows])
    success = (empty_effect > 0) & (empty_effect > full_effect)
    chosen = np.random.default_rng(args.seed).choice(np.flatnonzero(success), size=min(args.n_examples, int(success.sum())), replace=False)
    textures = load_all_textures(args.block_size)
    panel_textures = textures if args.display_scale == 1 else load_all_textures(args.block_size * args.display_scale)
    out_dir = run_dir / "water_success_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, index in enumerate(chosen):
        counterfactuals = _water_counterfactuals(states[index])
        frames = []
        for key, label in PANELS:
            view = render_local_view(counterfactuals[key], block_size=args.block_size, include_inventory=True, display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
            if args.minimal:
                footer = Image.new("RGB", (view.width, 76), "white")
                ImageDraw.Draw(footer).text((4, 8), f"value: {float(rows[index][key]):.4f}", fill="black", font=_font(35, bold=not args.regular_text))
                frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
                frame.paste(view, (0, 0))
                frame.paste(footer, (0, view.height))
            else:
                header = Image.new("RGB", (view.width, 112), "white")
                draw = ImageDraw.Draw(header)
                draw.multiline_text((8, 6), label, fill="black", font=_font(22, bold=not args.regular_text), spacing=0)
                draw.text((8, 78), f"value: {float(rows[index][key]):.4f}", fill="black", font=_font(23, bold=not args.regular_text))
                frame = Image.new("RGB", (view.width, view.height + header.height), "white")
                frame.paste(header, (0, 0))
                frame.paste(view, (0, header.height))
            frames.append(frame)
        width, height = frames[0].size
        margin = 22 if args.minimal else 0
        gap = 28 if args.minimal else 0
        sheet_height = 2 * height + gap + 2 * margin if args.minimal else 2 * height + 42
        sheet_width = 2 * width + gap + 2 * margin if args.minimal else 2 * width
        sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
        for frame_index, frame in enumerate(frames):
            sheet.paste(
                frame,
                (
                    margin + (frame_index % 2) * (width + gap),
                    margin + (frame_index // 2) * (height + gap),
                ),
            )
        if not args.minimal:
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 2 * height + 8), f"empty advantage: {empty_effect[index]:.4f}; full advantage: {full_effect[index]:.4f}", fill="black", font=_font(18))
        suffix = "_minimal" if args.minimal else "_2x2"
        if args.regular_text:
            suffix += "_regular"
        sheet.save(out_dir / f"example_{rank:02d}_idx_{index}{suffix}.png")
    print(f"Wrote {len(chosen)} water-success examples to {out_dir}")


if __name__ == "__main__":
    main()

"""Render saved collectability counterfactuals with only bold value labels.

Usage: python concept_mapping/render_collectability_minimal_grids.py --task diamond
       python concept_mapping/render_collectability_minimal_grids.py --task iron
       python concept_mapping/render_collectability_minimal_grids.py --task plant
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from craftax.craftax_classic.constants import load_all_textures

from concept_mapping.visualize_symbolic_trajectory import render_local_view


ROOT = Path(__file__).resolve().parent / "runs/collectability_three_tasks_10b_pilot"
CONDITIONS = {
    "diamond": ("A", "B", "D", "C"),
    "iron": ("A", "B", "D", "C"),
    "plant": ("ripe", "unripe"),
}


def render(args):
    source = args.source_dir or ROOT / args.task
    destination = args.out_dir or source / "minimal_grids"
    destination.mkdir(parents=True, exist_ok=True)

    with (source / "per_state_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(source / "counterfactual_observations.npz") as saved:
        observations = saved["observations"]
    keys = CONDITIONS[args.task]
    if observations.shape[:2] != (len(rows), len(keys)):
        raise ValueError("Saved observation and result counts do not match")

    font = ImageFont.truetype(args.font, args.font_size)
    after_suffix = getattr(args, 'after_values_suffix', None)
    textures = load_all_textures(args.block_size)
    panel_textures = (textures if args.display_scale == 1 else
                      load_all_textures(args.block_size * args.display_scale))
    examples_per_grid = args.examples_per_grid or (2 if args.task == "plant" else 1)
    if examples_per_grid < 1:
        raise ValueError("--examples-per-grid must be positive")
    sheets = []
    selections = []
    for start in range(0, len(rows), examples_per_grid):
        example_rows = rows[start:start + examples_per_grid]
        example_groups = observations[start:start + examples_per_grid]
        panels = []
        for row, group in zip(example_rows, example_groups):
            for key, observation in zip(keys, group):
                view = render_local_view(
                    observation, block_size=args.block_size, include_inventory=True,
                    display_scale=args.display_scale, textures=textures,
                    panel_textures=panel_textures,
                )
                value = float(row[f"V_{key}"])
                labels = ([f"before: {value:.4f}",
                           f"after: {float(row[f'V_{key}_{after_suffix}']):.4f}"]
                          if after_suffix else [f"value: {value:.4f}"])
                footer_height = ((args.font_size + 10) * 2 + 12
                                 if after_suffix else args.font_size + 22)
                panel = Image.new("RGB", (view.width, view.height + footer_height), "white")
                panel.paste(view, (0, 0))
                draw = ImageDraw.Draw(panel)
                for line, label in enumerate(labels):
                    draw.text((4, view.height + 5 + line * (args.font_size + 10)),
                              label, fill="black", font=font)
                panels.append(panel)

        width, height = panels[0].size
        columns = 2
        lines = (len(panels) + columns - 1) // columns
        gap = args.gap
        sheet = Image.new("RGB", (columns * width + gap,
                                  lines * height + (lines - 1) * gap), "white")
        for position, panel in enumerate(panels):
            sheet.paste(panel, ((position % columns) * (width + gap),
                                (position // columns) * (height + gap)))
        filename = f"example_{len(sheets) + 1:02d}.png"
        sheet.save(destination / filename)
        sheets.append(sheet)
        selections.append(dict(
            file=filename,
            seeds=",".join(row["seed"] for row in example_rows),
            episode_ids=",".join(row["episode_id"] for row in example_rows),
            timesteps=",".join(row["timestep"] for row in example_rows),
            panel_order=";".join(",".join(keys) for _ in example_rows),
        ))

    if sheets:
        sheets[0].save(destination / "all_examples.pdf", save_all=True,
                       append_images=sheets[1:])
    with (destination / "selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selections[0]))
        writer.writeheader()
        writer.writerows(selections)
    print(f"Rendered {len(sheets)} {args.task} examples to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=CONDITIONS, required=True)
    parser.add_argument("--source-dir", type=Path,
                        help="Directory containing per_state_results.csv and counterfactual_observations.npz")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--display-scale", type=int, default=4)
    parser.add_argument("--font-size", type=int, default=25)
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--examples-per-grid", type=int,
                        help="Defaults to two for plant and one for other tasks")
    parser.add_argument("--after-values-suffix",
                        help="Also show V_<condition>_<suffix> below the original value, as before/after labels")
    render(parser.parse_args())


if __name__ == "__main__":
    main()

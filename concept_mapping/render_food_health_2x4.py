"""Render minimal 2x4 grids for the cow health x food experiment."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from craftax.craftax_classic.constants import load_all_textures

from concept_mapping.analyze_food_health_interaction_10k import make_counterfactuals
from concept_mapping.visualize_symbolic_trajectory import render_local_view


DEFAULT_RUN = Path(__file__).resolve().parent / "runs/food_health_interaction_10b_10000episodes"
COLS = ((2, 0, "health 2, food 0"), (2, 9, "health 2, food 9"),
        (9, 0, "health 9, food 0"), (9, 9, "health 9, food 9"))
ROWS = ((False, "no cow"), (True, "cow"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--display-scale", type=int, default=4)
    args = parser.parse_args()
    output = args.out_dir or args.run_dir / "random_success_examples_2x4"
    output.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "per_state_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(args.run_dir / "base_states.npz") as saved:
        factual = saved["observations"]
    branches = make_counterfactuals(factual)
    eligible = np.asarray([row["primary_success"].lower() == "true" for row in rows])
    chosen = np.random.default_rng(args.seed).choice(
        np.flatnonzero(eligible), min(args.n_examples, int(eligible.sum())), replace=False)
    textures = load_all_textures(args.block_size)
    panel_textures = load_all_textures(args.block_size * args.display_scale)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    selection, sheets = [], []
    for rank, index in enumerate(chosen, 1):
        panels = []
        for cow, _row_label in ROWS:
            for health, food, _column_label in COLS:
                key = f"h{health}_f{food}_{'cow' if cow else 'no_cow'}"
                view = render_local_view(branches[key][index], block_size=args.block_size,
                                         include_inventory=True, display_scale=args.display_scale,
                                         textures=textures, panel_textures=panel_textures)
                footer = Image.new("RGB", (view.width, 55), "white")
                ImageDraw.Draw(footer).text((4, 5), f"value: {float(rows[index]['V_' + key]):.4f}",
                                            fill="black", font=bold)
                panel = Image.new("RGB", (view.width, view.height + footer.height), "white")
                panel.paste(view, (0, 0)); panel.paste(footer, (0, view.height)); panels.append(panel)
        width, height = panels[0].size
        left, top, col_gap, row_gap = 105, 45, 16, 42
        sheet = Image.new("RGB", (left + 4 * width + 3 * col_gap,
                                  top + 2 * height + row_gap), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (_health, _food, label) in enumerate(COLS):
            box = draw.textbbox((0, 0), label, font=small)
            x = left + column * (width + col_gap) + (width - (box[2] - box[0])) // 2
            draw.text((x, 10), label, fill="black", font=small)
        for row_index, (_cow, label) in enumerate(ROWS):
            draw.text((4, top + row_index * (height + row_gap) + height // 2 - 12),
                      label, fill="black", font=small)
        for position, panel in enumerate(panels):
            row_index, column = divmod(position, 4)
            sheet.paste(panel, (left + column * (width + col_gap),
                                top + row_index * (height + row_gap)))
        filename = f"example_{rank:02d}_idx_{index}.png"
        sheet.save(output / filename); sheets.append(sheet)
        selection.append(dict(file=filename, index=int(index), seed=rows[index]["seed"],
                              episode_id=rows[index]["episode_id"],
                              base_timestep=rows[index]["base_timestep"],
                              restoration_timestep=rows[index]["restoration_timestep"]))
    if sheets:
        sheets[0].save(output / "all_examples.pdf", save_all=True, append_images=sheets[1:])
    with (output / "selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection[0]))
        writer.writeheader(); writer.writerows(selection)
    print(f"Rendered {len(sheets)} food examples to {output}")


if __name__ == "__main__":
    main()

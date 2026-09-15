"""Render reproducibly sampled 2x2 grids from the final stone/pickaxe experiment."""

from __future__ import annotations

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

from concept_mapping.visualize_symbolic_trajectory import render_local_view


DEFAULT_EXPERIMENT_DIR = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "stone_wood_pickaxe_interaction_final_1p5b"
)
DEFAULT_BANK = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "hierarchy_concept_banks/wood_pickaxe_functional/wood_pickaxe_functional_bank.npz"
)


def font(size: int, bold: bool = True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{name}"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def load_values(path: Path) -> dict[int, dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["sample_index"]): row for row in rows}


def labeled_frame(observation: np.ndarray, label: str, description: str, value: float,
                  textures, panel_textures, block_size: int, display_scale: int) -> Image.Image:
    view = render_local_view(
        observation, block_size=block_size, include_inventory=True,
        display_scale=display_scale, textures=textures, panel_textures=panel_textures,
    )
    footer = Image.new("RGB", (view.width, 82), "white")
    draw = ImageDraw.Draw(footer)
    draw.text((6, 4), label, fill="#111827", font=font(18))
    draw.text((6, 30), description, fill="#334155", font=font(13, bold=False))
    draw.text((6, 54), f"V = {value:.4f}", fill="#111827", font=font(17))
    frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
    frame.paste(view, (0, 0))
    frame.paste(footer, (0, view.height))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--positive_only", action="store_true",
                        help="Sample only states whose saved interaction I is positive.")
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")

    with np.load(args.bank) as source:
        mask = np.asarray(source["resource"]).astype(str) == "stone"
        bank = {key: np.asarray(source[key])[mask] for key in source.files}
    value_rows = load_values(
        args.experiment_dir / "stone_wood_pickaxe_interaction_per_state.csv"
    )
    if len(value_rows) != len(bank["A"]):
        raise RuntimeError("Value table and stone bank have different lengths")
    candidates = np.asarray([
        index for index in range(len(bank["A"]))
        if not args.positive_only or float(value_rows[index]["I"]) > 0
    ], dtype=int)
    if not len(candidates):
        raise RuntimeError("No states satisfy the requested interaction filter")
    chosen = np.sort(np.random.default_rng(args.seed).choice(
        candidates, size=min(args.count, len(candidates)), replace=False
    ))
    selection_name = "positive" if args.positive_only else "random"

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    # Bank A/B/C/D correspond to s00/s10/s01/s11.
    conditions = (
        ("A", "s00", "no stone, no wood pickaxe", "V_s00"),
        ("B", "s10", "no stone, wood pickaxe", "V_s10"),
        ("C", "s01", "stone in front, no wood pickaxe", "V_s01"),
        ("D", "s11", "stone in front, wood pickaxe", "V_s11"),
    )
    sheets = []
    selection_rows = []
    for rank, index in enumerate(chosen, start=1):
        row = value_rows[int(index)]
        frames = [
            labeled_frame(
                bank[bank_key][index], label, description, float(row[value_field]),
                textures, panel_textures, args.block_size, args.display_scale,
            )
            for bank_key, label, description, value_field in conditions
        ]
        frame_width, frame_height = frames[0].size
        gap, banner_height = 16, 72
        sheet = Image.new(
            "RGB", (2 * frame_width + gap, 2 * frame_height + gap + banner_height), "white"
        )
        for position, frame in enumerate(frames):
            sheet.paste(frame, (
                (position % 2) * (frame_width + gap),
                (position // 2) * (frame_height + gap),
            ))
        draw = ImageDraw.Draw(sheet)
        banner_y = 2 * frame_height + gap
        draw.text(
            (6, banner_y + 6),
            f"Episode {int(row['episode_id'])}, step {int(row['timestep'])}",
            fill="#111827", font=font(16),
        )
        draw.text(
            (6, banner_y + 35),
            f"I = [V(s11)-V(s01)] - [V(s10)-V(s00)] = {float(row['I']):+.4f}",
            fill="#7c3aed", font=font(16),
        )
        output = args.experiment_dir / (
            f"{selection_name}_2x2_grid_{rank:02d}_episode_{int(row['episode_id']):04d}.png"
        )
        sheet.save(output)
        sheets.append(sheet)
        selection_rows.append({
            "grid_rank": rank, "sample_index": int(index),
            "episode_id": int(row["episode_id"]), "timestep": int(row["timestep"]),
            "I": float(row["I"]), "output_file": output.name,
        })

    # Also save one convenient contact sheet containing the five sampled grids.
    gap, margin = 20, 20
    width = max(sheet.width for sheet in sheets)
    height = sum(sheet.height for sheet in sheets) + gap * (len(sheets) - 1)
    contact = Image.new("RGB", (width + 2 * margin, height + 2 * margin), "white")
    y = margin
    for sheet in sheets:
        contact.paste(sheet, (margin, y))
        y += sheet.height + gap
    contact.save(args.experiment_dir / f"five_{selection_name}_2x2_grids.png")

    selection_path = args.experiment_dir / f"five_{selection_name}_2x2_grid_selection.csv"
    temporary = selection_path.with_name(f".{selection_path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    temporary.replace(selection_path)
    print(f"Saved {len(sheets)} {selection_name} 2x2 grids (seed={args.seed}) to {args.experiment_dir}")
    for row in selection_rows:
        print(f"  {row['output_file']}: sample={row['sample_index']}, I={row['I']:+.4f}")


if __name__ == "__main__":
    main()

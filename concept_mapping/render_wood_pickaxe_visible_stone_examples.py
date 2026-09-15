"""Render representative state pairs from the visible-stone value experiment."""

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

from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_DIM, render_local_view


DEFAULT_EXPERIMENT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "wood_pickaxe_value_by_visible_stone_final_1p5b"
)
WOOD_PICKAXE_OFFSET = MAP_DIM + INV_NAMES.index("wood_pickaxe")


def font(size: int, bold: bool = True):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{filename}"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def representative_rows(rows: list[dict], visible: bool, count: int) -> list[dict]:
    selected_group = sorted(
        (row for row in rows if parse_bool(row["stone_visible"]) == visible),
        key=lambda row: float(row["delta_V"]),
    )
    if not selected_group:
        raise RuntimeError(f"No rows for stone_visible={visible}")
    quantiles = np.linspace(.1, .9, min(count, len(selected_group)))
    indices = np.unique(np.round(quantiles * (len(selected_group) - 1)).astype(int))
    return [selected_group[index] for index in indices]


def load_observation(row: dict) -> np.ndarray:
    path = Path(row["source_npz"])
    state_step = int(row["state_step"])
    with np.load(path) as episode:
        observations = np.asarray(episode["observations"], dtype=np.float32)
        if state_step < len(observations):
            return observations[state_step]
        return np.asarray(episode["terminal_observation"], dtype=np.float32)


def make_frame(observation: np.ndarray, title: str, value: float, textures,
               panel_textures, block_size: int, display_scale: int) -> Image.Image:
    view = render_local_view(
        observation, block_size=block_size, include_inventory=True,
        display_scale=display_scale, textures=textures, panel_textures=panel_textures,
    )
    footer = Image.new("RGB", (view.width, 62), "white")
    draw = ImageDraw.Draw(footer)
    draw.text((6, 4), title, fill="#111827", font=font(16))
    draw.text((6, 34), f"V = {value:.4f}", fill="#111827", font=font(17))
    frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
    frame.paste(view, (0, 0))
    frame.paste(footer, (0, view.height))
    return frame


def make_pair(row: dict, textures, panel_textures, block_size: int,
              display_scale: int) -> Image.Image:
    plus = load_observation(row)
    minus = plus.copy()
    minus[WOOD_PICKAXE_OFFSET] = 0.0
    frames = [
        make_frame(minus, "s−: no wood pickaxe", float(row["V_s_minus"]),
                   textures, panel_textures, block_size, display_scale),
        make_frame(plus, "s+: natural wood pickaxe", float(row["V_s_plus"]),
                   textures, panel_textures, block_size, display_scale),
    ]
    gap, footer_height = 14, 74
    width, height = frames[0].size
    pair = Image.new("RGB", (2 * width + gap, height + footer_height), "white")
    pair.paste(frames[0], (0, 0))
    pair.paste(frames[1], (width + gap, 0))
    draw = ImageDraw.Draw(pair)
    context = "stone visible" if parse_bool(row["stone_visible"]) else "no stone visible"
    nearest = int(row["nearest_stone_manhattan"])
    distance = f", nearest distance {nearest}" if nearest >= 0 else ""
    draw.text(
        (6, height + 5),
        f"Episode {int(row['episode_id'])}, step {int(row['state_step'])}: {context}{distance}",
        fill="#334155", font=font(14),
    )
    color = "#047857" if float(row["delta_V"]) > 0 else "#b91c1c"
    draw.text(
        (6, height + 37), f"ΔV = V(s+) − V(s−) = {float(row['delta_V']):+.4f}",
        fill=color, font=font(17),
    )
    return pair


def contact_sheet(pairs: list[Image.Image]) -> Image.Image:
    columns, gap, margin = 2, 18, 18
    width, height = pairs[0].size
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (2 * margin + columns * width + (columns - 1) * gap,
         2 * margin + rows * height + (rows - 1) * gap),
        "white",
    )
    for index, pair in enumerate(pairs):
        sheet.paste(pair, (
            margin + (index % columns) * (width + gap),
            margin + (index // columns) * (height + gap),
        ))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--count_per_group", type=int, default=5)
    parser.add_argument("--single_distance", type=int, default=None,
                        help="Render one stone-visible example closest to the mean delta_V at this distance.")
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    if args.count_per_group <= 0:
        raise ValueError("count_per_group must be positive")
    source_csv = args.experiment_dir / "wood_pickaxe_value_by_visible_stone_per_state.csv"
    with source_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output_dir = args.experiment_dir / "representative_state_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )

    if args.single_distance is not None:
        candidates = [
            row for row in rows
            if parse_bool(row["stone_visible"])
            and int(row["nearest_stone_manhattan"]) == args.single_distance
        ]
        if not candidates:
            raise RuntimeError(f"No visible-stone states at distance {args.single_distance}")
        mean_effect = np.mean([float(row["delta_V"]) for row in candidates])
        selected = min(candidates, key=lambda row: abs(float(row["delta_V"]) - mean_effect))
        pair = make_pair(
            selected, textures, panel_textures, args.block_size, args.display_scale
        )
        output = output_dir / (
            f"distance_{args.single_distance}_representative_episode_"
            f"{int(selected['episode_id']):04d}.png"
        )
        pair.save(output)
        print(
            f"Saved {output}; group mean delta_V={mean_effect:+.4f}, "
            f"example delta_V={float(selected['delta_V']):+.4f}"
        )
        return

    audit_rows = []
    all_pairs = []
    for visible, stem in ((False, "no_stone_visible"), (True, "stone_visible")):
        selected = representative_rows(rows, visible, args.count_per_group)
        pairs = []
        for rank, row in enumerate(selected, start=1):
            pair = make_pair(
                row, textures, panel_textures, args.block_size, args.display_scale
            )
            output = output_dir / f"{stem}_{rank:02d}_episode_{int(row['episode_id']):04d}.png"
            pair.save(output)
            pairs.append(pair)
            all_pairs.append(pair)
            audit_rows.append({
                "context": stem, "rank": rank, "episode_id": int(row["episode_id"]),
                "state_step": int(row["state_step"]),
                "nearest_stone_manhattan": int(row["nearest_stone_manhattan"]),
                "delta_V": float(row["delta_V"]), "file": output.name,
            })
        contact_sheet(pairs).save(output_dir / f"five_{stem}_pairs.png")
    contact_sheet(all_pairs).save(output_dir / "ten_representative_pairs.png")

    audit_path = output_dir / "selection.csv"
    temporary = audit_path.with_name(f".{audit_path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    temporary.replace(audit_path)
    print(f"Saved {len(all_pairs)} representative pairs to {output_dir}")
    for row in audit_rows:
        print(f"  {row['context']} episode={row['episode_id']} delta_V={row['delta_V']:+.4f}")


if __name__ == "__main__":
    main()

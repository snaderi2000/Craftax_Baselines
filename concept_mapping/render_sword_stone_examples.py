"""Render random successes and failures of the stone-versus-wood comparison."""

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
from PIL import Image, ImageDraw

from concept_mapping.collect_water_sword_counterfactuals import _sword_counterfactuals
from concept_mapping.visualize_symbolic_trajectory import render_local_view


PANELS = (
    ("factual", "Factual base"),
    ("no_sword", "No sword"),
    ("wood_only", "Wood only"),
    ("stone_only", "Stone only"),
    ("iron_only", "Iron only"),
)


def _render(obs, row, out_path: Path, textures, panel_textures, block_size, display_scale, outcome):
    counterfactuals = _sword_counterfactuals(obs)
    frames = []
    for key, title in PANELS:
        view = render_local_view(
            counterfactuals[key], block_size=block_size, include_inventory=True,
            display_scale=display_scale, textures=textures, panel_textures=panel_textures,
        )
        header = Image.new("RGB", (view.width, 60), "white")
        draw = ImageDraw.Draw(header)
        draw.text((7, 7), title, fill="black")
        draw.text((7, 30), f"V={float(row[key]):.4f}", fill="black")
        frame = Image.new("RGB", (view.width, view.height + header.height), "white")
        frame.paste(header, (0, 0))
        frame.paste(view, (0, header.height))
        frames.append(frame)
    width, height = frames[0].size
    sheet = Image.new("RGB", (5 * width, height + 46), "white")
    for index, frame in enumerate(frames):
        sheet.paste(frame, (index * width, 0))
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (7, height + 8),
        f"{outcome}; merged sample={row['merged_index']}; "
        f"stone={float(row['stone_only']):.4f}, wood={float(row['wood_only']):.4f}",
        fill="black",
    )
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged",
    )
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max_health", type=float, default=None)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    states = np.load(run_dir / "sword_base_states.npz")["base_obs"]
    with (run_dir / "sword_counterfactual_values.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (run_dir / "sword_metadata.csv").open(newline="") as handle:
        metadata = {int(row["merged_index"]): row for row in csv.DictReader(handle)}
    if len(rows) != len(states):
        raise RuntimeError("Sword state/value count mismatch")
    eligible = np.asarray([
        args.max_health is None or float(metadata[int(row["merged_index"])]["health"]) <= args.max_health
        for row in rows
    ])
    success = np.asarray([float(row["stone_only"]) > float(row["wood_only"]) for row in rows]) & eligible
    rng = np.random.default_rng(args.seed)
    textures = load_all_textures(args.block_size)
    panel_textures = textures if args.display_scale == 1 else load_all_textures(args.block_size * args.display_scale)
    for name, mask, description in (
        ("stone_gt_wood_success", success, "Success: stone > wood"),
        ("stone_le_wood_failure", (~success) & eligible, "Failure: stone <= wood"),
    ):
        candidates = np.flatnonzero(mask)
        chosen = rng.choice(candidates, size=min(args.n_examples, len(candidates)), replace=False)
        health_dir = "all_health" if args.max_health is None else f"health_at_most_{args.max_health:g}"
        out_dir = run_dir / "sword_stone_examples" / health_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, index in enumerate(chosen):
            _render(states[index], rows[index], out_dir / f"example_{rank:02d}_idx_{index}.png", textures, panel_textures, args.block_size, args.display_scale, description)
        print(f"Wrote {len(chosen)} {name} examples to {out_dir}")


if __name__ == "__main__":
    main()

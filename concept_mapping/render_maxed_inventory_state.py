"""Render a standalone Craftax Classic observation with all inventory fields maxed."""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from craftax.craftax_classic.constants import load_all_textures

from concept_mapping.visualize_symbolic_trajectory import MAP_DIM, render_local_view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_states",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/water_base_states.npz",
    )
    parser.add_argument("--source_index", type=int, default=844)
    parser.add_argument(
        "--out_path",
        default="concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end/merged/paper_figures/maxed_inventory_state.png",
    )
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    states = np.load(args.source_states)["base_obs"]
    obs = np.array(states[args.source_index], dtype=np.float32, copy=True)
    # Classic encodes the 12 inventory slots and four survival meters as value / 10.
    obs[MAP_DIM : MAP_DIM + 16] = 1.0
    textures = load_all_textures(args.block_size)
    panel_textures = load_all_textures(args.block_size * args.display_scale)
    image = render_local_view(
        obs,
        block_size=args.block_size,
        include_inventory=True,
        display_scale=args.display_scale,
        textures=textures,
        panel_textures=panel_textures,
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved maxed-inventory state to {out_path}")


if __name__ == "__main__":
    main()

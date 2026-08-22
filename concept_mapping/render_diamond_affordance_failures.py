import csv
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from PIL import Image, ImageDraw

from craftax.craftax_classic.constants import BlockType, load_all_textures

from concept_mapping.diamond_affordance_over_time import (
    _make_counterfactuals,
    _render_counterfactual_sheet,
)


RUN_DIR = Path("concept_mapping/runs/diamond_affordance_over_time")
FAIL_DIR = RUN_DIR / "partial_order_failures_10b"


def main() -> None:
    states = np.load(RUN_DIR / "diamond_tminus_states.npz")
    base_obs = states["base_obs"]
    episodes = states["episodes"]
    base_steps = states["base_steps"]
    collect_steps = states["collect_steps"]

    rows = []
    with (FAIL_DIR / "failures_10b.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    textures = load_all_textures(7)
    panel_textures = load_all_textures(28)
    for r in rows:
        idx = int(r["sample_index"])
        cf = _make_counterfactuals(base_obs[idx])
        values = {
            "A_no_diamond_no_pickaxe": float(r["A"]),
            "B_no_diamond_pickaxe": float(r["B"]),
            "C_diamond_no_pickaxe": float(r["C"]),
            "D_diamond_pickaxe": float(r["D"]),
        }
        meta = {
            "episode": int(episodes[idx]),
            "base_step": int(base_steps[idx]),
            "collect_step": int(collect_steps[idx]),
        }
        out_path = FAIL_DIR / f"failure_idx_{idx:03d}_{r['reason'].replace(';','_')}.png"
        _render_counterfactual_sheet(
            cf,
            values,
            meta,
            str(out_path),
            7,
            4,
            textures,
            panel_textures,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

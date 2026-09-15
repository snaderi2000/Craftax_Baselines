"""Build a cumulative empirical inventory ladder on fixed pre-stone states.

Every external observation feature comes from a natural state immediately
before the 1.5B policy's first successful stone collection. Inventory is first
canonicalized to empty and then extended one component at a time in the median
first-acquisition order measured from the same policy's 1,000 master episodes.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from craftax.craftax_classic.constants import BlockType, OBS_DIM

from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_CHANNELS, MAP_DIM


SOURCE_BANK = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "hierarchy_concept_banks/wood_pickaxe_functional/wood_pickaxe_functional_bank.npz"
)
OUT_DIR = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/"
    "hierarchy_concept_banks/progression_vs_pickaxe_value"
)
# Median first-presence order measured from the same 1.5B policy's 1,000
# master episodes. Each resource is set to one unit and each tool to one copy.
EMPIRICAL_ORDER = (
    "sapling", "wood", "wood_pickaxe", "stone", "wood_sword",
    "stone_pickaxe", "coal", "stone_sword", "iron", "iron_sword",
    "iron_pickaxe", "diamond",
)
MEDIAN_STEPS = {
    "sapling": 7.0, "wood": 10.0, "wood_pickaxe": 24.0, "stone": 26.0,
    "wood_sword": 42.0, "stone_pickaxe": 77.0, "coal": 90.0,
    "stone_sword": 107.0, "iron": 110.0, "iron_sword": 157.0,
    "iron_pickaxe": 223.0, "diamond": 274.5,
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_BANK) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    indices = np.flatnonzero(data["resource"].astype(str) == "stone")
    natural = data["D"][indices].astype(np.float32)
    n = len(natural)

    # levels[0] has empty inventory. levels[k] contains the first k empirical
    # components, so each adjacent pair differs in exactly one inventory field.
    levels = np.repeat(natural[None, :, :], len(EMPIRICAL_ORDER) + 1, axis=0)
    levels[:, :, MAP_DIM : MAP_DIM + len(INV_NAMES)] = 0.0
    for level, item in enumerate(EMPIRICAL_ORDER, start=1):
        levels[level:, :, MAP_DIM + INV_NAMES.index(item)] = 0.1

    # The entire map and all non-inventory scalar features remain identical.
    assert np.array_equal(
        levels[:, :, :MAP_DIM], np.repeat(natural[None, :, :MAP_DIM], len(levels), axis=0)
    )
    assert np.array_equal(
        levels[:, :, MAP_DIM + len(INV_NAMES):],
        np.repeat(natural[None, :, MAP_DIM + len(INV_NAMES):], len(levels), axis=0),
    )
    blocks = np.argmax(
        natural[:, :MAP_DIM].reshape(n, *OBS_DIM, MAP_CHANNELS)[..., :len(BlockType)], axis=-1
    )
    rows, cols = data["target_row"][indices], data["target_col"][indices]
    assert np.all(blocks[np.arange(n), rows, cols] == BlockType.STONE.value)
    for level, item in enumerate(EMPIRICAL_ORDER, start=1):
        changed = np.flatnonzero(np.any(levels[level] != levels[level - 1], axis=0))
        expected = MAP_DIM + INV_NAMES.index(item)
        assert changed.tolist() == [expected], (level, item, changed)
        assert np.allclose(levels[level, :, expected], 0.1, atol=1e-4)
    pickaxe_level = EMPIRICAL_ORDER.index("wood_pickaxe") + 1
    pickaxe_offset = MAP_DIM + INV_NAMES.index("wood_pickaxe")
    assert np.all(levels[:pickaxe_level, :, pickaxe_offset] == 0)
    assert np.allclose(levels[pickaxe_level:, :, pickaxe_offset], 0.1, atol=1e-4)

    np.savez_compressed(
        OUT_DIR / "progression_bank.npz",
        natural=natural, levels=levels,
        level_added_item=np.asarray(("empty",) + EMPIRICAL_ORDER),
        empirical_order=np.asarray(EMPIRICAL_ORDER),
        empirical_median_steps=np.asarray([MEDIAN_STEPS[x] for x in EMPIRICAL_ORDER]),
        wood_pickaxe_level=np.int32(pickaxe_level),
        source_index=data["source_index"][indices], episode_id=data["episode_id"][indices],
        timestep=data["timestep"][indices], source_npz=data["source_npz"][indices],
        target_row=rows, target_col=cols,
    )

    with (OUT_DIR / "progression_bank_audit.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bank_index", "source_index", "episode_id", "timestep", "source_npz",
                         "target_row", "target_col", "n_levels", "pickaxe_level"])
        for i, source_index in enumerate(indices):
            writer.writerow([i, int(data["source_index"][source_index]), int(data["episode_id"][source_index]),
                             int(data["timestep"][source_index]), str(data["source_npz"][source_index]),
                             int(rows[i]), int(cols[i]), len(levels), pickaxe_level])

    selection = {
        "n_states": n, "source_bank": str(SOURCE_BANK),
        "state_definition": "Natural state immediately before first successful stone collection",
        "inventory_intervention": "Canonical empty inventory followed by cumulative presence-only additions",
        "empirical_order": list(EMPIRICAL_ORDER), "median_first_acquisition_steps": MEDIAN_STEPS,
        "order_source": "1,000 master episodes generated by the frozen 1.5B policy",
        "wood_pickaxe_level": pickaxe_level,
        "functional_transition": "level 2 -> 3 adds wood_pickaxe and makes the visible stone collectible",
        "progression_segment": "levels 3 -> 12; wood_pickaxe and immediate stone collectability remain fixed",
        "primary_progression_metric_levels": [
            "wood_pickaxe", "stone", "wood_sword", "stone_pickaxe", "coal", "stone_sword", "iron"
        ],
        "primary_metric": "fraction of per-state OLS slopes below zero across levels 3 through 9",
        "validation": (
            "Each adjacent level changes exactly one inventory field; map, stone action cell, intrinsic stats, "
            "direction, light, and sleep are unchanged. Achievements are not encoded in the critic observation."
        ),
    }
    (OUT_DIR / "progression_item_selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()

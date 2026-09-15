"""Build the frozen 2x2 wood-pickaxe harvesting-affordance bank.

The natural D states come from the existing first-successful stone and coal
collection banks. For each state:
  A = target resource replaced by PATH, wood pickaxe absent
  B = target resource replaced by PATH, wood pickaxe present
  C = target resource present, wood pickaxe absent
  D = target resource present, wood pickaxe present (natural state)
"""

from __future__ import annotations

import argparse
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


DEFAULT_BANK_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks"
WOOD_PICKAXE_INDEX = INV_NAMES.index("wood_pickaxe")


def _set_path(obs: np.ndarray, row: int, col: int) -> np.ndarray:
    out = np.array(obs, dtype=np.float32, copy=True)
    view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    view[row, col, :len(BlockType)] = 0.0
    view[row, col, BlockType.PATH.value] = 1.0
    return out


def _remove_pickaxe(obs: np.ndarray) -> np.ndarray:
    out = np.array(obs, dtype=np.float32, copy=True)
    out[MAP_DIM + WOOD_PICKAXE_INDEX] = 0.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_dir", default=DEFAULT_BANK_DIR)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    bank_dir = Path(args.bank_dir)
    out_dir = Path(args.out_dir) if args.out_dir else bank_dir / "wood_pickaxe_functional"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for resource in ("stone", "coal"):
        with np.load(bank_dir / f"{resource}_bank.npz") as bank:
            for index in range(len(bank["base_obs"])):
                d = np.asarray(bank["base_obs"][index], dtype=np.float32)
                c = np.asarray(bank["counterfactual_obs"][index], dtype=np.float32)
                row, col = int(bank["target_row"][index]), int(bank["target_col"][index])
                b = _set_path(d, row, col)
                a = _remove_pickaxe(b)
                # Validate the pre-existing C and newly constructed A/B exactly.
                expected_c = _remove_pickaxe(d)
                if not np.array_equal(c, expected_c):
                    raise RuntimeError(f"{resource}[{index}] C differs from a pure pickaxe removal")
                if not np.array_equal(a, _set_path(c, row, col)):
                    raise RuntimeError(f"{resource}[{index}] A is not the exact two-factor edit")
                d_blocks = np.argmax(d[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)[..., :len(BlockType)], axis=-1)
                expected_block = BlockType.STONE if resource == "stone" else BlockType.COAL
                if int(d_blocks[row, col]) != expected_block.value:
                    raise RuntimeError(f"{resource}[{index}] target is not the requested front resource")
                records.append({
                    "resource": resource, "source_index": index, "A": a, "B": b, "C": c, "D": d,
                    "episode_id": int(bank["episode_id"][index]), "timestep": int(bank["timestep"][index]),
                    "source_npz": str(bank["source_npz"][index]), "target_row": row, "target_col": col,
                })
    output = out_dir / "wood_pickaxe_functional_bank.npz"
    np.savez_compressed(
        output,
        A=np.stack([r["A"] for r in records]), B=np.stack([r["B"] for r in records]),
        C=np.stack([r["C"] for r in records]), D=np.stack([r["D"] for r in records]),
        resource=np.asarray([r["resource"] for r in records]),
        source_index=np.asarray([r["source_index"] for r in records], dtype=np.int32),
        episode_id=np.asarray([r["episode_id"] for r in records], dtype=np.int32),
        timestep=np.asarray([r["timestep"] for r in records], dtype=np.int32),
        source_npz=np.asarray([r["source_npz"] for r in records]),
        target_row=np.asarray([r["target_row"] for r in records], dtype=np.int8),
        target_col=np.asarray([r["target_col"] for r in records], dtype=np.int8),
    )
    with (out_dir / "bank_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "resource", "source_index", "episode_id", "timestep", "source_npz", "target_row", "target_col", "validation_passed"])
        writer.writeheader()
        for i, record in enumerate(records):
            writer.writerow({"sample_index": i, **{k: record[k] for k in ("resource", "source_index", "episode_id", "timestep", "source_npz", "target_row", "target_col")}, "validation_passed": True})
    metadata = {
        "n_states": len(records), "n_stone": sum(r["resource"] == "stone" for r in records),
        "n_coal": sum(r["resource"] == "coal" for r in records),
        "conditions": {
            "A": "resource absent (target replaced by PATH), no wood pickaxe",
            "B": "resource absent (target replaced by PATH), wood pickaxe present",
            "C": "resource present directly in front, no wood pickaxe",
            "D": "natural pre-collection state: resource present directly in front, wood pickaxe present",
        },
        "primary_value_interaction": "(V_D - V_C) - (V_B - V_A)",
        "primary_representation_interaction": "(h_D - h_C) - (h_B - h_A)",
        "validation": "All C states are exact one-slot pickaxe removals; all B states are exact one-cell PATH replacements; A combines exactly those edits.",
    }
    (out_dir / "bank_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved validated functional bank: {len(records)} states ({metadata['n_stone']} stone, {metadata['n_coal']} coal) to {output}")


if __name__ == "__main__":
    main()

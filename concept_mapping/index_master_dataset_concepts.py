"""Index the master symbolic dataset and freeze a reusable diamond state bank.

The input dataset consists of one compressed NPZ per frozen-policy episode.  This
script creates a *logical row per pre-action timestep* in ``state_index.csv.gz``
without duplicating the 1,345-float observation: ``source_npz`` and
``state_index`` identify the exact stored observation.  It also counts a small,
explicit set of candidate conditions for future counterfactual probes.

For the first replication, it samples a fixed bank of states where a diamond is
visible, Crafty holds an iron pickaxe, and it has not already collected a
diamond.  The bank can then be passed unchanged to every PPO checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from craftax.craftax_classic.constants import Achievement, BlockType
from concept_mapping.visualize_symbolic_trajectory import INTRINSIC_NAMES, INV_NAMES, MOB_NAMES


DEFAULT_DATASET_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes"
DEFAULT_OUT_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/concept_index"

# "Visible" means within the 7x9 symbolic observation.  ``near`` means one of
# the eight cells around the agent, which is centred at (3, 4).
PLAYER_ROW, PLAYER_COL = 3, 4
NEIGHBOUR_ROWS = np.array([2, 2, 2, 3, 3, 4, 4, 4])
NEIGHBOUR_COLS = np.array([3, 4, 5, 3, 5, 3, 4, 5])
IRON_PICKAXE_INDEX = INV_NAMES.index("iron_pickaxe")
WOOD_PICKAXE_INDEX = INV_NAMES.index("wood_pickaxe")
DIAMOND_INDEX = INV_NAMES.index("diamond")


def _candidate_masks(blocks: np.ndarray, inventory: np.ndarray) -> dict[str, np.ndarray]:
    """Return named [T] candidate masks using only saved observation metadata."""
    visible = lambda block: np.any(blocks == block.value, axis=(1, 2))
    near = lambda block: np.any(
        blocks[:, NEIGHBOUR_ROWS, NEIGHBOUR_COLS] == block.value, axis=1
    )
    diamond_visible = visible(BlockType.DIAMOND)
    has_iron_pickaxe = inventory[:, IRON_PICKAXE_INDEX] > 0
    no_diamond_yet = inventory[:, DIAMOND_INDEX] == 0
    return {
        # Candidate definitions are intentionally conservative and are written
        # to the output CSV so later experiments do not rely on implicit rules.
        "wood_tree_visible": visible(BlockType.TREE),
        "wood_tree_near": near(BlockType.TREE),
        "table_visible": visible(BlockType.CRAFTING_TABLE),
        "table_near": near(BlockType.CRAFTING_TABLE),
        "wood_pickaxe_relevant": visible(BlockType.STONE) & (inventory[:, WOOD_PICKAXE_INDEX] == 0),
        "stone_visible": visible(BlockType.STONE),
        "stone_near": near(BlockType.STONE),
        "furnace_visible": visible(BlockType.FURNACE),
        "furnace_near": near(BlockType.FURNACE),
        "iron_visible": visible(BlockType.IRON),
        "iron_near": near(BlockType.IRON),
        "diamond_visible": diamond_visible,
        "diamond_collectible_observed": diamond_visible & has_iron_pickaxe & no_diamond_yet,
    }


def _fieldnames() -> list[str]:
    fields = ["episode", "t", "source_npz", "state_index"]
    fields += [f"inventory_{name}" for name in INV_NAMES]
    fields += list(INTRINSIC_NAMES)
    fields += [f"visible_{block.name.lower()}_count" for block in BlockType]
    fields += [f"visible_{name}_count" for name in MOB_NAMES]
    fields += [f"achievement_{achievement.name.lower()}" for achievement in Achievement]
    return fields


def _row(
    episode: int, t: int, source_npz: str, inventory: np.ndarray,
    intrinsics: np.ndarray, blocks: np.ndarray, mobs: np.ndarray, achievements: np.ndarray,
) -> dict[str, int | str]:
    result: dict[str, int | str] = {
        "episode": episode, "t": int(t), "source_npz": source_npz, "state_index": int(t),
    }
    result.update({f"inventory_{name}": int(inventory[i]) for i, name in enumerate(INV_NAMES)})
    result.update({name: int(intrinsics[i]) for i, name in enumerate(INTRINSIC_NAMES)})
    result.update({
        f"visible_{block.name.lower()}_count": int((blocks == block.value).sum())
        for block in BlockType
    })
    result.update({f"visible_{name}_count": int(mobs[..., i].sum()) for i, name in enumerate(MOB_NAMES)})
    result.update({
        f"achievement_{achievement.name.lower()}": int(achievements[i])
        for i, achievement in enumerate(Achievement)
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--diamond_bank_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    if args.diamond_bank_size <= 0:
        raise ValueError("--diamond_bank_size must be positive")

    dataset_dir = Path(args.dataset_dir)
    episode_paths = sorted((dataset_dir / "episodes").glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode NPZ files in {dataset_dir / 'episodes'}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_counts: dict[str, int] = {}
    diamond_refs: list[tuple[int, int, str]] = []
    observations: list[np.ndarray] = []
    total_states = 0
    index_path = out_dir / "state_index.csv.gz"
    with gzip.open(index_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        for path_number, path in enumerate(episode_paths, start=1):
            with np.load(path) as data:
                episode = int(data["episode_id"])
                obs = np.asarray(data["observations"], dtype=np.float32)
                inventory = np.asarray(data["inventory"], dtype=np.uint8)
                intrinsics = np.asarray(data["intrinsics"], dtype=np.uint8)
                blocks = np.asarray(data["visible_blocks"], dtype=np.uint8)
                mobs = np.asarray(data["visible_mobs"], dtype=np.uint8)
                achievements = np.asarray(data["achievements"], dtype=np.uint8)
            masks = _candidate_masks(blocks, inventory)
            total_states += len(obs)
            for name, mask in masks.items():
                candidate_counts[name] = candidate_counts.get(name, 0) + int(mask.sum())
            for t in np.flatnonzero(masks["diamond_collectible_observed"]):
                diamond_refs.append((episode, int(t), path.name))
                observations.append(obs[t])
            for t in range(len(obs)):
                writer.writerow(_row(episode, t, path.name, inventory[t], intrinsics[t], blocks[t], mobs[t], achievements[t]))
            if path_number % 100 == 0 or path_number == len(episode_paths):
                print(f"indexed {path_number}/{len(episode_paths)} episodes; diamond candidates={len(diamond_refs)}", flush=True)

    candidate_path = out_dir / "candidate_counts.csv"
    definitions = {
        "wood_tree_visible": "at least one tree tile is visible",
        "wood_tree_near": "a tree occupies an adjacent cell",
        "table_visible": "a crafting-table tile is visible",
        "table_near": "a crafting table occupies an adjacent cell",
        "wood_pickaxe_relevant": "stone is visible and no wood pickaxe is held",
        "stone_visible": "at least one stone tile is visible",
        "stone_near": "stone occupies an adjacent cell",
        "furnace_visible": "a furnace tile is visible",
        "furnace_near": "a furnace occupies an adjacent cell",
        "iron_visible": "at least one iron tile is visible",
        "iron_near": "iron occupies an adjacent cell",
        "diamond_visible": "at least one diamond tile is visible",
        "diamond_collectible_observed": "diamond visible, iron pickaxe held, and no diamond already held",
    }
    with candidate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["concept_candidate", "definition", "n_states"])
        writer.writeheader()
        for name in definitions:
            writer.writerow({"concept_candidate": name, "definition": definitions[name], "n_states": candidate_counts.get(name, 0)})

    if not observations:
        raise RuntimeError("The master dataset contains no observable collectible-diamond states.")
    selection_size = min(args.diamond_bank_size, len(observations))
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(len(observations), size=selection_size, replace=False))
    selected_obs = np.stack([observations[i] for i in selected]).astype(np.float32)
    selected_refs = [diamond_refs[i] for i in selected]
    bank_path = out_dir / "diamond_collectible_state_bank.npz"
    np.savez_compressed(
        bank_path,
        base_obs=selected_obs,
        episodes=np.asarray([x[0] for x in selected_refs], dtype=np.int32),
        base_steps=np.asarray([x[1] for x in selected_refs], dtype=np.int32),
        source_npz=np.asarray([x[2] for x in selected_refs]),
    )
    with (out_dir / "diamond_collectible_state_bank.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "episode", "t", "source_npz"])
        writer.writeheader()
        for i, (episode, t, source_npz) in enumerate(selected_refs):
            writer.writerow({"sample_index": i, "episode": episode, "t": t, "source_npz": source_npz})
    metadata = {
        "dataset_dir": str(dataset_dir),
        "n_episode_files": len(episode_paths),
        "n_indexed_states": total_states,
        "index": "state_index.csv.gz; source_npz plus state_index reconstructs the exact observation",
        "candidate_counts_csv": candidate_path.name,
        "diamond_bank": bank_path.name,
        "diamond_bank_definition": definitions["diamond_collectible_observed"],
        "n_diamond_candidates": len(observations),
        "n_diamond_bank_states": selection_size,
        "selection": f"uniform sample without replacement, numpy default_rng seed={args.seed}",
    }
    (out_dir / "index_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {index_path}")
    print(f"Saved candidate counts: {candidate_path}")
    print(f"Frozen {selection_size}/{len(observations)} collectible-diamond states: {bank_path}")


if __name__ == "__main__":
    main()

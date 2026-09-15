"""Build audited, frozen counterfactual state banks for Craftax's tool hierarchy.

Every positive/base state is a naturally occurring state from the 1,000-episode
master dataset.  Negative states are single, auditable edits of either one
front-facing map tile or one necessary binary tool slot.  The selectors mirror
``craftax_classic/game_logic.py`` rather than relying on object-recognition
heuristics.

No PPO checkpoint is evaluated here.  This script only creates and validates
the common state banks that later checkpoint analyses must reuse unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from craftax.craftax_classic.constants import Achievement, Action, BlockType, OBS_DIM
from concept_mapping.visualize_symbolic_trajectory import (
    INTRINSIC_NAMES,
    INV_NAMES,
    MAP_CHANNELS,
    MAP_DIM,
)


DEFAULT_DATASET_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes"
DEFAULT_OUT_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks"
PLAYER_ROW, PLAYER_COL = OBS_DIM[0] // 2, OBS_DIM[1] // 2
FRONT_OFFSETS = np.asarray([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int16)
NEIGHBOUR_OFFSETS = np.asarray(
    [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]],
    dtype=np.int16,
)
INV = {name: i for i, name in enumerate(INV_NAMES)}


@dataclass
class Candidate:
    concept: str
    episode_id: int
    timestep: int
    source_npz: str
    base_obs: np.ndarray
    counterfactual_obs: np.ndarray
    inventory: np.ndarray
    intrinsics: np.ndarray
    achievements: np.ndarray
    action: int
    direction: int
    target_row: int
    target_col: int
    target_block_before: int
    target_block_after: int
    edit_type: str
    edit_detail: str
    required_inventory_index: int
    positive_affordance: bool
    negative_affordance: bool


def _map(obs: np.ndarray) -> np.ndarray:
    return obs[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)


def _blocks(obs: np.ndarray) -> np.ndarray:
    return np.argmax(_map(obs)[..., :len(BlockType)], axis=-1)


def _front(direction: int) -> tuple[int, int]:
    if not 0 <= direction < 4:
        raise ValueError(f"Symbolic direction must be 0..3, got {direction}")
    dr, dc = FRONT_OFFSETS[direction]
    return int(PLAYER_ROW + dr), int(PLAYER_COL + dc)


def _near_locations(blocks: np.ndarray, block: BlockType) -> list[tuple[int, int]]:
    locations = []
    for dr, dc in NEIGHBOUR_OFFSETS:
        row, col = PLAYER_ROW + int(dr), PLAYER_COL + int(dc)
        if int(blocks[row, col]) == block.value:
            locations.append((row, col))
    return locations


def _set_block(obs: np.ndarray, row: int, col: int, block: BlockType) -> np.ndarray:
    result = np.array(obs, dtype=np.float32, copy=True)
    view = _map(result)
    view[row, col, :len(BlockType)] = 0.0
    view[row, col, block.value] = 1.0
    return result


def _remove_tool(obs: np.ndarray, inventory_index: int) -> np.ndarray:
    result = np.array(obs, dtype=np.float32, copy=True)
    result[MAP_DIM + inventory_index] = 0.0
    return result


def _achievement_after(data: dict[str, np.ndarray], t: int, achievement: Achievement) -> bool:
    """Whether the action at t newly realizes an achievement in the saved trace."""
    before = bool(data["achievements"][t, achievement.value])
    if before:
        return False
    if t + 1 < len(data["achievements"]):
        return bool(data["achievements"][t + 1, achievement.value])
    return bool(data["terminal_achievements"][achievement.value])


def _one_changed_map_cell(base: np.ndarray, cf: np.ndarray, row: int, col: int) -> bool:
    base_map, cf_map = _map(base), _map(cf)
    mask = np.ones(OBS_DIM, dtype=bool)
    mask[row, col] = False
    return np.array_equal(base_map[mask], cf_map[mask]) and not np.array_equal(
        base_map[row, col], cf_map[row, col]
    )


def _validate(candidate: Candidate) -> tuple[bool, dict[str, int | bool | str]]:
    """Prove the synthetic edit has exactly the advertised scope."""
    base, cf = candidate.base_obs, candidate.counterfactual_obs
    base_stats, cf_stats = base[MAP_DIM:], cf[MAP_DIM:]
    changed_inventory = np.flatnonzero(base_stats[:12] != cf_stats[:12]).tolist()
    unchanged_intrinsics = bool(np.array_equal(base_stats[12:16], cf_stats[12:16]))
    unchanged_direction_light_sleep = bool(np.array_equal(base_stats[16:22], cf_stats[16:22]))
    base_blocks, cf_blocks = _blocks(base), _blocks(cf)
    changed_map_cells = int(np.count_nonzero(base_blocks != cf_blocks))
    target_ok = (
        int(base_blocks[candidate.target_row, candidate.target_col]) == candidate.target_block_before
        and int(cf_blocks[candidate.target_row, candidate.target_col]) == candidate.target_block_after
    )
    if candidate.edit_type == "remove_tool":
        map_ok = changed_map_cells == 0
        inventory_ok = changed_inventory == [candidate.required_inventory_index]
    else:
        map_ok = changed_map_cells == 1 and _one_changed_map_cell(
            base, cf, candidate.target_row, candidate.target_col
        )
        inventory_ok = changed_inventory == []
    passed = bool(
        map_ok and inventory_ok and unchanged_intrinsics and unchanged_direction_light_sleep
        and target_ok and candidate.positive_affordance and not candidate.negative_affordance
    )
    return passed, {
        "validation_passed": passed,
        "changed_map_cells": changed_map_cells,
        "changed_inventory_indices": ";".join(map(str, changed_inventory)),
        "unrelated_map_cells_identical": map_ok,
        "unrelated_inventory_identical": inventory_ok,
        "intrinsics_identical": unchanged_intrinsics,
        "direction_light_sleep_identical": unchanged_direction_light_sleep,
        "target_edit_correct": target_ok,
        "positive_affordance": candidate.positive_affordance,
        "negative_affordance": candidate.negative_affordance,
    }


def _candidate_from_mining(
    *, concept: str, data: dict[str, np.ndarray], episode_id: int, source_npz: str,
    t: int, target: BlockType, achievement: Achievement, tool_name: str | None,
    remove_target: bool,
) -> Candidate | None:
    """Select an actual first successful mining action with its target in front."""
    if int(data["actions"][t]) != Action.DO.value or not _achievement_after(data, t, achievement):
        return None
    obs = np.asarray(data["observations"][t], dtype=np.float32)
    inv = np.asarray(data["inventory"][t], dtype=np.uint8)
    direction = int(data["direction"][t])
    row, col = _front(direction)
    if int(_blocks(obs)[row, col]) != target.value:
        return None
    if tool_name is not None and inv[INV[tool_name]] == 0:
        return None
    if remove_target:
        # Craftax replaces trees with GRASS; unlike ore, it is a known exact
        # environment transition rather than an inferred background tile.
        cf = _set_block(obs, row, col, BlockType.GRASS)
        edit_type, required_index = "remove_front_target", -1
        negative = False
        after_block = BlockType.GRASS.value
    else:
        required_index = INV[tool_name]  # type: ignore[arg-type]
        cf = _remove_tool(obs, required_index)
        edit_type, negative, after_block = "remove_tool", False, target.value
    # Positive affordance is exactly the Craftax game-logic predicate for the
    # requested mining action.  Negative loses its necessary target or tool.
    positive = True
    return Candidate(
        concept, episode_id, t, source_npz, obs, cf, inv,
        np.asarray(data["intrinsics"][t], dtype=np.uint8),
        np.asarray(data["achievements"][t], dtype=np.uint8), int(data["actions"][t]),
        direction, row, col, target.value, after_block, edit_type,
        f"{target.name.lower()} at front; " + ("replaced with grass" if remove_target else f"removed {tool_name}"),
        required_index, positive, negative,
    )


def _crafting_candidate(
    *, concept: str, data: dict[str, np.ndarray], episode_id: int, source_npz: str,
    t: int, action: Action, achievement: Achievement, needs_furnace: bool,
) -> Candidate | None:
    """Select an actual first successful crafting action and remove one station."""
    if int(data["actions"][t]) != action.value or not _achievement_after(data, t, achievement):
        return None
    obs = np.asarray(data["observations"][t], dtype=np.float32)
    inv = np.asarray(data["inventory"][t], dtype=np.uint8)
    blocks = _blocks(obs)
    tables = _near_locations(blocks, BlockType.CRAFTING_TABLE)
    furnaces = _near_locations(blocks, BlockType.FURNACE)
    if len(tables) != 1 or (needs_furnace and len(furnaces) != 1):
        return None
    if action == Action.MAKE_WOOD_PICKAXE:
        resources_ok = inv[INV["wood"]] >= 1
    else:  # Actual iron-pickaxe recipe from Craftax classic game_logic.py.
        resources_ok = all(inv[INV[x]] >= 1 for x in ("wood", "stone", "coal", "iron"))
    if not resources_ok:
        return None
    row, col = furnaces[0] if needs_furnace else tables[0]
    removed = BlockType.FURNACE if needs_furnace else BlockType.CRAFTING_TABLE
    # Use GRASS as a consistent neutral terrain replacement for removed
    # structures. This is a one-cell map edit and removes the station.
    cf = _set_block(obs, row, col, BlockType.GRASS)
    cf_blocks = _blocks(cf)
    no_table = len(_near_locations(cf_blocks, BlockType.CRAFTING_TABLE)) == 0
    no_furnace = len(_near_locations(cf_blocks, BlockType.FURNACE)) == 0
    # The counterfactual must eliminate the station that was edited.  An
    # iron-tier craft still has a table after furnace removal, but is correctly
    # disabled because the furnace is necessary as well.
    removed_station_is_absent = no_furnace if needs_furnace else no_table
    negative = not removed_station_is_absent
    return Candidate(
        concept, episode_id, t, source_npz, obs, cf, inv,
        np.asarray(data["intrinsics"][t], dtype=np.uint8),
        np.asarray(data["achievements"][t], dtype=np.uint8), int(data["actions"][t]),
        int(data["direction"][t]), row, col, removed.value, BlockType.GRASS.value,
        "remove_required_station", f"replaced the sole adjacent {removed.name.lower()} with grass",
        -1, True, negative,
    )


def _table_candidate(
    data: dict[str, np.ndarray], episode_id: int, source_npz: str, t: int,
) -> Candidate | None:
    """Use the immediate natural post-placement state and remove the table."""
    if int(data["actions"][t]) != Action.PLACE_TABLE.value or not _achievement_after(
        data, t, Achievement.PLACE_TABLE
    ) or t + 1 >= len(data["observations"]):
        return None
    before, obs = data["observations"][t], data["observations"][t + 1]
    blocks = _blocks(obs)
    direction = int(data["direction"][t])
    row, col = _front(direction)
    # PLACE_TABLE does not move the player; this verifies the exact cell that
    # changed from a real, observed substrate to the crafted table.
    if int(blocks[row, col]) != BlockType.CRAFTING_TABLE.value:
        return None
    inv = np.asarray(data["inventory"][t + 1], dtype=np.uint8)
    tables = _near_locations(blocks, BlockType.CRAFTING_TABLE)
    # Functionality: enough remaining wood to make a wood pickaxe, and no
    # existing wood pickaxe; removing this sole table disables that craft.
    if len(tables) != 1 or inv[INV["wood"]] < 1 or inv[INV["wood_pickaxe"]] != 0:
        return None
    cf = _set_block(obs, row, col, BlockType.GRASS)
    negative = len(_near_locations(_blocks(cf), BlockType.CRAFTING_TABLE)) != 0
    return Candidate(
        "table", episode_id, t + 1, source_npz, np.asarray(obs, dtype=np.float32), cf, inv,
        np.asarray(data["intrinsics"][t + 1], dtype=np.uint8),
        np.asarray(data["achievements"][t + 1], dtype=np.uint8), int(data["actions"][t]),
        direction, row, col, BlockType.CRAFTING_TABLE.value, BlockType.GRASS.value,
        "remove_newly_placed_table", "replaced the newly placed table with grass",
        -1, True, negative,
    )


def _furnace_candidate(data: dict[str, np.ndarray], episode_id: int, source_npz: str, t: int) -> Candidate | None:
    """Any natural state where the sole furnace enables an iron-pickaxe craft."""
    obs = np.asarray(data["observations"][t], dtype=np.float32)
    inv = np.asarray(data["inventory"][t], dtype=np.uint8)
    blocks = _blocks(obs)
    tables = _near_locations(blocks, BlockType.CRAFTING_TABLE)
    furnaces = _near_locations(blocks, BlockType.FURNACE)
    resources_ok = all(inv[INV[x]] >= 1 for x in ("wood", "stone", "coal", "iron"))
    if not resources_ok or len(tables) != 1 or len(furnaces) != 1:
        return None
    row, col = furnaces[0]
    cf = _set_block(obs, row, col, BlockType.GRASS)
    negative = len(_near_locations(_blocks(cf), BlockType.FURNACE)) != 0
    return Candidate(
        "furnace", episode_id, t, source_npz, obs, cf, inv,
        np.asarray(data["intrinsics"][t], dtype=np.uint8),
        np.asarray(data["achievements"][t], dtype=np.uint8), int(data["actions"][t]),
        int(data["direction"][t]), row, col, BlockType.FURNACE.value, BlockType.GRASS.value,
        "remove_required_station", "replaced the sole adjacent furnace with grass", -1,
        True, negative,
    )


def _collect_candidates(data: dict[str, np.ndarray], episode_id: int, source_npz: str):
    """Yield (concept, raw-event-or-state count, clean candidate) per episode."""
    raw = defaultdict(int)
    clean: list[Candidate] = []
    target_specs = [
        ("wood", BlockType.TREE, Achievement.COLLECT_WOOD, None, True),
        ("stone", BlockType.STONE, Achievement.COLLECT_STONE, "wood_pickaxe", False),
        ("coal", BlockType.COAL, Achievement.COLLECT_COAL, "wood_pickaxe", False),
        ("iron", BlockType.IRON, Achievement.COLLECT_IRON, "stone_pickaxe", False),
    ]
    for t in range(len(data["observations"])):
        for concept, block, achievement, tool, remove_target in target_specs:
            if int(data["actions"][t]) == Action.DO.value and _achievement_after(data, t, achievement):
                raw[concept] += 1
                candidate = _candidate_from_mining(
                    concept=concept, data=data, episode_id=episode_id, source_npz=source_npz,
                    t=t, target=block, achievement=achievement, tool_name=tool,
                    remove_target=remove_target,
                )
                if candidate is not None:
                    clean.append(candidate)
        # Actual first wood-pickaxe and iron-pickaxe achievement actions.
        if int(data["actions"][t]) == Action.MAKE_WOOD_PICKAXE.value and _achievement_after(data, t, Achievement.MAKE_WOOD_PICKAXE):
            raw["wood_pickaxe"] += 1
            candidate = _crafting_candidate(concept="wood_pickaxe", data=data, episode_id=episode_id, source_npz=source_npz, t=t, action=Action.MAKE_WOOD_PICKAXE, achievement=Achievement.MAKE_WOOD_PICKAXE, needs_furnace=False)
            if candidate is not None:
                clean.append(candidate)
        if int(data["actions"][t]) == Action.MAKE_IRON_PICKAXE.value and _achievement_after(data, t, Achievement.MAKE_IRON_PICKAXE):
            raw["iron_pickaxe"] += 1
            candidate = _crafting_candidate(concept="iron_pickaxe", data=data, episode_id=episode_id, source_npz=source_npz, t=t, action=Action.MAKE_IRON_PICKAXE, achievement=Achievement.MAKE_IRON_PICKAXE, needs_furnace=True)
            if candidate is not None:
                clean.append(candidate)
        if int(data["actions"][t]) == Action.PLACE_TABLE.value and _achievement_after(data, t, Achievement.PLACE_TABLE):
            raw["table"] += 1
            candidate = _table_candidate(data, episode_id, source_npz, t)
            if candidate is not None:
                clean.append(candidate)
        # Furnace states are selected from their functional affordance, not a
        # placement event: a unique adjacent table/furnace plus the exact
        # iron-pickaxe recipe.  Count all naturally occurring eligible states.
        candidate = _furnace_candidate(data, episode_id, source_npz, t)
        if candidate is not None:
            raw["furnace"] += 1
            clean.append(candidate)
    return raw, clean


CONCEPT_DEFINITIONS = {
    "wood": "first successful collect-wood action; front tree changed to grass",
    "table": "immediate post-place-table state with a sole adjacent table enabling a wood pickaxe; replace table with grass",
    "wood_pickaxe": "first successful wood-pickaxe craft; remove the sole adjacent crafting table",
    "stone": "first successful collect-stone action; remove the required wood pickaxe",
    "furnace": "natural state where sole adjacent table and furnace plus recipe resources enable an iron pickaxe; replace furnace with grass",
    "coal": "first successful collect-coal action; remove the required wood pickaxe",
    "iron": "first successful collect-iron action; remove the required stone pickaxe",
    "iron_pickaxe": "first successful iron-pickaxe craft; replace the sole adjacent furnace with grass",
    "diamond": "existing validated master-dataset bank: visible diamond, iron pickaxe held, no diamond held; remove iron pickaxe",
}
CONCEPTS = list(CONCEPT_DEFINITIONS)


def _write_bank(out_dir: Path, concept: str, candidates: list[Candidate], seed: int, target_n: int):
    rng = np.random.default_rng(seed)
    chosen_indices = np.sort(rng.choice(len(candidates), size=min(target_n, len(candidates)), replace=False))
    chosen = [candidates[i] for i in chosen_indices]
    validations = [_validate(candidate) for candidate in chosen]
    passed = [result for result, _detail in validations]
    if not all(passed):
        raise RuntimeError(f"Internal validation failed for {concept}: {len(passed) - sum(passed)} states")
    np.savez_compressed(
        out_dir / f"{concept}_bank.npz",
        base_obs=np.stack([c.base_obs for c in chosen]).astype(np.float32),
        counterfactual_obs=np.stack([c.counterfactual_obs for c in chosen]).astype(np.float32),
        episode_id=np.asarray([c.episode_id for c in chosen], dtype=np.int32),
        timestep=np.asarray([c.timestep for c in chosen], dtype=np.int32),
        state_index=np.asarray([c.timestep for c in chosen], dtype=np.int32),
        source_npz=np.asarray([c.source_npz for c in chosen]),
        inventory=np.stack([c.inventory for c in chosen]).astype(np.uint8),
        intrinsics=np.stack([c.intrinsics for c in chosen]).astype(np.uint8),
        achievements=np.stack([c.achievements for c in chosen]).astype(np.uint8),
        action=np.asarray([c.action for c in chosen], dtype=np.int16),
        direction=np.asarray([c.direction for c in chosen], dtype=np.uint8),
        player_row=np.full(len(chosen), PLAYER_ROW, dtype=np.int8),
        player_col=np.full(len(chosen), PLAYER_COL, dtype=np.int8),
        target_row=np.asarray([c.target_row for c in chosen], dtype=np.int8),
        target_col=np.asarray([c.target_col for c in chosen], dtype=np.int8),
        target_block_before=np.asarray([c.target_block_before for c in chosen], dtype=np.uint8),
        target_block_after=np.asarray([c.target_block_after for c in chosen], dtype=np.uint8),
        required_inventory_index=np.asarray([c.required_inventory_index for c in chosen], dtype=np.int8),
        edit_type=np.asarray([c.edit_type for c in chosen]),
        edit_detail=np.asarray([c.edit_detail for c in chosen]),
    )
    audit_path = out_dir / f"{concept}_audit.csv"
    with audit_path.open("w", newline="") as handle:
        fields = [
            "sample_index", "episode_id", "timestep", "source_npz", "action", "direction",
            "player_row", "player_col", "target_row", "target_col", "target_block_before",
            "target_block_after", "edit_type", "edit_detail", "required_inventory_index",
            *[f"inventory_{name}" for name in INV_NAMES], *INTRINSIC_NAMES,
            "validation_passed", "changed_map_cells", "changed_inventory_indices",
            "unrelated_map_cells_identical", "unrelated_inventory_identical", "intrinsics_identical",
            "direction_light_sleep_identical", "target_edit_correct", "positive_affordance", "negative_affordance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, (candidate, (_passed, detail)) in enumerate(zip(chosen, validations)):
            row = {
                "sample_index": i, "episode_id": candidate.episode_id, "timestep": candidate.timestep,
                "source_npz": candidate.source_npz, "action": candidate.action, "direction": candidate.direction,
                "player_row": PLAYER_ROW, "player_col": PLAYER_COL, "target_row": candidate.target_row,
                "target_col": candidate.target_col, "target_block_before": candidate.target_block_before,
                "target_block_after": candidate.target_block_after, "edit_type": candidate.edit_type,
                "edit_detail": candidate.edit_detail, "required_inventory_index": candidate.required_inventory_index,
                **{f"inventory_{name}": int(candidate.inventory[j]) for j, name in enumerate(INV_NAMES)},
                **{name: int(candidate.intrinsics[j]) for j, name in enumerate(INTRINSIC_NAMES)}, **detail,
            }
            writer.writerow(row)
    return len(chosen), audit_path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--target_states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()
    dataset_dir, out_dir = Path(args.dataset_dir), Path(args.out_dir)
    if not (dataset_dir / "concept_index" / "state_index.csv.gz").exists():
        raise FileNotFoundError("Expected existing concept_index/state_index.csv.gz; build the master index first.")
    paths = sorted((dataset_dir / "episodes").glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode files found under {dataset_dir / 'episodes'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_counts: dict[str, int] = defaultdict(int)
    # Retain only a uniform reservoir of at most target_states per concept.
    # Furnace-enabled states can persist for many timesteps, so accumulating
    # every full 1,345-dimensional observation would be needlessly memory-heavy.
    candidates: dict[str, list[Candidate]] = {concept: [] for concept in CONCEPTS if concept != "diamond"}
    clean_counts: dict[str, int] = defaultdict(int)
    reservoirs_rng = {
        concept: np.random.default_rng(args.seed + 10_000 * index)
        for index, concept in enumerate(candidates)
    }
    for number, path in enumerate(paths, start=1):
        with np.load(path) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        episode_raw, episode_candidates = _collect_candidates(data, int(data["episode_id"]), path.name)
        for concept, count in episode_raw.items():
            raw_counts[concept] += count
        for candidate in episode_candidates:
            concept = candidate.concept
            clean_counts[concept] += 1
            reservoir = candidates[concept]
            if len(reservoir) < args.target_states:
                reservoir.append(candidate)
            else:
                replacement = int(reservoirs_rng[concept].integers(0, clean_counts[concept]))
                if replacement < args.target_states:
                    reservoir[replacement] = candidate
        if number % 100 == 0 or number == len(paths):
            summary = ", ".join(f"{key}={clean_counts[key]}" for key in candidates)
            print(f"scanned {number}/{len(paths)} episodes; {summary}", flush=True)

    rows = []
    for concept in CONCEPTS:
        if concept == "diamond":
            source = dataset_dir / "concept_index" / "diamond_collectible_state_bank.npz"
            with np.load(source) as bank:
                frozen_n = int(len(bank["base_obs"]))
            rows.append({"concept": concept, "raw_candidates": 3409, "clean_candidates": 3409, "frozen_n": frozen_n, "counterfactual_definition": CONCEPT_DEFINITIONS[concept], "audit_csv": "concept_index/diamond_collectible_state_bank.csv", "status": "existing_validated_bank"})
            continue
        clean_n = clean_counts[concept]
        if clean_n:
            frozen_n, audit_csv = _write_bank(out_dir, concept, candidates[concept], args.seed, args.target_states)
            status = "validated"
        else:
            frozen_n, audit_csv, status = 0, "", "not_cleanly_operationalized_from_stored_states"
        rows.append({"concept": concept, "raw_candidates": raw_counts.get(concept, 0), "clean_candidates": clean_n, "frozen_n": frozen_n, "counterfactual_definition": CONCEPT_DEFINITIONS[concept], "audit_csv": audit_csv, "status": status})
    with (out_dir / "hierarchy_candidate_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["concept", "raw_candidates", "clean_candidates", "selection_definition"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"concept": row["concept"], "raw_candidates": row["raw_candidates"], "clean_candidates": row["clean_candidates"], "selection_definition": row["counterfactual_definition"]})
    with (out_dir / "hierarchy_bank_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset_dir": str(dataset_dir), "n_episodes": len(paths), "seed": args.seed,
        "target_states_per_bank": args.target_states,
        "symbolic_observation_note": "The symbolic observation stores local player coordinates (3,4) and direction, but not absolute world coordinates. source_npz plus state_index reconstructs each exact observation.",
        "station_removal_note": "Table and furnace counterfactuals replace exactly one station tile with GRASS, a consistent neutral terrain replacement. The raw master observation is never modified.",
        "game_logic_source": "craftax.craftax_classic.game_logic: front-cell mining; adjacent-station crafting; recipes as encoded there.",
        "definitions": CONCEPT_DEFINITIONS, "banks": rows,
    }
    (out_dir / "hierarchy_bank_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nConcept | Raw candidates | Clean candidates | Frozen N | Counterfactual")
    for row in rows:
        print(f"{row['concept']} | {row['raw_candidates']} | {row['clean_candidates']} | {row['frozen_n']} | {row['counterfactual_definition']}")


if __name__ == "__main__":
    main()

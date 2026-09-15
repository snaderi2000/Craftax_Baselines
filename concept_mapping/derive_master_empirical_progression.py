"""Derive empirical achievement and inventory order from the master episodes."""

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
from craftax.craftax_classic.constants import Achievement

from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_DIM


MASTER = Path("concept_mapping/runs/master_symbolic_1p5b_1000episodes")
OUT = MASTER / "hierarchy_concept_banks" / "progression_vs_pickaxe_value"


def _summaries(names: list[str], first: np.ndarray) -> list[dict]:
    rows = []
    for column, name in enumerate(names):
        values = first[:, column]
        acquired = values >= 0
        rows.append(
            {
                "name": name,
                "episodes_observed": int(acquired.sum()),
                "observation_rate": float(acquired.mean()),
                "median_first_step_among_observed": (
                    float(np.median(values[acquired])) if acquired.any() else None
                ),
                "mean_first_step_among_observed": (
                    float(values[acquired].mean()) if acquired.any() else None
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float("inf") if row["median_first_step_among_observed"] is None
            else row["median_first_step_among_observed"],
            row["name"],
        ),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    files = sorted((MASTER / "episodes").glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(MASTER / "episodes")
    achievement_names = [x.name.lower() for x in Achievement]
    first_achievement = np.full((len(files), len(achievement_names)), -1, np.int32)
    first_inventory = np.full((len(files), len(INV_NAMES)), -1, np.int32)

    for episode_index, path in enumerate(files):
        with np.load(path) as data:
            achievements = np.concatenate(
                [np.asarray(data["achievements"]), np.asarray(data["terminal_achievements"])[None]], axis=0
            )
            terminal_inventory = np.rint(
                np.asarray(data["terminal_observation"])[MAP_DIM : MAP_DIM + len(INV_NAMES)] * 10
            ).astype(np.uint8)
            inventory = np.concatenate(
                [np.asarray(data["inventory"]), terminal_inventory[None]], axis=0
            )
        for column in range(achievements.shape[1]):
            hits = np.flatnonzero(achievements[:, column] > 0)
            if len(hits):
                first_achievement[episode_index, column] = int(hits[0])
        for column in range(inventory.shape[1]):
            hits = np.flatnonzero(inventory[:, column] > 0)
            if len(hits):
                first_inventory[episode_index, column] = int(hits[0])

    achievement_rows = _summaries(achievement_names, first_achievement)
    inventory_rows = _summaries(list(INV_NAMES), first_inventory)
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "empirical_achievement_order_1p5b.csv", achievement_rows)
    _write_csv(OUT / "empirical_inventory_order_1p5b.csv", inventory_rows)
    np.savez_compressed(
        OUT / "empirical_first_acquisition_steps_1p5b.npz",
        achievement_names=np.asarray(achievement_names),
        first_achievement=first_achievement,
        inventory_names=np.asarray(INV_NAMES),
        first_inventory=first_inventory,
    )
    metadata = {
        "master_dataset": str(MASTER),
        "n_episodes": len(files),
        "checkpoint": int(np.load(files[0])["checkpoint"]),
        "step_semantics": "First stored state (including terminal state) where the achievement/item is present",
        "achievement_order": [row["name"] for row in achievement_rows],
        "inventory_order": [row["name"] for row in inventory_rows],
    }
    (OUT / "empirical_progression_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

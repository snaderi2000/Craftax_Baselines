"""Merge and deduplicate bounded water/sword counterfactual collection shards."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _obs_key(obs: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()


def _merge_task(shard_dirs: list[Path], out_dir: Path, task: str) -> dict[str, np.ndarray]:
    samples, metadata, values, seen = [], [], [], set()
    for shard_dir in shard_dirs:
        states = np.load(shard_dir / f"{task}_base_states.npz")
        with (shard_dir / f"{task}_metadata.csv").open(newline="") as handle:
            shard_metadata = list(csv.DictReader(handle))
        with (shard_dir / f"{task}_counterfactual_values.csv").open(newline="") as handle:
            shard_values = list(csv.DictReader(handle))
        if not (len(states["base_obs"]) == len(shard_metadata) == len(shard_values)):
            raise RuntimeError(f"Mismatched {task} files in {shard_dir}")
        for index, obs in enumerate(states["base_obs"]):
            key = _obs_key(obs)
            if key in seen:
                continue
            seen.add(key)
            samples.append(np.asarray(obs, dtype=np.float32))
            row = dict(shard_metadata[index])
            row.update({"shard": shard_dir.name, "merged_index": len(samples) - 1})
            metadata.append(row)
            value_row = dict(shard_values[index])
            value_row.update({"shard": shard_dir.name, "merged_index": len(samples) - 1})
            values.append(value_row)

    if not samples:
        raise RuntimeError(f"No {task} states found")
    np.savez_compressed(out_dir / f"{task}_base_states.npz", base_obs=np.stack(samples))
    for filename, rows in ((f"{task}_metadata.csv", metadata), (f"{task}_counterfactual_values.csv", values)):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    numeric = {key: np.asarray([float(row[key]) for row in values]) for key in values[0] if key not in {"sample_index", "episode", "base_step", "shard", "merged_index"}}
    return numeric


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge water/sword collection shards.")
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    shard_root, out_dir = Path(args.shard_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise RuntimeError(f"No shard directories found in {shard_root}")

    water = _merge_task(shard_dirs, out_dir, "water")
    sword = _merge_task(shard_dirs, out_dir, "sword")
    empty_effect = water["empty_water"] - water["empty_no_water"]
    full_effect = water["full_water"] - water["full_no_water"]
    iron, stone, wood = sword["iron_only"], sword["stone_only"], sword["wood_only"]
    summary = {
        "water_n": len(empty_effect),
        "water_empty_effect_positive_rate": float((empty_effect > 0).mean()),
        "water_interaction_positive_rate": float(((empty_effect - full_effect) > 0).mean()),
        "sword_n": len(iron),
        "sword_iron_gt_stone_rate": float((iron > stone).mean()),
        "sword_stone_gt_wood_rate": float((stone > wood).mean()),
        "sword_full_ranking_rate": float(((iron > stone) & (stone > wood)).mean()),
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Merged {summary['water_n']} water and {summary['sword_n']} sword states into {out_dir}")


if __name__ == "__main__":
    main()

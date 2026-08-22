"""Merge independently collected diamond-affordance shards."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


VALUE_COLUMNS = (
    "A_no_diamond_no_pickaxe",
    "B_no_diamond_pickaxe",
    "C_diamond_no_pickaxe",
    "D_diamond_pickaxe",
)


def _obs_key(obs: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate and merge four diamond-affordance collector shards."
    )
    parser.add_argument("--shard_root", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise RuntimeError(f"No shard directories found in {shard_root}")

    rows = []
    base_obs, episodes, base_steps, collect_steps = [], [], [], []
    seen = set()
    for shard_dir in shard_dirs:
        states_path = shard_dir / "clean_diamond_tminus_states.npz"
        values_path = shard_dir / "clean_counterfactual_values.csv"
        if not states_path.exists() or not values_path.exists():
            raise RuntimeError(f"Incomplete shard: {shard_dir}")
        states = np.load(states_path)
        with values_path.open(newline="") as handle:
            shard_rows = list(csv.DictReader(handle))
        if len(states["base_obs"]) != len(shard_rows):
            raise RuntimeError(f"State/value count mismatch in {shard_dir}")
        for index, obs in enumerate(states["base_obs"]):
            key = _obs_key(obs)
            if key in seen:
                continue
            seen.add(key)
            row = dict(shard_rows[index])
            row["shard"] = shard_dir.name
            row["merged_index"] = len(rows)
            rows.append(row)
            base_obs.append(np.asarray(obs, dtype=np.float32))
            episodes.append(int(states["episodes"][index]))
            base_steps.append(int(states["base_steps"][index]))
            collect_steps.append(int(states["collect_steps"][index]))

    if not rows:
        raise RuntimeError("No states found after merging shards")

    np.savez_compressed(
        out_dir / "clean_diamond_tminus_states.npz",
        base_obs=np.stack(base_obs),
        episodes=np.asarray(episodes, dtype=np.int32),
        base_steps=np.asarray(base_steps, dtype=np.int32),
        collect_steps=np.asarray(collect_steps, dtype=np.int32),
    )

    output_columns = ["merged_index", "shard", *rows[0].keys()]
    output_columns = list(dict.fromkeys(output_columns))
    with (out_dir / "clean_counterfactual_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(rows)

    values = {
        key: np.asarray([float(row[key]) for row in rows]) for key in VALUE_COLUMNS
    }
    a, b, c, d = (values[key] for key in VALUE_COLUMNS)
    middle_free = (d > a) & (d > c) & (a > b) & (c > b)
    tantalization = a - c
    summary = [
        ("n_states", len(rows)),
        ("middle_free_success_rate", float(middle_free.mean())),
        ("D_gt_A_rate", float((d > a).mean())),
        ("D_gt_C_rate", float((d > c).mean())),
        ("A_gt_B_rate", float((a > b).mean())),
        ("C_gt_B_rate", float((c > b).mean())),
        ("tantalization_C_lt_A_rate", float((tantalization > 0).mean())),
        ("mean_tantalization_A_minus_C", float(tantalization.mean())),
        ("median_tantalization_A_minus_C", float(np.median(tantalization))),
    ]
    with (out_dir / "clean_ordering_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)

    print(f"Merged {len(rows)} unique states from {len(shard_dirs)} shards into {out_dir}")


if __name__ == "__main__":
    main()

"""Compare offline spatial predictors on the step-512 Craftax reconstruction split."""

import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import torch

from concept_mapping.reconstruct_partial_terrain import (
    ARCHIVE, N_TILES, RUNS, SEED, SIDE, STEP, UNKNOWN, TerrainCompletion,
    expected_top_decile_hits, snapshot,
)


OUTPUT = RUNS / "offline_map_benchmark"
CHECKPOINT = RUNS / "terrain_reconstruction_512/model.pt"
DISTANCES = (1, 2, 4, 8, 16)
REGION = 8
TILE_INDEX = np.indices((SIDE, SIDE))
REGIONS = (TILE_INDEX[0] // REGION) * (SIDE // REGION) + TILE_INDEX[1] // REGION


def load_snapshots():
    files = sorted(ARCHIVE.glob("episode_*.npz"))
    order = np.random.default_rng(SEED).permutation(len(files))
    files = [files[i] for i in order]
    identity = hashlib.sha256()
    for path in files:
        st = path.stat()
        identity.update(f"{path.name}/{st.st_size}/{st.st_mtime_ns}\n".encode())
    signature = identity.hexdigest()
    cache = OUTPUT / f"snapshots_{STEP}_{signature[:12]}.npz"
    if not cache.exists():
        memory, ages, truth, _ = map(np.stack, zip(*(snapshot(path) for path in files)))
        np.savez_compressed(cache, memory=memory, ages=ages, truth=truth)
    with np.load(cache) as data:
        return files, data["memory"], data["ages"], data["truth"]


def neighbor_features(memory):
    distances, indices = distance_transform_edt(memory == UNKNOWN, return_indices=True)
    bands = np.digitize(distances, DISTANCES, right=True)
    nearest = memory[tuple(indices)]
    return nearest * (len(DISTANCES) + 1) + bands, distances


def fit_spatial_tables(memory, truth):
    n_features = N_TILES * (len(DISTANCES) + 1)
    counts = np.zeros((SIDE // REGION * (SIDE // REGION), n_features, N_TILES))
    for grid, target in zip(memory, truth):
        features, _ = neighbor_features(grid)
        unknown = grid == UNKNOWN
        np.add.at(counts, (REGIONS[unknown], features[unknown], target[unknown]), 1)
    coord_counts = np.stack([((truth == tile) & (memory == UNKNOWN)).sum(0)
                             for tile in range(N_TILES)]).astype(float)
    coord_counts = gaussian_filter(coord_counts, sigma=(0, 1.5, 1.5)) + 1e-10
    prior = coord_counts / coord_counts.sum(0)
    # Pool across regions for rare combinations of distance and observed terrain.
    pooled = counts.sum(0) + .01
    pooled /= pooled.sum(-1, keepdims=True)
    prob = counts + 128 * pooled
    return prior, prob / prob.sum(-1, keepdims=True)


def hit_in_three_regions(score, unknown, diamonds):
    # Sum scores only over the still-unseen part of each 8x8 region.
    mass = np.bincount(REGIONS[unknown], weights=score[unknown], minlength=64)
    found = np.bincount(REGIONS[diamonds], minlength=64) > 0
    ranked = np.argsort(-mass, kind="stable")[:3]
    return int(found[ranked].any())


def main():
    start = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files, memory, ages, truth = load_snapshots()
    cached_at = perf_counter()
    n = len(files) * 4 // 5
    torch.set_num_threads(4)
    net = TerrainCompletion().eval()
    net.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    prior, table = fit_spatial_tables(memory[:n], truth[:n])
    rows = []
    with torch.inference_mode():
        for i in range(n, len(files)):
            grid = memory[i]
            features, dist = neighbor_features(grid)
            unknown = grid == UNKNOWN
            gold = truth[i]
            diamonds = (gold == 10) & unknown
            near = (dist <= 4) & unknown
            unet = net(torch.from_numpy(grid[None]), torch.from_numpy(ages[i:i+1])).softmax(1)[0].numpy()
            local = table[REGIONS, features].transpose(2, 0, 1)
            for name, prob, diamond_score in [
                ("coordinate", prior, prior[10]),
                ("neighbor_table", local, local[10]),
                ("unet", unet, unet[10]),
                ("unet_rock_score", unet, unet[[4, 8, 9, 10]].sum(axis=0)),
            ]:
                matches = gold == prob.argmax(0)
                rows.append(dict(model=name, episode=files[i].stem,
                                 unseen=int(unknown.sum()), correct=int(matches[unknown].sum()),
                                 near=int(near.sum()), near_correct=int(matches[near].sum()),
                                 diamonds=int(diamonds.sum()),
                                 tile_hits=expected_top_decile_hits(diamond_score[unknown], diamonds[unknown]),
                                 has_diamond=int(diamonds.any()),
                                 region_hit=hit_in_three_regions(diamond_score, unknown, diamonds)))
    with (OUTPUT / "per_world.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    draws = np.random.default_rng(SEED + 1).integers(0, len(files)-n, (5000, len(files)-n))
    scores, boot = {}, {}
    metrics = dict(terrain=("correct", "unseen"), near_terrain=("near_correct", "near"),
                   diamond_top_decile=("tile_hits", "diamonds"), region_top3=("region_hit", "has_diamond"))
    for name in dict.fromkeys(row["model"] for row in rows):
        subset = [row for row in rows if row["model"] == name]
        scores[name], boot[name] = {}, {}
        for metric, (a, b) in metrics.items():
            num, den = (np.array([r[col] for r in subset]) for col in (a, b))
            scores[name][metric] = num.sum() / den.sum()
            boot[name][metric] = num[draws].sum(1) / den[draws].sum(1)
    intervals = {name: {metric: np.quantile(values - boot["coordinate"][metric], [.025, .975]).tolist()
                        for metric, values in boot[name].items()} for name in boot if name != "coordinate"}
    data = dict(train_worlds=n, evaluation_worlds=len(files)-n, snapshot_max_step=STEP,
                evaluation_worlds_with_unseen_diamond=sum(r["has_diamond"] for r in rows if r["model"] == "unet"),
                scores=scores, paired_intervals_vs_coordinate=intervals,
                read_or_build_cache_seconds=round(cached_at-start, 2),
                fit_and_compare_seconds=round(perf_counter()-cached_at, 2))
    (OUTPUT / "metrics.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

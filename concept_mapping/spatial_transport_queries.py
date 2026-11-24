"""Paired offscreen localization queries from previously observed terrain only."""
import csv
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from concept_mapping.sru_reconstruction_data import ARCHIVE

HORIZON = 2304
STRIDE = 32
BANDS = ((1, 99), (100, 131), (500, 531), (1000, 1031))
RESOURCES = (3, 10)  # water and diamond in the Classic symbolic observation
LOCAL = np.stack(np.meshgrid(np.arange(-3, 4), np.arange(-4, 5), indexing="ij"), -1).reshape(-1, 2)
BOARD = np.stack(np.unravel_index(np.arange(4096), (64, 64)), -1)


def read_world(row):
    rng = np.random.default_rng([41853, int(row["episode"])])
    with np.load(ARCHIVE / row["file"]) as data:
        length = min(int(row["length"]), HORIZON)
        position = data["player_position"][:length].astype(int)
        bits = np.unpackbits(data["local_bits"][:length], axis=-1, count=1323)
        view = bits.reshape(length, 63, 21)[:, :, :17].argmax(-1)
    motion = np.diff(position, axis=0, prepend=position[:1])
    assert (abs(motion).sum(-1) <= 1).all()
    times = np.full(4096, -1, int)
    terrain = np.full(4096, -1, int)
    # Padding carries no supervision. Off-board tiles are present in the policy view,
    # but can never become probes.
    tiles = np.zeros((HORIZON, 63), np.uint8)
    tiles[:length] = view
    delta = np.zeros((HORIZON, 2), np.int8)
    delta[:length] = motion
    query = np.zeros((HORIZON // STRIDE, len(BANDS), 2, 2, 2), np.int8)
    valid = np.zeros(query.shape[:-2], bool)
    coords = position[:, None] + LOCAL
    inside = ((coords >= 0) & (coords < 64)).all(-1)
    cells = coords[..., 0] * 64 + coords[..., 1]
    for tick in range(STRIDE - 1, length, STRIDE):
        start = tick + 1 - STRIDE
        active = inside[start:tick + 1]
        latest = np.full(4096, -1)
        np.maximum.at(latest, cells[start:tick + 1][active], np.flatnonzero(active))
        updated = latest >= 0
        terrain[updated] = view[start:tick + 1].ravel()[latest[updated]]
        times[updated] = start + latest[updated] // 63
        offsets = BOARD - position[tick]
        distance = abs(offsets).sum(-1)
        offscreen = (abs(offsets) > (3, 4)).any(-1)
        age = tick - times
        for b, (lo, hi) in enumerate(BANDS):
            eligible = (times >= 0) & offscreen & (age >= lo) & (age <= hi)
            for r, code in enumerate(RESOURCES):
                pos = np.flatnonzero(eligible & (terrain == code))
                neg = np.flatnonzero(eligible & (terrain != code))
                if len(pos) and len(neg):
                    p = rng.choice(pos)
                    # Prefer equal Manhattan distance, then the closest available.
                    gap = abs(distance[neg] - distance[p])
                    n = rng.choice(neg[gap == gap.min()])
                    query[tick // STRIDE, b, r] = offsets[[p, n]]
                    valid[tick // STRIDE, b, r] = True
    return dict(tiles=tiles, motion=delta, query=query, valid=valid, length=np.int32(length))


def prepare(out, train=256, test=736):
    with (ARCHIVE / "episodes.csv").open() as f:
        rows = list(csv.DictReader(f))
    chosen = [rows[i] for i in np.random.default_rng(41853).permutation(len(rows))[:train + test]]
    assert len({r["initial_map_sha256"] for r in chosen}) == train + test
    out.mkdir(parents=True, exist_ok=True)
    split = {}
    for name, panel in (("train", chosen[:train]), ("test", chosen[train:])):
        with ThreadPoolExecutor(max_workers=4) as pool:
            items = list(pool.map(read_world, panel))
        arrays = {k: np.stack([x[k] for x in items]) for k in items[0]}
        np.savez(out / f"{name}.npz", **arrays)
        split[name] = [{k: r[k] for k in ("episode", "initial_map_sha256")} for r in panel]
        print(name, "eligible pairs by age and resource", arrays["valid"].sum((0, 1)).tolist(), flush=True)
    (out / "worlds.json").write_text(json.dumps(split, indent=2))


if __name__ == "__main__":
    import sys
    prepare(Path(sys.argv[1]))

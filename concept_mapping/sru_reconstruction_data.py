"""Build causal, agent-relative terrain queries from archived local GRU trajectories."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / 'concept_mapping' / 'runs'
ARCHIVE = RUNS / 'lava_only_gru_concat_dp2_10b_seed50/world_transitions_1000'
OUT = RUNS / 'sru_reconstruction_10p6b_pilot'
SEED = 31751
HORIZON = 1024
STRIDE = 32
UNKNOWN = 17
# Three uniform groups, followed by positive/negative pairs for each resource.
UNIFORM = 16
PAIRS = 4
RESOURCES = {'water': (3,), 'plant': (15, 16), 'diamond': (10,)}
N_QUERY = 3 * UNIFORM + len(RESOURCES) * 2 * PAIRS


def motion_inputs(positions):
    """Match ClassicMotionInputWrapper, including the zero reset displacement."""
    motion = np.zeros((HORIZON, 2), np.int8)
    motion[1:len(positions)] = np.diff(positions, axis=0)[:, ::-1]
    return motion


def make_queries(rng, position, last_time, remembered, truth, tick):
    age = tick - last_time
    seen = last_time >= 0
    old = seen & (age >= 100) & (age <= 1000)
    selections = []

    def sample(mask, count):
        cells = np.flatnonzero(mask)
        if not len(cells):
            return np.full(count, -1)
        return rng.choice(cells, count, replace=len(cells) < count)

    for mask in (seen & (age > 0) & (age < 100), old, ~seen):
        selections.extend(sample(mask, UNIFORM))
    for codes in RESOURCES.values():
        positive = np.isin(remembered, codes)
        for cell in sample(positive & old, PAIRS):
            if cell < 0:
                selections.extend((-1, -1))
                continue
            # Match distractor recency to the positive within the same world/time.
            low, high = next((a, b) for a, b in ((100, 249), (250, 499), (500, 1000))
                             if a <= age[cell] <= b)
            other = sample(seen & ~positive & (age >= low) & (age <= high), 1)[0]
            selections.extend((cell if other >= 0 else -1, other))
    cells = np.asarray(selections)
    valid = cells >= 0
    safe = np.maximum(cells, 0)
    offsets = np.stack(np.unravel_index(safe, (64, 64)), axis=-1) - position
    assert (np.abs(offsets[valid]) <= 63).all()
    return offsets, np.where(valid, remembered[safe], -1), np.where(valid, truth[safe], -1)


def read_prefix(path, rng):
    with np.load(path) as episode:
        length = min(len(episode['actions']), HORIZON)  # Exclude the terminal successor.
        bits = episode['local_bits'][:length]
        stats = episode['local_stats'][:length]
        actions = episode['actions'][:length]
        positions = episode['player_position'][:length]
        truth = episode['initial_map'].astype(np.int16).ravel().copy()
        change_steps, change_cells, change_tiles = (
            episode[key] for key in ('change_step', 'change_cell', 'change_tile'))
    tiles = np.unpackbits(bits, axis=-1, count=1323).reshape(length, 7, 9, 21)[..., :17].argmax(-1)
    ticks = HORIZON // STRIDE
    result = dict(bits=np.zeros((HORIZON, 166), np.uint8), stats=np.zeros((HORIZON, 22), np.float32),
                  previous=np.full(HORIZON, 17, np.uint8), motion=motion_inputs(positions),
                  length=np.int32(length),
                  offsets=np.zeros((ticks, N_QUERY, 2), np.int8),
                  labels=np.full((ticks, N_QUERY), -1, np.int8),
                  terrain=np.full((ticks, N_QUERY), -1, np.int8))
    result['bits'][:length] = bits
    result['stats'][:length] = stats
    result['previous'][1:length] = actions[:length-1]
    last_time = np.full(4096, -1, np.int32)
    memory = np.full(4096, UNKNOWN, np.int16)
    rr = positions[:, 0, None, None] + np.arange(-3, 4)[None, :, None]
    cc = positions[:, 1, None, None] + np.arange(-4, 5)[None, None, :]
    inside = (rr >= 0) & (rr < 64) & (cc >= 0) & (cc < 64)
    cells = 64 * rr + cc
    change_start = 0
    for start in range(0, length - STRIDE + 1, STRIDE):
        sl = slice(start, start + STRIDE)
        tick = sl.stop - 1
        # Repeated writes select the most recent observation of each tile.
        times = np.arange(STRIDE * 63).reshape(STRIDE, 7, 9)
        latest = np.full(4096, -1, np.int32)
        np.maximum.at(latest, cells[sl][inside[sl]], times[inside[sl]])
        written = latest >= 0
        memory[written] = tiles[sl].ravel()[latest[written]]
        last_time[written] = start + latest[written] // 63
        change_end = np.searchsorted(change_steps, tick, side='right')
        for i in range(change_start, change_end):
            truth[change_cells[i]] = change_tiles[i]
        change_start = change_end
        assert np.array_equal(truth[last_time == tick], memory[last_time == tick])
        values = make_queries(rng, positions[tick], last_time, memory, truth, tick)
        for name, value in zip(('offsets', 'labels', 'terrain'), values):
            result[name][start // STRIDE] = value
    return result


def prepare(out, train, test):
    with (ARCHIVE / 'episodes.csv').open() as f:
        rows = list(csv.DictReader(f))
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in rng.permutation(len(rows))[:train + test]]
    if len({r['initial_map_sha256'] for r in rows}) != train + test:
        raise ValueError('Expected independent worlds in the train/test split')
    out.mkdir(parents=True, exist_ok=False)
    splits = {}
    for name, split in [('train', rows[:train]), ('test', rows[train:])]:
        samples = [read_prefix(ARCHIVE / 'episodes' / f"episode_{int(r['episode']):04d}.npz",
                               np.random.default_rng([SEED, int(r['episode'])])) for r in split]
        np.savez(out / f'{name}.npz', **{k: np.stack([s[k] for s in samples]) for k in samples[0]})
        splits[name] = [{k:r[k] for k in ('episode','initial_map_sha256')} for r in split]
        print(f'Prepared {name}: {len(split)} worlds', flush=True)
    (out / 'split.json').write_text(json.dumps(splits, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--train', type=int, default=256)
    parser.add_argument('--test', type=int, default=128)
    args = parser.parse_args()
    prepare(args.out, args.train, args.test)

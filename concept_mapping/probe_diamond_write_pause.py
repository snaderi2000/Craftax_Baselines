"""Fork held-out diamond queries at their last visible frame and replay with writes paused."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from concept_mapping.spatial_transport_queries import BANDS, LOCAL, RESOURCES, STRIDE

DEFAULT = Path(__file__).resolve().parent / "runs/sru_spatial_transport_offline"
DIAMOND = RESOURCES.index(10)
TILE = RESOURCES[DIAMOND]
SEEDS = (32150, 32151)


def query_events(data):
    events = []
    for world, block, band in np.argwhere(data["valid"][:, :, 1:, DIAMOND]):
        band += 1
        end = (block + 1) * STRIDE - 1
        assert end < data["length"][world]
        path = data["motion"][world, :end + 1].astype(int).cumsum(0)
        offsets = data["query"][world, block, band, DIAMOND].astype(int)
        pair = path[end] + offsets
        in_view = (abs(pair[None] - path[:, None]) <= [3, 4]).all(-1)
        last = [int(np.flatnonzero(visible)[-1]) for visible in in_view.T]
        for i, tick in enumerate(last):
            r, c = pair[i] - path[tick]
            code = data["tiles"][world, tick, (r+3)*9+c+4]
            assert (code == TILE) == (i == 0)
            assert BANDS[band][0] <= end - tick <= BANDS[band][1]
        assert not in_view[last[0]+1:, 0].any()
        known_neg = np.flatnonzero(in_view[:last[0]+1, 1])
        neg_confirmed = False
        if known_neg.size:
            tick = known_neg[-1]
            r, c = pair[1] - path[tick]
            neg_confirmed = data["tiles"][world, tick, (r+3)*9+c+4] != TILE
        events.append((world, band, end, last[0], offsets, pair-path[last[0]], neg_confirmed))
    return events


def address(offset, frequency):
    angle = offset @ frequency
    return np.concatenate([np.cos(angle), np.sin(angle)], axis=-1) / np.sqrt(frequency.shape[1])


def rotate(memory, cos, sin):
    x, y = np.split(memory, 2, axis=-1)
    return np.concatenate([cos*x-sin*y, sin*x+cos*y], axis=-1)


def replay(data, events, fitted, arm):
    # The saved update is separable by terrain channel. The scalar write gate
    # still depends on all 63 observed terrain IDs, exactly as during fitting.
    direction = fitted["direction"].astype(np.float64)
    freq = (direction[0] - direction[1]) / 2
    if arm == "reversible":
        direction = np.stack([freq, -freq])
    local = address(LOCAL, freq)
    gram = local.T @ local
    norm = np.linalg.norm(gram) + 1e-6
    gain = np.logaddexp(0., fitted["gain"])
    worlds, bands, end, fork = np.array([e[:4] for e in events]).T
    at_end = address(np.stack([e[4] for e in events]), freq)
    at_fork = address(np.stack([e[5] for e in events]), freq)
    normal = np.zeros((len(events), local.shape[1]))
    paused = normal.copy()
    scores = np.full((len(events), 3, 2), np.nan)
    for t in range(end.max()+1):
        delta = data["motion"][worlds, t].astype(float)
        angles = -(delta.clip(min=0) @ direction[0] + (-delta).clip(min=0) @ direction[1])
        cos, sin = np.cos(angles), np.sin(angles)
        view = data["tiles"][worlds, t]
        weight = 1 / (1 + np.exp(-fitted["gate"][view].mean(-1)))
        moved = rotate(normal, cos, sin)
        error = (view == TILE) @ local - moved @ gram
        normal = moved + weight[:, None] * error / norm
        paused = np.where((t <= fork)[:, None], normal, rotate(paused, cos, sin))
        leaving = fork == t
        assert np.array_equal(paused[leaving], normal[leaving])
        scores[leaving, 0] = (at_fork[leaving]*normal[leaving, None]).sum(-1)*gain
        finished = end == t
        for index, memory in enumerate((normal, paused), 1):
            scores[finished, index] = (at_end[finished]*memory[finished, None]).sum(-1)*gain
    assert np.isfinite(scores).all()
    if arm == "reversible":
        # A pure sequence of transports preserves both world-coordinate scores.
        np.testing.assert_allclose(scores[:, 0], scores[:, 2], atol=1e-8)
    correct = (scores[:, :, 0] > scores[:, :, 1]).astype(float) + .5*(scores[:, :, 0] == scores[:, :, 1])
    for band in range(1, len(BANDS)):
        keep = bands == band
        observed = np.bincount(worlds[keep], weights=correct[keep, 1], minlength=len(data["length"]))
        np.testing.assert_array_equal(observed, fitted["counts"][:, band, DIAMOND, 0])
        np.testing.assert_array_equal(np.bincount(worlds[keep], minlength=len(observed)),
                                      fitted["counts"][:, band, DIAMOND, 1])
    return scores, correct


def paired_summary(rows, band, restricted):
    # Equal weights by fitting seed within query, then by world.
    selected = [r for r in rows if r["band"] == band and (not restricted or r["negative_known_at_fork"])]
    worlds = sorted({r["episode"] for r in selected})
    metrics = ["fork_correct","normal_correct","paused_correct","target_change","distractor_change"]
    by_world = np.array([[np.mean([r[m] for r in selected if r["episode"] == w]) for m in metrics] for w in worlds])
    fork, normal, paused, target, distractor = by_world.mean(0)
    diff = by_world[:, 2] - by_world[:, 1]
    rng = np.random.default_rng(73111)
    ci = np.quantile(diff[rng.integers(len(diff), size=(10000,len(diff)))].mean(1), [.025,.975])
    return dict(age_min=BANDS[band][0], age_max=BANDS[band][1], known_distractor_only=restricted,
                worlds=len(worlds), queries=len(selected)//len(SEEDS), fork=fork, normal=normal, paused=paused,
                gain=paused-normal, ci_low=ci[0], ci_high=ci[1], target_score_gain=target, distractor_score_gain=distractor)


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(base):
    out = base / "diamond_write_pause"
    out.mkdir(exist_ok=True)
    with np.load(base / "test.npz") as archive:
        data = dict(archive)
    panel = json.loads((base / "worlds.json").read_text())["test"]
    events = query_events(data)
    rows, summary = [], []
    for arm in ("free", "reversible"):
        for seed in SEEDS:
            with np.load(base / f"{arm}_{seed}.npz") as fitted:
                scores, correct = replay(data, events, fitted, arm)
            for e, s, c in zip(events, scores, correct):
                world, band, end, fork, offsets, _, known = e
                rows.append(dict(model=arm, seed=seed, episode=panel[world]["episode"], band=band,
                                 query_step=end, last_visible_step=fork, age=end-fork,
                                 diamond_row_offset=offsets[0,0], diamond_col_offset=offsets[0,1],
                                 distractor_row_offset=offsets[1,0], distractor_col_offset=offsets[1,1],
                                 negative_known_at_fork=bool(known), fork_correct=c[0], normal_correct=c[1],
                                 paused_correct=c[2], fork_margin=s[0,0]-s[0,1], normal_margin=s[1,0]-s[1,1],
                                 paused_margin=s[2,0]-s[2,1], target_change=s[2,0]-s[1,0], distractor_change=s[2,1]-s[1,1]))
        for band in range(1,len(BANDS)):
            for restricted in (False,True):
                summary.append(dict(model=arm, **paired_summary([r for r in rows if r["model"]==arm],band,restricted)))
    write_csv(out/"per_query.csv", rows)
    write_csv(out/"summary.csv", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT)
    main(parser.parse_args().base)

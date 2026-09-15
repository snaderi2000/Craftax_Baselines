"""Analyze observed depletion continuations; no new rollouts or policy changes."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / 'runs/master_symbolic_10b_5000episodes'
OUT = ROOT / 'depletion_return_analysis'
GAMMA = 0.99


def returns(rewards):
    result = np.zeros(len(rewards) + 1, dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        result[t] = rewards[t] + GAMMA * result[t + 1]
    return result


def summarize(rows, episode_count):
    if not rows:
        return {'stretches': 0}
    values = np.array([r['G_total'] for r in rows])
    sums = np.zeros(episode_count)
    counts = np.zeros(episode_count)
    for r in rows:
        sums[r['episode_id']] += r['G_total']
        counts[r['episode_id']] += 1
    rng = np.random.default_rng(50)
    boot = []
    for _ in range(100):
        sample = rng.integers(0, episode_count, size=(100, episode_count))
        denominator = counts[sample].sum(axis=1)
        boot.extend((sums[sample].sum(axis=1)[denominator > 0] /
                     denominator[denominator > 0]).tolist())
    return {
        'stretches': len(rows), 'distinct_episodes': int((counts > 0).sum()),
        'mean_total_return': float(values.mean()),
        'mean_health_return': float(np.mean([r['G_health'] for r in rows])),
        'mean_achievement_return': float(np.mean([r['G_achievement'] for r in rows])),
        'episode_bootstrap_mean_95ci': np.percentile(boot, [2.5, 97.5]).tolist(),
        'negative_count': int((values < -1e-8).sum()),
        'zero_count': int((np.abs(values) <= 1e-8).sum()),
        'positive_count': int((values > 1e-8).sum()),
        'negative_fraction': float((values < -1e-8).mean()),
        'zero_fraction': float((np.abs(values) <= 1e-8).mean()),
        'positive_fraction': float((values > 1e-8).mean()),
        'lava_ending_stretches': sum(r['ended_in_lava'] for r in rows),
    }


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    paths = sorted((ROOT / 'episodes').glob('episode_*.npz'))
    max_error = 0.0
    terminal_only_starts = 0
    for path in paths:
        with np.load(path) as d:
            eid = int(d['episode_id'])
            rewards = d['rewards'].astype(np.float64)
            n = len(rewards)
            terminal = d['terminal_observation']
            # Signed integers: terminal health can be negative after combat.
            stats = np.vstack([d['intrinsics'].astype(int),
                               np.rint(terminal[1335:1339] * 10).astype(int)])
            achievements = np.vstack([d['achievements'], d['terminal_achievements']]).astype(int)
            ar = np.diff(achievements.sum(axis=1)).astype(float)
            hr = np.diff(stats[:, 0]).astype(float) * 0.1
            error = float(np.max(np.abs(rewards - ar - hr)))
            max_error = max(max_error, error)
            assert error < 1e-6, (eid, error)
            gh, ga, gt = returns(hr), returns(ar), returns(rewards)
            future_achievements = np.cumsum(ar[::-1])[::-1]
            zero = (stats[:, 1:] == 0).any(axis=1)
            starts = np.flatnonzero(zero & ~np.r_[False, zero[:-1]])
            ends = np.flatnonzero(~zero & np.r_[False, zero[:-1]])
            lava = bool(d['dones'][-1] and terminal[:1323].reshape(7, 9, 21)[3, 4, :17].argmax() == 14)
            for start in starts:
                if start == n:
                    terminal_only_starts += 1
                later = ends[ends > start]
                end = int(later[0]) if len(later) else n
                recovered = bool(len(later) and end < n)
                rows.append(dict(
                    episode_id=eid, start_step=int(start), end_step=end,
                    health=int(stats[start, 0]), food=int(stats[start, 1]),
                    water=int(stats[start, 2]), energy=int(stats[start, 3]),
                    duration=end-int(start), recovered_before_termination=recovered,
                    positive_only_at_terminal=bool(len(later) and end == n),
                    starts_at_terminal=bool(start == n), ended_in_lava=lava,
                    episode_steps=n, G_health=float(gh[start]), G_achievement=float(ga[start]),
                    G_total=float(gt[start]),
                    no_subsequent_achievements=bool(start == n or future_achievements[start] == 0)))
    with (OUT / 'depletion_stretches.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    eligible = [r for r in rows if not r['starts_at_terminal']]
    late = [r for r in eligible if r['no_subsequent_achievements']]
    summary = {
        'gamma': GAMMA, 'episodes': len(paths), 'all_stretches': len(rows),
        'terminal_only_starts_excluded': terminal_only_starts,
        'max_reward_decomposition_error': max_error,
        'bootstrap': '10000 resamples of all 5000 whole episodes, seed 50; stretch-weighted mean; percentile 95% CI',
        'return_alignment': 'Start at first zero observation; reward[start] has exponent zero; include terminal transition. Terminal-only starts saved but excluded from continuation statistics.',
        'late_game_definition': 'No achievement reward from stretch start through terminal transition; no timestep cutoff.',
        'recovery_definition': 'All three resources positive strictly before terminal state.',
        'zero_return_tolerance': 1e-8,
        'late_game_all': summarize(late, len(paths)),
        'late_game_recovered': summarize([r for r in late if r['recovered_before_termination']], len(paths)),
        'late_game_not_recovered': summarize([r for r in late if not r['recovered_before_termination']], len(paths)),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

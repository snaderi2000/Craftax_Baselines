"""Collect matched frozen-policy temperature sweeps and plot their outcomes."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

from monitor_ppo_continuation import read_outcomes
from run_policy_randomness_sweep import label


def read(path):
    return json.loads(Path(path).read_text())


def interval(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return [100*max(0, center-half), 100*min(1, center+half)]


def collect(protocol, split):
    rows, vectors = [], {}
    for model in protocol['models']:
        for temperature in protocol['temperatures']:
            folder = Path(model['study']) / split / label(temperature)
            if not (folder / 'summary.json').exists():
                continue
            try:
                counts, episodes = read_outcomes(folder)
                raw = read(folder / 'summary.json')
            except (OSError, ValueError):
                continue
            assert counts['episodes'] == protocol['worlds_per_split']
            assert len(episodes) == protocol['worlds_per_split']
            row = dict(model=model['label'], temperature=temperature,
                episodes=counts['episodes'], all22_count=counts['all_22'],
                hit_10k_count=counts['reached_10000'], both_count=counts['both'],
                timeout_count=counts['survived_to_time_limit'], lava_deaths=counts['lava_deaths'],
                mean_achievements=raw['mean_achievements'], mean_length=raw['mean_length'],
                checkpoint=counts['checkpoint'], parameter_count=model['parameter_count'])
            for metric in ('all22', 'hit_10k', 'both'):
                lo, hi = interval(row[metric+'_count'], row['episodes'])
                row[metric+'_pct'] = 100*row[metric+'_count']/row['episodes']
                row[metric+'_ci_low'], row[metric+'_ci_high'] = lo, hi
            rows.append(row)
            vectors[(model['label'], temperature)] = episodes
    return rows, vectors


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def paired_differences(protocol, vectors):
    import numpy as np
    left, right = [m['label'] for m in protocol['models']]
    rows = []
    for temperature in protocol['temperatures']:
        a, b = vectors.get((left, temperature)), vectors.get((right, temperature))
        if a is None or b is None:
            continue
        assert a.keys() == b.keys(), 'Policies were evaluated on different worlds'
        keys = sorted(a)
        rng = np.random.default_rng(protocol['bootstrap_seed'])
        indices = rng.integers(0, len(keys), (protocol['bootstrap_samples'], len(keys)))
        row = dict(temperature=temperature, difference=f'{left} minus {right}')
        for metric, field in (('all22', 'all_22'), ('hit_10k', 'hit_limit'), ('both', None)):
            def value(x):
                return x[field] if field else x['all_22'] and x['hit_limit']
            delta = np.array([int(value(a[k]))-int(value(b[k])) for k in keys])
            bounds = 100*np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
            row[metric+'_difference_pp'] = float(100*delta.mean())
            row[metric+'_paired_ci_low'], row[metric+'_paired_ci_high'] = map(float, bounds)
        rows.append(row)
    return rows


def plot(root, protocol, rows):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    colors = ('#2563B9', '#0D8B6F')
    for axis, metric, title in zip(axes, ('all22', 'hit_10k', 'both'),
            ('All 22 achievements', 'Reach 10,000 steps', 'Both')):
        for model, color, marker in zip(protocol['models'], colors, ('o', 's')):
            values = sorted((r for r in rows if r['model'] == model['label']),
                            key=lambda r: r['temperature'])
            if not values:
                continue
            y = np.array([r[metric+'_pct'] for r in values])
            error = np.array([[r[metric+'_pct']-r[metric+'_ci_low'] for r in values],
                              [r[metric+'_ci_high']-r[metric+'_pct'] for r in values]])
            axis.errorbar([r['temperature'] for r in values], y, yerr=error,
                color=color, marker=marker, linewidth=2, markersize=5,
                elinewidth=1, capsize=3, label=model['label'])
        axis.set(title=title, xlabel='Action temperature', ylim=(0, 103), xlim=(-.04, 1.04))
        axis.set_xticks([0, .1, .25, .5, .75, 1.])
        axis.grid(axis='y', alpha=.25)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Episodes (%)')
    axes[0].legend(frameon=False, loc='lower right')
    fig.suptitle('10B checkpoints: GRU + concat versus map-MLP', fontsize=16, weight='bold')
    fig.text(.5, .018, '100 identical worlds per setting · one episode per world · 0 = greedy, 1 = normal sampling · 95% Wilson intervals',
             ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .055, 1, .93))
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(root / ('temperature_comparison.'+extension), dpi=180,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def summarize(root, protocol):
    rows, vectors = collect(protocol, 'validation')
    confirmed, _ = collect(protocol, 'confirmation')
    write_csv(root / 'temperature_comparison.csv', rows)
    write_csv(root / 'confirmation_comparison.csv', confirmed)
    differences = paired_differences(protocol, vectors)
    write_csv(root / 'paired_model_differences.csv', differences)
    best = {}
    for model in protocol['models']:
        path = Path(model['study']) / 'selected_temperature.json'
        if path.exists():
            best[model['label']] = read(path)
    states = {}
    for model in protocol['models']:
        path = Path(model['study']) / 'progress.json'
        states[model['label']] = read(path) if path.exists() else {'state': 'starting'}
    done = all(state.get('state') == 'complete' for state in states.values())
    report = dict(state='complete' if done else 'evaluating',
        validation_rows=rows, confirmation_rows=confirmed, best_temperatures=best,
        paired_model_differences=differences, model_states=states,
        episodes_per_setting=protocol['worlds_per_split'], attempts_per_world=1,
        note='Best temperatures are selected on validation worlds and independently confirmed on disjoint worlds.')
    (root / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    root = args.study.resolve();protocol = read(root / 'comparison_protocol.json')
    previous = -1
    while True:
        report = summarize(root, protocol)
        count = len(report['validation_rows'])
        if count != previous:
            plot(root, protocol, report['validation_rows'])
            print(f'Completed {count}/{2*len(protocol["temperatures"])} sweep settings', flush=True)
            previous = count
        if any(s.get('state') == 'failed' for s in report['model_states'].values()):
            raise RuntimeError('A model sweep failed; inspect its worker.log')
        if report['state'] == 'complete':
            print(json.dumps(report, indent=2), flush=True)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()

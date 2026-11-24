"""Summarize the four saved diamond value distributions over training.

Mean/median, population standard deviation and variance, and distribution
quantiles describe variation across the fixed probe states, not seed uncertainty.
No inference, training, or new rollouts are needed.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import numpy as np


DEFAULT_ROOT = Path('concept_mapping/runs/diamond_collectability_10b_10000episodes_all_visible_stone/four_comparisons_over_checkpoints')
BRANCHES = {
    'A': ('Diamond + iron pickaxe', '#2563eb', '-', 'o'),
    'B': ('Diamond, no iron pickaxe', '#d97706', '--', 's'),
    'C': ('No diamond, no iron pickaxe', '#0f766e', '-.', '^'),
    'D': ('No diamond + iron pickaxe', '#dc2626', ':', 'D'),
}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def save(fig, out, stem):
    for extension in ('pdf', 'png', 'jpeg'):
        fig.savefig(out / f'{stem}.{extension}', dpi=300)
    plt.close(fig)


def style(ax, xlabel):
    ax.set_xscale('log')
    ax.set_xlim(1e5, 1e10)
    ax.xaxis.set_major_locator(FixedLocator([10.**k for k in range(5, 11)]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Predicted value')
    ax.axhline(0, lw=.7, color='#6b7280', alpha=.5)
    ax.grid(alpha=.22)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontweight('bold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.results_dir
    out = root / 'value_distributions'
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / 'metadata.json').read_text())
    checkpoints = [int(r['checkpoint']) for r in read_csv(root / 'ordering_over_checkpoints.csv')]
    stats, first_ids = [], None
    for step in checkpoints:
        rows = read_csv(root / 'per_checkpoint' / f'{step}.csv')
        ids = [(r['state_index'], r['seed'], r['episode_id'], r['timestep']) for r in rows]
        if first_ids is None:
            first_ids = ids
        assert ids == first_ids, 'Probe identities or ordering changed between checkpoints'
        for key, (label, _, _, _) in BRANCHES.items():
            values = np.array([float(r['V_' + key]) for r in rows], dtype=np.float64)
            assert np.isfinite(values).all()
            q = np.quantile(values, [.1, .25, .5, .75, .9])
            stats.append(dict(checkpoint=step, branch=key, condition=label, n=len(values),
                              mean=float(values.mean()), std=float(values.std(ddof=0)),
                              variance=float(values.var(ddof=0)),
                              q10=float(q[0]), q25=float(q[1]), median=float(q[2]),
                              q75=float(q[3]), q90=float(q[4]),
                              minimum=float(values.min()), maximum=float(values.max()),
                              fraction_negative=float((values < 0).mean())))
    with (out / 'value_distribution_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold', 'mathtext.default': 'bf',
                         'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    frozen = bool(metadata.get('policy_frozen', False))
    xlabel = 'Critic-training environment steps' if frozen else 'Training environment steps'
    x = np.asarray(checkpoints)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for key, (label, color, line, marker) in BRANCHES.items():
        data = [r for r in stats if r['branch'] == key]
        ax.plot(x, [r['mean'] for r in data], color=color, ls=line, lw=1.8,
                marker=marker, ms=3, label=label)
    style(ax, xlabel)
    ax.set_title('Diamond: mean counterfactual values')
    ax.legend(loc='best', frameon=False, prop={'size': 8, 'weight': 'bold'})
    save(fig, out, 'mean_values_over_training')

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
    # Match the 2x2 state grids: rows vary diamond, columns vary pickaxe.
    for ax, key in zip(axes.flat, ('A', 'B', 'D', 'C')):
        label, color, _, _ = BRANCHES[key]
        data = [r for r in stats if r['branch'] == key]
        mean = np.asarray([r['mean'] for r in data])
        sd = np.asarray([r['std'] for r in data])
        ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=.18, linewidth=0,
                        label='Mean ±1 SD across states')
        ax.plot(x, mean, color=color, lw=1.9, marker='o', ms=3.2, label='Mean')
        ax.plot(x, [r['median'] for r in data], color=color, lw=1.3, ls='--', label='Median')
        style(ax, xlabel)
        ax.set_title(label)
        ax.legend(loc='best', frameon=False, prop={'size': 7.3, 'weight': 'bold'})
        ax.tick_params(labelbottom=True, labelleft=True)
    save(fig, out, 'value_distributions_over_training')
    final = [r for r in stats if r['checkpoint'] == checkpoints[-1]]
    details = dict(source_results=str(root), run=metadata['run'], policy_frozen=frozen,
                   checkpoint_count=len(checkpoints), n_states=len(first_ids),
                   fixed_probe_identities_verified=True, final=final,
                   statistic='Population variance/std (ddof=0) over the fixed set of states.',
                   bands='Mean ± one standard deviation across states. Not confidence intervals or training-seed variability.',
                   quantiles='10th, 25th, 50th, 75th, 90th percentiles saved in CSV.',
                   limitations=metadata.get('limitations'), no_new_training_or_rollouts=True)
    (out / 'metadata.json').write_text(json.dumps(details, indent=2) + '\n')
    text = ['# Diamond value distributions over training', '',
            f'Four counterfactual value distributions across {len(checkpoints)} checkpoints '
            f'and the same {len(first_ids):,} states. Policy frozen: {frozen}.', '',
            '[Four means on one graph](mean_values_over_training.pdf) and '
            '[four panels showing the distributions](value_distributions_over_training.pdf).', '',
            'Solid curves show means, dashed curves show medians, and shaded regions show '
            '**mean ± one standard deviation across states**. These are descriptive spread bands, '
            'not confidence intervals or variability across training seeds. The same state IDs are '
            'verified at every checkpoint. A band is not a bound on the values or a percentile interval.', '',
            'The CSV also saves variance, standard deviation, median, 10/25/75/90th percentiles, '
            'minimum/maximum, and the fraction of negative values.', '',
            '| Condition | Final mean | Final SD |', '| --- | ---: | ---: |']
    text.extend(f'| {r["condition"]} | {r["mean"]:.4f} | {r["std"]:.4f} |' for r in final)
    text.extend(['', 'Reproduce from the repository root:', '', '```bash',
                 f'python concept_mapping/plot_diamond_value_distributions.py --results-dir {root}', '```', ''])
    (out / 'README.md').write_text('\n'.join(text))
    print(json.dumps(dict(checkpoints=len(checkpoints), final=final), indent=2))


if __name__ == '__main__':
    main()

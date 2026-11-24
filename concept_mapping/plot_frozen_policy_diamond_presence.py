"""Plot pairwise diamond or iron-pickaxe valuation under a frozen policy.

Uses saved per-state values only; no inference, training, or new rollouts.
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


DEFAULT_ROOT = Path('concept_mapping/runs/cross_actor_critic_transfer_100b_10b/diamond_ordering')
CRITERIA = {
    'diamond-presence': dict(other='D', stem='diamond_presence_with_pickaxe',
        criterion='V(diamond, iron pickaxe) > V(no diamond, iron pickaxe)',
        label='Diamond > no diamond (pickaxe held)',
        intervention='All visible diamond tiles replaced with stone; factual iron pickaxe retained in both branches.',
        interpretation='Diamond-presence valuation with the tool held fixed; does not test pickaxe necessity. Stone is mineable.'),
    'pickaxe': dict(other='B', stem='pickaxe_presence_with_diamond',
        criterion='V(diamond, iron pickaxe) > V(diamond, no iron pickaxe)',
        label='Pickaxe > no pickaxe (diamond held)',
        intervention='Remove only the iron pickaxe inventory entry; all visible diamonds and other observation features retained in both branches.',
        interpretation='Pickaxe-retention valuation with diamonds held fixed; does not establish optimal valuation or isolate combat/environment interactions.'),
    'diamond-no-pickaxe': dict(left='B', other='C', stem='diamond_presence_without_pickaxe',
        criterion='V(diamond, no iron pickaxe) > V(no diamond, no iron pickaxe)',
        label='Diamond > no diamond (no pickaxe)',
        intervention='Iron pickaxe absent in both branches; replace all visible diamond tiles with stone in the no-diamond branch; all other features fixed.',
        interpretation='Diamond-presence valuation without an iron pickaxe. Other inventory is retained and replacement stone is mineable; positive values do not imply immediate diamond collectability.'),
}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--criterion', choices=CRITERIA, default='diamond-presence')
    args = parser.parse_args()
    comparison = CRITERIA[args.criterion]
    left = comparison.get('left', 'A')
    other = comparison['other']
    root = args.results_dir
    out = root / comparison['stem']
    out.mkdir(parents=True, exist_ok=True)
    original = read_csv(root / 'ordering_over_checkpoints.csv')
    rows = []
    for checkpoint in original:
        states = read_csv(root / 'per_checkpoint' / f"{checkpoint['checkpoint']}.csv")
        delta = np.array([float(r['V_' + left]) - float(r['V_' + other]) for r in states])
        assert len(delta) == int(checkpoint['n_states']) and np.isfinite(delta).all()
        count = int((delta > 0).sum())
        rate = 100 * count / len(delta)
        cached_column = f'{left}_gt_{other}_pct'
        if cached_column in checkpoint:
            np.testing.assert_allclose(rate, float(checkpoint[cached_column]), atol=1e-10)
        rows.append(dict(checkpoint=int(checkpoint['checkpoint']), n_states=len(delta),
                         success_count=count, success_pct=rate,
                         mean_delta=float(delta.mean()), median_delta=float(np.median(delta)),
                         margin_gt_1e5_pct=float(100 * (delta > 1e-5).mean())))
    with (out / 'ordering_over_checkpoints.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    behavior = read_csv(root / 'diamond_success_smoothed.csv')
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold', 'mathtext.default': 'bf',
                         'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    ax.plot([int(r['environment_steps']) for r in behavior],
            [float(r['success_smoothed_pct']) for r in behavior],
            color='#2563eb', lw=2, label='Frozen-policy collection (smoothed)')
    ax.plot([r['checkpoint'] for r in rows], [r['success_pct'] for r in rows],
            color='#dc2626', lw=1.8, marker='o', ms=3.8,
            label=comparison['label'])
    ax.set(xscale='log', xlim=(1e5, 1e10), ylim=(-2, 102), title='Diamond',
           xlabel='Critic-training environment steps', ylabel='Success rate (%)')
    ax.xaxis.set_major_locator(FixedLocator([10.**i for i in range(5, 11)]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=.22)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight('bold')
    ax.legend(loc='best', frameon=False, prop={'size': 7.2, 'weight': 'bold'})
    for suffix in ('pdf', 'png', 'jpeg'):
        fig.savefig(out / f"{comparison['stem']}_panel.{suffix}", dpi=300)
    plt.close(fig)
    metadata = dict(criterion=comparison['criterion'],
                    equivalent_saved_columns=f'V_{left} > V_{other}', source=str(root),
                    n_states=rows[0]['n_states'], checkpoints=len(rows),
                    intervention=comparison['intervention'],
                    first_100pct_checkpoint=next((r['checkpoint'] for r in rows if r['success_pct']==100), None),
                    final=rows[-1], values=f'Recomputed from saved per-state values; checked against {left}_gt_{other}_pct when available.',
                    behavior='Previously cached frozen-policy W&B history with centered 1024-update smoothing.',
                    interpretation=comparison['interpretation'])
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

"""Plot effective PPO learning rates before and after a checkpoint restart."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def rates(config, updates):
    batch = config['NUM_ENVS'] * config['NUM_STEPS']
    horizon = config['LR_HORIZON_TIMESTEPS'] // batch
    assert config['ANNEAL_LR'] and horizon > 0
    return config['LR'] * (1.0 - updates / horizon)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-config', type=Path, required=True)
    parser.add_argument('--continuation-study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.original_config.read_text())
    study = args.continuation_study
    continued = json.loads((study / 'map_mlp/inference_config.json').read_text())
    plan = json.loads((study / 'continuation_plan.json').read_text())
    progress = json.loads((study / 'map_mlp/progress.json').read_text())
    batch = original['NUM_ENVS'] * original['NUM_STEPS']
    assert batch == continued['NUM_ENVS'] * continued['NUM_STEPS']
    restart = plan['source_steps']
    end = progress['actual_training_steps']
    assert restart % batch == end % batch == 0
    u1 = np.arange(restart // batch + 1, dtype=np.int64)
    u2 = np.arange(restart // batch, end // batch + 1, dtype=np.int64)
    x1, x2 = u1 * batch / 1e9, u2 * batch / 1e9
    y1, y2 = rates(original, u1), rates(continued, u2)
    assert np.isclose(y1[0], 2e-4) and np.isclose(y1[-1], 0)
    assert np.isclose(y2[0], plan['effective_learning_rate_at_restart'])
    assert np.isclose(y2[-1], plan['effective_learning_rate_at_target'])
    args.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 18,
                         'axes.labelsize': 12, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(11.5, 6.3), constrained_layout=True)
    blue, orange = '#2563B9', '#D86518'
    ax.axvspan(0, x1[-1], color=blue, alpha=.035, lw=0)
    ax.axvspan(x2[0], x2[-1], color=orange, alpha=.065, lw=0)
    ax.plot(x1, y1, color=blue, lw=3, label='Original training: 0–10B steps')
    ax.plot(x2, y2, color=orange, lw=3, label='Continuation: 10–12B steps')
    ax.axvline(x2[0], color='#94A3B8', lw=1.2, ls=(0, (4, 4)))
    ax.scatter([0, x1[-1]], [y1[0], y1[-1]], color=blue, s=43, zorder=5)
    ax.scatter([x2[0], x2[-1]], [y2[0], y2[-1]], color=orange, s=43, zorder=6)
    ax.annotate('', xy=(x2[0], y2[0]), xytext=(x2[0], y1[-1]),
                arrowprops=dict(arrowstyle='->', color=orange, lw=1.8, linestyle='--'))
    ax.annotate('Start: 2 × 10⁻⁴', xy=(0, y1[0]), xytext=(15, -24),
                textcoords='offset points', color=blue, fontsize=11)
    ax.annotate('LR restart\n~0 → 5 × 10⁻⁵', xy=(x2[0], y2[0]), xytext=(-20, 40),
                textcoords='offset points', ha='center', color=orange,
                arrowprops=dict(arrowstyle='-', color=orange, lw=1), fontsize=11)
    ax.annotate('At 12B: 4 × 10⁻⁵', xy=(x2[-1], y2[-1]), xytext=(-15, -35),
                textcoords='offset points', ha='right', color=orange, fontsize=11)
    ax.set_title('Learning-rate schedule used', loc='left', pad=24, weight='bold')
    ax.text(0, 1.025, 'Map-MLP · original annealing and the subsequent 2B-step continuation',
            transform=ax.transAxes, color='#475569', fontsize=10)
    ax.set_xlabel('Total environment steps (billions)')
    ax.set_ylabel('Effective learning rate (×10⁻⁴)')
    ax.set_xlim(-.15, 12.5)
    ax.set_ylim(-7e-6, 2.22e-4)
    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_yticks(np.arange(0, 2.01e-4, .5e-4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1e-4:.1f}'))
    ax.grid(axis='y', color='#CBD5E1', lw=.7, alpha=.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#94A3B8')
    ax.legend(loc='upper right', frameon=False)
    fig.text(.5, -.015, 'Reconstructed from saved training configurations and preserved optimizer counters; '
             'values shown at PPO-update boundaries.', ha='center', fontsize=9, color='#64748B')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(args.out / f'learning_rate_schedule.{extension}', dpi=200,
                    facecolor='white', bbox_inches='tight')
    plt.close(fig)

    with (args.out / 'learning_rate_schedule.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['phase', 'environment_steps', 'learning_rate'])
        for phase, updates, values in (('original', u1, y1), ('continuation', u2, y2)):
            writer.writerows((phase, int(update) * batch, float(value))
                             for update, value in zip(updates, values))
    metadata = dict(original_config=str(args.original_config.resolve()),
        continuation_study=str(study.resolve()), restart_steps=restart, end_steps=end,
        original_initial_learning_rate=float(y1[0]), original_rate_at_restart=float(y1[-1]),
        continuation_initial_learning_rate=float(y2[0]), continuation_final_learning_rate=float(y2[-1]),
        optimizer_reset=False, original_horizon=original['LR_HORIZON_TIMESTEPS'],
        continuation_horizon=continued['LR_HORIZON_TIMESTEPS'])
    (args.out / 'schedule_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    print(args.out / 'learning_rate_schedule.png')


if __name__ == '__main__':
    main()

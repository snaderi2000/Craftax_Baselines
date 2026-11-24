"""Evaluate four diamond orderings across ordinary, jointly trained PPO checkpoints.

Uses the existing fixed counterfactual bank and cached training success curve.
No new training or rollouts. Saves per-state values and matching paper plots.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concept_mapping.plot_collectability_emergence import (
    BANKS, DIAMOND_BEHAVIOR, checkpoints, make_branches, read_rows,
    _init_network, _load_wandb_config, _restore_train_state, make_craftax_env_from_name,
)
from concept_mapping.probe_frozen_policy_collectability import component_arrays, fingerprint, write_csv
from concept_mapping.plot_frozen_policy_diamond_presence import CRITERIA
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext


def draw(ax, rows, behavior, key, label):
    ax.plot([int(r['environment_steps']) for r in behavior],
            [float(r['diamond_success_smoothed_pct']) for r in behavior],
            color='#2563eb', lw=2, label='Diamond collection (smoothed)')
    ax.plot([r['checkpoint'] for r in rows], [r[key + '_pct'] for r in rows],
            color='#dc2626', lw=1.8, marker='o', ms=3.8, label=label)
    ax.set(xscale='log', xlim=(1e5, 1e10), ylim=(-2, 102),
           xlabel='Training environment steps', ylabel='Success rate (%)', title='Diamond')
    ax.xaxis.set_major_locator(FixedLocator([10.**k for k in range(5, 11)]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=.22)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontweight('bold')
    ax.legend(loc='best', frameon=False, prop={'size': 7.2, 'weight': 'bold'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('wandb/run-20260630_214658-m0mw4end'))
    parser.add_argument('--out', type=Path, default=BANKS['diamond'] / 'four_comparisons_over_checkpoints')
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = _load_wandb_config(str(args.run))
    assert cfg.get('TRAIN_COMPONENT', 'both') == 'both'
    assert not cfg.get('COMPONENT_INIT_FROM')
    assert cfg.get('DEATH_PENALTY', 0) == 0
    assert cfg['ENV_NAME'] == 'Craftax-Classic-Symbolic-v1'
    assert cfg.get('ACTIVATION', 'tanh') == 'tanh'
    bank = BANKS['diamond']
    branches = make_branches('diamond', bank)
    reference = read_rows(bank / 'per_state_results.csv')
    n = len(reference)
    assert n == 3483
    assert len({(r['seed'], r['episode_id']) for r in reference}) == n
    selected = checkpoints(args.run)
    prior_rows = {int(r['checkpoint']): r for r in read_rows(bank / 'emergence_over_checkpoints/ordering_over_checkpoints.csv')}
    behavior = read_rows(DIAMOND_BEHAVIOR)
    assert int(behavior[-1]['environment_steps']) >= max(selected) - 65536
    env = make_craftax_env_from_name(cfg['ENV_NAME'], auto_reset=False)
    net = _init_network(cfg, env, env.default_params)
    first, _ = _restore_train_state(str(args.run), cfg, net, env, env.default_params, selected[0])
    manager = ocp.CheckpointManager(str(args.run / 'files/policies'), ocp.PyTreeCheckpointer())

    @jax.jit
    def evaluate_batch(params, obs):
        return net.apply(params, obs)[1]

    def evaluate(params, obs):
        values = []
        for start in range(0, len(obs), args.batch_size):
            chunk = obs[start:start + args.batch_size]
            # Match the original sweep's unpadded final batch; changing its
            # shape can move floating-point ties across a strict threshold.
            values.append(np.asarray(evaluate_batch(params, jnp.asarray(chunk))))
        result = np.concatenate(values).astype(np.float64)
        assert np.isfinite(result).all()
        return result

    comparisons = [('full_ordering', 'Full critic ordering', 'A > max(B,C,D)')]
    for name in ('diamond-presence', 'pickaxe', 'diamond-no-pickaxe'):
        c = CRITERIA[name]
        comparisons.append((c['stem'], c['label'], c['criterion']))
    rows, validation, final_error, curve_checks = [], [], {}, []
    per_state_dir = args.out / 'per_checkpoint'
    per_state_dir.mkdir(exist_ok=True)
    for step in selected:
        print(f'Joint-training checkpoint {step:,} ({len(rows)+1}/{len(selected)})', flush=True)
        state = first if step == selected[0] else manager.restore(step, items=first)
        v = {key: evaluate(state.params, obs) for key, obs in branches.items()}
        deltas = {'full_ordering': v['A'] - np.maximum.reduce([v['B'], v['C'], v['D']]),
                  'diamond_presence_with_pickaxe': v['A'] - v['D'],
                  'pickaxe_presence_with_diamond': v['A'] - v['B'],
                  'diamond_presence_without_pickaxe': v['B'] - v['C']}
        row = dict(checkpoint=step, n_states=n)
        for key, delta in deltas.items():
            row[key + '_count'] = int((delta > 0).sum())
            row[key + '_pct'] = 100 * float((delta > 0).mean())
            row[key + '_mean_delta'] = float(delta.mean())
            row[key + '_median_delta'] = float(np.median(delta))
            row[key + '_margin_gt_1e5_pct'] = 100 * float((delta > 1e-5).mean())
        if step in prior_rows:
            prior_count = int(prior_rows[step]['success_count'])
            near_ties = int((np.abs(deltas['full_ordering']) <= 1e-5).sum())
            count_difference = row['full_ordering_count'] - prior_count
            # Strict comparisons can flip at float32 ties across GPU/library
            # versions. Record discrepancies and require them to fit inside
            # the near-zero-margin population rather than silently overwriting.
            assert abs(count_difference) <= near_ties
            curve_checks.append(dict(checkpoint=step, previous_success_count=prior_count,
                                     new_success_count=row['full_ordering_count'],
                                     count_difference=count_difference,
                                     near_zero_margin_count=near_ties))
        if step == 9999941632:
            for key in 'ABCD':
                expected = np.array([float(r['V_' + key]) for r in reference])
                final_error[key] = float(np.max(np.abs(v[key] - expected)))
                np.testing.assert_allclose(v[key], expected, atol=1e-5, rtol=1e-5)
            assert row['full_ordering_count'] == 3330
        records = [dict(state_index=i, seed=r['seed'], episode_id=r['episode_id'], timestep=r['timestep'],
                        **{f'V_{key}': float(v[key][i]) for key in 'ABCD'},
                        **{key + '_success': bool(delta[i] > 0) for key, delta in deltas.items()})
                   for i, r in enumerate(reference)]
        write_csv(per_state_dir / f'{step}.csv', records)
        rows.append(row)
        write_csv(args.out / 'ordering_over_checkpoints.csv', rows)
        validation.append(dict(checkpoint=step, actor_sha256=fingerprint(component_arrays(state.params, range(4))),
                               critic_sha256=fingerprint(component_arrays(state.params, range(4, 8)))))
        print({k: round(row[k + '_pct'], 3) for k in deltas}, flush=True)
    assert len({r['actor_sha256'] for r in validation}) == len(selected)
    assert len({r['critic_sha256'] for r in validation}) == len(selected)
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                         'mathtext.default': 'bf', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    for key, label, _ in comparisons:
        fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
        draw(ax, rows, behavior, key, label)
        for ext in ('pdf', 'png', 'jpeg'):
            fig.savefig(args.out / f'{key}_panel.{ext}', dpi=300)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (key, label, _) in zip(axes.flat, comparisons):
        draw(ax, rows, behavior, key, label)
    for ext in ('pdf', 'png', 'jpeg'):
        fig.savefig(args.out / f'four_comparisons.{ext}', dpi=300)
    plt.close(fig)
    metadata = dict(run=str(args.run), wandb_run='synaderi-uc-davis/Craftax_Baselines/' + args.run.name.rsplit('-', 1)[-1],
                    train_component='both', policy_frozen=False, seed=int(cfg['SEED']),
                    n_states=n, bank=str(bank), checkpoints=selected,
                    comparisons={key: text for key, _, text in comparisons},
                    branch_definitions=dict(A='diamond + iron pickaxe', B='diamond, no iron pickaxe',
                                            C='no diamond, no iron pickaxe', D='no diamond + iron pickaxe'),
                    behavior_csv=str(DIAMOND_BEHAVIOR), behavior_smoothing='Centered 1024-update rolling mean of W&B Achievements/collect_diamond.',
                    prior_full_ordering_checkpoints_verified=len(set(selected) & set(prior_rows)),
                    prior_full_ordering_curve_checks=curve_checks,
                    final_original_values_max_abs_error=final_error,
                    checkpoint_fingerprints=validation, final=rows[-1],
                    limitations='One training seed, fixed states selected by the final 10B policy. Counterfactuals may be out of distribution. No-diamond branches replace all visible diamonds with mineable stone. Descriptive orderings, not proof of optimal or causal valuation.',
                    frozen_comparison_note='The critic-only study used a pretrained frozen 100B actor; differences are not a matched intervention isolating actor updates.')
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    lines = ['# Diamond comparisons during ordinary PPO training', '',
             f'Seed {cfg["SEED"]}; actor and critic both learned from scratch on original Craftax rewards. '
             f'Evaluated {len(selected)} saved checkpoints on the same {n:,} fixed states; no new training or rollouts.', '',
             'A = diamond + iron pickaxe; B = diamond without iron pickaxe; '
             'C = no diamond, no iron pickaxe; D = no diamond + iron pickaxe. '
             'All visible diamonds are replaced with stone in no-diamond branches. Other inventory remains factual.', '',
             '| Comparison | Final success | Plot |', '| --- | ---: | --- |']
    for key, _, text in comparisons:
        lines.append(f'| {text} | {rows[-1][key + "_pct"]:.2f}% | [PDF]({key}_panel.pdf) |')
    lines.extend(['', '[All four plots](four_comparisons.pdf). Blue: smoothed ordinary-policy diamond collection. '
                  'Red: the indicated comparison. X-axis: ordinary joint-training interactions.', '',
                  f'Validation checked the prior full-ordering curve at {metadata["prior_full_ordering_checkpoints_verified"]} checkpoints '
                  f'(maximum count discrepancy {max(abs(c["count_difference"]) for c in curve_checks)} states; discrepancies were confined to the near-zero-margin count envelope). '
                  f'All four final-state value arrays matched their saved references (max errors {final_error}). Both actor and critic fingerprints changed at every saved checkpoint. '
                  'Strict comparisons and a 1e-5 margin sensitivity are saved for every comparison. See metadata for the individual numerical checks.', '',
                  metadata['limitations'], '', metadata['frozen_comparison_note'], '',
                  'Reproduce from the repository root in `jax-v100-cuda12`:', '',
                  '```bash', 'XLA_PYTHON_CLIENT_PREALLOCATE=false python concept_mapping/analyze_joint_policy_diamond_comparisons.py', '```', ''])
    (args.out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(checkpoints=len(selected), final=rows[-1]), indent=2))


if __name__ == '__main__':
    main()

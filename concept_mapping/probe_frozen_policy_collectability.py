"""Probe collectability as a fresh critic learns under an unchanged actor.

Reuses the fixed, validated counterfactual banks and ordering criteria from
plot_collectability_emergence.py. Does not train or collect new rollouts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concept_mapping.plot_collectability_emergence import (
    BANKS, checkpoints, make_branches, read_rows, _init_network,
    _load_wandb_config, _restore_train_state, make_craftax_env_from_name,
)
from concept_mapping.plot_smoothed_diamond_success import (
    _read_local_history, _read_wandb_history, _centered_rolling_mean,
)
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext


DEFAULT_RUN = Path('wandb/run-20261006_044214-d6nkqz6d')
DEFAULT_OUT = Path('concept_mapping/runs/cross_actor_critic_transfer_100b_10b/diamond_ordering')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def component_arrays(params, indices):
    return {f'Dense_{i}/{leaf}': np.asarray(value)
            for i in indices for leaf, value in params['params'][f'Dense_{i}'].items()}


def fingerprint(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--checkpoints', help='Optional comma-separated subset (smoke checks).')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--window-updates', type=int, default=1024)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = _load_wandb_config(str(args.run))
    assert cfg['TRAIN_COMPONENT'] == 'critic'
    assert cfg['ENV_NAME'] == 'Craftax-Classic-Symbolic-v1'
    assert cfg.get('ACTIVATION', 'tanh') == 'tanh'
    bank = BANKS['diamond']
    branches = make_branches('diamond', bank)  # Includes exact edit checks.
    reference = read_rows(bank / 'per_state_results.csv')
    n = len(reference)
    assert n == len(branches['A'])
    selected = ([int(x) for x in args.checkpoints.split(',')]
                if args.checkpoints else checkpoints(args.run))
    env = make_craftax_env_from_name(cfg['ENV_NAME'], auto_reset=False)
    net = _init_network(cfg, env, env.default_params)
    source_folder = Path(cfg['COMPONENT_INIT_FROM'])
    if source_folder.name != 'policies':
        raise ValueError('Expected component source policies directory')
    source_run = source_folder.parent.parent
    source_step = int(cfg['COMPONENT_INIT_STEP'])
    source_cfg = _load_wandb_config(str(source_run))
    source_state, _ = _restore_train_state(
        str(source_run), source_cfg, net, env, env.default_params, source_step)
    source_actor = component_arrays(source_state.params, range(4))
    source_actor_hash = fingerprint(source_actor)
    manager = ocp.CheckpointManager(str(args.run / 'files/policies'), ocp.PyTreeCheckpointer())

    @jax.jit
    def values(params, obs):
        return net.apply(params, obs)[1]

    def evaluate(params, observations):
        batches = []
        for start in range(0, len(observations), args.batch_size):
            chunk = observations[start:start + args.batch_size]
            padded = np.pad(chunk, ((0, args.batch_size - len(chunk)), (0, 0)), mode='edge')
            batches.append(np.asarray(values(params, jnp.asarray(padded)))[:len(chunk)])
        result = np.concatenate(batches).astype(np.float64)
        assert np.isfinite(result).all()
        return result

    # Independently reproduce the original bank's values, confirming that the
    # shared branch builder and present-day network still match the old probe.
    original_run = 'wandb/run-20260630_214658-m0mw4end'
    original_cfg = _load_wandb_config(original_run)
    original, _ = _restore_train_state(original_run, original_cfg, net, env, env.default_params, 9999941632)
    reproduction = {}
    for key, obs in branches.items():
        actual = evaluate(original.params, obs)
        expected = np.array([float(row['V_' + key]) for row in reference])
        reproduction[key] = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    del original

    rows, validation = [], []
    first_critic = None
    for index, step in enumerate(selected):
        print(f'Checkpoint {step:,} ({index+1}/{len(selected)})', flush=True)
        state = manager.restore(step, items=source_state)
        actor = component_arrays(state.params, range(4))
        actor_hash = fingerprint(actor)
        max_diff = max(float(np.max(np.abs(actor[key] - source_actor[key]))) for key in actor)
        assert actor_hash == source_actor_hash and max_diff == 0, 'Frozen actor changed'
        critic = component_arrays(state.params, range(4, 8))
        if first_critic is None:
            first_critic = critic
        critic_change = max(float(np.max(np.abs(critic[key] - first_critic[key]))) for key in critic)
        result = {key: evaluate(state.params, obs) for key, obs in branches.items()}
        margin = result['A'] - np.maximum.reduce([result['B'], result['C'], result['D']])
        success = margin > 0
        row = dict(checkpoint=step, n_states=n, success_count=int(success.sum()),
                   success_pct=float(100 * success.mean()), mean_margin=float(margin.mean()),
                   median_margin=float(np.median(margin)),
                   margin_gt_1e5_pct=float(100 * (margin > 1e-5).mean()))
        for key in 'BCD':
            row[f'A_gt_{key}_pct'] = float(100 * (result['A'] > result[key]).mean())
        rows.append(row)
        validation.append(dict(checkpoint=step, actor_sha256=actor_hash,
                               actor_max_abs_diff_from_source=max_diff,
                               critic_sha256=fingerprint(critic),
                               critic_max_abs_change_from_first=critic_change))
        records = []
        for i, ref in enumerate(reference):
            records.append(dict(state_index=i, **{k: ref[k] for k in ('seed', 'episode_id', 'timestep') if k in ref},
                                **{f'V_{key}': float(result[key][i]) for key in 'ABCD'},
                                ordering_margin=float(margin[i]), success=bool(success[i])))
        per_state = args.out / 'per_checkpoint'
        per_state.mkdir(exist_ok=True)
        write_csv(per_state / f'{step}.csv', records)
        write_csv(args.out / 'ordering_over_checkpoints.csv', rows)
        print(f"  success={row['success_pct']:.3f}%; actor max diff={max_diff}; critic change={critic_change:.6g}", flush=True)
    if len(selected) > 1:
        assert validation[-1]['critic_max_abs_change_from_first'] > 0

    history_path = args.out / 'diamond_success_smoothed.csv'
    run_ref = 'synaderi-uc-davis/Craftax_Baselines/' + args.run.name.rsplit('-', 1)[-1]
    history_source = 'local W&B archive'
    if history_path.exists():
        history = read_rows(history_path)
        update = np.array([int(r['wandb_update']) for r in history])
        raw = np.array([float(r['success_raw_pct']) for r in history])
        history_source = 'cached W&B history'
    else:
        history = _read_local_history(str(args.run))
        update, raw = history[:, 0].astype(np.int64), history[:, 1]
    stride = int(cfg['NUM_ENVS']) * int(cfg['NUM_STEPS'])
    if (int(update[-1]) + 1) * stride < max(checkpoints(args.run)) - stride:
        print('Local history incomplete; reading complete synced W&B history.', flush=True)
        history = _read_wandb_history(run_ref)
        update, raw = history[:, 0].astype(np.int64), history[:, 1]
        history_source = 'synced W&B history'
    assert (int(update[-1]) + 1) * stride >= max(checkpoints(args.run)) - stride
    assert np.isfinite(raw).all() and raw.min() >= -1e-3 and raw.max() <= 100.001
    raw = np.clip(raw, 0, 100)
    smooth = _centered_rolling_mean(raw, min(args.window_updates, len(raw)))
    x = (update + 1) * stride
    write_csv(history_path, [dict(wandb_update=int(u), environment_steps=int(s),
                                 success_raw_pct=float(r), success_smoothed_pct=float(m))
                             for u, s, r, m in zip(update, x, raw, smooth)])
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    ax.plot(x, smooth, color='#2563eb', lw=2, label='Frozen-policy collection (smoothed)')
    ax.plot([r['checkpoint'] for r in rows], [r['success_pct'] for r in rows],
            color='#dc2626', lw=1.8, marker='o', ms=3.8, label='Critic ordering')
    ax.set(xscale='log', xlim=(1e5, 1e10), ylim=(-2, 102),
           xlabel='Critic-training environment steps', ylabel='Success rate (%)', title='Diamond')
    ax.xaxis.set_major_locator(FixedLocator([10.**i for i in range(5, 11)]))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=.22)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight('bold')
    ax.legend(loc='best', frameon=False, prop={'size': 7.4, 'weight': 'bold'})
    for suffix in ('pdf', 'png', 'jpeg'):
        fig.savefig(args.out / f'diamond_frozen_policy_emergence_panel.{suffix}', dpi=300)
    plt.close(fig)
    metadata = dict(run=str(args.run), wandb_run=run_ref, source_policy_run=str(source_run),
                    source_policy_step=source_step, policy_frozen=True, train_component='critic',
                    critic_initialized_from_scratch=True, training_seed=int(cfg['SEED']),
                    state_bank=str(bank), n_states=n, checkpoints=selected,
                    criterion='A > max(B,C,D)', branches=dict(A='diamond + iron pickaxe',
                    B='diamond, no iron pickaxe', C='no diamond, no iron pickaxe', D='no diamond + iron pickaxe'),
                    intervention='Remove all visible diamond tiles by replacing with stone; remove only iron pickaxe inventory entry.',
                    actor_source_sha256=source_actor_hash, checkpoint_validation=validation,
                    original_bank_reproduction_max_abs_error=reproduction,
                    history_source=history_source, history_first_step=int(x[0]), history_last_step=int(x[-1]),
                    behavior_smoothing_updates=args.window_updates,
                    final_tail_1024_updates_diamond_pct=float(raw[-1024:].mean()),
                    final_ordering=rows[-1],
                    interpretation='Critic learning under a fixed policy and stationary task, not improved policy behavior. Blue fluctuations are sampling variation.',
                    limitations='One trained critic seed; fixed probe states selected with the separate 10B policy m0mw4end, not this 100B actor. Stone is itself mineable. No new training or rollouts.')
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(final=rows[-1], actor_frozen_at_every_checkpoint=True,
                          behavior_tail_pct=metadata['final_tail_1024_updates_diamond_pct']), indent=2))


if __name__ == '__main__':
    main()

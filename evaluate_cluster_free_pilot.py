"""Frozen normal-env evaluation with policy-independent per-episode world seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement
from models.actor_critic import ActorCritic
from concept_mapping.compare_transport_achievement_groups import save_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=int, required=True)
    p.add_argument('--episodes', type=int, default=512)
    p.add_argument('--seed', type=int, default=6050)
    args = p.parse_args()
    if (args.out/'evaluation.json').exists():
        raise FileExistsError('Refusing to overwrite completed evaluation')
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
    params = env.default_params
    n = 32
    if args.episodes % n:
        raise ValueError('Episode count must be divisible by 32')
    quota = args.episodes//n
    uids = jnp.arange(n, dtype=jnp.int32)*quota
    root = jax.random.PRNGKey(args.seed)
    world_root, action_root, dynamics_root = jax.random.split(root, 3)
    def world_keys(uids):
        return jax.vmap(lambda uid: jax.random.fold_in(world_root, uid))(uids)
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    obs, state = reset(world_keys(uids), params)
    manager = CheckpointManager(str(Path(args.run)/'files/policies'), PyTreeCheckpointer(), CheckpointManagerOptions(create=False))
    weights = manager.restore(args.checkpoint)['params']
    net = ActorCritic(17, 512)
    def transition(carry, unused):
        obs, state, uids, returns, lengths = carry
        def keys(source):
            return jax.vmap(lambda uid, t: jax.random.fold_in(jax.random.fold_in(source, uid), t))(uids, state.timestep)
        pi, _ = net.apply(weights, obs)
        actions = jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(keys(action_root), pi.logits)
        after, successor, reward, done, _ = step(keys(dynamics_root), state, actions, params)
        returns, lengths = returns+reward, lengths+1
        records = (done, uids, returns, lengths, successor.achievements)
        next_uids = uids+done.astype(jnp.int32)
        robs, rstate = reset(world_keys(next_uids), params)
        raw = jax.tree_util.tree_map(lambda a, b: jnp.where(done.reshape((n,)+(1,)*(a.ndim-1)), a, b), rstate, successor)
        return (jnp.where(done[:, None], robs, after), raw, next_uids,
                jnp.where(done, 0., returns), jnp.where(done, 0, lengths)), records
    block = jax.jit(lambda c: jax.lax.scan(transition, c, None, length=64))
    carry = (obs, state, uids, jnp.zeros(n), jnp.zeros(n, jnp.int32))
    completed = np.zeros(n, int)
    rows = []
    names = [a.name.lower() for a in Achievement]
    while len(rows) < args.episodes:
        carry, records = block(carry)
        done, uids, returns, lengths, flags = jax.tree_util.tree_map(np.asarray, records)
        for t, worker in np.argwhere(done):
            if completed[worker] >= quota:
                continue
            row = dict(episode_uid=int(uids[t, worker]), return_=float(returns[t, worker]), length=int(lengths[t, worker]))
            row.update({name: int(flags[t, worker, i]) for i, name in enumerate(names)})
            rows.append(row)
            completed[worker] += 1
    rows.sort(key=lambda r: r['episode_uid'])
    assert [r['episode_uid'] for r in rows] == list(range(args.episodes))
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out/'evaluation_episodes.csv', rows)
    values = np.array([r['return_'] for r in rows])
    draws = np.random.default_rng(args.seed+100).integers(0, len(rows), (2000, len(rows)))
    summary = dict(episodes=len(rows), checkpoint=args.checkpoint, evaluation_seed=args.seed,
        mean_return=float(values.mean()), median_return=float(np.median(values)),
        mean_return_95_ci=np.percentile(values[draws].mean(1), [2.5, 97.5]).tolist(),
        mean_length=float(np.mean([r['length'] for r in rows])), median_length=float(np.median([r['length'] for r in rows])),
        capped_episodes=sum(r['length'] >= params.max_timesteps for r in rows),
        achievement_rates={name: float(np.mean([r[name] for r in rows])) for name in names},
        reset='normal; per-episode initial-world key depends only on evaluation seed and episode UID',
        stochastic_policy=True)
    (args.out/'evaluation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    # These evaluators are launched only after training. Refresh comparisons
    # automatically; a lock prevents simultaneous workers truncating tables.
    experiment = args.out.parent.parent
    if args.episodes == 512 and experiment.name == 'cluster_free_practice_10m':
        import fcntl
        with (experiment/'comparison.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run([sys.executable, str(Path(__file__).with_name('compare_cluster_free_pilot.py')),
                            '--verify-warmup'], check=True,
                env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': ''},
                stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()

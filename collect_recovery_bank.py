"""Collect early failures and branch the frozen map policy on fresh training worlds."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import orbax.checkpoint as ocp
from craftax.craftax_env import make_craftax_env_from_name
from models.actor_critic_map_mlp import map_mlp_network
from symbolic_map_memory import with_map_memory


def select(mask, new, old):
    def choose(x, y):
        return jnp.where(mask.reshape((len(mask),) + (1,) * (x.ndim - 1)), x, y)
    return jax.tree_util.tree_map(choose, new, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--worlds', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=724039)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'bank.msgpack').exists():
        raise FileExistsError(out/'bank.msgpack')
    cfg = json.loads(args.config.read_text())
    env = with_map_memory(make_craftax_env_from_name(cfg['ENV_NAME'], False), cfg)
    net = map_mlp_network(cfg)
    payload = args.checkpoint/'default'
    params = ocp.PyTreeCheckpointer().restore(str(payload if payload.is_dir() else args.checkpoint))['params']
    vkey = jax.vmap(jax.random.PRNGKey)
    fold = jax.vmap(jax.random.fold_in, in_axes=(0, 0))
    reset = jax.jit(jax.vmap(lambda k: env.reset(k, env.default_params)))
    step = jax.vmap(lambda k, s, a: env.step(k, s, a, env.default_params))
    @jax.jit
    def roll(params, start, keys, until):
        # Use absolute age in each lane's RNG so replay from a snapshot is exact.
        def active(c):
            (_, state), done = c
            return ~done & (state.elapsed < until)
        def tick(c):
            (obs, state), done = c
            live = active(c)
            pair = jax.vmap(jax.random.split)(fold(keys, state.elapsed))
            pi, _ = net.apply(params, obs)
            actions = jax.vmap(jax.random.categorical)(pair[:,0], pi.logits)
            obs2, state2, _, died, _ = step(pair[:,1], state, actions)
            return select(live, (obs2,state2), (obs,state)), done | (live & died)
        return jax.lax.while_loop(lambda c: active(c).any(), tick,
                                  (start, jnp.zeros(len(keys), bool)))
    batch = 128
    if args.worlds % batch:
        raise ValueError('world count must be a multiple of 128')
    rng = np.random.default_rng(args.seed)
    seeds = rng.choice(2**31, args.worlds, replace=False).astype(np.uint32)
    snapshots, records = [], []
    for start in range(0,len(seeds),batch):
        world = seeds[start:start+batch]
        keys = vkey(world)
        initial = reset(keys)
        stream = fold(keys, jnp.full(batch, 901))
        (_, end), dead = roll(params, initial, stream, jnp.full(batch,1000))
        ages = np.asarray(end.elapsed)
        bad = np.asarray(dead) & (ages<1000) & ~np.asarray(end.env_state.achievements).all(1)
        for lag in (50,100,200):
            lanes = np.flatnonzero(bad & (ages>=lag))
            if not len(lanes): continue
            target = np.where(bad, np.maximum(ages-lag,0),0)
            candidate, _ = roll(params, initial, stream, jnp.asarray(target))
            # Group 16 worlds x 8 independent continuations per compiled batch.
            for pos in range(0,len(lanes),16):
                group = lanes[pos:pos+16]
                index = np.repeat(np.resize(group,16),8)
                branch_start = jax.tree_util.tree_map(lambda x: x[index],candidate)
                ids = jnp.tile(jnp.arange(8),16) + 10000 + lag*8
                branch_keys = fold(keys[index],ids)
                (_, final), died = roll(params,branch_start,branch_keys,jnp.full(batch,1000))
                survived = np.asarray((~died) & (final.elapsed==1000)).reshape(16,8).sum(1)
                for lane,count in zip(group,survived):
                    record = dict(world=int(world[lane]),death_age=int(ages[lane]),lead=lag,
                                  restored_age=int(target[lane]),survived=int(count),branches=8)
                    records.append(record)
                    if 0<count<8:
                        snap=jax.tree_util.tree_map(lambda x: np.asarray(x[lane]),candidate)
                        snapshots.append((record,snap))
            # Reapply original action/dynamics RNG to verify factual restoration.
            (_, replay), replay_done = roll(params,candidate,stream,jnp.asarray(np.where(bad,ages,0)))
            if not np.asarray(replay_done)[lanes].all(): raise RuntimeError('Factual replay missed death')
            for a,b in zip(jax.tree_util.tree_leaves(replay),jax.tree_util.tree_leaves(end)):
                np.testing.assert_array_equal(np.asarray(a)[lanes],np.asarray(b)[lanes])
        print(f'Worlds {start+batch}/{args.worlds}, mixed-outcome candidates {len(snapshots)}',flush=True)
    # One equally sampled snapshot per world, prefer intermediate difficulty.
    rng.shuffle(snapshots)
    chosen={}
    for rec,snap in sorted(snapshots,key=lambda x:abs(x[0]['survived']-4)):
        chosen.setdefault(rec['world'],(rec,snap))
    report=dict(source=str(args.checkpoint),seed=args.seed,worlds=args.worlds,world_seeds=seeds.tolist(),
                candidates=records,selected=[v[0] for v in chosen.values()])
    (out/'diagnostic.json').write_text(json.dumps(report,indent=2))
    if len(chosen)<32:
        raise RuntimeError(f'Only {len(chosen)} recoverable worlds. Inspect diagnostic before training.')
    bank=jax.tree_util.tree_map(lambda *xs: np.stack(xs),*(v[1] for v in chosen.values()))
    tmp=out/'bank.msgpack.tmp'
    tmp.write_bytes(serialization.to_bytes(bank))
    tmp.replace(out/'bank.msgpack')
    print(f'Bank ready: {len(chosen)} distinct worlds',flush=True)


if __name__=='__main__':main()

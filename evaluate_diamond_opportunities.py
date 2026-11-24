"""Short, on-policy diamond opportunity evaluation; no training or edits to Craftax."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import BlockType, Achievement, Action, OBS_DIM, DIRECTIONS
from models.actor_critic import ActorCritic
from wrappers import GymnaxWrapper, LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper


class OpportunityWrapper(GymnaxWrapper):
    def step(self, rng, state, action, params=None):
        unseen = ~state.achievements[Achievement.COLLECT_DIAMOND.value]
        equipped = state.inventory.iron_pickaxe > 0
        grid = jnp.pad(state.map, ((3, 3), (4, 4)), constant_values=-1)
        view = jax.lax.dynamic_slice(grid, state.player_position, OBS_DIM)
        visible = unseen & equipped & jnp.any(view == BlockType.DIAMOND.value)
        front = state.player_position + DIRECTIONS[state.player_direction]
        immediate = (unseen & equipped & ~state.is_sleeping
                     & (state.map[front[0], front[1]] == BlockType.DIAMOND.value)
                     & ~state.mob_map[front[0], front[1]])
        obs, next_state, reward, done, info = self._env.step(rng, state, action, params)
        collected = unseen & next_state.achievements[Achievement.COLLECT_DIAMOND.value]
        info.update(visible_opportunity=visible, immediate_opportunity=immediate,
                    first_collection=collected, immediate_mined=immediate & collected)
        return obs, next_state, reward, done, info


def proportion(k, n):
    if not n:
        return dict(successes=int(k), opportunities=0, rate=None, wilson_95_ci=None)
    p = k / n
    z = 1.95996398454
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return dict(successes=int(k), opportunities=int(n), rate=float(p),
                wilson_95_ci=[float(center-half), float(center+half)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--seed', type=int, default=2050)
    parser.add_argument('--checkpoint', type=int, default=None)
    args = parser.parse_args()
    manager = CheckpointManager(str(Path(args.run).resolve()/'files/policies'),
                                PyTreeCheckpointer(), CheckpointManagerOptions(create=False))
    step = args.checkpoint if args.checkpoint is not None else max(manager.all_steps())
    if step not in manager.all_steps():
        raise ValueError(f'Checkpoint {step} is not available')
    weights = manager.restore(step)['params']
    base = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
    env = BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(OpportunityWrapper(base))), 32)
    params = env.default_params
    net = ActorCritic(17, 512)
    key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed))
    obs, state = env.reset(reset_key, params)

    def transition(carry, _):
        key, obs, state = carry
        key, ak, ek = jax.random.split(key, 3)
        pi, _ = net.apply(weights, obs)
        obs, state, _, done, info = env.step(ek, state, pi.sample(seed=ak), params)
        return (key, obs, state), (done, info)

    block = jax.jit(lambda carry: jax.lax.scan(transition, carry, None, length=64))
    carry = (key, obs, state)
    quotas = np.full(32, args.episodes//32)
    quotas[:args.episodes % 32] += 1
    completed = np.zeros(32, dtype=int)
    accum = [dict(visible_steps=0, immediate_steps=0, immediate_mined=0,
                  first_visible_step=-1, first_immediate_step=-1,
                  collection_step=-1, length=0) for _ in range(32)]
    rows = []
    while len(rows) < args.episodes:
        carry, (done, info) = block(carry)
        done = np.asarray(done)
        info = jax.tree_util.tree_map(np.asarray, info)
        for t in range(64):
            for worker in range(32):
                if completed[worker] >= quotas[worker]:
                    continue
                a = accum[worker]
                for source, count, first in [('visible_opportunity', 'visible_steps', 'first_visible_step'),
                                             ('immediate_opportunity', 'immediate_steps', 'first_immediate_step')]:
                    if info[source][t, worker]:
                        a[count] += 1
                        if a[first] < 0:
                            a[first] = a['length']
                a['immediate_mined'] += int(info['immediate_mined'][t, worker])
                if info['first_collection'][t, worker]:
                    a['collection_step'] = a['length']
                a['length'] += 1
                if done[t, worker]:
                    rows.append(dict(worker=worker, worker_episode=int(completed[worker]), **a))
                    completed[worker] += 1
                    accum[worker] = dict.fromkeys(a, 0)
                    for name in ['first_visible_step', 'first_immediate_step', 'collection_step']:
                        accum[worker][name] = -1
        if len(rows) % 64 == 0:
            print(f'Completed {len(rows)}/{args.episodes}', flush=True)
    visible = [r for r in rows if r['visible_steps']]
    immediate = [r for r in rows if r['immediate_steps']]
    summary = dict(episodes=len(rows), seed=args.seed, checkpoint=step, run=args.run,
                   diamond_collections=sum(r['collection_step'] >= 0 for r in rows),
                   visible_episode_conversion=proportion(sum(r['collection_step'] >= r['first_visible_step'] for r in visible), len(visible)),
                   immediate_episode_conversion=proportion(sum(r['collection_step'] >= r['first_immediate_step'] for r in immediate), len(immediate)),
                   immediate_step_mining=proportion(sum(r['immediate_mined'] for r in rows), sum(r['immediate_steps'] for r in rows)))
    summary['note'] = ('Episode rates count each episode once, before its first diamond collection. '
                       'Visible means within the symbolic 7x9 view, not necessarily reachable. '
                       'Immediate means in front, awake, iron pickaxe present, no mob on target. '
                       'Step observations are correlated; the step-level Wilson interval is descriptive only.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Local checkpoint evaluation with explicit accounting of every simulator step."""
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
from craftax.craftax_env import make_craftax_env_from_name
from models.actor_critic import ActorCritic
from wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper


def evaluate(checkpoint_dir, out, seed, episodes, max_steps, width=512):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(str(Path(checkpoint_dir).resolve()), PyTreeCheckpointer(),
                                CheckpointManagerOptions(create=False))
    step = manager.latest_step()
    params = manager.restore(step)['params']
    net = ActorCritic(17, width)
    env = BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(
        make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False))), 32)
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = env.reset(reset_key, env.default_params)

    def transition(carry, _):
        key, obs, state = carry
        key, action_key, step_key = jax.random.split(key, 3)
        pi, _ = net.apply(params, obs)
        obs, state, _, done, info = env.step(step_key, state, pi.sample(seed=action_key), env.default_params)
        return (key, obs, state), (done, info)
    block = jax.jit(lambda carry: jax.lax.scan(transition, carry, None, length=64))
    quotas = np.full(32, episodes // 32, np.int32)
    quotas[:episodes % 32] += 1
    completed = np.zeros(32, np.int32)
    carry = key, obs, state
    rows = []; steps = 0
    while len(rows) < episodes:
        if steps + 2048 > max_steps:
            result = dict(complete=False, environment_steps=steps, episodes=len(rows),
                          required_episodes=episodes, reason='evaluation budget exhausted')
            (out / 'summary.json').write_text(json.dumps(result, indent=2))
            return result
        carry, (done, info) = block(carry)
        steps += 2048
        info = jax.tree_util.tree_map(np.asarray, info)
        for t, e in np.argwhere(np.asarray(done)):
            if completed[e] >= quotas[e]:
                continue
            row = {k: float(v[t, e]) for k, v in info.items() if np.asarray(v).ndim == 2
                   and (k.startswith('Achievements/') or k in ('returned_episode_returns', 'returned_episode_lengths'))}
            row['environment'] = int(e)
            row['episode_in_environment'] = int(completed[e])
            rows.append(row); completed[e] += 1
    manager.close()
    result = dict(complete=True, checkpoint=int(step), environment_steps=steps,
                  evaluation_seed=seed, episodes=len(rows),
                  mean_return=float(np.mean([r['returned_episode_returns'] for r in rows])),
                  achievements={k.split('/', 1)[1]:float(np.mean([r[k] for r in rows]))
                                for k in rows[0] if k.startswith('Achievements/')})
    with (out / 'episodes.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    return result

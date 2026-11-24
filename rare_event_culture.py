"""Oracle diamond-event memory for an explicitly privileged expert-transfer pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from culture import CultureMemory, encode
from wrappers import GymnaxWrapper
from craftax.craftax_classic.constants import Achievement, BlockType, OBS_DIM


class DiamondOracleWrapper(GymnaxWrapper):
    def step(self, rng, state, action, params=None):
        padded = jnp.pad(state.map, ((3, 3), (4, 4)), constant_values=-1)
        view = jax.lax.dynamic_slice(padded, state.player_position, OBS_DIM)
        opportunity = (jnp.any(view == BlockType.DIAMOND.value)
                       & (state.inventory.iron_pickaxe > 0)
                       & ~state.achievements[Achievement.COLLECT_DIAMOND.value])
        obs, next_state, reward, done, info = self._env.step(rng, state, action, params)
        info['diamond_oracle'] = opportunity
        return obs, next_state, reward, done, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--window', type=int, default=20)
    p.add_argument('--seed', type=int, default=50)
    args = p.parse_args()
    source = Path('concept_mapping/runs/master_symbolic_10b_5000episodes')
    candidates = []
    for path in sorted((source/'episodes').glob('episode_*.npz')):
        with np.load(path) as d:
            flags = np.r_[d['achievements'][:, 19], d['terminal_achievements'][19]]
            events = np.flatnonzero(np.diff(flags.astype(int)) > 0)
            if len(events):
                candidates.append((path, int(events[0])))
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(candidates), min(args.episodes, len(candidates)), replace=False)
    obs, actions, episodes, steps = [], [], [], []
    for i in chosen:
        path, t = candidates[i]
        with np.load(path) as d:
            for s in range(max(0, t-args.window+1), t+1):
                if (d['inventory'][s, 8] > 0 and not d['achievements'][s, 19]
                        and np.any(d['visible_blocks'][s] == 10)):
                    obs.append(d['observations'][s].copy())
                    actions.append(int(d['actions'][s]))
                    episodes.append(int(d['episode_id']))
                    steps.append(s)
    if not obs:
        raise RuntimeError('No eligible successful diamond-event examples')
    # Reuse the fixed expert embedding index, never its scalar values.
    parent = CultureMemory.load('concept_mapping/runs/cppo_100m_pilot/memory/culture.npz')
    encoder_path = parent.metadata['encoder_path']
    with np.load(encoder_path) as data:
        weights = {'params': {f'Dense_{i}': {k: jnp.asarray(data[f'Dense_{i}_{k}'])
                                           for k in ['kernel', 'bias']} for i in range(8)}}
    observations = np.stack(obs)
    embeddings = np.asarray(jax.jit(lambda x: encode(weights, x))(observations))
    n = len(obs)
    memory = CultureMemory(dict(observations=observations, actions=np.array(actions)[:, None],
                                embeddings=embeddings, outcomes=np.ones(n, np.float32),
                                returns=np.ones(n, np.float32), episode_id=np.array(episodes),
                                timestep=np.array(steps)),
                           dict(source_dataset=str(source.resolve()), source_policy=parent.metadata['source_policy'],
                                source_checkpoint=parent.metadata['source_checkpoint'],
                                encoder_path=encoder_path, representation='critic_hidden',
                                seed=args.seed, window=args.window, success_episodes=len(chosen),
                                entries=n, importance='unit event success, not return-to-episode-end',
                                selection='eligible pre-first-diamond frames in successful final window; immutable expert replay',
                                candidate_success_episodes=len(candidates)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    memory.save(out/'culture.npz')
    (out/'metadata.json').write_text(json.dumps(memory.metadata, indent=2)+'\n')
    print(json.dumps(memory.metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()

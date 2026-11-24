"""Opt-in reset distribution over explicit original-generator world seeds."""
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from wrappers import GymnaxWrapper


def read_world_seeds(path):
    data=json.loads(Path(path).read_text())
    seeds=data['world_seeds'] if isinstance(data,dict) else data
    if not isinstance(seeds,list) or not seeds:
        raise ValueError('Expected a nonempty world_seeds list')
    if any(not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32 for seed in seeds):
        raise ValueError('World seeds must be uint32 integers')
    if len(set(seeds))!=len(seeds):raise ValueError('World seeds must be unique')
    return seeds


class FixedWorldSeedWrapper(GymnaxWrapper):
    """Choose a saved world uniformly on reset; subsequent steps stay stochastic.

    Put inside autoreset/batch wrappers, outside reward wrappers. Only the reset
    key changes; the generator, initial inventory, rewards and mechanics do not.
    """
    def __init__(self,env,seeds):
        super().__init__(env)
        if not seeds:raise ValueError('Empty world bank')
        self.seeds=jnp.asarray(seeds,jnp.uint32)

    @partial(jax.jit,static_argnums=(0,2))
    def reset(self,key,params=None):
        index=jax.random.randint(key,(),0,len(self.seeds))
        return self._env.reset(jax.random.PRNGKey(self.seeds[index]),params)


def with_fixed_world_seeds(env,config):
    path=config.get('RESET_SEED_BANK')
    return FixedWorldSeedWrapper(env,read_world_seeds(path)) if path else env

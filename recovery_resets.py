"""Start a configurable fraction of training episodes at verified pre-death states."""
from functools import partial
from pathlib import Path

from flax import serialization, struct
import jax
import jax.numpy as jnp
import numpy as np

from wrappers import GymnaxWrapper


@struct.dataclass
class RecoveryState:
    env_state: object
    recovery: jnp.ndarray
    first: jnp.ndarray


class RecoveryResetWrapper(GymnaxWrapper):
    def __init__(self, env, bank_path, probability):
        super().__init__(env)
        if not 0 < probability <= 1:
            raise ValueError('Recovery probability must be in (0,1]')
        self.probability = probability
        template = jax.eval_shape(lambda: env.reset(jax.random.PRNGKey(0),env.default_params))
        bank = serialization.from_bytes(template, Path(bank_path).read_bytes())
        self.size = bank[0].shape[0]
        if not self.size:
            raise ValueError('Empty recovery bank')
        for saved, expected in zip(jax.tree_util.tree_leaves(bank), jax.tree_util.tree_leaves(template)):
            if saved.shape != (self.size, *expected.shape) or saved.dtype != expected.dtype:
                raise ValueError('Recovery bank does not match environment observation/state schema')
        self.bank = jax.tree_util.tree_map(jnp.asarray,bank)

    @partial(jax.jit, static_argnums=(0,2))
    def reset(self,key,params=None):
        decision, index = jax.random.split(jax.random.fold_in(key,104))
        recovery = jax.random.bernoulli(decision,self.probability)
        slot = jax.random.randint(index,(),0,self.size)
        obs,state = jax.lax.cond(recovery,
            lambda: jax.tree_util.tree_map(lambda leaf: leaf[slot],self.bank),
            lambda: self._env.reset(key,params))
        return obs,RecoveryState(state,recovery,jnp.asarray(True))

    @partial(jax.jit, static_argnums=(0,4))
    def step(self,key,state,action,params=None):
        obs,next_state,reward,done,info = self._env.step(key,state.env_state,action,params)
        info=dict(info)
        info['practice/recovery_step']=state.recovery
        info['practice/start']=state.first
        info['practice/recovery_start']=state.recovery & state.first
        return obs,RecoveryState(next_state,state.recovery,jnp.asarray(False)),reward,done,info

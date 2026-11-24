"""Expose existing map odometry to the learned spatial memory, no hidden tiles."""
from functools import partial
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from wrappers import GymnaxWrapper


class NeuralMapPositionWrapper(GymnaxWrapper):
    def observation_space(self, params=None):
        size = self._env.observation_space(params).shape[0]
        return spaces.Box(-jnp.inf, jnp.inf, (size+2,), jnp.float32)

    @staticmethod
    def augment(obs, state):
        terrain_state = state
        while not hasattr(terrain_state, 'terrain'):
            terrain_state = terrain_state.env_state
        if terrain_state.terrain.shape != (64, 64):
            raise ValueError('Neural map currently supports Classic 64x64 worlds')
        return jnp.concatenate((obs, terrain_state.position.astype(jnp.float32)))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.augment(obs, state), state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.augment(obs, state), state, reward, done, info

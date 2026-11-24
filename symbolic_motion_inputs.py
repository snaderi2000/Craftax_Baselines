"""Append the preceding action and actual displacement to Classic observations.

Place inside logging/autoreset/batch wrappers, around a NO-auto-reset simulator.
At s_t, the suffix is [one_hot(a_{t-1}, 17), dx_t, dy_t]. Reset uses 19 zeros.
Only actual coordinate differences are exposed, never absolute coordinates or
unseen terrain. The original 1,345 observation entries are unchanged.
"""
from functools import partial

from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from symbolic_map_memory import simulator_state
from wrappers import GymnaxWrapper

BASE_OBSERVATION_SIZE = 1345
ACTION_COUNT = 17
MOTION_INPUT_SIZE = ACTION_COUNT + 2


@struct.dataclass
class MotionInputState:
    env_state: object
    previous_position: jnp.ndarray  # Simulator (row, column); not a policy input.


class ClassicMotionInputWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        if env.observation_space(env.default_params).shape != (BASE_OBSERVATION_SIZE,):
            raise ValueError('Motion inputs currently require the raw 1,345-element Classic symbolic observation')
        if env.action_space(env.default_params).n != ACTION_COUNT:
            raise ValueError('Motion inputs require the 17 Classic actions')

    def observation_space(self, params=None):
        return spaces.Box(-jnp.inf, jnp.inf,
                          (BASE_OBSERVATION_SIZE + MOTION_INPUT_SIZE,), jnp.float32)

    def initialize(self, obs, env_state):
        position = simulator_state(env_state).player_position
        return (jnp.concatenate((obs.astype(jnp.float32), jnp.zeros(MOTION_INPUT_SIZE, jnp.float32))),
                MotionInputState(env_state, position))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.initialize(obs, state)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, next_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        position = simulator_state(next_state).player_position
        delta_yx = position - state.previous_position
        # Input order is dx,dy, despite simulator coordinates being row,column.
        suffix = jnp.concatenate((jax.nn.one_hot(action, ACTION_COUNT, dtype=jnp.float32),
                                  delta_yx[::-1].astype(jnp.float32)))
        augmented = jnp.concatenate((obs.astype(jnp.float32), suffix))
        return augmented, MotionInputState(next_state, position), reward, done, info


def with_motion_inputs(env, config):
    """Shared opt-in factory for collection, evaluation and video inference."""
    if not config.get('MOTION_INPUTS', False):
        return env
    if config.get('ENV_NAME', 'Craftax-Classic-Symbolic-v1') != 'Craftax-Classic-Symbolic-v1':
        raise ValueError('Motion inputs require Classic symbolic observations')
    if config.get('SYMBOLIC_ENCODER', 'mlp') != 'mlp':
        raise ValueError('Motion inputs currently support the flat symbolic encoder only')
    env = ClassicMotionInputWrapper(env)
    if config.get('WORLD_VIEW'):
        from symbolic_world_view import WorldViewWrapper
        env = WorldViewWrapper(env,config['WORLD_VIEW'])
    return env

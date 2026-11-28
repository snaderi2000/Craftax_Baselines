"""First observed diamond, first planted sapling, and episode-local footprints."""
from functools import partial

from craftax.craftax_classic.constants import Achievement, Action, BlockType, DIRECTIONS
from flax import struct
from gymnax.environments import spaces
import jax
import jax.numpy as jnp

from symbolic_map_memory import LOCAL_GRID_SIZE, LOCAL_SHAPE, simulator_state
from wrappers import GymnaxWrapper

HEIGHT, WIDTH = LOCAL_SHAPE[:2]
FEATURES = 14 + HEIGHT * WIDTH
MODES = ('control', 'landmarks', 'trail', 'both')


@struct.dataclass
class EpisodeMemory:
    env_state: object
    landmarks: jnp.ndarray  # [diamond, plant] by [column, row, timestep]
    visited: jnp.ndarray


def observe(obs, memory, mode):
    raw = simulator_state(memory.env_state)
    row, col = raw.player_position
    tick = raw.timestep
    diamond = obs[:LOCAL_GRID_SIZE].reshape(LOCAL_SHAPE)[..., BlockType.DIAMOND.value] > 0
    dy = jnp.arange(7)[:, None] - 3
    dx = jnp.arange(9)[None, :] - 4
    index = jnp.argmin(jnp.where(diamond, abs(dx) + abs(dy), 100))
    first = jnp.array([col + index % 9 - 4, row + index // 9 - 3, tick])
    slots = memory.landmarks.at[0].set(jnp.where(
        (memory.landmarks[0, 2] < 0) & diamond.any(), first, memory.landmarks[0]))
    visited = memory.visited.at[row, col].set(tick.astype(jnp.int16))
    x, y, seen = slots.T
    valid = seen >= 0
    features = jnp.stack((valid, (x-32)/32, (y-32)/32, seen/10000,
                          (x-col)/64, (y-row)/64, (tick-seen)/10000), axis=1)
    features = jnp.where(valid[:, None], features, 0).ravel()
    rows = row + dy
    cols = col + dx
    height, width = visited.shape
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    last = visited[rows.clip(0, height-1), cols.clip(0, width-1)].astype(jnp.int32)
    trail = jnp.where(inside & (last >= 0), jnp.exp2((last-tick)/100.), 0).ravel()
    if mode not in ('landmarks', 'both'):
        features = jnp.zeros_like(features)
    if mode not in ('trail', 'both'):
        trail = jnp.zeros_like(trail)
    return jnp.concatenate((obs, features, trail)), memory.replace(landmarks=slots, visited=visited)


class EpisodeMemoryWrapper(GymnaxWrapper):
    def __init__(self, env, mode):
        super().__init__(env)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode

    def observation_space(self, params=None):
        size, = self._env.observation_space(params).shape
        return spaces.Box(-jnp.inf, jnp.inf, (size + FEATURES,), jnp.float32)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        world_shape = simulator_state(state).map.shape
        memory = EpisodeMemory(state, jnp.full((2, 3), -1, jnp.int32),
                               jnp.full(world_shape, -1, jnp.int16))
        return observe(obs, memory, self.mode)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, memory, action, params=None):
        before = simulator_state(memory.env_state)
        obs, state, reward, done, info = self._env.step(key, memory.env_state, action, params)
        after = simulator_state(state)
        event = Achievement.PLACE_PLANT.value
        planted = ((action == Action.PLACE_PLANT.value) & ~before.achievements[event]
                   & after.achievements[event] & (memory.landmarks[1, 2] < 0))
        row, col = before.player_position + DIRECTIONS[before.player_direction]
        slots = memory.landmarks.at[1].set(jnp.where(
            planted, jnp.array([col, row, after.timestep]), memory.landmarks[1]))
        obs, memory = observe(obs, memory.replace(env_state=state, landmarks=slots), self.mode)
        return obs, memory, reward, done, info

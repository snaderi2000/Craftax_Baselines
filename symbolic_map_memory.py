"""Observed-terrain memory with explicit odometry for symbolic Classic.

Wrap a NO-auto-reset environment before log/reset/batch wrappers. Only the
existing symbolic observation supplies terrain values. Simulator positions
supply privileged actual displacement, never unseen map contents. The backing
map persists for the episode and its readout depends on the policy.
"""
from functools import partial

from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from wrappers import GymnaxWrapper

LOCAL_SIZE = 1345
LOCAL_SHAPE = (7, 9, 21)
TERRAIN_TYPES = 17
GLOBAL_MAP_SIZE = 64
GLOBAL_MAP_FEATURES = 2 * GLOBAL_MAP_SIZE**2 + 2
# Each world cell carries (terrain ID, observation age, current occupancy bits).
BOARD_CELL_FIELDS = 3
PLAYER_BIT = 16
LOCAL_GRID_SIZE = LOCAL_SHAPE[0] * LOCAL_SHAPE[1] * LOCAL_SHAPE[2]
PLAYER_FEATURES = LOCAL_SIZE - LOCAL_GRID_SIZE
BOARD_OBS_SIZE = GLOBAL_MAP_SIZE**2 * BOARD_CELL_FIELDS + PLAYER_FEATURES


def _visible_indices(position, shape):
    rows = position[0] + jnp.arange(7)[:, None] - 3
    cols = position[1] + jnp.arange(9)[None, :] - 4
    height, width = shape
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    # Out-of-bounds sentinels work for both drop-mode writes and fill-mode reads.
    return jnp.where(valid, rows, height), jnp.where(valid, cols, width)


def map_discovery_reward(new_cells, done, coefficient):
    """The initial view and terminal transitions earn no exploration reward."""
    return jnp.where(done, 0., coefficient * new_cells)


@struct.dataclass
class MapMemoryState:
    env_state: object
    terrain: jnp.ndarray  # uint8: 0 unknown, observed tile ID + 1 otherwise
    last_seen: jnp.ndarray  # step number, -1 for unknown
    position: jnp.ndarray  # map address, advanced using actual displacement
    last_position: jnp.ndarray  # simulator position used only for odometry
    elapsed: jnp.ndarray


class ClassicMapMemoryWrapper(GymnaxWrapper):
    def __init__(self, env, view_size=33, global_readout=False, full_terrain=False, single_map=False, board_only=False,
                 track_coverage=False):
        super().__init__(env)
        if not isinstance(view_size, int) or view_size < 9 or view_size % 2 != 1:
            raise ValueError('Map view size must be an odd integer >= 9')
        self.view_size = view_size
        self.global_readout = global_readout
        if full_terrain and not global_readout:
            raise ValueError('Full terrain requires a global map readout')
        self.full_terrain = full_terrain
        if single_map and (not global_readout or full_terrain):
            raise ValueError('Single-map policy requires an observed-only global readout')
        self.single_map = single_map
        if board_only and not single_map:
            raise ValueError('Board-only input requires a single observed map')
        self.board_only = board_only
        self.track_coverage = track_coverage

    def observation_space(self, params=None):
        if self.board_only:
            return spaces.Box(0., 31., (BOARD_OBS_SIZE,), jnp.float32)
        global_size = GLOBAL_MAP_FEATURES if self.global_readout else 0
        return spaces.Box(0., float(TERRAIN_TYPES),
                          (LOCAL_SIZE + (0 if self.single_map else 2*self.view_size**2) + global_size,), jnp.float32)

    def initialize(self, obs, env_state):
        # Allocate by world SHAPE only. No world terrain is copied here.
        shape = env_state.map.shape
        if self.global_readout and shape != (GLOBAL_MAP_SIZE, GLOBAL_MAP_SIZE):
            raise ValueError('Global map readout requires a 64×64 Classic world')
        position = env_state.player_position
        state = MapMemoryState(env_state, jnp.zeros(shape, jnp.uint8),
            jnp.full(shape, -1, jnp.int32), position, position, jnp.asarray(0, jnp.int32))
        return self.observe(obs, state)

    def observe(self, obs, state):
        """Write the visible tiles, then construct the configured policy input."""
        local_grid = obs[:LOCAL_GRID_SIZE].reshape(LOCAL_SHAPE)
        patch = local_grid[..., :TERRAIN_TYPES]
        codes = (jnp.argmax(patch, axis=-1)+1).astype(jnp.uint8)
        rows, cols = _visible_indices(state.position, state.terrain.shape)
        terrain = state.terrain.at[rows, cols].set(codes, mode='drop')
        last_seen = state.last_seen.at[rows, cols].set(state.elapsed, mode='drop')
        state = state.replace(terrain=terrain, last_seen=last_seen)
        def age(seen):
            elapsed = jnp.where(seen >= 0, state.elapsed-seen, 10000)
            return jnp.log1p(jnp.minimum(elapsed, 10000).astype(jnp.float32))/jnp.log(10001.)

        if self.board_only:
            entities = local_grid[..., TERRAIN_TYPES:].astype(jnp.uint8)
            bits = jnp.sum(entities * jnp.array([1, 2, 4, 8], jnp.uint8), axis=-1, dtype=jnp.uint8)
            flags = jnp.zeros_like(terrain).at[rows, cols].set(bits, mode='drop')
            flags = flags.at[tuple(state.position)].add(PLAYER_BIT)
            board = jnp.stack((terrain, age(last_seen), flags), axis=-1)
            return jnp.concatenate((board.ravel(), obs[LOCAL_GRID_SIZE:])), state

        augmented = obs.astype(jnp.float32)
        if not self.single_map:
            radius = self.view_size//2
            shape = (self.view_size, self.view_size)
            start = tuple(state.position)
            terrain_view = jax.lax.dynamic_slice(jnp.pad(terrain, radius), start, shape)
            seen_view = jax.lax.dynamic_slice(jnp.pad(last_seen, radius, constant_values=-1), start, shape)
            augmented = jnp.concatenate((augmented, terrain_view.reshape(-1), age(seen_view).reshape(-1)))
        if self.global_readout:
            global_terrain = state.env_state.map.astype(jnp.uint8) + 1 if self.full_terrain else terrain
            augmented = jnp.concatenate((augmented, global_terrain.reshape(-1),
                age(last_seen).reshape(-1), state.position.astype(jnp.float32)/(GLOBAL_MAP_SIZE-1)))
        return augmented, state

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        return self.initialize(obs, env_state)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        displacement = env_state.player_position-state.last_position
        state = state.replace(env_state=env_state,
            position=state.position+displacement, last_position=env_state.player_position,
            elapsed=state.elapsed+1)
        if self.track_coverage:
            rows, cols = _visible_indices(state.position, state.terrain.shape)
            previous_ids = state.terrain.at[rows, cols].get(mode='fill', fill_value=1)
            info = {**info, 'map_new_cells': jnp.count_nonzero(previous_ids == 0)}
        obs, state = self.observe(obs, state)
        return obs, state, reward, done, info


def with_map_memory(env, config, *, track_coverage=False):
    """Shared training/evaluation/video factory; off for historical configs."""
    if config.get('SYMBOLIC_ENCODER', 'mlp') in ('spatial_map','map_mlp'):
        env=ClassicMapMemoryWrapper(env, int(config.get('MAP_VIEW_SIZE', 33)),
            global_readout=config.get('MAP_GLOBAL_READOUT',False),
            full_terrain=config.get('MAP_FULL_TERRAIN',False), single_map=config.get('MAP_SINGLE_READOUT',False),
            board_only=config.get('MAP_BOARD_ONLY',False), track_coverage=track_coverage)
        mode=config.get('MAP_SURVIVAL_FEATURES','none')
        if mode!='none':
            from symbolic_survival_map import ClassicSurvivalMapWrapper
            env=ClassicSurvivalMapWrapper(env,mode)
    if config.get('NEURAL_MAP_CHANNELS',0):
        if config.get('SYMBOLIC_ENCODER')!='map_mlp':
            raise ValueError('Neural map requires map_mlp terrain inputs')
        from symbolic_neural_map import NeuralMapPositionWrapper
        env=NeuralMapPositionWrapper(env)
    if config.get('MAP_LANDMARKS'):
        from symbolic_landmarks import LandmarkWrapper
        if not config.get('MAP_GLOBAL_READOUT') or config.get('MAP_SINGLE_READOUT') or config.get('MAP_FULL_TERRAIN'):
            raise ValueError('Landmarks require the crop plus observed world memory')
        env=LandmarkWrapper(env,config['MAP_LANDMARKS'])
    return env


def simulator_state(state):
    """Recover the raw state for metrics/rendering without exposing it to PPO."""
    while hasattr(state, 'env_state'):
        state = state.env_state
    return state


def display_local_view(obs, raw_state, config):
    """Recover the spectator's 7×9 view for replay."""
    if config.get('MAP_BOARD_ONLY'):
        from craftax.craftax_classic.renderer import render_craftax_symbolic
        return render_craftax_symbolic(raw_state)
    return obs[:LOCAL_SIZE]

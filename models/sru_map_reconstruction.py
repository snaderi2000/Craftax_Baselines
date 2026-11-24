"""Terrain supervision for the SRU. The action network receives only its usual observation."""
from functools import partial

import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from symbolic_map_memory import LOCAL_SHAPE, TERRAIN_TYPES, simulator_state
from wrappers import GymnaxWrapper

STRIDE = 8
QUERY_COUNT = 12
SIDE = 64
RESOURCE_IDS = (3, 15, 16, 10)  # water, plant, ripe plant, diamond


@struct.dataclass
class TargetState:
    env_state: object
    # Padded so updating the visible patch at a world edge never wraps.
    # Zero means unknown. Positive values encode (last_seen_step + 1, terrain_id).
    grid: object
    landmarks: object
    tick: object


class TargetWrapper(GymnaxWrapper):
    def record(self, obs, state):
        pos = simulator_state(state.env_state).player_position
        tiles = obs[:1323].reshape(LOCAL_SHAPE)[..., :TERRAIN_TYPES].argmax(-1)
        grid = jax.lax.dynamic_update_slice(state.grid, (state.tick + 1) * 32 + tiles, tuple(pos))
        matches = (tiles.ravel()[:, None] == jnp.array(RESOURCE_IDS))
        ix = matches.argmax(0)
        coords = pos + jnp.stack((ix // 9 - 3, ix % 9 - 4), axis=-1)
        landmarks = jnp.where(matches.any(0)[:, None], coords, state.landmarks)
        return state.replace(grid=grid, landmarks=landmarks)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = TargetState(state, jnp.zeros((SIDE + 6, SIDE + 8), jnp.int32),
                            jnp.full((len(RESOURCE_IDS), 2), -1, jnp.int32), jnp.int32(0))
        return obs, self.record(obs, state)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        state = self.record(obs, state.replace(env_state=env_state, tick=state.tick + 1))
        return obs, state, reward, done, info


def query_labels(state, coords):
    raw = simulator_state(state.env_state)
    offset = coords - raw.player_position
    valid = ((coords >= 0) & (coords < SIDE)).all(-1)
    offscreen = (jnp.abs(offset[:, 0]) > 3) | (jnp.abs(offset[:, 1]) > 4)
    row, col = jnp.clip(coords, 0, SIDE - 1).T
    packed = state.grid[row + 3, col + 4]
    # For old terrain use the last observation, even if the tile has since changed.
    terrain = jnp.where(packed > 0, packed % 32, raw.map[row, col])
    age = jnp.where(packed > 0, state.tick - (packed // 32 - 1), -1)
    return jnp.column_stack((offset, terrain, age, valid & offscreen)).astype(jnp.int16)


def sample_queries(state, key):
    local_key, global_key = jax.random.split(key)
    pos = simulator_state(state.env_state).player_position
    nearby = pos + jax.random.randint(local_key, (4, 2), -16, 17)
    board = jax.random.randint(global_key, (4, 2), 0, SIDE)
    return query_labels(state, jnp.concatenate((nearby, board, state.landmarks), axis=0))


class MapDecoder(nn.Module):
    @nn.compact
    def __call__(self, hidden, offset):
        # Each query is independent. Its sampling group and ground truth are hidden.
        context = nn.Dense(64, name='memory')(hidden)[..., None, :]
        relative = offset.astype(jnp.float32)
        angles = relative[..., None] * jnp.array([.1, .25, .5, 1.])
        features = jnp.concatenate((relative / 63., jnp.sin(angles).reshape((*offset.shape[:-1], 8)),
                                    jnp.cos(angles).reshape((*offset.shape[:-1], 8))), axis=-1)
        query = nn.Dense(64, name='offset')(features)
        out = nn.Dense(TERRAIN_TYPES + 1, name='prediction')(nn.tanh(context + query))
        return out[..., :TERRAIN_TYPES], out[..., -1]


class MapTrainState(TrainState):
    decoder: TrainState


def init_decoder(width, key):
    net = MapDecoder()
    params = net.init(key, jnp.zeros((1, width)), jnp.zeros((1, 1, 2)))
    return TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(3e-4, eps=1e-5))


def masked_mean(values, mask):
    return (values * mask).sum() / jnp.maximum(mask.sum(), 1)


def reconstruction_loss(params, hidden, queries):
    terrain, seen_logits = MapDecoder().apply(params, hidden, queries[..., :2])
    targets, age, valid = queries[..., 2], queries[..., 3], queries[..., 4] > 0
    known = age >= 0
    ce = -jnp.take_along_axis(jax.nn.log_softmax(terrain), targets[..., None].astype(jnp.int32), axis=-1)[..., 0]
    seen_ce = optax.sigmoid_binary_cross_entropy(seen_logits, known.astype(jnp.float32))
    loss = (masked_mean(ce, valid & known) + .1 * masked_mean(ce, valid & ~known)
            + .25 * masked_mean(seen_ce, valid))
    old = valid & (age >= 100)
    correct = terrain.argmax(-1) == targets
    return loss, (masked_mean(correct, valid & known), (correct & old).sum(), old.sum())

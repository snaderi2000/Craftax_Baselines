"""Reset Classic on-policy rollouts to fresh worlds or unresolved failed worlds."""
from functools import partial

from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp

from symbolic_map_memory import simulator_state
from wrappers import GymnaxWrapper


SLOTS = 8
MAX_WEIGHT = 8
REPLAY_PROBABILITY = .5


class ReplayTrainState(TrainState):
    replay_keys: jnp.ndarray
    replay_failures: jnp.ndarray


@struct.dataclass
class ReplayState:
    env_state: object
    world_key: jnp.ndarray
    slot: jnp.ndarray  # -1 on a fresh episode
    keys: jnp.ndarray
    failures: jnp.ndarray  # zero means empty


def next_world(state, success, fresh_key, choice_key):
    """Each lane owns its bank. Occupied slots persist until their first success."""
    weights = state.failures
    retry = state.slot >= 0
    available = (weights == 0).any()
    slot = jnp.where(retry, state.slot, jnp.argmin(weights))
    admit = ~success & ~retry & available
    keys = state.keys.at[slot].set(jnp.where(admit, state.world_key, state.keys[slot]))
    failed_weight = jnp.minimum(weights[slot] + 1, MAX_WEIGHT)
    weight = jnp.where(retry, jnp.where(success, 0, failed_weight),
                       jnp.where(admit, 1, weights[slot]))
    weights = weights.at[slot].set(weight)
    coin, draw = jax.random.split(choice_key)
    total = weights.sum()
    # Integer cumulative sampling is exactly proportional to the failure counts.
    ticket = jax.random.randint(draw, (), 0, jnp.maximum(1, total))
    selected = jnp.argmax(jnp.cumsum(weights) > ticket)
    replay = (total > 0) & jax.random.bernoulli(coin, REPLAY_PROBABILITY)
    return state.replace(keys=keys, failures=weights,
                         slot=jnp.where(replay, selected, -1),
                         world_key=jnp.where(replay, keys[selected], fresh_key))


class FailureSeedAutoReset(GymnaxWrapper):
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        return obs, ReplayState(env_state, rng, jnp.int32(-1),
                                jnp.zeros((SLOTS, 2), jnp.uint32), jnp.zeros(SLOTS, jnp.int32))

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, dynamics = jax.random.split(rng)
        choice, fresh = jax.random.split(rng)
        obs, after, reward, done, info = self._env.step(dynamics, state.env_state, action, params)
        success = simulator_state(after).achievements.all()
        retry = state.slot >= 0
        info = dict(info)
        info['curriculum/replayed_episode'] = retry.astype(jnp.float32)
        info['curriculum/replayed_all22'] = (retry & success).astype(jnp.float32)
        info['curriculum/fresh_all22'] = (~retry & success).astype(jnp.float32)
        info['curriculum/unresolved_seeds_per_lane'] = (state.failures > 0).sum()
        info['curriculum/fresh_failure_bank_full'] = (~retry & ~success & (state.failures > 0).all()).astype(jnp.float32)

        def restart(_):
            selected = next_world(state, success, fresh, choice)
            initial, restarted = self._env.reset(selected.world_key, params)
            return initial, selected.replace(env_state=restarted)

        obs, state = jax.lax.cond(done, restart, lambda _: (obs, state.replace(env_state=after)), None)
        return obs, state, reward, done, info

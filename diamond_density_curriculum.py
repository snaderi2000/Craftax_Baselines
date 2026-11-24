"""Opt-in episode-start diamond enrichment; never edits the installed environment."""
from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from flax import struct
from wrappers import GymnaxWrapper
from craftax.craftax_classic.constants import BlockType
from craftax.craftax_classic.renderer import render_craftax_symbolic


def enrich_map(key, grid, multiplier=10):
    """Add (multiplier-1)*D diamonds on random stone tiles, capped by availability."""
    stone = grid == BlockType.STONE.value
    diamonds = (grid == BlockType.DIAMOND.value).sum()
    extra = jnp.minimum((multiplier-1)*diamonds, stone.sum())
    scores = jax.random.uniform(key, grid.shape)
    scores = jnp.where(stone, scores, 2.)
    ranks = jnp.argsort(jnp.argsort(scores.reshape(-1))).reshape(grid.shape)
    return jnp.where(stone & (ranks < extra), BlockType.DIAMOND.value, grid)


@struct.dataclass
class CurriculumState:
    inner: Any
    interactions: Any
    dense: Any
    original_diamonds: Any
    starting_diamonds: Any


class DiamondDensityCurriculum(GymnaxWrapper):
    """Wrap an already-vectorized, logged, auto-reset Classic symbolic environment.

    Drawing with fold_in leaves the underlying environment/action RNG unchanged.
    Episodes are mixed by reset count, not by transition count. Longer episodes
    may make the dense transition share differ from the dense episode share.
    """
    def __init__(self, env, start=50_000_000, probability=.25, multiplier=10):
        super().__init__(env)
        self.start, self.probability, self.multiplier = int(start), float(probability), int(multiplier)

    def _enrich(self, key, obs, inner, chosen):
        def apply(_):
            keys = jax.random.split(jax.random.fold_in(key, 172903), self.num_envs)
            maps = jax.vmap(lambda k, m: enrich_map(k, m, self.multiplier))(keys, inner.env_state.map)
            maps = jnp.where(chosen[:, None, None], maps, inner.env_state.map)
            updated = inner.replace(env_state=inner.env_state.replace(map=maps))
            return jax.vmap(render_craftax_symbolic)(updated.env_state), updated
        return jax.lax.cond(jnp.any(chosen), apply, lambda _: (obs, inner), operand=None)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, inner = self._env.reset(key, params)
        chosen = (jax.random.uniform(jax.random.fold_in(key, 172904), (self.num_envs,))
                  < self.probability) & (self.start <= 0)
        counts = (inner.env_state.map == BlockType.DIAMOND.value).sum(axis=(1, 2))
        obs, inner = self._enrich(key, obs, inner, chosen)
        after = (inner.env_state.map == BlockType.DIAMOND.value).sum(axis=(1, 2))
        return obs, CurriculumState(inner, jnp.asarray(0, jnp.int32), chosen, counts, after)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, inner, reward, done, info = self._env.step(key, state.inner, action, params)
        interactions = state.interactions + self.num_envs
        chosen = (done & (interactions >= self.start)
                  & (jax.random.uniform(jax.random.fold_in(key, 172904), (self.num_envs,)) < self.probability))
        before = (inner.env_state.map == BlockType.DIAMOND.value).sum(axis=(1, 2))
        obs, inner = self._enrich(key, obs, inner, chosen)
        after = (inner.env_state.map == BlockType.DIAMOND.value).sum(axis=(1, 2))
        info['curriculum/dense_episode'] = state.dense.astype(jnp.float32)
        info['curriculum/original_diamonds'] = state.original_diamonds.astype(jnp.float32)
        info['curriculum/starting_diamonds'] = state.starting_diamonds.astype(jnp.float32)
        info['curriculum/dense_transition'] = state.dense.astype(jnp.float32)
        next_state = CurriculumState(inner, interactions, jnp.where(done, chosen, state.dense),
                                     jnp.where(done, before, state.original_diamonds),
                                     jnp.where(done, after, state.starting_diamonds))
        return obs, next_state, reward, done, info

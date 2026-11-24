"""Fresh normal-map resets into reachable visible-diamond drills."""
from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from flax import struct
from wrappers import GymnaxWrapper, LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper
from craftax.craftax_classic.constants import BlockType, Achievement, OBS_DIM
from craftax.craftax_classic.envs.common import compute_score


def drill_position(key, grid):
    """Sample one diamond and a visible, walkable spawn connected to its boundary."""
    dk, pk = jax.random.split(key)
    scores = jnp.where(grid == BlockType.DIAMOND.value, jax.random.uniform(dk, grid.shape), -1.)
    index = jnp.argmax(scores)
    target = jnp.array([index//grid.shape[1], index % grid.shape[1]])
    padded = jnp.pad(grid, ((3, 3), (4, 4)), constant_values=BlockType.OUT_OF_BOUNDS.value)
    local = jax.lax.dynamic_slice(padded, target, OBS_DIM)
    walkable = (local == BlockType.GRASS.value) | (local == BlockType.PATH.value) | (local == BlockType.SAND.value)
    rr, cc = jnp.indices(OBS_DIM)
    reached = walkable & (jnp.abs(rr-3)+jnp.abs(cc-4) == 1)
    def expand(_, reached):
        pad = jnp.pad(reached, 1)
        neighbors = pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
        return (neighbors | reached) & walkable
    reached = jax.lax.fori_loop(0, 12, expand, reached)
    candidate_scores = jnp.where(reached, jax.random.uniform(pk, OBS_DIM), -1.)
    chosen = jnp.argmax(candidate_scores)
    position = target + jnp.array([chosen//9-3, chosen % 9-4])
    return jnp.any(reached) & jnp.any(grid == BlockType.DIAMOND.value), position, target


@struct.dataclass
class DrillState:
    env_state: Any
    elapsed: Any
    reset_attempts: Any


class DiamondDrill(GymnaxWrapper):
    def __init__(self, env, horizon=64):
        super().__init__(env)
        self.horizon = horizon

    def reset(self, key, params=None):
        def sample(key):
            key, world_key, position_key = jax.random.split(key, 3)
            _, state = self._env.reset(world_key, params)
            valid, position, _ = drill_position(position_key, state.map)
            return key, state, valid, position
        key, state, valid, position = sample(key)
        def resample(carry):
            key, _, _, _, attempts = carry
            key, state, valid, position = sample(key)
            return key, state, valid, position, attempts+1
        key, state, valid, position, attempts = jax.lax.while_loop(
            lambda carry: ~carry[2], resample, (key, state, valid, position, jnp.asarray(1)))
        achievements = state.achievements
        for flag in [Achievement.COLLECT_WOOD, Achievement.PLACE_TABLE, Achievement.COLLECT_STONE,
                     Achievement.COLLECT_COAL, Achievement.COLLECT_IRON, Achievement.PLACE_FURNACE,
                     Achievement.MAKE_WOOD_PICKAXE, Achievement.MAKE_STONE_PICKAXE, Achievement.MAKE_IRON_PICKAXE]:
            achievements = achievements.at[flag.value].set(True)
        state = state.replace(player_position=position,
                              player_direction=jax.random.randint(key, (), 1, 5),
                              inventory=state.inventory.replace(wood_pickaxe=1, stone_pickaxe=1, iron_pickaxe=1),
                              achievements=achievements, player_health=9, player_food=9,
                              player_drink=9, player_energy=9)
        return self._env.get_obs(state), DrillState(state, jnp.asarray(0), attempts)

    def step(self, key, state, action, params=None):
        obs, next_state, reward, terminal, info = self._env.step(key, state.env_state, action, params)
        success = next_state.achievements[Achievement.COLLECT_DIAMOND.value]
        elapsed = state.elapsed+1
        truncated = (elapsed >= self.horizon) & ~terminal & ~success
        done = terminal | success | truncated
        info.update(compute_score(next_state, done))
        info.update({'drill/success': success.astype(jnp.float32),
                     'drill/truncated': truncated, 'drill/bootstrap_obs': obs,
                     'drill/is_drill': jnp.asarray(1., jnp.float32),
                     'drill/reset_attempts': state.reset_attempts.astype(jnp.float32)})
        return obs, DrillState(next_state, elapsed, state.reset_attempts), reward, done, info


@struct.dataclass
class MixedState:
    normal: Any
    drill: Any


class MixedDiamondDrills(GymnaxWrapper):
    def __init__(self, base, num_envs=128, num_drills=32, horizon=64):
        super().__init__(base)
        self.num_envs, self.num_drills = num_envs, num_drills
        self.normal = BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(base)), num_envs-num_drills)
        self.drill = BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(DiamondDrill(base, horizon))), num_drills)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        nk, dk = jax.random.split(key)
        normal_obs, normal = self.normal.reset(nk, params)
        drill_obs, drill = self.drill.reset(dk, params)
        return jnp.concatenate([normal_obs, drill_obs]), MixedState(normal, drill)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        nk, dk = jax.random.split(key)
        n = self.num_envs-self.num_drills
        no, ns, nr, nd, ni = self.normal.step(nk, state.normal, action[:n], params)
        do, ds, dr, dd, di = self.drill.step(dk, state.drill, action[n:], params)
        ni.update({'drill/success': jnp.zeros(n), 'drill/truncated': jnp.zeros(n, dtype=bool),
                   'drill/bootstrap_obs': jnp.zeros_like(no), 'drill/is_drill': jnp.zeros(n),
                   'drill/reset_attempts': jnp.zeros(n)})
        info = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), ni, di)
        return (jnp.concatenate([no, do]), MixedState(ns, ds), jnp.concatenate([nr, dr]),
                jnp.concatenate([nd, dd]), info)

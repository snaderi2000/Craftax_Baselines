import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any

from jax.tree_util import tree_map 

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class ClassicDeathPenaltyWrapper(GymnaxWrapper):
    """Add a death cost before reset, preserving time-limit-only rewards.

    Wrap a Classic environment without automatic resets. Place reset and log
    wrappers outside this wrapper so both PPO and episode returns see the cost.
    """

    def __init__(self, env, death_penalty):
        super().__init__(env)
        self.death_penalty = float(death_penalty)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        from craftax.craftax_classic.constants import BlockType

        obs, next_state, reward, done, info = self._env.step(
            rng, state, action, params
        )
        in_lava = next_state.map[
            next_state.player_position[0], next_state.player_position[1]
        ] == BlockType.LAVA.value
        died = done & ((next_state.player_health <= 0) | in_lava)
        return obs, next_state, reward - self.death_penalty * died, done, info


class ClassicDiamondRewardWrapper(GymnaxWrapper):
    """Change the FIRST diamond achievement's total reward, before resets.

    The base achievement reward is +1. Add only (target - 1), once; retain
    health rewards, any death cost, all other achievements and game mechanics.
    """
    def __init__(self, env, diamond_reward=1.):
        super().__init__(env)
        if not np.isfinite(diamond_reward) or diamond_reward < 0:
            raise ValueError('Diamond reward must be finite and nonnegative')
        self.diamond_reward = float(diamond_reward)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        from craftax.craftax_classic.constants import Achievement
        obs, next_state, reward, done, info = self._env.step(rng, state, action, params)
        index = Achievement.COLLECT_DIAMOND.value
        newly_collected = next_state.achievements[index] & ~state.achievements[index]
        return obs, next_state, reward+(self.diamond_reward-1.)*newly_collected, done, info


class ClassicLavaPenaltyWrapper(GymnaxWrapper):
    """Add a cost only for terminal lava entry; preserve all other rewards."""

    def __init__(self, env, lava_penalty=2.):
        super().__init__(env)
        if not np.isfinite(lava_penalty) or lava_penalty < 0:
            raise ValueError('lava_penalty must be finite and nonnegative')
        self.lava_penalty=float(lava_penalty)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        from craftax.craftax_classic.constants import BlockType
        obs, next_state, reward, done, info=self._env.step(rng,state,action,params)
        in_lava=next_state.map[next_state.player_position[0],next_state.player_position[1]]==BlockType.LAVA.value
        return obs,next_state,reward-self.lava_penalty*(done&in_lava),done,info


class ClassicAchievementRewardWrapper(GymnaxWrapper):
    """Set first-diamond and first-plant achievement rewards before reset."""

    def __init__(self, env, diamond_reward=1., plant_reward=1.):
        super().__init__(env)
        for name, value in [('diamond_reward', diamond_reward), ('plant_reward', plant_reward)]:
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        self.diamond_reward = float(diamond_reward)
        self.plant_reward = float(plant_reward)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        from craftax.craftax_classic.constants import Achievement
        obs, next_state, reward, done, info = self._env.step(rng, state, action, params)
        newly_earned = next_state.achievements & ~state.achievements
        reward = (reward
                  + (self.diamond_reward - 1.) * newly_earned[Achievement.COLLECT_DIAMOND.value]
                  + (self.plant_reward - 1.) * newly_earned[Achievement.EAT_PLANT.value])
        return obs, next_state, reward, done, info


class ClassicCompletionWrapper(GymnaxWrapper):
    """Treat earning the final missing Classic achievement as a terminal win."""

    def __init__(self, env, bonus):
        super().__init__(env)
        if not np.isfinite(bonus) or bonus < 0:
            raise ValueError("Completion bonus must be finite and nonnegative")
        self.bonus = float(bonus)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        from craftax.craftax_classic.envs.common import compute_score
        obs, next_state, reward, done, info = self._env.step(rng, state, action, params)
        won = next_state.achievements.all() & ~state.achievements.all()
        done = done | won
        info = {**info, **compute_score(next_state, done),
                "discount": jnp.where(won, 0., info["discount"]), "all22_win": won}
        return obs, next_state, reward + self.bonus * won, done, info


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

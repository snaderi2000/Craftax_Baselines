"""Advantage estimators for the CartPole PPO experiments.

``compute_gae`` is vanilla generalized advantage estimation. New estimators
should take the same inputs and return ``(advantages, value_targets)``, both
shaped ``[time, batch]``, then get registered in ``ADVANTAGE_FNS``.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Rollout(NamedTuple):
    """One PPO rollout.

    Leading axes are ``[time, batch]``. ``rewards`` is what the learner sees
    (possibly delayed). ``true_rewards`` is the raw CartPole +1-per-step signal
    and is only for scoring, not for the advantage.
    """

    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    true_rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray


def compute_gae(
    rollout: Rollout,
    last_values: jnp.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    **_unused,
):
    """GAE(lambda), the vanilla PPO advantage.

    ``last_values`` bootstraps the state after the final transition and has
    shape ``[batch]``. Completed episodes are cut with ``dones``.
    """

    def _step(carry, xs):
        gae, next_value = carry
        reward, value, done = xs
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_values), last_values),
        (rollout.rewards, rollout.values, rollout.dones),
        reverse=True,
    )
    return advantages, advantages + rollout.values, rollout


def _cartpole_state(obs: jnp.ndarray):
    from gymnax.environments.classic_control.cartpole import EnvState

    return EnvState(
        x=obs[0],
        x_dot=obs[1],
        theta=obs[2],
        theta_dot=obs[3],
        time=jnp.int32(0),
    )


def _horizon_return(
    obs, action, rng, *, params, network, env, env_params, horizon, gamma
):
    """H-step simulator return starting with ``action``, then bootstrap V(s_H).

    The bootstrap is zero once the pole has fallen. It lets the value function
    represent survival beyond the horizon instead of cutting the return off at
    ``horizon`` steps.
    """

    def _body(carry, _):
        state, next_action, rng, alive, ret, discount, _last_value = carry
        rng, env_rng, policy_rng = jax.random.split(rng, 3)
        next_obs, next_state, reward, done, _info = env.step_env(
            env_rng, state, next_action, env_params
        )
        step_reward = reward.astype(jnp.float32) * alive
        ret = ret + discount * step_reward
        alive = alive * (1.0 - done.astype(jnp.float32))
        discount = discount * gamma
        pi, value = network.apply(params, next_obs)
        next_action = pi.sample(seed=policy_rng)
        return (next_state, next_action, rng, alive, ret, discount, value), None

    _pi0, value0 = network.apply(params, obs)
    start = (
        _cartpole_state(obs),
        action.astype(jnp.int32),
        rng,
        jnp.float32(1.0),
        jnp.float32(0.0),
        jnp.float32(1.0),
        value0,
    )
    (_state, _action, _rng, alive, ret, discount, last_value), _ = jax.lax.scan(
        _body, start, None, horizon
    )
    bootstrap = discount * alive * jax.lax.stop_gradient(last_value)
    return ret + bootstrap


def compute_group_lookahead(
    rollout: Rollout,
    last_values: jnp.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    params,
    rng,
    network,
    env,
    env_params,
    horizon: int,
    group_size: int,
    **_unused,
):
    """Value target from a group of perfect-model rollouts.

    Each action at a visited state is unrolled ``horizon`` steps, ``group_size``
    times. The action score is the mean bootstrapped return of that group.
    The critic target is the policy-weighted average of those scores, so one
    real visit supplies several on-policy returns. The advantage is the taken
    action's score minus that target. The behavior action is left unchanged.
    """
    del last_values, gae_lambda
    obs = rollout.obs
    time_steps, batch = obs.shape[:2]
    flat_obs = obs.reshape((-1, obs.shape[-1]))
    flat_actions = rollout.actions.reshape((-1,)).astype(jnp.int32)
    n_states = flat_obs.shape[0]
    n_actions = env.num_actions
    keys = jax.random.split(rng, n_states * n_actions * group_size).reshape(
        (n_states, n_actions, group_size, -1)
    )

    def _scores(obs_i, keys_i):
        def _one_action(action, action_keys):
            returns = jax.vmap(
                lambda key: _horizon_return(
                    obs_i,
                    action,
                    key,
                    params=params,
                    network=network,
                    env=env,
                    env_params=env_params,
                    horizon=horizon,
                    gamma=gamma,
                )
            )(action_keys)
            return returns.mean()

        return jax.vmap(_one_action)(jnp.arange(n_actions), keys_i)

    scores = jax.vmap(_scores)(flat_obs, keys)

    def _probs(obs_i):
        pi, _value = network.apply(params, obs_i)
        return pi.probs

    probs = jax.vmap(_probs)(flat_obs)
    value_target = (probs * scores).sum(axis=-1)
    taken_score = jnp.take_along_axis(scores, flat_actions[:, None], axis=-1)[:, 0]
    advantages = (taken_score - value_target).reshape((time_steps, batch))
    targets = value_target.reshape((time_steps, batch))
    return advantages, targets, rollout


AdvantageFn = Callable[..., tuple]

ADVANTAGE_FNS: dict[str, AdvantageFn] = {
    "gae": compute_gae,
    "group_lookahead": compute_group_lookahead,
}

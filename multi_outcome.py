"""Observed achievement/inventory changes with per-output balanced replay."""
from dataclasses import fields
import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Achievement
from craftax.craftax_classic.envs.craftax_state import Inventory
from wrappers import GymnaxWrapper

INVENTORY_FIELDS = tuple(f.name for f in fields(Inventory))
OUTCOMES = tuple('achievement_' + a.name.lower() for a in Achievement) + tuple(
    'inventory_increase_' + f for f in INVENTORY_FIELDS)
N_OUTCOMES = len(OUTCOMES)


def transition_labels(before, after):
    earned = after.achievements & ~before.achievements
    increases = jnp.stack([getattr(after.inventory, f) > getattr(before.inventory, f)
                           for f in INVENTORY_FIELDS])
    return jnp.concatenate([earned, increases])


class MultiOutcomeWrapper(GymnaxWrapper):
    def step(self, key, state, action, params=None):
        obs, next_state, reward, done, info = self._env.step(key, state, action, params)
        labels = transition_labels(state, next_state)
        info.update({'outcome/' + name: labels[i] for i, name in enumerate(OUTCOMES)})
        return obs, next_state, reward, done, info


def initialize_head(params, key, width):
    params = dict(params)
    params['params'] = dict(params['params'])
    k1, k2 = jax.random.split(key)
    params['params']['MultiOutcome'] = dict(
        hidden_kernel=jax.random.normal(k1, (width + 17, 64)) / jnp.sqrt(width + 17),
        hidden_bias=jnp.zeros(64), kernel=jax.random.normal(k2, (64, N_OUTCOMES)) * .01,
        bias=jnp.zeros(N_OUTCOMES))
    return params


def predict_logits(params, obs, actions):
    root = params['params']
    h = obs
    for i in range(3):
        h = jnp.tanh(h @ root[f'Dense_{i}']['kernel'] + root[f'Dense_{i}']['bias'])
    h = jnp.concatenate([h, jax.nn.one_hot(actions, 17)], axis=-1)
    head = root['MultiOutcome']
    h = jnp.tanh(h @ head['hidden_kernel'] + head['hidden_bias'])
    return h @ head['kernel'] + head['bias']


def empty_replay(capacity, obs_dim, n_outcomes=N_OUTCOMES):
    # Two factual queues per output; store only input and the indexed binary label.
    return dict(obs=jnp.zeros((2 * n_outcomes, capacity, obs_dim)),
                actions=jnp.zeros((2 * n_outcomes, capacity), jnp.int32),
                counts=jnp.zeros(2 * n_outcomes, jnp.int32),
                totals=jnp.zeros(2 * n_outcomes, jnp.int32))


def append_replay(buffer, obs, actions, labels, eligible=None):
    capacity = buffer['obs'].shape[1]
    k = min(capacity, len(actions))
    def insert(queue, result):
        matches = labels[:, queue // 2] == (queue % 2)
        if eligible is not None:
            matches = matches & eligible[:, queue // 2]
        values, indices = jax.lax.top_k(jnp.where(matches, jnp.arange(len(actions)), -1), k)
        valid = values >= 0
        slots = (buffer['totals'][queue] + jnp.arange(k)) % capacity
        slots = jnp.where(valid, slots, capacity)
        result = dict(result)
        result['obs'] = result['obs'].at[queue].set(
            buffer['obs'][queue].at[slots].set(obs[indices], mode='drop'))
        result['actions'] = result['actions'].at[queue].set(
            buffer['actions'][queue].at[slots].set(actions[indices], mode='drop'))
        n = valid.sum()
        result['totals'] = result['totals'].at[queue].add(n)
        result['counts'] = result['counts'].at[queue].set(jnp.minimum(capacity, buffer['counts'][queue] + n))
        return result
    return jax.lax.fori_loop(0, buffer['counts'].shape[0], insert, buffer)


def sample_replay(buffer, key, batch_size):
    offset_key, index_key = jax.random.split(key)
    offset = jax.random.randint(offset_key, (), 0, 2 * N_OUTCOMES)
    ranks = jnp.arange(batch_size) + offset
    outputs = ranks % N_OUTCOMES
    labels = (ranks // N_OUTCOMES) % 2
    counts = buffer['counts'].reshape((N_OUTCOMES, 2))
    labels = jnp.where(counts[outputs, labels] > 0, labels, 1 - labels)
    queues = 2 * outputs + labels
    indices = jax.random.randint(index_key, (batch_size,), 0, jnp.maximum(buffer['counts'][queues], 1))
    return (buffer['obs'][queues, indices], buffer['actions'][queues, indices], outputs,
            labels.astype(jnp.float32), (buffer['counts'][queues] > 0).astype(jnp.float32))


def outcome_loss(params, batch):
    obs, actions, outputs, labels, mask = batch
    logits = jnp.take_along_axis(predict_logits(params, obs, actions), outputs[:, None], axis=-1)[:, 0]
    ce = jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return (ce * mask).sum() / jnp.maximum(mask.sum(), 1)

"""Actor-shared next-reward sign classifier trained on factual transitions."""
import jax
import jax.numpy as jnp


def labels_from_rewards(rewards):
    # Class 0: negative, 1: exactly zero, 2: positive. No reward edits.
    return (jnp.sign(rewards).astype(jnp.int32) + 1)


def initialize_head(params, key, width):
    params = dict(params)
    params['params'] = dict(params['params'])
    k1, k2 = jax.random.split(key)
    params['params']['RewardSign'] = dict(
        hidden_kernel=jax.random.normal(k1, (width + 17, 64)) / jnp.sqrt(width + 17),
        hidden_bias=jnp.zeros(64),
        kernel=jax.random.normal(k2, (64, 3)) * .01, bias=jnp.zeros(3))
    return params


def predict_logits(params, observations, actions):
    root = params['params']
    h = observations
    for i in range(3):
        h = jnp.tanh(h @ root[f'Dense_{i}']['kernel'] + root[f'Dense_{i}']['bias'])
    h = jnp.concatenate([h, jax.nn.one_hot(actions, 17)], axis=-1)
    head = root['RewardSign']
    h = jnp.tanh(h @ head['hidden_kernel'] + head['hidden_bias'])
    return h @ head['kernel'] + head['bias']


def empty_replay(capacity, obs_dim):
    return dict(obs=jnp.zeros((3, capacity, obs_dim)),
                actions=jnp.zeros((3, capacity), jnp.int32),
                counts=jnp.zeros(3, jnp.int32), totals=jnp.zeros(3, jnp.int32))


def append_replay(buffer, observations, actions, rewards):
    labels = labels_from_rewards(rewards)
    capacity = buffer['obs'].shape[1]
    result = dict(buffer)
    k = min(capacity, len(actions))
    for label in range(3):
        ranks = jnp.where(labels == label, jnp.arange(len(actions)), -1)
        values, indices = jax.lax.top_k(ranks, k)
        valid = values >= 0
        slots = (buffer['totals'][label] + jnp.arange(k)) % capacity
        slots = jnp.where(valid, slots, capacity)
        result['obs'] = result['obs'].at[label].set(
            buffer['obs'][label].at[slots].set(observations[indices], mode='drop'))
        result['actions'] = result['actions'].at[label].set(
            buffer['actions'][label].at[slots].set(actions[indices], mode='drop'))
        inserted = valid.sum()
        result['totals'] = result['totals'].at[label].add(inserted)
        result['counts'] = result['counts'].at[label].set(
            jnp.minimum(capacity, buffer['counts'][label] + inserted))
    return result


def sample_replay(buffer, key, batch_size):
    available = buffer['counts'] > 0
    classes = jnp.nonzero(available, size=3, fill_value=0)[0]
    labels = classes[jnp.arange(batch_size) % jnp.maximum(available.sum(), 1)]
    indices = jax.random.randint(key, (batch_size,), 0, jnp.maximum(buffer['counts'][labels], 1))
    return (buffer['obs'][labels, indices], buffer['actions'][labels, indices], labels,
            available.any() * jnp.ones(batch_size))


def outcome_loss(params, batch):
    obs, actions, labels, mask = batch
    logs = jax.nn.log_softmax(predict_logits(params, obs, actions))
    ce = -jnp.take_along_axis(logs, labels[:, None], axis=1)[:, 0]
    return (ce * mask).sum() / jnp.maximum(mask.sum(), 1)

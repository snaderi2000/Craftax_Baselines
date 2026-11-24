"""Searchable actor-encoder auxiliary tasks on factual PPO rollouts.

No simulator labels, altered rewards, replay restarts, or pretrained encoders.
Future windows crossing an episode boundary are excluded, because PPO next_obs
is the reset observation on terminal transitions. Targets are built before
minibatch shuffling. Projection and sampling RNGs are separate from PPO's RNG.
"""
import jax
import jax.numpy as jnp

TASKS = ('inverse', 'observation_delta', 'stats_delta', 'event')
HORIZONS = (1, 4, 16)
OUTPUTS = dict(inverse=17, observation_delta=32, stats_delta=16, event=1)


def actor_hidden(params, obs):
    h = obs
    for i in range(3):
        layer = params['params'][f'Dense_{i}']
        h = jnp.tanh(h @ layer['kernel'] + layer['bias'])
    return h


def initialize_head(params, key, width, task):
    root = dict(params['params'])
    k1, k2 = jax.random.split(key)
    inputs = width * 2 if task == 'inverse' else width + 17
    root['AuxPrediction'] = dict(
        hidden_kernel=jax.random.normal(k1, (inputs, 64)) / jnp.sqrt(inputs),
        hidden_bias=jnp.zeros(64),
        kernel=jax.random.normal(k2, (64, OUTPUTS[task])) * .01,
        bias=jnp.zeros(OUTPUTS[task]))
    return dict(params, params=root)


def projection(obs_dim, seed):
    # Shared fixed target across tasks and training seeds; never optimized.
    return jax.random.normal(jax.random.PRNGKey(seed), (obs_dim, 32)) / jnp.sqrt(obs_dim)


def targets(obs, next_obs, actions, rewards, done, task, horizon, matrix):
    """Return [T,E,...] examples and masks with no cross-reset target leakage."""
    t = obs.shape[0]
    start = jnp.arange(t)
    end = jnp.minimum(start + horizon - 1, t - 1)
    boundary_prefix = jnp.concatenate([jnp.zeros_like(done[:1], dtype=jnp.int32),
                                      jnp.cumsum(done.astype(jnp.int32), axis=0)], axis=0)
    valid = (start + horizon <= t)[:, None] & (boundary_prefix[end + 1] == boundary_prefix[start])
    after = next_obs[end]
    if task == 'inverse':
        target = actions[..., None]
    elif task == 'observation_delta':
        target = (after - obs) @ matrix
    elif task == 'stats_delta':
        target = after[..., 1323:1339] - obs[..., 1323:1339]
    elif task == 'event':
        events = jnp.isclose(rewards, 1., rtol=0., atol=1e-6).astype(jnp.int32)
        prefix = jnp.concatenate([jnp.zeros_like(events[:1]), jnp.cumsum(events, axis=0)], axis=0)
        target = (prefix[end + 1] > prefix[start])[..., None].astype(jnp.float32)
    else:
        raise ValueError(task)
    return obs, after, actions, jax.lax.stop_gradient(target), valid.astype(jnp.float32)


def sample_batch(data, key, batch_size):
    flat = tuple(x.reshape((-1,) + x.shape[2:]) for x in data)
    mask = flat[-1]
    count = (mask > 0).sum()
    positions = jnp.nonzero(mask > 0, size=len(mask), fill_value=0)[0]
    draw = jax.random.randint(key, (batch_size,), 0, jnp.maximum(count, 1))
    return tuple(x[positions[draw]] for x in flat)


def predict(params, batch, task):
    obs, after, actions = batch[:3]
    h = actor_hidden(params, obs)
    if task == 'inverse':
        h = jnp.concatenate([h, actor_hidden(params, after)], axis=-1)
    else:
        h = jnp.concatenate([h, jax.nn.one_hot(actions, 17)], axis=-1)
    head = params['params']['AuxPrediction']
    h = jnp.tanh(h @ head['hidden_kernel'] + head['hidden_bias'])
    return h @ head['kernel'] + head['bias']


def loss(params, batch, task):
    logits = predict(params, batch, task)
    target, mask = batch[3:]
    if task == 'inverse':
        error = -jnp.take_along_axis(jax.nn.log_softmax(logits), target.astype(jnp.int32), axis=-1)[:, 0]
    elif task == 'event':
        error = (jnp.maximum(logits, 0) - logits * target + jnp.log1p(jnp.exp(-jnp.abs(logits))))[:, 0]
    else:
        error = ((logits - target) ** 2).mean(axis=-1)
    return (error * mask).sum() / jnp.maximum(mask.sum(), 1)

"""Checkpoint-preserving SRU-GRU gate and its staged PPO optimizer.

Candidate modulation follows Yang et al. (2025), section 4.4, and the
authors' GRUSRUCell (MIT): https://arxiv.org/abs/2506.05997
https://github.com/leggedrobotics/sru-pytorch-spatial-learning/blob/main/network/gru_sru.py
The caller keeps Flax's GRU reset convention and pretrained parameter names.
"""

import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import optax


GATE_PATH = ('params', 'ScannedRNN_0', 'spatial_transform')


class ResidualDecisionHead(nn.Module):
    """Normalized actor or critic with two width-preserving residual layers."""
    width: int
    outputs: int
    output_scale: float

    @nn.compact
    def __call__(self, features):
        from models.actor_critic import DenseResBlock
        x = nn.LayerNorm()(features)
        x = nn.relu(nn.Dense(self.width, kernel_init=nn.initializers.orthogonal(2.))(x))
        for _ in range(2):
            x = nn.relu(DenseResBlock(self.width)(x))
        return nn.Dense(self.outputs, kernel_init=nn.initializers.orthogonal(self.output_scale))(nn.LayerNorm()(x))


def spatial_multiplier(inputs):
    return nn.Dense(inputs.shape[-1], kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.ones, name=GATE_PATH[-1])(inputs)


def has_spatial_gate(parameters):
    return GATE_PATH[-1] in parameters['params'].get(GATE_PATH[-2], {})


def transplant_gru(target, source):
    """Copy every existing leaf and retain only the newly initialized gate."""
    old = flatten_dict(unfreeze(source))
    new = flatten_dict(unfreeze(target))
    gate = {path for path in new if path[:-1] == GATE_PATH}
    if not gate or set(old) != set(new) - gate:
        raise ValueError('Expected the same GRU policy and a newly added spatial transform')
    for path, value in old.items():
        if (value.shape, value.dtype) != (new[path].shape, new[path].dtype):
            raise ValueError(f'GRU checkpoint differs at {path}')
    result = unflatten_dict({**new, **old})
    return freeze(result) if isinstance(target, FrozenDict) else result


def gate_penalty(parameters):
    gate = parameters['params'][GATE_PATH[-2]][GATE_PATH[-1]]
    # Squared distance from the affine identity multiplier per hidden unit.
    return jnp.mean(jnp.sum(jnp.square(gate['kernel']), axis=0) + jnp.square(gate['bias']-1.))


def warmup_minibatches(config):
    batch = config['NUM_ENVS'] * config['NUM_STEPS']
    warmup_updates = (config.get('SRU_WARMUP_STEPS', 0) + batch - 1) // batch
    return warmup_updates * config['NUM_MINIBATCHES'] * config['UPDATE_EPOCHS']


def is_gate(path):
    return tuple(getattr(key, 'key', None) for key in path[:-1]) == GATE_PATH


def make_optimizer(config, base_lr, gate_lr=None):
    """Advance the phase counter while keeping pretrained Adam moments frozen."""
    labels = lambda params: jax.tree_util.tree_map_with_path(
        lambda path, _: 'gate' if is_gate(path) else 'pretrained', params)
    base = optax.conditionally_mask(optax.adam(base_lr, eps=1e-5),
                                    lambda step: step >= warmup_minibatches(config))
    gate = optax.adam(config['SRU_GATE_LR'] if gate_lr is None else gate_lr, eps=1e-5)
    return optax.chain(optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
                       optax.partition({'gate': gate, 'pretrained': base}, labels))


def mask_warmup_gradients(gradients, step, config):
    """Exclude frozen leaves from global clipping as well as Adam updates."""
    return jax.lax.cond(step < warmup_minibatches(config),
        lambda grads: jax.tree_util.tree_map_with_path(
            lambda path, gradient: gradient if is_gate(path) else jnp.zeros_like(gradient), grads),
        lambda grads: grads, gradients)


def learning_rate_scale(config, update):
    """Scale both SRU optimizer partitions on an absolute step schedule."""
    batch = config['NUM_ENVS'] * config['NUM_STEPS']
    start = config['SRU_LR_ANNEAL_FROM']
    end = config['SRU_LR_ANNEAL_UNTIL']
    # Keep the absolute environment step labels out of JAX's int32 arithmetic.
    elapsed = update + (config['START_TIMESTEPS'] - start) // batch
    progress = jnp.clip(elapsed / ((end - start) // batch), 0., 1.)
    floor = config['SRU_LR_FINAL_FRACTION']
    return (1. - progress) * (1. - floor) + floor


def apply_scaled_updates(state, gradients, scale):
    """Scale Adam's parameter deltas while retaining its moments and counters."""
    updates, optimizer_state = state.tx.update(gradients, state.opt_state, state.params)
    updates = jax.tree_util.tree_map(lambda delta: delta * scale, updates)
    return state.replace(step=state.step + 1, params=optax.apply_updates(state.params, updates),
                         opt_state=optimizer_state)

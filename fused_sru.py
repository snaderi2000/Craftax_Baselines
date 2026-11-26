"""Evaluate the restored Flax SRU with packed gates and one input projection per sequence."""
import jax
import jax.numpy as jnp


def run_sru(params, hidden, inputs, done):
    gates = params['GRUCell_1']
    input_layers = [gates[name] for name in ('ir', 'iz', 'in')] + [params['spatial_transform']]
    input_kernel = jnp.concatenate([p['kernel'] for p in input_layers], axis=1)
    input_bias = jnp.concatenate([p['bias'] for p in input_layers])
    projected = inputs @ input_kernel + input_bias
    recurrent_kernel = jnp.concatenate([gates[k]['kernel'] for k in ('hr', 'hz', 'hn')], axis=1)

    def step(h, args):
        features, reset = args
        h = jnp.where(reset[:, None], 0., h)
        ir, iz, candidate, scale = jnp.split(features, 4, axis=-1)
        hr, hz, hn = jnp.split(h @ recurrent_kernel, 3, axis=-1)
        r = jax.nn.sigmoid(ir + hr)
        z = jax.nn.sigmoid(iz + hz)
        n = jnp.tanh(scale * (candidate + r * (hn + gates['hn']['bias'])))
        output = (1. - z) * n + z * h
        return output, output

    return jax.lax.scan(step, hidden, (projected, done))

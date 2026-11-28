"""Evaluate the restored Flax SRU with packed gates and one input projection per sequence."""
import jax
import jax.numpy as jnp


def run_sru(params, hidden, inputs, done, compute_dtype=None):
    gates = params['GRUCell_1']
    input_layers = [gates[name] for name in ('ir', 'iz', 'in')] + [params['spatial_transform']]
    input_kernel = jnp.concatenate([p['kernel'] for p in input_layers], axis=1)
    input_bias = jnp.concatenate([p['bias'] for p in input_layers])
    recurrent_kernel = jnp.concatenate([gates[k]['kernel'] for k in ('hr', 'hz', 'hn')], axis=1)
    hn_bias = gates['hn']['bias']

    if compute_dtype is not None:
        inputs = inputs.astype(compute_dtype)
        input_kernel = input_kernel.astype(compute_dtype)
        input_bias = input_bias.astype(compute_dtype)
        recurrent_kernel = recurrent_kernel.astype(compute_dtype)
        hn_bias = hn_bias.astype(compute_dtype)

    # Put batch axis first before flattening so GSPMD keeps (B*T, D) contiguously sharded
    # across GPUs with zero all-gather while cuBLAS executes one large Tensor Core GEMM.
    if inputs.shape[0] == 1:
        projected = (inputs[0] @ input_kernel + input_bias)[None]
    else:
        t_steps, batch_size, feat_dim = inputs.shape
        flat_in = jnp.swapaxes(inputs, 0, 1).reshape((batch_size * t_steps, feat_dim))
        projected = jnp.swapaxes(
            (flat_in @ input_kernel + input_bias).reshape((batch_size, t_steps, -1)), 0, 1
        )

    def step(h, args):
        features, reset = args
        h = jnp.where(reset[:, None], jnp.zeros_like(h), h)
        h_in = h.astype(compute_dtype) if compute_dtype is not None else h
        ir, iz, candidate, scale = jnp.split(features, 4, axis=-1)
        hr, hz, hn = jnp.split(h_in @ recurrent_kernel, 3, axis=-1)
        if compute_dtype is not None:
            r = jax.nn.sigmoid((ir + hr).astype(jnp.float32))
            z = jax.nn.sigmoid((iz + hz).astype(jnp.float32))
            n = jnp.tanh(
                scale.astype(jnp.float32)
                * (candidate.astype(jnp.float32) + r * (hn + hn_bias).astype(jnp.float32))
            )
        else:
            r = jax.nn.sigmoid(ir + hr)
            z = jax.nn.sigmoid(iz + hz)
            n = jnp.tanh(scale * (candidate + r * (hn + hn_bias)))
        output = (1. - z) * n + z * h
        return output, output

    if inputs.shape[0] == 1:
        output, _ = step(hidden, (projected[0], done[0]))
        return output, output[None]

    return jax.lax.scan(step, hidden, (projected, done))

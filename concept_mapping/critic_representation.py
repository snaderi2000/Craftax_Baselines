"""Utilities for extracting the symbolic PPO critic representation.

The symbolic ``ActorCritic`` network uses four actor Dense modules followed by
four critic Dense modules.  The critic's final scalar is produced by
``Dense_7`` from the post-tanh output of ``Dense_6``.  This module extracts
that [batch, layer_width] tensor without changing the trained architecture.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


CRITIC_HIDDEN_DENSE_NAMES = ("Dense_4", "Dense_5", "Dense_6")
VALUE_HEAD_DENSE_NAME = "Dense_7"


def _parameter_collection(params: Mapping) -> Mapping:
    """Return the Flax ``params`` collection from a TrainState parameter tree."""
    return params["params"] if "params" in params else params


def _dense(x: jnp.ndarray, dense_params: Mapping) -> jnp.ndarray:
    """Apply a Flax Dense layer to a rank-two [batch, features] tensor."""
    return jnp.matmul(x, dense_params["kernel"]) + dense_params["bias"]


def critic_pre_value_representation(
    params: Mapping,
    observations: jnp.ndarray,
    activation: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the final critic hidden state and reconstructed scalar value.

    Args:
        params: Flax parameters of the symbolic ``ActorCritic`` network.
        observations: Symbolic observations with shape ``[B, 1345]``.
        activation: The checkpoint's critic activation (``"tanh"`` or ``"relu"``).

    Returns:
        h: The post-activation output of critic ``Dense_6``, shape ``[B, 512]``.
        value: The scalar reconstructed by critic ``Dense_7``, shape ``[B]``.
    """
    collection = _parameter_collection(params)
    required = (*CRITIC_HIDDEN_DENSE_NAMES, VALUE_HEAD_DENSE_NAME)
    missing = [name for name in required if name not in collection]
    if missing:
        raise KeyError(
            "Expected symbolic ActorCritic parameter names are missing: "
            f"{missing}. Available names: {sorted(collection.keys())}"
        )
    if activation == "tanh":
        nonlinearity = jnp.tanh
    elif activation == "relu":
        nonlinearity = jax.nn.relu
    else:
        raise ValueError(f"Unsupported ActorCritic activation: {activation!r}")

    # h has shape [B, 1345] initially, then [B, 512] after each critic layer.
    h = observations
    for dense_name in CRITIC_HIDDEN_DENSE_NAMES:
        h = nonlinearity(_dense(h, collection[dense_name]))

    # ``h`` is exactly the representation immediately before Dense_7: [B, 512].
    value = _dense(h, collection[VALUE_HEAD_DENSE_NAME]).squeeze(axis=-1)  # [B]
    return h, value


def validate_critic_representation(
    network,
    params: Mapping,
    observations: jnp.ndarray,
    activation: str,
) -> dict[str, float | list[int]]:
    """Check that the manual critic path reproduces ordinary network values."""
    h, reconstructed_value = critic_pre_value_representation(params, observations, activation)
    _policy, network_value = network.apply(params, observations)
    max_abs_error = float(jnp.max(jnp.abs(network_value - reconstructed_value)))
    return {
        "representation_shape": list(h.shape),
        "value_shape": list(reconstructed_value.shape),
        "value_reconstruction_max_abs_error": max_abs_error,
    }

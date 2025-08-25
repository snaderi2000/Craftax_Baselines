# slicer.py (Complete and Corrected)

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Sequence
import numpy as np

class Slicer(nn.Module):
    """
    A JIT-compatible Flax module for calculating sequence slices.
    """
    max_blocks: int
    block_mask: np.ndarray # It now expects a NumPy array

    def setup(self):
        """
        This setup computes static attributes for the slicer. Because this is a Flax
        Module, Flax ensures this logic runs correctly during model initialization
        before JIT compilation begins.
        """
        # This setup logic now runs with concrete NumPy arrays.
        self.block_size = self.block_mask.shape[0]
        self.num_kept_tokens = self.block_mask.sum().item()

        kept_indices = np.where(self.block_mask)[0]
        kept_indices_repeated = np.tile(kept_indices, self.max_blocks)
        offsets = np.repeat(np.arange(self.max_blocks), self.num_kept_tokens)

        # Compute indices with NumPy, then convert to a JAX array
        # for use in the JIT-compiled computation.
        self.indices = jnp.array(kept_indices_repeated + self.block_size * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> jnp.ndarray:
        """Computes the relevant indices for the current step."""
        total_steps = num_steps + prev_steps
        # Use ceiling division to get the number of blocks as a Python integer
        num_blocks = (total_steps + self.block_size - 1) // self.block_size

        slice_end = num_blocks * self.num_kept_tokens
        indices_subset = self.indices[:slice_end]

        valid_mask = jnp.logical_and(prev_steps <= indices_subset, indices_subset < total_steps)
        return indices_subset[valid_mask] - prev_steps

    def __call__(self, *args, **kwargs):
        # This module is not meant to be called directly.
        raise NotImplementedError

# --- Head and Embedder Modules ---

class Head(nn.Module):
    """Slices an input tensor and applies a head module."""
    slicer: Slicer
    head_module: nn.Module

    def __call__(self, x: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        indices = self.slicer.compute_slice(num_steps, prev_steps)
        x_sliced = x[:, indices, :]
        return self.head_module(x_sliced)

class Embedder(nn.Module):
    """Embeds tokens from different partitions of a sequence."""
    slicers: List[Slicer]
    embedding_tables: Sequence[nn.Module]

    def __call__(self, tokens: jnp.ndarray, num_steps: int, prev_steps: int) -> jnp.ndarray:
        embedding_dim = self.embedding_tables[0].features
        output = jnp.zeros((*tokens.shape, embedding_dim), dtype=jnp.float32)

        for slicer, emb_table in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            embedded_slice = emb_table(tokens[:, s])
            output = output.at[:, s].set(embedded_slice)

        return output
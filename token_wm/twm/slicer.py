import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, List

class Slicer:
    """
    JAX Helper to generate masks.
    """
    @staticmethod
    def compute_mask(num_steps: int, prev_steps: int, block_mask: jnp.ndarray) -> jnp.ndarray:
        block_size = block_mask.shape[0]
        global_indices = jnp.arange(num_steps) + prev_steps
        block_pos = global_indices % block_size
        valid_mask = block_mask[block_pos]
        return valid_mask

class Head(nn.Module):
    """
    Projects hidden states to logits, but ONLY at token positions selected by block_mask.
    This matches the PyTorch intent: the head "fires" on a subset of tokens.
    """
    embed_dim: int
    output_dim: int
    block_mask: jnp.ndarray            # (tokens_per_block,) float32 0/1
    hidden_dim: int = None
    mask_before_mlp: bool = True       # stronger (recommended)

    @nn.compact
    def __call__(self, x, num_steps: int, prev_steps: int):
        # x: (B, T, E) where T == num_steps
        h_dim = self.hidden_dim if self.hidden_dim else self.embed_dim

        # valid_mask: (T,) in {0,1}
        valid_mask = Slicer.compute_mask(num_steps=num_steps, prev_steps=prev_steps, block_mask=self.block_mask)
        m = valid_mask[None, :, None].astype(x.dtype)  # (1, T, 1)

        # Option A: mask hidden states BEFORE MLP (closest to PyTorch behavior)
        if self.mask_before_mlp:
            x = x * m

        y = nn.Dense(h_dim)(x)
        y = nn.relu(y)
        y = nn.Dense(self.output_dim)(y)

        # Also mask logits (extra safety; ensures zero grads from invalid positions)
        y = y * m
        return y

class Embedder(nn.Module):
    """
    Combines multiple embedding tables based on block masks.
    """
    max_blocks: int
    embed_dim: int
    block_masks: Sequence[jnp.ndarray] # List of masks
    vocab_sizes: Sequence[int]         # List of vocab sizes

    def setup(self):
        self.tables = [
            nn.Embed(num_embeddings=v_size, features=self.embed_dim) 
            for v_size in self.vocab_sizes
        ]

    def __call__(self, tokens, start_step=0):
        # We don't need num_steps arg if we just use tokens.shape[1]
        num_steps = tokens.shape[1]
        prev_steps = start_step

        output = jnp.zeros((*tokens.shape, self.embed_dim), dtype=jnp.float32)
        
        for i, (mask_pattern, table) in enumerate(zip(self.block_masks, self.tables)):
            valid_mask = Slicer.compute_mask(num_steps, prev_steps, mask_pattern)
            
            # Expand for broadcasting (1, T, 1) -> (B, T, E)
            m = valid_mask[None, :, None]
            
            embeddings = table(tokens)
            output = output + (embeddings * m)
            
        return output
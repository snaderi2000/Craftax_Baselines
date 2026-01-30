import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Sequence

class Slicer:
    """
    JAX Helper to generate masks instead of indices.
    In JAX, we avoid dynamic slicing (changing shapes). 
    Instead, we generate boolean masks to 'keep' or 'ignore' tokens.
    """
    @staticmethod
    def compute_mask(num_steps: int, prev_steps: int, block_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Generates a boolean mask (B, T) where True indicates the token belongs to this slice.
        """
        # block_mask shape: (block_size,)
        block_size = block_mask.shape[0]
        
        # Create global indices [prev_steps, ..., prev_steps + num_steps - 1]
        # shape: (num_steps,)
        global_indices = jnp.arange(num_steps) + prev_steps
        
        # Map to local block position
        block_pos = global_indices % block_size
        
        # Lookup validity in the block_mask
        # shape: (num_steps,)
        # valid_mask[t] is True if token t is of this type
        valid_mask = block_mask[block_pos]
        
        return valid_mask

class Head(nn.Module):
    """
    Projects the hidden states to logits.
    
    Translation Note: 
    The PyTorch version slices the input 'x' to a smaller shape (B, N_kept, E).
    In JAX, to maintain static shapes for JIT, we output the FULL sequence (B, T, Out).
    We rely on the Loss function (in world_model.py) to mask out the 'invalid' steps.
    """
    max_blocks: int
    block_mask: jnp.ndarray # Shape (block_size,)
    output_dim: int
    hidden_dim: int = None
    
    @nn.compact
    def __call__(self, x, num_steps=None, prev_steps=None):
        # x: (Batch, Time, Dim)
        
        # Standard MLP Head
        # We apply this to ALL tokens. The 'inactive' ones will be ignored by the loss.
        h_dim = self.hidden_dim if self.hidden_dim else x.shape[-1]
        
        y = nn.Dense(h_dim)(x)
        y = nn.relu(y)
        y = nn.Dense(self.output_dim)(y)
        
        return y

class Embedder(nn.Module):
    """
    Combines multiple embedding tables based on block masks.
    """
    max_blocks: int
    block_masks: Sequence[jnp.ndarray] # List of masks, one per table
    vocab_sizes: Sequence[int]         # Vocab size for each table
    embed_dim: int

    def setup(self):
        # Check partitions
        # In JAX we assume inputs are correct, but can assert in non-jitted init
        # masks_sum = sum(self.block_masks)
        # assert jnp.all(masks_sum == 1)
        
        self.tables = [
            nn.Embed(num_embeddings=v_size, features=self.embed_dim) 
            for v_size in self.vocab_sizes
        ]

    def __call__(self, tokens, num_steps: int, prev_steps: int):
        """
        Args:
            tokens: (Batch, Time)
            num_steps: int
            prev_steps: int
        Returns:
            Mixed embeddings: (Batch, Time, EmbedDim)
        """
        # Initialize output accumulator
        output = jnp.zeros((*tokens.shape, self.embed_dim), dtype=jnp.float32)
        
        # Iterate over each partition (e.g., Obs vs Actions)
        for i, (mask_pattern, table) in enumerate(zip(self.block_masks, self.tables)):
            
            # 1. Compute which tokens belong to this table
            # valid_mask: (Time,) boolean
            valid_mask = Slicer.compute_mask(num_steps, prev_steps, mask_pattern)
            
            # Expand for broadcasting: (1, Time, 1)
            # Match (Batch, Time, Embed)
            m = valid_mask[None, :, None]
            
            # 2. Embed ALL tokens using this table
            # (Note: table(token) is efficient; we don't worry about embedding 'wrong' tokens 
            # because they get multiplied by 0.0 in the next step)
            embeddings = table(tokens)
            
            # 3. Add to output where mask is True
            output = output + (embeddings * m)
            
        return output
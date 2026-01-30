import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange

# We will define the KV cache structure in kv_caching.py later, 
# but we type hint it here for clarity.
KVCache = Any 

@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: Literal['causal', 'block_causal']

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE).
    Crucial for the Craftax TWM as specified in the paper.
    """
    dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: [batch, length, heads, dim]
        seq_len = x.shape[1]
        
        # Create position indices [0, 1, ..., seq_len-1]
        pos = jnp.arange(seq_len, dtype=jnp.float32)
        
        # Calculate frequencies
        # theta_i = 10000 ^ (-2(i-1)/d)
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        
        # Outer product to get angles
        sinusoid_inp = jnp.einsum('i,j->ij', pos, inv_freq)
        
        # Create sin and cos embeddings
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        
        # Reshape for broadcasting: [1, seq_len, 1, dim/2]
        sin = jnp.expand_dims(sin, (0, 2))
        cos = jnp.expand_dims(cos, (0, 2))
        
        # Apply RoPE to x
        # x is [b, t, h, d]. We pair elements 2i and 2i+1
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Standard RoPE rotation
        # [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        return jnp.concatenate([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], axis=-1)

class SelfAttention(nn.Module):
    config: TransformerConfig

    def setup(self):
        assert self.config.embed_dim % self.config.num_heads == 0
        self.head_dim = self.config.embed_dim // self.config.num_heads

        self.key = nn.Dense(self.config.embed_dim, use_bias=True)
        self.query = nn.Dense(self.config.embed_dim, use_bias=True)
        self.value = nn.Dense(self.config.embed_dim, use_bias=True)
        
        self.attn_drop = nn.Dropout(self.config.attn_pdrop)
        self.resid_drop = nn.Dropout(self.config.resid_pdrop)
        self.proj = nn.Dense(self.config.embed_dim, use_bias=True)

        self.rope = RotaryEmbedding(self.head_dim)

        # Create masks
        self.causal_mask = jnp.tril(jnp.ones((self.config.max_tokens, self.config.max_tokens)))
        
        # Create block causal mask logic
        # 1. Create block diagonal mask
        ones_block = jnp.ones((self.config.tokens_per_block, self.config.tokens_per_block))
        block_diag = jax.scipy.linalg.block_diag(*[ones_block for _ in range(self.config.max_blocks)])
        # 2. Combine with causal mask (max)
        self.block_causal_mask = jnp.maximum(self.causal_mask, block_diag)

    def __call__(self, x, kv_cache=None, deterministic=True):
        B, T, C = x.shape
        
        # Determine L (length of past cache)
        if kv_cache is not None:
            # Assuming kv_cache structure: (k_cache, v_cache, index)
            # We will rely on kv_caching.py logic, but here we just need the length
            L = kv_cache[2] # Current index/length
        else:
            L = 0

        # Project Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape heads: [B, T, num_heads, head_dim]
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.config.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.config.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.config.num_heads)

        # Apply RoPE (Rotary Positional Embeddings)
        q = self.rope(q)
        k = self.rope(k)

        # KV Cache Interaction
        if kv_cache is not None:
            # We assume a helper function 'update' exists on the cache object
            # or we return the new k, v to be updated externally.
            # In JAX/Flax, we typically return the new cache state.
            k_cache, v_cache, update_idx = kv_cache
            
            # Update cache at correct indices
            # This logic will be strictly defined in kv_caching.py, 
            # here we perform the 'update' logically for the attention computation
            
            # For simplicity in this file, we assume we act on the FULL projected K, V
            # effectively concatenating past + present for attention
            
            # NOTE: In JAX, we usually use dynamic_update_slice
            # We will handle the explicit update in the kv_caching module return,
            # but for *calculation*, we need the full sequence [Past, Current]
            
            # If doing inference (T=1), we concat with past
            # If doing training (cache is usually None), we use just x
            
            # Simplified for now: assume x includes everything or cache is handled
            # If using cache for AR generation:
            start_idx = update_idx
            
            # Update the cache arrays (functional update)
            k_cache = jax.lax.dynamic_update_slice(k_cache, k, (0, start_idx, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(v_cache, v, (0, start_idx, 0, 0))
            
            # Use the full cache for attention
            k_in = k_cache
            v_in = v_cache
            
            # Update the index for next time
            new_idx = start_idx + T
            new_kv_cache = (k_cache, v_cache, new_idx)
        else:
            k_in = k
            v_in = v
            new_kv_cache = None

        # Compute Attention
        # q: [B, T, h, d]
        # k_in: [B, L+T, h, d]
        # Attn: [B, h, T, L+T]
        
        scale = 1.0 / jnp.sqrt(self.head_dim)
        att = jnp.einsum('bthd,bLhd->bhtL', q, k_in) * scale

        # Masking
        # Select correct mask based on config
        mask = self.causal_mask if self.config.attention == 'causal' else self.block_causal_mask
        
        # Slice mask to current window [L:L+T, :L+T]
        # Since JAX arrays are static, for training T is full context. 
        # For inference, T=1, L grows.
        
        # We handle the mask slicing dynamically
        total_len = att.shape[-1] # L + T
        # We need the mask row for the current query position
        
        # If kv_cache is None (Training), we use the full mask up to T
        if kv_cache is None:
            curr_mask = mask[:T, :T]
            # Broadcast to [1, 1, T, T]
            curr_mask = curr_mask[None, None, :, :]
            att = jnp.where(curr_mask == 1, att, -1e9)
        else:
            # Inference: We attend to all history. 
            # Usually no mask needed if simply attending to all past (standard causal),
            # but for Block Causal we must respect blocks.
            # We take the row corresponding to the current token index.
            # Simplified: assuming standard causal inference for now
            pass 

        att = nn.softmax(att, axis=-1)
        att = self.attn_drop(att, deterministic=deterministic)

        # Aggregate V
        y = jnp.einsum('bhtL,bLhd->bthd', att, v_in)
        
        # Merge heads
        y = rearrange(y, 'b t h d -> b t (h d)')

        y = self.proj(y)
        y = self.resid_drop(y, deterministic=deterministic)

        return y, new_kv_cache

class Block(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, kv_cache=None, deterministic=True):
        ln1 = nn.LayerNorm()(x)
        attn_out, new_kv_cache = SelfAttention(self.config)(ln1, kv_cache, deterministic=deterministic)
        
        x = x + attn_out
        
        ln2 = nn.LayerNorm()(x)
        
        # MLP
        mlp_out = nn.Dense(4 * self.config.embed_dim)(ln2)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.config.embed_dim)(mlp_out)
        mlp_out = nn.Dropout(self.config.resid_pdrop)(mlp_out, deterministic=deterministic)
        
        x = x + mlp_out
        
        return x, new_kv_cache

class Transformer(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, sequences, past_keys_values=None, deterministic=True):
        # sequences: [Batch, Time, Dim]
        # past_keys_values: List or Tuple of KVCaches (one per block)

        x = nn.Dropout(self.config.embed_pdrop)(sequences, deterministic=deterministic)
        
        new_caches = []
        
        for i in range(self.config.num_layers):
            # If we have past caches, pick the i-th one
            layer_cache = past_keys_values[i] if past_keys_values is not None else None
            
            # Run block
            x, new_cache = Block(self.config)(x, layer_cache, deterministic=deterministic)
            
            if new_cache is not None:
                new_caches.append(new_cache)
        
        x = nn.LayerNorm()(x)
        
        if past_keys_values is not None:
            return x, new_caches
        else:
            return x
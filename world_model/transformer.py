# transformer.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass
from typing import Optional, Tuple, List
from functools import partial

# Import the functional KV cache structures and functions
from kv_caching import KVCacheState, create_keys_values, update_kv_cache

# --- Configuration ---

@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    tokens_per_block: int
    max_blocks: int
    attention: str # 'causal' or 'block_causal'

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self) -> int:
        return self.tokens_per_block * self.max_blocks

# --- Model Components ---

class SelfAttention(nn.Module):
    """Self-Attention block with functional KV Caching."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, kv_cache: Optional[KVCacheState] = None, *, train: bool) -> Tuple[jnp.ndarray, Optional[KVCacheState]]:
        B, T, C = x.shape
        n_head, n_embd = self.config.num_heads, self.config.embed_dim
        head_size = C // n_head
        
        # Projections for Q, K, V
        qkv = nn.Dense(3 * n_embd, name="qkv_proj")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, n_head, head_size).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_head, head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_head, head_size).transpose(0, 2, 1, 3)
        
        # Handle KV Cache for efficient inference
        new_kv_cache = None
        current_seq_len = 0
        if kv_cache is not None:
            current_seq_len = kv_cache.k.size
            new_kv_cache = update_kv_cache(kv_cache, k, v)
            k, v = new_kv_cache.k.data, new_kv_cache.v.data

        # Attention calculation
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(head_size))
        
        # Create mask on the fly
        L = k.shape[2]
        if self.config.attention == 'causal':
            mask = nn.make_causal_mask(jnp.ones((1, L)))
        elif self.config.attention == 'block_causal':
            causal_mask = jnp.tril(jnp.ones((L, L)))
            block_diag_mask = jax.scipy.linalg.block_diag(
                *[jnp.ones((self.config.tokens_per_block, self.config.tokens_per_block)) for _ in range(self.config.max_blocks)]
            )
            full_block_mask = block_diag_mask[:L, :L]
            mask = jnp.maximum(causal_mask, full_block_mask)
        
        # Apply mask to the relevant slice of the attention matrix
        mask_slice = jax.lax.dynamic_slice(mask, (current_seq_len, 0), (T, L))
        att = jnp.where(mask_slice, att, -jnp.inf)
        
        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(self.config.attn_pdrop)(att, deterministic=not train)
        
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        y = nn.Dense(n_embd, name="out_proj")(y)
        y = nn.Dropout(self.config.resid_pdrop)(y, deterministic=not train)
        
        return y, new_kv_cache

class MLP(nn.Module):
    """Standard MLP for the Transformer block."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = nn.Dense(4 * self.config.embed_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.embed_dim)(x)
        x = nn.Dropout(self.config.resid_pdrop)(x, deterministic=not train)
        return x

class Block(nn.Module):
    """A single Transformer block."""
    config: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, kv_cache: Optional[KVCacheState] = None, *, train: bool) -> Tuple[jnp.ndarray, Optional[KVCacheState]]:
        attn_out, new_kv_cache = SelfAttention(self.config, name="attention")(
            nn.LayerNorm()(x), kv_cache=kv_cache, train=train
        )
        x = x + attn_out
        x = x + MLP(self.config, name="mlp")(nn.LayerNorm()(x), train=train)
        return x, new_kv_cache

# --- Main Transformer Model ---

class Transformer(nn.Module):
    """The main Transformer model, integrated with functional KV Caching."""
    config: TransformerConfig

    def setup(self):
        # We don't need a top-level dropout on embeddings as per the original structure
        self.blocks = [
            Block(self.config, name=f'block_{i}') for i in range(self.config.num_layers)
        ]
        self.ln_f = nn.LayerNorm(name="final_layernorm")

        self.embed_drop = nn.Dropout(self.config.embed_pdrop)
        
    def __call__(self, sequences: jnp.ndarray, past_keys_values: Optional[List[KVCacheState]] = None, *, train: bool) -> Tuple[jnp.ndarray, Optional[List[KVCacheState]]]:
        
        # Initial embedding dropout is often applied here
        x = self.embed_drop(sequences, deterministic=not train)

        new_keys_values = [] if past_keys_values is not None else None

        for i, block in enumerate(self.blocks):
            cache = past_keys_values[i] if past_keys_values is not None else None
            x, new_cache = block(x, kv_cache=cache, train=train)
            if new_keys_values is not None:
                new_keys_values.append(new_cache)
        
        x = self.ln_f(x)
        
        return x, new_keys_values
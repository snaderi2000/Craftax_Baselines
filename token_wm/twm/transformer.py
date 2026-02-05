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
    
    IMPORTANT: Must pass start_pos for KV caching to work correctly!
    """
    dim: int

    @nn.compact
    def __call__(self, x, start_pos: int = 0):
        # x shape: [batch, length, heads, dim]
        seq_len = x.shape[1]
        
        # Create position indices starting from start_pos
        # This is CRITICAL for KV caching - new tokens need absolute positions!
        pos = jnp.arange(seq_len, dtype=jnp.float32) + start_pos
        
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

        # Create masks (unchanged)
        self.causal_mask = jnp.tril(jnp.ones((self.config.max_tokens, self.config.max_tokens)))
        ones_block = jnp.ones((self.config.tokens_per_block, self.config.tokens_per_block))
        block_diag = jax.scipy.linalg.block_diag(*[ones_block for _ in range(self.config.max_blocks)])
        self.block_causal_mask = jnp.maximum(self.causal_mask, block_diag)

    def __call__(self, x, kv_cache=None, deterministic=True):
        B, T, C = x.shape
        
        # 1. Project Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 2. Reshape heads: [B, T, num_heads, head_dim]
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.config.num_heads)
        k = rearrange(k, 'b t (h d) -> b t h d', h=self.config.num_heads)
        v = rearrange(v, 'b t (h d) -> b t h d', h=self.config.num_heads)

        # 4. KV Cache Interaction - need to get start_pos BEFORE applying RoPE
        if kv_cache is not None:
            start_pos = kv_cache.index
        else:
            start_pos = 0

        # 3. Apply RoPE with correct absolute positions
        # This is CRITICAL for generation - tokens must know their absolute position!
        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)

        # 4. KV Cache Update
        if kv_cache is not None:
            # Update cache using the class method
            new_kv_cache = kv_cache.update(k, v)
            
            # Use full history for attention
            k_in = new_kv_cache.key
            v_in = new_kv_cache.value
            # start_pos already set above for RoPE
        else:
            k_in = k
            v_in = v
            new_kv_cache = None
            start_pos = 0

        # 5. Compute Attention
        # q: [B, T, H, D]
        # k_in: [B, L_total, H, D]
        scale = 1.0 / jnp.sqrt(self.head_dim)
        
        # Einstein Summation for Dot Product Attention
        # b: batch, t: query_time, l: key_time, h: heads, d: head_dim
        att = jnp.einsum('bthd,blhd->bhtl', q, k_in) * scale

        # 6. Masking
        # We need to slice the global mask to match the current [start_pos : start_pos+T] window
        global_mask = self.causal_mask if self.config.attention == 'causal' else self.block_causal_mask
        
        # Slice the mask:
        # Rows (Query): start_pos to start_pos + T
        # Cols (Key):   0 to start_pos + T (assuming k_in is the full buffer or valid slice)
        
        # Note: If k_in is the FULL buffer (max_tokens), we just take the first L+T columns
        key_len = k_in.shape[1]
        
        # Safe slicing for dynamic shapes (JAX prefers fixed, but we use limits)
        # For training, start_pos=0, T=Max. For inference, T=1.
        
        # Create mask based on positions
        # Query positions: range(start_pos, start_pos + T)
        # Key positions:   range(0, key_len)
        q_idx = jnp.arange(T) + start_pos
        k_idx = jnp.arange(key_len)
        
        # Broadcast to create grid
        # mask_val[i, j] is True if q_i can attend to k_j
        # We assume global_mask is [Max, Max]. We gather relevant rows/cols.
        
        # Simple Logic: 
        # Causal: q_idx >= k_idx
        # We can just use the pre-computed global mask if we slice it correctly.
        
        # Extract the relevant sub-block from the global mask
        # We use jnp.take to gather rows/cols
        # (Batching this might be tricky, but usually masks are constant)
        
        # Efficient Masking:
        # mask_slice = global_mask[start_pos : start_pos+T, :key_len]
        # We use dynamic_slice for JIT compatibility if needed, or simple slicing if shapes are static enough.
        
        # Let's use simple logic valid for both Training and Inference:
        # 1. Expand dims for broadcasting: (1, 1, T, L)
        # 2. Use large negative number for masking
        
        # Construct mask dynamically to avoid slicing issues:
        q_idx_b = q_idx[:, None] # (T, 1)
        k_idx_b = k_idx[None, :] # (1, L)
        
        if self.config.attention == 'causal':
             mask_bool = q_idx_b >= k_idx_b
        else:
             # Block Causal: (q_idx >= k_idx) OR (same_block)
             # Same block means floor(q/block_size) == floor(k/block_size)
             # This is expensive to recompute. Let's trust the global mask slice.
             pass 

        # Using the pre-computed mask is safest for Block Causal logic.
        # We slice the global mask.
        mask_slice = jax.lax.dynamic_slice(
            global_mask,
            (start_pos, 0),
            (T, key_len)
        )
        
        # Apply Mask (where mask is 0, set to -inf)
        att = jnp.where(mask_slice[None, None, :, :] > 0, att, -1e9)

        # 7. Softmax & Weighted Sum
        att = nn.softmax(att, axis=-1)
        att = self.attn_drop(att, deterministic=deterministic)
        
        # y = att @ v_in
        # bhtl, blhd -> bthd
        y = jnp.einsum('bhtl,blhd->bthd', att, v_in)
        
        # 8. Output Projection
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
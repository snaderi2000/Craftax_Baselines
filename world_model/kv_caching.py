# kv_caching.py

from typing import List, Tuple
import jax
import jax.numpy as jnp
from flax.struct import dataclass

# In JAX, we define the state of the cache as an explicit dataclass.
# This makes it a "Pytree," which JAX functions (like jit) can work with.

@dataclass
class CacheState:
    """Holds the data and current size for a single cache (e.g., for keys)."""
    data: jnp.ndarray
    size: int

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

@dataclass
class KVCacheState:
    """Holds the CacheState for both keys and values for a single attention layer."""
    k: CacheState
    v: CacheState

# --- Helper Functions to Manipulate Cache State ---
# Instead of methods that modify `self`, we use pure functions that
# take a state as input and return a new state as output.

def create_empty_cache(batch_size: int, num_heads: int, max_tokens: int, head_dim: int) -> CacheState:
    """Creates a new, empty cache state."""
    shape = (batch_size, num_heads, max_tokens, head_dim)
    return CacheState(data=jnp.zeros(shape), size=0)

def get_from_cache(cache: CacheState) -> jnp.ndarray:
    """Extracts the valid part of the cache data."""
    # Use jax.lax.dynamic_slice for jit-compatibility
    return jax.lax.dynamic_slice(
        cache.data,
        (0, 0, 0, 0),
        (cache.shape[0], cache.shape[1], cache.size, cache.shape[3])
    )

def update_cache(cache: CacheState, x: jnp.ndarray) -> CacheState:
    """
    Updates the cache with new data (e.g., new keys or values).
    This is the functional equivalent of the PyTorch `update` method.
    """
    batch_size, n_head, seq_len, head_dim = x.shape
    
    # Use dynamic_update_slice, the JAX equivalent of in-place assignment
    new_data = jax.lax.dynamic_update_slice(
        cache.data,
        x,
        (0, 0, cache.size, 0)
    )
    
    return cache.replace(data=new_data, size=cache.size + seq_len)

def prune_cache(cache: CacheState, mask: jnp.ndarray) -> CacheState:
    """Prunes the cache along the batch dimension based on a boolean mask."""
    new_data = cache.data[mask]
    return cache.replace(data=new_data)

# --- Top-Level Functions for Managing KV Caches Across Layers ---

def create_kv_cache(batch_size: int, num_heads: int, max_tokens: int, embed_dim: int) -> KVCacheState:
    """Creates an empty Key-Value cache state for one layer."""
    assert embed_dim % num_heads == 0
    head_dim = embed_dim // num_heads
    
    k_cache = create_empty_cache(batch_size, num_heads, max_tokens, head_dim)
    v_cache = create_empty_cache(batch_size, num_heads, max_tokens, head_dim)
    return KVCacheState(k=k_cache, v=v_cache)

def update_kv_cache(kv_cache: KVCacheState, k: jnp.ndarray, v: jnp.ndarray) -> KVCacheState:
    """Updates the Key-Value cache for one layer."""
    new_k_cache = update_cache(kv_cache.k, k)
    new_v_cache = update_cache(kv_cache.v, v)
    return KVCacheState(k=new_k_cache, v=new_v_cache)

def create_keys_values(n: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int) -> List[KVCacheState]:
    """Creates a list of empty KV Caches, one for each transformer layer."""
    return [
        create_kv_cache(n, num_heads, max_tokens, embed_dim)
        for _ in range(num_layers)
    ]

def prune_keys_values(keys_values: List[KVCacheState], mask: jnp.ndarray) -> List[KVCacheState]:
    """Prunes the list of KV Caches."""
    return [
        KVCacheState(k=prune_cache(kv.k, mask), v=prune_cache(kv.v, mask))
        for kv in keys_values
    ]
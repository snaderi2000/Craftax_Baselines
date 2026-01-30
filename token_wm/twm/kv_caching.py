import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple, List, Union

@struct.dataclass
class KVCache:
    """
    JAX replacement for the PyTorch Cache and KVCache classes.
    Stores Key and Value buffers for a single layer.
    """
    key: jnp.ndarray   # Shape: (Batch, MaxLen, NumHeads, HeadDim)
    value: jnp.ndarray # Shape: (Batch, MaxLen, NumHeads, HeadDim)
    index: jnp.int32   # Scalar integer tracking current position

    @classmethod
    def init(cls, batch_size, num_heads, max_tokens, embed_dim, dtype=jnp.float32):
        head_dim = embed_dim // num_heads
        
        # Note: We use (Batch, MaxLen, NumHeads, HeadDim) layout 
        # to match the JAX Transformer implementation efficiently.
        k_init = jnp.zeros((batch_size, max_tokens, num_heads, head_dim), dtype=dtype)
        v_init = jnp.zeros((batch_size, max_tokens, num_heads, head_dim), dtype=dtype)
        idx_init = jnp.array(0, dtype=jnp.int32)
        
        return cls(key=k_init, value=v_init, index=idx_init)

    def update(self, k_new: jnp.ndarray, v_new: jnp.ndarray):
        """
        Writes new k/v into the buffer at 'self.index'.
        Returns a NEW KVCache object with updated buffers.
        """
        # k_new shape: (Batch, Time, NumHeads, HeadDim)
        start_idx = self.index
        
        # JAX functional update: array[idx:idx+len] = new_val
        new_key = jax.lax.dynamic_update_slice(self.key, k_new, (0, start_idx, 0, 0))
        new_value = jax.lax.dynamic_update_slice(self.value, v_new, (0, start_idx, 0, 0))
        
        new_index = start_idx + k_new.shape[1]
        
        return self.replace(key=new_key, value=new_value, index=new_index)

    def prune(self, mask: jnp.ndarray):
        """
        Filters the batch dimension based on a boolean mask.
        NOTE: This changes the shape of the arrays. 
        Do not use inside a JIT-compiled loop that expects fixed shapes.
        """
        # mask shape: (Batch,)
        new_key = self.key[mask]
        new_value = self.value[mask]
        return self.replace(key=new_key, value=new_value)

    def reset(self):
        """Resets index to 0. Does not clear memory (optimization)."""
        return self.replace(index=jnp.array(0, dtype=jnp.int32))

    # Compatibility: Allow unpacking like a tuple (key, value, index)
    def __iter__(self):
        return iter((self.key, self.value, self.index))

class KeysValues(struct.PyTreeNode):
    """
    Collection of KVCaches for all Transformer layers.
    Matches the PyTorch KeysValues class structure.
    """
    layers: List[KVCache]

    @classmethod
    def init(cls, n, num_heads, max_tokens, embed_dim, num_layers):
        layers = [
            KVCache.init(n, num_heads, max_tokens, embed_dim) 
            for _ in range(num_layers)
        ]
        return cls(layers=layers)
    
    def __getitem__(self, i):
        return self.layers[i]
    
    def __len__(self):
        return len(self.layers)

    def reset(self):
        return self.replace(layers=[l.reset() for l in self.layers])
        
    def prune(self, mask):
        return self.replace(layers=[l.prune(mask) for l in self.layers])
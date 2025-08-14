import sys
import pathlib

# Add submodule to path
submodule_path = pathlib.Path(__file__).resolve().parent / "third_party" / "nanoGPT-jax"
sys.path.append(str(submodule_path))

from model import GPTConfig, GPT

print("✅ Successfully imported GPT from nanoGPT-jax")

import jax
import jax.numpy as jnp

# Config: vocab=520, seq_len=10
conf = GPTConfig(vocab_size=520, block_size=10, n_layer=2, n_head=2, n_embd=32)
model = GPT(conf)

# Dummy tokens
B, S = 4, 10
dummy_input = jnp.ones((B, S), dtype=jnp.int32)

# Init params
rng = jax.random.PRNGKey(0)
params = model.init(rng, dummy_input)

# Forward
logits, _ = model.apply(params, dummy_input)

print("Logits shape:", logits.shape)  # Expect (B, S, vocab_size)

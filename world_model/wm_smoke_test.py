import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

# Import all the modules and functions we've built
from tokenizer import Tokenizer, EncoderDecoderConfig
from transformer import TransformerConfig
from world_model import WorldModel, compute_wm_loss

def run_world_model_smoke_test():
    print("🚀 Starting World Model smoke test...")

    # --- 1. Define Hyperparameters ---
    # These values are based on the paper for the MBRL setup
    batch_size = 4
    seq_len = 20 # T_WM, the rollout horizon for the TWM
    obs_res = 63
    tokens_per_obs = 256 # L = 256 for token-based models
    obs_vocab_size = 4096 # K = 4096 for models with NNT
    act_vocab_size = 17 # For Craftax-classic
    embed_dim = 128
    
    # --- 2. Generate Fake Data ---
    print("🎲 Generating fake data batch...")
    key = jax.random.PRNGKey(0)
    key, obs_key, act_key, reward_key, done_key = jax.random.split(key, 5)

    # Fake batch dictionary, mimicking what your replay buffer would provide
    fake_batch = {
        # Shape: (batch, seq_len, height, width, channels)
        'observations': jax.random.uniform(obs_key, (batch_size, seq_len, obs_res, obs_res, 3)),
        # Shape: (batch, seq_len)
        'actions': jax.random.randint(act_key, (batch_size, seq_len), 0, act_vocab_size),
        # Shape: (batch, seq_len)
        'rewards': jax.random.randint(reward_key, (batch_size, seq_len), -1, 2),
        # Shape: (batch, seq_len)
        'ends': jax.random.randint(done_key, (batch_size, seq_len), 0, 2).astype(jnp.bool_),
        # Shape: (batch, seq_len) - mask is False where there is padding
        'mask_padding': jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
    }
    print(f"✅ Fake data generated.")

    # --- 3. Initialize Models ---
    print("🧠 Initializing Tokenizer and World Model...")
    key, tokenizer_key, wm_key = jax.random.split(key, 3)

    # A simplified Tokenizer config
    tokenizer_config = EncoderDecoderConfig(
        resolution=obs_res, in_channels=3, z_channels=128, ch=64,
        ch_mult=[1, 2, 4], num_res_blocks=1, attn_resolutions=[],
        out_ch=3, dropout=0.0
    )
    tokenizer = Tokenizer(
        vocab_size=obs_vocab_size, embed_dim=embed_dim,
        encoder_config=tokenizer_config, decoder_config=tokenizer_config
    )

    # The Transformer config from the paper
    wm_config = TransformerConfig(
        tokens_per_block=tokens_per_obs + 1,
        max_blocks=seq_len,
        attention='block_causal',
        num_layers=3,
        num_heads=8,
        embed_dim=embed_dim,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    world_model = WorldModel(
        obs_vocab_size=obs_vocab_size,
        act_vocab_size=act_vocab_size,
        config=wm_config
    )

    # Initialize Tokenizer
    dummy_obs_batch_5d = fake_batch['observations']
    B, T, H, W, C = dummy_obs_batch_5d.shape
    dummy_obs_batch_4d = dummy_obs_batch_5d.reshape(B * T, H, W, C)
    tokenizer_params = tokenizer.init(tokenizer_key, dummy_obs_batch_4d, train=False)['params']

    # Initialize World Model with a correctly shaped dummy token sequence
    # Note: We get the shape from the tokenizer's output, making this robust.
    dummy_wm_input_shape = (batch_size, seq_len * (tokens_per_obs + 1))
    dummy_wm_input = jnp.zeros(dummy_wm_input_shape, dtype=jnp.int32)
    wm_params = world_model.init(wm_key, dummy_wm_input, train=False)['params']

    print("✅ Models initialized successfully.")

    # --- 4. Run One Training Step ---
    print("⚡ Running a single training step for the World Model...")
    key, dropout_key = jax.random.split(key)
    rngs = {'dropout': dropout_key}

    # Use jax.jit to compile the loss function for performance
    jit_loss_fn = jax.jit(partial(
        compute_wm_loss,
        world_model=world_model,
        tokenizer=tokenizer,
    ))
    
    try:
        loss_output = jit_loss_fn(
            world_model_params=wm_params,
            tokenizer_params=tokenizer_params,
            batch=fake_batch,
            rngs=rngs,
        )
        print("\n--- Smoke Test PASSED! ---")
        print(f"  Total Loss: {loss_output.total_loss:.4f}")
        print(f"  Observation Loss: {loss_output.loss_obs:.4f}")
        print(f"  Reward Loss: {loss_output.loss_rewards:.4f}")
        print(f"  Ends Loss: {loss_output.loss_ends:.4f}")
    except Exception as e:
        print("\n--- ❌ Smoke Test FAILED ---")
        print("An error occurred during the training step:")
        raise e

if __name__ == "__main__":
    run_world_model_smoke_test()
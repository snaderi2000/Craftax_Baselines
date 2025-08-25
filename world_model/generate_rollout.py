# generate_rollout.py (Corrected)

import jax
import jax.numpy as jnp
from flax import serialization
import matplotlib.pyplot as plt
import os

from tokenizer import Tokenizer, EncoderDecoderConfig
from transformer import TransformerConfig
from world_model import WorldModel
from sample import make_replay_samplers

def generate_and_visualize_rollout():
    print("🔬 Generating a rollout for visual inspection...")

    # --- 1. Configuration ---
    VAULT_UID = "my_first_buffer_run"
    MODEL_DIR = "./trained_models"
    seq_len = 20
    batch_size = 4
    obs_res = 63
    tokens_per_obs = 64
    obs_vocab_size = 512
    act_vocab_size = 17
    embed_dim = 128

    # --- 2. Load Data ---
    _, sample_val, _ = make_replay_samplers(
        vault_uid=VAULT_UID, T_wm=seq_len, batch_envs=batch_size
    )
    key = jax.random.PRNGKey(69)
    key, sample_key = jax.random.split(key)
    batch = sample_val(sample_key)
    print("✅ Sampled a validation batch.")

    # --- 3. Initialize Models and Load Parameters ---
    key, tokenizer_key, wm_key = jax.random.split(key, 3)

    tokenizer_config = EncoderDecoderConfig(
        resolution=obs_res, in_channels=3, z_channels=128, ch=64, ch_mult=[1, 2, 4, 8], 
        num_res_blocks=1, attn_resolutions=[], out_ch=3, dropout=0.0
    )
    tokenizer = Tokenizer(
        vocab_size=obs_vocab_size, embed_dim=embed_dim, 
        encoder_config=tokenizer_config, decoder_config=tokenizer_config
    )

    wm_config = TransformerConfig(
        tokens_per_block=tokens_per_obs + 1, max_blocks=seq_len, attention='block_causal',
        num_layers=3, num_heads=8, embed_dim=embed_dim,
        embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
    )
    world_model = WorldModel(obs_vocab_size, act_vocab_size, config=wm_config)

    with open(os.path.join(MODEL_DIR, "tokenizer_params_32_25k.msgpack"), "rb") as f:
        tokenizer_params = serialization.from_bytes(None, f.read())
    with open(os.path.join(MODEL_DIR, "world_model_params_32_25k.msgpack"), "rb") as f:
        wm_params = serialization.from_bytes(None, f.read())
    print("✅ Loaded trained parameters.")

    # --- 4. Generate the Imagined Rollout ---

    real_obs_sequence = batch['obs'][0]
    action_sequence = batch['actions'][0]

    @jax.jit
    def TWM_step(token_input, past_kv):
        # The WorldModel.__call__ already computes everything we need.
        wm_output, new_kv = world_model.apply(
            {'params': wm_params}, 
            token_input, 
            past_keys_values=past_kv,
            train=False
        )

        # Get the logits directly from the output object.
        logits = wm_output.logits_observations

        # Sample the next tokens (greedy argmax for simplicity)
        next_obs_tokens = jnp.argmax(logits, axis=-1)

        return next_obs_tokens, new_kv

    imagined_obs_sequence = []

    # Encode the first frame
    initial_frame = real_obs_sequence[0][None, ...]
    encode_output = tokenizer.apply(
        {'params': tokenizer_params}, initial_frame, train=False, method=tokenizer.encode
    )
    current_obs_tokens = encode_output.tokens

    # Decode it back to an image for the first frame
    first_reconstruction = tokenizer.apply(
        {'params': tokenizer_params}, encode_output.z_quantized, train=False, method=tokenizer.decode
    )
    imagined_obs_sequence.append(first_reconstruction[0])

    kv_cache = None

    print("🧠 Generating imagined rollout step-by-step...")
    for t in range(seq_len - 1):
        # Combine the observation and action tokens for this timestep
        action_token = action_sequence[t][None, None] # Add batch and token dims
        token_input_flat = jnp.concatenate([current_obs_tokens, action_token], axis=1)

        # Predict the next set of observation tokens
        next_obs_tokens, kv_cache = TWM_step(token_input_flat, kv_cache)

        # Decode the predicted tokens into an image
        reconstruction = tokenizer.apply(
            {'params': tokenizer_params},
            next_obs_tokens,
            method=Tokenizer.decode_from_tokens # Call our new method
        )

        imagined_obs_sequence.append(reconstruction[0])

        current_obs_tokens = next_obs_tokens

    # --- 5. Save the Visualization ---
    print("💾 Saving comparison to `rollout_comparison.png`...")
    fig, axes = plt.subplots(2, seq_len, figsize=(seq_len * 2, 4))
    for t in range(seq_len):
        axes[0, t].imshow(real_obs_sequence[t])
        axes[0, t].set_title(f"Real (t={t+1})")
        axes[0, t].axis('off')

        axes[1, t].imshow(jnp.clip(imagined_obs_sequence[t], 0, 1))
        axes[1, t].set_title(f"Imagined (t={t+1})")
        axes[1, t].axis('off')

    plt.tight_layout()
    plt.savefig("rollout_comparison.png", dpi=200)
    print("✅ Done.")


if __name__ == "__main__":
    generate_and_visualize_rollout()
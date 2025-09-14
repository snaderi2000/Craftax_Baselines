# generate_rollout.py (Corrected for Orbax Checkpoints)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import optax
from flax.training import train_state
from orbax.checkpoint import PyTreeCheckpointer

from .tokenizer import Tokenizer, EncoderDecoderConfig
from .transformer import TransformerConfig
from .world_model import WorldModel
from .sample import make_replay_samplers

# Copied from your ppo_m2.py to define the saved object's structure
class WMTrainState(train_state.TrainState):
    rng: jax.random.PRNGKey

def generate_and_visualize_rollout():
    print("🔬 Generating a rollout for visual inspection...")

    # --- 1. Configuration ---
    VAULT_UID = "chilla"
    
    # --- Point to your new checkpoints ---
    # Find your latest run ID in the wandb/ directory
    RUN_ID = "run-20250913_165616-cp2mk218"
    CHECKPOINT_STEP = 43  # The update step you want to visualize
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(script_dir, '..', 'wandb', RUN_ID, 'files')

    #MODEL_DIR = os.path.join(script_dir, '..', 'trained_models_orbax')


    seq_len = 16  # Must match the T_WM of the saved model
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
    print("🧠 Initializing model templates...")
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

    # --- Orbax loading logic ---
    orbax_checkpointer = PyTreeCheckpointer()
    
    # Create empty state templates for Orbax to restore into
    dummy_optimizer = optax.adam(1e-3)
    
    dummy_obs = jnp.zeros((1, obs_res, obs_res, 3))
    tokenizer_template_params = tokenizer.init(tokenizer_key, dummy_obs, train=False)['params']
    empty_tok_state = WMTrainState.create(
        apply_fn=tokenizer.apply, params=tokenizer_template_params, tx=dummy_optimizer, rng=jax.random.PRNGKey(0)
    )

    dummy_tokens = jnp.zeros((1, seq_len * (tokens_per_obs + 1)), dtype=jnp.int32)
    wm_template_params = world_model.init(wm_key, dummy_tokens, train=False)['params']
    empty_wm_state = WMTrainState.create(
        apply_fn=world_model.apply, params=wm_template_params, tx=dummy_optimizer, rng=jax.random.PRNGKey(0)
    )
    
    print(f"✅ Loading parameters from step {CHECKPOINT_STEP} of run {RUN_ID}...")
    
    tok_ckpt_path = os.path.join(MODEL_DIR, f"tokenizer_checkpoints/update_{CHECKPOINT_STEP}/")
    wm_ckpt_path = os.path.join(MODEL_DIR, f"wm_checkpoints/update_{CHECKPOINT_STEP}/")
    
    #tok_ckpt_path = os.path.join(MODEL_DIR, "tokenizer")
    #wm_ckpt_path = os.path.join(MODEL_DIR, "world_model")


    tokenizer_state = orbax_checkpointer.restore(os.path.abspath(tok_ckpt_path), item=empty_tok_state)
    wm_state = orbax_checkpointer.restore(os.path.abspath(wm_ckpt_path), item=empty_wm_state)
    
    tokenizer_params = tokenizer_state.params
    wm_params = wm_state.params
    print("✅ Loaded trained parameters.")

    # --- 4. Generate the Imagined Rollout ---
    real_obs_sequence = batch['obs'][0]
    action_sequence = batch['actions'][0]

    @jax.jit
    def TWM_step(token_input, past_kv):
        wm_output, new_kv = world_model.apply(
            {'params': wm_params}, 
            token_input, 
            past_keys_values=past_kv,
            train=False
        )
        logits = wm_output.logits_observations
        next_obs_tokens = jnp.argmax(logits, axis=-1)
        return next_obs_tokens, new_kv

    imagined_obs_sequence = []

    initial_frame = real_obs_sequence[0][None, ...]
    encode_output = tokenizer.apply(
        {'params': tokenizer_params}, initial_frame, train=False, method=tokenizer.encode
    )
    current_obs_tokens = encode_output.tokens

    first_reconstruction = tokenizer.apply(
        {'params': tokenizer_params}, encode_output.z_quantized, train=False, method=tokenizer.decode
    )
    imagined_obs_sequence.append(first_reconstruction[0])

    kv_cache = None

    print("🧠 Generating imagined rollout step-by-step...")
    for t in range(seq_len - 1):
        action_token = action_sequence[t][None, None]
        token_input_flat = jnp.concatenate([current_obs_tokens, action_token], axis=1)
        
        next_obs_tokens, kv_cache = TWM_step(token_input_flat, kv_cache)

        # To decode, we must first get the quantized vectors from the tokens
        codebook = tokenizer_params['embedding']['embedding']
        z_quantized_flat = codebook[next_obs_tokens]


        B, L, C = z_quantized_flat.shape
        H = W = int(L**0.5)
        
        z_quantized = z_quantized_flat.reshape(B, H, W, C)


        reconstruction = tokenizer.apply(
            {'params': tokenizer_params}, z_quantized, train=False, method=tokenizer.decode
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

# train.py (Modified to save with Orbax)

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial
import os

# --- 💡 ADDED IMPORT 💡 ---
from orbax.checkpoint import PyTreeCheckpointer

# Import all our modules and the data loader
from .sample import make_replay_samplers
from .tokenizer import Tokenizer, EncoderDecoderConfig, compute_loss as compute_tokenizer_loss
from .transformer import TransformerConfig
from .world_model import WorldModel, compute_wm_loss

# A class to manage the training state for a model
class TrainState(train_state.TrainState):
    rng: jax.random.PRNGKey

def create_train_state(module, params, learning_rate, rng):
    """Creates an initial TrainState."""
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        rng=rng,
    )

def main():
    print("🚀 Starting full training process...")
    
    # --- 1. Hyperparameters ---
    VAULT_UID = "chilla"
    
    # Training loop settings
    N_ITERS_TOK = 3000
    N_ITERS_TWM = 3000
    
    # Model and data settings
    LEARNING_RATE = 1e-3
    batch_size = 32
    seq_len = 16
    obs_res = 63
    tokens_per_obs = 64
    obs_vocab_size = 512
    act_vocab_size = 17
    embed_dim = 128

    # --- 2. Load Real Data ---
    print(f"🔄 Initializing data sampler for vault UID: {VAULT_UID}...")
    sample_train, _, _ = make_replay_samplers(
        vault_uid=VAULT_UID, T_wm=seq_len, batch_envs=batch_size
    )
    print("✅ Data sampler ready.")

    # --- 3. Initialize Models and TrainStates ---
    print("🧠 Initializing models and optimizers...")
    key = jax.random.PRNGKey(0)
    key, tokenizer_init_key, wm_init_key = jax.random.split(key, 3)

    tokenizer_config = EncoderDecoderConfig(
        resolution=obs_res, in_channels=3, z_channels=128, ch=64, ch_mult=[1, 2, 4, 8], 
        num_res_blocks=1, attn_resolutions=[], out_ch=3, dropout=0.0
    )
    tokenizer = Tokenizer(
        vocab_size=obs_vocab_size, embed_dim=embed_dim, 
        encoder_config=tokenizer_config, decoder_config=tokenizer_config
    )
    dummy_obs = jnp.zeros((batch_size * seq_len, obs_res, obs_res, 3))
    tokenizer_params = tokenizer.init(tokenizer_init_key, dummy_obs, train=False)['params']
    tokenizer_state = create_train_state(tokenizer, tokenizer_params, LEARNING_RATE, tokenizer_init_key)

    wm_config = TransformerConfig(
        tokens_per_block=tokens_per_obs + 1, max_blocks=seq_len, attention='block_causal',
        num_layers=3, num_heads=8, embed_dim=embed_dim,
        embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
    )
    world_model = WorldModel(obs_vocab_size, act_vocab_size, config=wm_config)
    dummy_tokens = jnp.zeros((batch_size, seq_len * (tokens_per_obs + 1)), dtype=jnp.int32)
    wm_params = world_model.init(wm_init_key, dummy_tokens, train=False)['params']
    wm_state = create_train_state(world_model, wm_params, LEARNING_RATE, wm_init_key)
    
    print("✅ Models initialized.")
    # (Removed the legacy resume logic for simplicity)

    # --- 4. Define JIT-compiled Training Steps ---
    @jax.jit
    def tokenizer_train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)

        def loss_fn(params):
            obs_5d = batch['observations']
            B, T, H, W, C = obs_5d.shape
            obs_4d = obs_5d.reshape(B * T, H, W, C)
            training_batch = batch.copy()
            training_batch['observations'] = obs_4d
            
            loss_info = compute_tokenizer_loss(
                model=tokenizer, 
                params=params, 
                batch=training_batch,
                rngs={'dropout': dropout_rng}
            )
            aux_losses = (
                loss_info.total_loss,
                loss_info.reconstruction_loss,
                loss_info.codebook_loss,
                loss_info.commitment_loss,
            )
            return loss_info.total_loss, aux_losses

        (loss, losses_tuple), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state.replace(rng=rng), losses_tuple

    @jax.jit
    def twm_train_step(wm_state, tokenizer_params, batch):
        rng, dropout_rng = jax.random.split(wm_state.rng)
        
        def loss_fn(params):
            loss_info = compute_wm_loss(
                world_model_params=params, tokenizer_params=tokenizer_params,
                world_model=world_model, tokenizer=tokenizer,
                batch=batch, rngs={'dropout': dropout_rng}
            )
            return loss_info.total_loss, loss_info
        
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(wm_state.params)
        new_wm_state = wm_state.apply_gradients(grads=grads)
        return new_wm_state.replace(rng=rng), loss_info

    # --- 5. Run Training Loops ---
    main_rng = jax.random.PRNGKey(42)

    print("\n--- Phase 1: Updating Tokenizer ---")
    for i in range(N_ITERS_TOK):
        main_rng, sample_key = jax.random.split(main_rng)
        batch = sample_train(sample_key)
        batch['observations'] = batch.pop('obs')

        tokenizer_state, tok_losses = tokenizer_train_step(tokenizer_state, batch)
        
        if (i + 1) % 50 == 0:
            total, recon, codebook, commit = tok_losses
            print(
                f"Step {i+1}/{N_ITERS_TOK} | "
                f"Total: {total:.4f}, "
                f"Recon: {recon:.4f}, "
                f"Codebook: {codebook:.4f}, "
                f"Commit: {commit:.4f}"
            )

    print("\n--- Phase 2: Updating Transformer World Model ---")
    frozen_tokenizer_params = jax.lax.stop_gradient(tokenizer_state.params)
    
    for i in range(N_ITERS_TWM):
        main_rng, sample_key = jax.random.split(main_rng)
        batch = sample_train(sample_key)
        
        batch['ends'] = batch.pop('dones')
        batch['observations'] = batch.pop('obs')
        batch['mask_padding'] = jnp.ones_like(batch['actions'], dtype=jnp.bool_)

        wm_state, loss_info = twm_train_step(wm_state, frozen_tokenizer_params, batch)

        if (i + 1) % 50 == 0:
            print(f"TWM training step {i+1}/{N_ITERS_TWM} | Total Loss: {loss_info.total_loss:.4f} | Obs Loss: {loss_info.loss_obs:.4f}")


    # --- 6. Save the Final Model Parameters with Orbax ---
    print("\n💾 Saving trained models with Orbax...")
    model_dir = "./trained_models_orbax"
    os.makedirs(model_dir, exist_ok=True)

    # Convert the relative path to an absolute path
    abs_model_dir = os.path.abspath(model_dir)

    orbax_checkpointer = PyTreeCheckpointer()

    # Save tokenizer state using the absolute path
    tok_path = os.path.join(abs_model_dir, "tokenizer")
    orbax_checkpointer.save(tok_path, tokenizer_state)

    # Save world model state using the absolute path
    wm_path = os.path.join(abs_model_dir, "world_model")
    orbax_checkpointer.save(wm_path, wm_state)

    print(f"\n✅ Training script finished. Models saved to {abs_model_dir}")		


if __name__ == "__main__":
    main()

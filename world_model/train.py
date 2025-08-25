# train.py

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import serialization
from functools import partial
import os

# Import all our modules and the data loader
from sample import make_replay_samplers
from tokenizer import Tokenizer, EncoderDecoderConfig, compute_loss as compute_tokenizer_loss
from transformer import TransformerConfig
from world_model import WorldModel, compute_wm_loss

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
    print("🚀 Starting full training process (Algorithm 5)...")
    TOKENIZER_CHECKPOINT = "./trained_models/tokenizer_params_32_25k.msgpack"
    WM_CHECKPOINT = "./trained_models/world_model_params_32_25k.msgpack"

    # --- 1. Hyperparameters ---
    VAULT_UID = "my_first_buffer_run"
    RESUME_TRAINING = True # Set to True to load a checkpoint
    MODEL_VERSION = "_2" # The suffix of the model you want to resume, e.g., "_2", "_3"
    # NT = f"./trained_models/tokenizer_params{MODEL_VERSION}.msgpack"
    # WM_CHECKPOINT = f"./trained_models/world_model_params{MODEL_VERSION}.msgpack"TOKENIZER_CHECKPOI

    
    # Training loop settings from the paper
    N_ITERS_TOK = 15000
    N_ITERS_TWM = 15000
    
    # Model and data settings
    LEARNING_RATE = 1e-3
    batch_size = 32
    seq_len = 20
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
    if RESUME_TRAINING:
        print(f"🔄 Resuming training from checkpoints with version '{MODEL_VERSION}'...")
        try:
            # Load Tokenizer parameters
            with open(TOKENIZER_CHECKPOINT, "rb") as f:
                tok_bytes = f.read()
                # Use the existing state to provide the structure for the loaded params
                loaded_tok_params = serialization.from_bytes(tokenizer_state.params, tok_bytes)
                # Replace the random params with the loaded ones
                tokenizer_state = tokenizer_state.replace(params=loaded_tok_params)

            # Load World Model parameters
            with open(WM_CHECKPOINT, "rb") as f:
                wm_bytes = f.read()
                loaded_wm_params = serialization.from_bytes(wm_state.params, wm_bytes)
                wm_state = wm_state.replace(params=loaded_wm_params)

            print(f"✅ Successfully loaded parameters.")
        except FileNotFoundError:
            print(f"⚠️ Checkpoint files not found. Starting training from scratch.")


    # --- 4. Define JIT-compiled Training Steps ---
    @jax.jit
    def tokenizer_train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)

        def loss_fn(params):
            # --- START of CHANGE ---
            # The batch has 5D observations: (B, T, H, W, C)
            obs_5d = batch['observations']
            B, T, H, W, C = obs_5d.shape

            # The Tokenizer's CNN expects a 4D batch of images: (B*T, H, W, C)
            obs_4d = obs_5d.reshape(B * T, H, W, C)

            # Create a new batch dictionary with the reshaped data
            training_batch = batch.copy()
            training_batch['observations'] = obs_4d
            # --- END of CHANGE ---

            loss_info = compute_tokenizer_loss(
                model=tokenizer, 
                params=params, 
                batch=training_batch, # Use the new batch with reshaped data
                rngs={'dropout': dropout_rng}
            )
            return loss_info.total_loss, loss_info.total_loss

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state.replace(rng=rng), loss


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

    # --- 5. Run Training Loops (Algorithm 5) ---
    main_rng = jax.random.PRNGKey(42)

    print("\n--- Phase 1: Updating Tokenizer ---")
    for i in range(N_ITERS_TOK):
        main_rng, sample_key = jax.random.split(main_rng)
        batch = sample_train(sample_key)
        batch['observations'] = batch.pop('obs') # Rename for consistency

        tokenizer_state, tok_loss = tokenizer_train_step(tokenizer_state, batch)
        
        if (i + 1) % 50 == 0:
            print(f"Tokenizer training step {i+1}/{N_ITERS_TOK} | Loss: {tok_loss:.4f}")

    print("\n--- Phase 2: Updating Transformer World Model ---")
    frozen_tokenizer_params = jax.lax.stop_gradient(tokenizer_state.params)
    
    for i in range(N_ITERS_TWM):
        main_rng, sample_key = jax.random.split(main_rng)
        batch = sample_train(sample_key)
        
        # Prepare batch for TWM loss function
        batch['ends'] = batch.pop('dones')
        batch['observations'] = batch.pop('obs')
        batch['mask_padding'] = jnp.ones_like(batch['actions'], dtype=jnp.bool_)

        wm_state, loss_info = twm_train_step(wm_state, frozen_tokenizer_params, batch)

        if (i + 1) % 50 == 0:
            print(f"TWM training step {i+1}/{N_ITERS_TWM} | Total Loss: {loss_info.total_loss:.4f} | Obs Loss: {loss_info.loss_obs:.4f}")

    # --- 6. Save the Final Model Parameters ---
    print("\n💾 Saving trained model parameters...")
    model_dir = "./trained_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save tokenizer
    tok_bytes = serialization.to_bytes(tokenizer_state.params)
    with open(os.path.join(model_dir, "tokenizer_params_32_35k.msgpack"), "wb") as f:
        f.write(tok_bytes)
        
    # Save world model
    wm_bytes = serialization.to_bytes(wm_state.params)
    with open(os.path.join(model_dir, "world_model_params_32_35k.msgpack"), "wb") as f:
        f.write(wm_bytes)

    print("\n✅ Training script finished.")

if __name__ == "__main__":
    main()
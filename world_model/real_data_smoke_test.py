# real_data_smoke_test.py

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial

# Import your data loader and all translated modules
from sample import make_replay_samplers
from tokenizer import Tokenizer, EncoderDecoderConfig, TokenizerEncoderOutput, LossWithIntermediateLosses, compute_loss
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

def run_real_data_smoke_test():
    print("🚀 Starting smoke test with REAL data...")
    
    # --- 1. Hyperparameters ---
    VAULT_UID = "my_first_buffer_run"  # IMPORTANT: Change this to your vault's UID
    
    seq_len = 20    # T_WM from the paper
    batch_size = 4 # N_env from the paper
    obs_res = 63
    tokens_per_obs = 256
    obs_vocab_size = 4096
    act_vocab_size = 17
    embed_dim = 128
    learning_rate = 1e-3

    # --- 2. Load Real Data Sampler ---
    print(f"🔄 Initializing data sampler for vault UID: {VAULT_UID}...")
    try:
        sample_train, _, _ = make_replay_samplers(
            vault_uid=VAULT_UID,
            T_wm=seq_len,
            batch_envs=batch_size,
        )
    except FileNotFoundError:
        print(f"❌ ERROR: Vault '{VAULT_UID}' not found.")
        print("Please ensure the UID is correct and the vault is in the expected directory.")
        return
    print("✅ Data sampler ready.")

    # --- 3. Initialize Models and TrainStates ---
    print("🧠 Initializing models and optimizers...")
    key = jax.random.PRNGKey(0)
    key, tokenizer_init_key, wm_init_key, sample_key = jax.random.split(key, 4)

    # Initialize Tokenizer
    tokenizer_config = EncoderDecoderConfig(
        resolution=obs_res, in_channels=3, z_channels=128, ch=64, ch_mult=[1, 2, 4], 
        num_res_blocks=1, attn_resolutions=[], out_ch=3, dropout=0.0
    )
    tokenizer = Tokenizer(
        vocab_size=obs_vocab_size, embed_dim=embed_dim, 
        encoder_config=tokenizer_config, decoder_config=tokenizer_config
    )
    dummy_obs = jnp.zeros((batch_size * seq_len, obs_res, obs_res, 3))
    tokenizer_params = tokenizer.init(tokenizer_init_key, dummy_obs, train=False)['params']
    tokenizer_state = create_train_state(tokenizer, tokenizer_params, learning_rate, tokenizer_init_key)

    # Initialize World Model
    wm_config = TransformerConfig(
        tokens_per_block=tokens_per_obs + 1, max_blocks=seq_len, attention='block_causal',
        num_layers=3, num_heads=8, embed_dim=embed_dim,
        embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1,
    )
    world_model = WorldModel(obs_vocab_size, act_vocab_size, config=wm_config)
    dummy_tokens = jnp.zeros((batch_size, seq_len * (tokens_per_obs + 1)), dtype=jnp.int32)
    wm_params = world_model.init(wm_init_key, dummy_tokens, train=False)['params']
    wm_state = create_train_state(world_model, wm_params, learning_rate, wm_init_key)
    print("✅ Models initialized.")

    # --- 4. Define JIT-compiled Training Step ---
    @jax.jit
    def twm_train_step(wm_state, tokenizer_params, batch):
        rng, dropout_rng = jax.random.split(wm_state.rng)
        
        def loss_fn(params):
            loss_info = compute_wm_loss(
                world_model_params=params,
                tokenizer_params=tokenizer_params,
                world_model=world_model,
                tokenizer=tokenizer,
                batch=batch,
                rngs={'dropout': dropout_rng}
            )
            return loss_info.total_loss, loss_info
        
        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(wm_state.params)
        new_wm_state = wm_state.apply_gradients(grads=grads)
        return new_wm_state.replace(rng=rng), loss_info

    # --- 5. Run a Single Step with Real Data ---
    print("⚡ Sampling a real batch and running one training step...")
    
    # Sample a batch from your dataset
    batch = sample_train(sample_key)
    
    # The `ends` key from your data loader is named `dones`. Let's rename it.
    batch['ends'] = batch.pop('dones')
    
    # Your data loader provides 'obs', but the loss function expects 'observations'.
    batch['observations'] = batch.pop('obs')
    
    # The loss function also needs a mask_padding key. We'll assume no padding for this test.
    batch['mask_padding'] = jnp.ones_like(batch['actions'], dtype=jnp.bool_)

    try:
        # Run the training step
        updated_wm_state, loss_output = twm_train_step(
            wm_state, 
            tokenizer_state.params, 
            batch
        )

        print("\n--- ✅ Smoke Test PASSED! ---")
        print("Successfully ran one training step on a real data batch.")
        print(f"  Total Loss: {loss_output.total_loss:.4f}")
        print(f"  Observation Loss: {loss_output.loss_obs:.4f}")
        print(f"  Reward Loss: {loss_output.loss_rewards:.4f}")
        print(f"  Ends Loss: {loss_output.loss_ends:.4f}")
        print("\nReady to start full training loop.")

    except Exception as e:
        print("\n--- ❌ Smoke Test FAILED ---")
        print("An error occurred during the training step:")
        raise e

if __name__ == "__main__":
    run_real_data_smoke_test()
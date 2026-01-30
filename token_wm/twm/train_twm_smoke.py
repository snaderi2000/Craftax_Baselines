import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
from PIL import Image
import functools    
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../tokenizer'))    

# --- Import your modules ---
from vqvae import VQVAE
from world_model import WorldModel
from transformer import TransformerConfig
from kv_caching import KeysValues

# --- Configuration ---
DATA_PATH = "../../replay_data/my_buffer/replay_data.npz"
VQVAE_PARAMS_PATH = "../tokenizer/vqvae_params.pkl"
TWM_SAVE_PATH = "twm_params.pkl"
OUTPUT_DIR = "twm_results"

# Paper Config
BATCH_SIZE = 32          # Smaller for smoke test
SEQ_LEN = 20             # T_WM = 20 steps
EPOCHS = 5               # Fast training to verify logic
LR = 1e-3
TOKENS_PER_BLOCK = 65    # 64 patches + 1 action
MAX_BLOCKS = SEQ_LEN     # 20 steps
EMBED_DIM = 128
NUM_LAYERS = 3
NUM_HEADS = 8
VOCAB_SIZE = 512
ACT_VOCAB_SIZE = 17      # Craftax actions

# --- 1. Data Loading & Tokenization ---

def load_and_tokenize():
    print("--- Phase 1: Loading & Tokenizing ---")
    
    # Load Raw Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cannot find data at {DATA_PATH}")
    data = np.load(DATA_PATH)
    obs = data['obs']        # (N, T, 63, 63, 3)
    actions = data['actions'] # (N, T)
    rewards = data['rewards'] # (N, T)
    dones = data['dones']     # (N, T)
    
    print(f"Loaded raw data: {obs.shape}")

    # Load VQ-VAE
    print("Loading VQ-VAE for tokenization...")
    with open(VQVAE_PARAMS_PATH, "rb") as f:
        vqvae_params = pickle.load(f)
    
    vqvae = VQVAE()
    
    # Jitted encoder
    @jax.jit
    def encode_batch(x):
        return vqvae.apply(vqvae_params, x, method=vqvae.encode_indices)

    # Tokenize in chunks to save VRAM
    num_eps, ep_len, h, w, c = obs.shape
    flat_obs = obs.reshape(-1, h, w, c)
    all_indices = []
    
    chunk_size = 128
    print("Tokenizing frames...")
    for i in tqdm(range(0, flat_obs.shape[0], chunk_size)):
        batch = jnp.array(flat_obs[i:i+chunk_size])
        idx = encode_batch(batch) # (B, 8, 8)
        # Flatten 8x8 -> 64 tokens
        idx = idx.reshape(idx.shape[0], -1)
        all_indices.append(np.array(idx))
    
    obs_tokens = np.concatenate(all_indices, axis=0)
    obs_tokens = obs_tokens.reshape(num_eps, ep_len, 64) # (N, T, 64)
    
    print(f"Tokenization complete. Shape: {obs_tokens.shape}")
    
    # CLEANUP: Free VQ-VAE memory!
    del vqvae
    del vqvae_params
    del obs
    jax.clear_caches()
    print("VQ-VAE memory freed.")
    
    return obs_tokens, actions, rewards, dones

# --- 2. TWM Training ---

def create_train_state(rng, config):
    model = WorldModel(
        obs_vocab_size=VOCAB_SIZE, 
        act_vocab_size=ACT_VOCAB_SIZE, 
        config=config
    )
    # Dummy input for init: [Batch, Time] (flat interleaved)
    # Length = SEQ_LEN * TOKENS_PER_BLOCK
    dummy_input = jnp.zeros((1, SEQ_LEN * TOKENS_PER_BLOCK), dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    
    tx = optax.adam(LR)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, dropout_rng):
    # WorldModel.compute_loss is a helper we added in world_model.py
    # It expects batch to contain: 'obs_tokens', 'actions', 'rewards', 'ends', 'mask_padding'
    
    def loss_fn(params):
        # We define the model blueprint
        # Ensure 'config' is available here. If strictly needed, pass it into train_step 
        # or access it from state.apply_fn.keywords if available.
        # For simplicity in this script, we can recreate the config or pull from global.
        model = WorldModel(
            obs_vocab_size=VOCAB_SIZE, 
            act_vocab_size=ACT_VOCAB_SIZE, 
            config=TransformerConfig(TOKENS_PER_BLOCK, MAX_BLOCKS, 'causal', NUM_LAYERS, NUM_HEADS, EMBED_DIM, 0.1, 0.1, 0.1)
        )
        
        # CORRECT CALL: Use .apply()
        # This binds 'params' to the model and then runs the 'method'
        loss_output = model.apply(
            params,             # The variables
            batch,                          # Arg 1 for compute_loss
            dropout_rng,                    # Arg 2 for compute_loss
            method=model.compute_loss,      # The function to run
            rngs={'dropout': dropout_rng}   # RNGs needed for Dropout
        )
        
        return loss_output.total_loss, loss_output

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics

def get_batch(obs_tokens, actions, rewards, dones, batch_size, seq_len):
    # Randomly sample trajectories and start indices
    num_eps, ep_len, _ = obs_tokens.shape
    
    # Valid start indices (must have room for seq_len)
    max_start = ep_len - seq_len
    
    ep_idxs = np.random.randint(0, num_eps, size=batch_size)
    start_idxs = np.random.randint(0, max_start, size=batch_size)
    
    batch_obs = []
    batch_act = []
    batch_rew = []
    batch_end = []
    
    for i, start in zip(ep_idxs, start_idxs):
        end = start + seq_len
        batch_obs.append(obs_tokens[i, start:end])
        batch_act.append(actions[i, start:end])
        batch_rew.append(rewards[i, start:end])
        batch_end.append(dones[i, start:end])
        
    return {
        'obs_tokens': jnp.array(np.stack(batch_obs)),   # (B, 20, 64)
        'actions': jnp.array(np.stack(batch_act)),      # (B, 20)
        'rewards': jnp.array(np.stack(batch_rew)),
        'ends': jnp.array(np.stack(batch_end)),
        'mask_padding': jnp.zeros((batch_size, seq_len), dtype=bool) # No padding for now
    }

# --- 3. Rollout & Viz ---

def run_rollout(state, initial_obs_tokens, action_sequence):
    """
    Autoregressive generation.
    initial_obs_tokens: (1, 64) - The starting frame
    action_sequence: (1, T) - The actions to take
    """
    print("Running TWM Rollout (Imagination)...")
    model = WorldModel(
        obs_vocab_size=VOCAB_SIZE, 
        act_vocab_size=ACT_VOCAB_SIZE, 
        config=TransformerConfig(TOKENS_PER_BLOCK, MAX_BLOCKS, 'causal', NUM_LAYERS, NUM_HEADS, EMBED_DIM, 0.1, 0.1, 0.1)
    )
    
    # 1. Setup KV Cache
    cache = KeysValues.init(n=1, num_heads=NUM_HEADS, max_tokens=SEQ_LEN*TOKENS_PER_BLOCK, 
                            embed_dim=EMBED_DIM, num_layers=NUM_LAYERS)
    
    # 2. Feed Initial Frame (Context)
    # Flat format: [O_1..O_64]
    # We feed these one by one or as a chunk to prime the cache
    curr_tokens = initial_obs_tokens.reshape(1, -1) # (1, 64)
    
    # To prime properly, we pass these through. 
    # BUT, we also need to append the first action to generate the NEXT frame.
    
    generated_frames_tokens = []
    
    # We will loop for the length of action_sequence
    T_rollout = action_sequence.shape[1]
    
    # Current input to model starts as the initial observation patches
    current_input_ids = curr_tokens # (1, 64)
    
    for t in range(T_rollout):
        # A. Process Observation Patches
        # This updates cache and lets model "see" the current state
        # In a real efficient loop we might do this differently, but for smoke test:
        # Pass obs tokens
        output, cache = model.apply(state.params, current_input_ids, past_keys_values=cache, deterministic=True)
        
        # B. Process Action
        # Now we feed the action token. The model output after this token 
        # is the prediction for the FIRST token of the NEXT observation.
        act_token = action_sequence[:, t:t+1] # (1, 1)
        output, cache = model.apply(state.params, act_token, past_keys_values=cache, deterministic=True)
        
        # C. Auto-regressive generation of the NEXT 64 observation tokens
        # The output from the action step contains the logit for the 1st patch of next frame
        next_frame_tokens = []
        
        # Get last logit (next token prediction)
        logits = output.logits_observations[:, -1, :] # (1, 512)
        next_token = jnp.argmax(logits, axis=-1).reshape(1, 1)
        next_frame_tokens.append(next_token)
        
        # Generate remaining 63 tokens
        for _ in range(63):
            # Feed the last generated token to get the next one
            output, cache = model.apply(state.params, next_token, past_keys_values=cache, deterministic=True)
            logits = output.logits_observations[:, -1, :]
            next_token = jnp.argmax(logits, axis=-1).reshape(1, 1)
            next_frame_tokens.append(next_token)
            
        # Stack this frame
        full_frame = jnp.concatenate(next_frame_tokens, axis=1) # (1, 64)
        generated_frames_tokens.append(full_frame)
        
        # This generated frame becomes the input for the next step (before the next action)
        current_input_ids = full_frame

    return jnp.stack(generated_frames_tokens, axis=1) # (1, T, 64)

def decode_and_viz(tokens, original_pixels, save_name):
    """
    Decodes tokens back to pixels using VQ-VAE and saves grid.
    tokens: (1, T, 64)
    """
    # Load VQ-VAE (Re-load, since we deleted it)
    with open(VQVAE_PARAMS_PATH, "rb") as f:
        vqvae_params = pickle.load(f)
    vqvae = VQVAE()
    
    # Decode
    # Reshape tokens to (T, 8, 8) assuming square grid
    T = tokens.shape[1]
    indices = tokens.reshape(T, 8, 8)
    
    @jax.jit
    def decode_batch(idxs):
        # VQVAE decode expects quantized vectors usually, or we add a helper
        # We need to map indices -> quantized vectors -> decode
        # Or easier: if VQVAE has a 'decode_from_indices' or we do it manually
        
        # Manual lookup from codebook
        codebook = vqvae_params['params']['quantizer']['embedding'] # (512, 128)
        # Normalize codebook if VQVAE was trained with normalized codes
        codebook = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-10)
        
        # Look up
        z_q = codebook[idxs] # (T, 8, 8, 128)
        
        # Decode
        return vqvae.apply(vqvae_params, z_q, method=lambda m, x: m.decoder(x, training=False))

    recon_pixels = decode_batch(indices) # (T, 63, 63, 3)
    
    # Visualization Grid
    # Top: Real, Bottom: Imagined
    # Ensure original_pixels matches T
    orig = original_pixels[0, :T] # (T, 63, 63, 3)
    
    # Convert to uint8
    orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
    recon = np.clip(recon_pixels * 255, 0, 255).astype(np.uint8)
    
    # Make grid
    row_orig = np.concatenate([orig[t] for t in range(T)], axis=1)
    row_recon = np.concatenate([recon[t] for t in range(T)], axis=1)
    
    grid = np.concatenate([row_orig, row_recon], axis=0)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Image.fromarray(grid).save(os.path.join(OUTPUT_DIR, save_name))
    print(f"Saved visualization to {OUTPUT_DIR}/{save_name}")

# --- Main Driver ---

def main():
    # 1. Prepare Data
    obs_tokens, actions, rewards, dones = load_and_tokenize()
    
    # 2. Train TWM
    print("--- Phase 2: Training TWM ---")
    config = TransformerConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        max_blocks=MAX_BLOCKS,
        attention='causal',
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_dim=EMBED_DIM,
        embed_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1
    )
    
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config)
    
    # Training Loop
    steps_per_epoch = obs_tokens.shape[0] // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        epoch_losses = []
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")
        
        for _ in pbar:
            batch = get_batch(obs_tokens, actions, rewards, dones, BATCH_SIZE, SEQ_LEN)
            rng, drop_rng = jax.random.split(rng)
            
            state, metrics = train_step(state, batch, drop_rng)
            
            loss_val = metrics.total_loss
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            
        print(f"Epoch {epoch+1} Avg Loss: {np.mean(epoch_losses):.4f}")

    # 3. Save Weights
    print("--- Phase 3: Saving Weights ---")
    with open(TWM_SAVE_PATH, "wb") as f:
        pickle.dump(state.params, f)
    print("TWM params saved.")
    
    # 4. Rollout Smoke Test
    print("--- Phase 4: Rollout Smoke Test ---")
    # Pick a test sample
    test_idx = 0
    test_start = 0
    test_len = 10 # 10 steps rollout
    
    # Inputs
    start_obs_tokens = obs_tokens[test_idx, test_start] # (64,)
    action_seq = actions[test_idx, test_start:test_start+test_len].reshape(1, -1) # (1, 10)
    
    # Ground Truth Pixels (Need to load raw again? Or just trust tokenizer reconstruction? 
    # Better to compare against RECONSTRUCTED ground truth to ignore VQVAE loss, 
    # OR real ground truth. Let's load a tiny slice of real data for viz)
    data = np.load(DATA_PATH)
    gt_pixels = data['obs'][test_idx:test_idx+1, test_start+1:test_start+1+test_len] # Next frames
    
    # Run Imagination
    imagined_tokens = run_rollout(state, jnp.array(start_obs_tokens), jnp.array(action_seq))
    
    # Visualize
    decode_and_viz(imagined_tokens, gt_pixels, "smoke_test_rollout.png")
    
    print("Smoke Test Complete!")

if __name__ == "__main__":
    main()
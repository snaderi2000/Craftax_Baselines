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
EPOCHS = 200               # Fast training to verify logic
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
    print(f"Token value range: [{obs_tokens.min()}, {obs_tokens.max()}]")
    if obs_tokens.max() >= VOCAB_SIZE:
        print(f"ERROR: Token values exceed VOCAB_SIZE={VOCAB_SIZE}! Clipping...")
        obs_tokens = np.clip(obs_tokens, 0, VOCAB_SIZE - 1)
    
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
    
    # Use gradient clipping for stability
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LR)
    )
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

def get_batch(obs_tokens, actions, rewards, dones, ep_lengths, batch_size, seq_len):
    """
    Sample a batch of sequences from the dataset.
    
    Args:
        obs_tokens: (N, T, 64) tokenized observations
        actions: (N, T) action indices
        rewards: (N, T) reward values  
        dones: (N, T) episode end flags
        ep_lengths: (N,) precomputed valid lengths for each episode
        batch_size: number of sequences to sample
        seq_len: length of each sequence
    """
    num_eps = obs_tokens.shape[0]
    
    batch_obs = []
    batch_act = []
    batch_rew = []
    batch_end = []
    batch_mask = []
    
    for _ in range(batch_size):
        # Sample an episode
        ep_idx = np.random.randint(0, num_eps)
        valid_len = ep_lengths[ep_idx]
        
        # Make sure we have at least seq_len valid steps
        if valid_len < seq_len:
            # Pad this trajectory - start from 0
            start = 0
        else:
            # Random start that fits within valid data
            max_start = valid_len - seq_len
            start = np.random.randint(0, max_start + 1)
        
        end = start + seq_len
        
        batch_obs.append(obs_tokens[ep_idx, start:end])
        batch_act.append(actions[ep_idx, start:end])
        batch_rew.append(rewards[ep_idx, start:end])
        batch_end.append(dones[ep_idx, start:end])
        
        # Create padding mask: True = padding (to be ignored)
        # If end > valid_len, mark those positions as padding
        mask = np.zeros(seq_len, dtype=bool)
        if end > valid_len:
            # Positions from (valid_len - start) onwards are padding
            pad_start = max(0, valid_len - start)
            mask[pad_start:] = True
        batch_mask.append(mask)
    
    obs_array = jnp.array(np.stack(batch_obs))
    act_array = jnp.array(np.stack(batch_act))
    rew_array = jnp.array(np.stack(batch_rew))
    end_array = jnp.array(np.stack(batch_end))
    mask_array = jnp.array(np.stack(batch_mask))
    
    # Debug: Check for invalid token values
    if obs_array.max() >= VOCAB_SIZE:
        print(f"WARNING: obs_tokens max {obs_array.max()} >= VOCAB_SIZE {VOCAB_SIZE}")
    if obs_array.min() < 0:
        print(f"WARNING: obs_tokens min {obs_array.min()} < 0")
    
    return {
        'obs_tokens': obs_array,   # (B, 20, 64)
        'actions': act_array,      # (B, 20)
        'rewards': rew_array,
        'ends': end_array,
        'mask_padding': mask_array  # (B, 20) - True = padding
    }

# --- 3. Rollout & Viz ---

def run_rollout(state, initial_obs_tokens, action_sequence, debug=True):
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
    curr_tokens = initial_obs_tokens.reshape(1, -1) # (1, 64)
    
    generated_frames_tokens = []
    T_rollout = action_sequence.shape[1]
    
    # Current input to model starts as the initial observation patches
    current_input_ids = curr_tokens # (1, 64)
    
    if debug:
        print(f"  Initial obs tokens sample (first 5): {curr_tokens[0, :5]}")
        print(f"  Action sequence: {action_sequence[0]}")
    
    for t in range(T_rollout):
        # A. Process Observation Patches (64 tokens)
        output, cache = model.apply(state.params, current_input_ids, past_keys_values=cache, deterministic=True)
        
        if debug and t == 0:
            print(f"  After obs, cache index: {cache[0].index}")
        
        # B. Process Action (1 token)
        act_token = action_sequence[:, t:t+1] # (1, 1)
        output, cache = model.apply(state.params, act_token, past_keys_values=cache, deterministic=True)
        
        if debug and t == 0:
            print(f"  After action, cache index: {cache[0].index}")
            # Check the logits at action position
            action_logits = output.logits_observations[:, -1, :]
            print(f"  Logits stats: min={float(action_logits.min()):.2f}, max={float(action_logits.max()):.2f}")
            print(f"  Logits sum: {float(action_logits.sum()):.2f} (should be non-zero)")
        
        # C. Auto-regressive generation of the NEXT 64 observation tokens
        next_frame_tokens = []
        
        # Get first token prediction from action output
        logits = output.logits_observations[:, -1, :] # (1, 512)
        next_token = jnp.argmax(logits, axis=-1).reshape(1, 1)
        next_frame_tokens.append(next_token)
        
        if debug and t == 0:
            print(f"  First generated token: {int(next_token[0, 0])}")
        
        # Generate remaining 63 tokens
        for i in range(63):
            output, cache = model.apply(state.params, next_token, past_keys_values=cache, deterministic=True)
            logits = output.logits_observations[:, -1, :]
            
            # Check if logits are all zeros (masked out)
            if debug and t == 0 and i < 3:
                print(f"    Token {i+1}: logits sum={float(logits.sum()):.2f}, max={float(logits.max()):.2f}")
            
            next_token = jnp.argmax(logits, axis=-1).reshape(1, 1)
            next_frame_tokens.append(next_token)
            
        # Stack this frame
        full_frame = jnp.concatenate(next_frame_tokens, axis=1) # (1, 64)
        generated_frames_tokens.append(full_frame)
        
        if debug and t == 0:
            print(f"  Generated frame {t} tokens (first 10): {full_frame[0, :10]}")
            print(f"  Generated frame {t} unique values: {len(np.unique(np.array(full_frame)))}")
        
        # This generated frame becomes the input for the next step
        current_input_ids = full_frame

    result = jnp.stack(generated_frames_tokens, axis=1) # (1, T, 64)
    
    if debug:
        print(f"  Final generated shape: {result.shape}")
        # Check token diversity across all frames
        all_tokens = np.array(result).flatten()
        print(f"  Token diversity: {len(np.unique(all_tokens))} unique values out of {len(all_tokens)}")
    
    return result

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
    recon_pixels = recon_pixels[:, :63, :63, :]
    
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
    
    # Debug: Check data statistics
    print("--- Data Statistics ---")
    print(f"  obs_tokens shape: {obs_tokens.shape}")
    print(f"  obs_tokens dtype: {obs_tokens.dtype}")
    print(f"  obs_tokens range: [{obs_tokens.min()}, {obs_tokens.max()}]")
    print(f"  actions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"  actions range: [{actions.min()}, {actions.max()}]")
    print(f"  rewards range: [{rewards.min()}, {rewards.max()}]")
    print(f"  dones unique values: {np.unique(dones)}")
    
    # Check for NaN in input data
    if np.any(np.isnan(rewards)):
        print("  WARNING: NaN in rewards!")
        rewards = np.nan_to_num(rewards, nan=0.0)
    if np.any(np.isnan(dones)):
        print("  WARNING: NaN in dones!")
        dones = np.nan_to_num(dones, nan=0.0)
    if actions.max() >= ACT_VOCAB_SIZE:
        print(f"  WARNING: actions max {actions.max()} >= ACT_VOCAB_SIZE {ACT_VOCAB_SIZE}!")
        actions = np.clip(actions, 0, ACT_VOCAB_SIZE - 1)
    
    # Ensure dones is boolean-like (0 or 1)
    dones = (dones > 0.5).astype(np.float32)
    
    # Pre-compute episode lengths for efficient batch sampling
    num_eps, ep_len, _ = obs_tokens.shape
    ep_lengths = np.zeros(num_eps, dtype=np.int32)
    for i in range(num_eps):
        done_idx = np.where(dones[i] > 0.5)[0]
        if len(done_idx) > 0:
            ep_lengths[i] = done_idx[0] + 1
        else:
            ep_lengths[i] = ep_len
    print(f"  Episode lengths: min={ep_lengths.min()}, max={ep_lengths.max()}, mean={ep_lengths.mean():.1f}")
    
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
        
        for step in pbar:
            batch = get_batch(obs_tokens, actions, rewards, dones, ep_lengths, BATCH_SIZE, SEQ_LEN)
            rng, drop_rng = jax.random.split(rng)
            
            state, metrics = train_step(state, batch, drop_rng)
            
            loss_val = float(metrics.total_loss)
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            
            # Early NaN detection
            if np.isnan(loss_val):
                print(f"\nNaN detected at epoch {epoch+1}, step {step}")
                print(f"  loss_obs: {float(metrics.loss_obs)}")
                print(f"  loss_rew: {float(metrics.loss_rewards)}")
                print(f"  loss_ends: {float(metrics.loss_ends)}")
                print(f"  Batch stats:")
                print(f"    obs_tokens range: [{batch['obs_tokens'].min()}, {batch['obs_tokens'].max()}]")
                print(f"    actions range: [{batch['actions'].min()}, {batch['actions'].max()}]")
                print(f"    mask_padding sum: {batch['mask_padding'].sum()}")
                break
        
        if np.isnan(loss_val):
            print("Stopping training due to NaN")
            break
            
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
    
    # Ground truth tokens for comparison
    gt_obs_tokens = obs_tokens[test_idx, test_start+1:test_start+1+test_len]  # (10, 64)
    
    print(f"  Ground truth obs tokens (frame 0, first 10): {gt_obs_tokens[0, :10]}")
    print(f"  Ground truth token range: [{gt_obs_tokens.min()}, {gt_obs_tokens.max()}]")
    
    # Ground Truth Pixels
    data = np.load(DATA_PATH)
    gt_pixels = data['obs'][test_idx:test_idx+1, test_start+1:test_start+1+test_len] # Next frames
    
    # Run Imagination
    imagined_tokens = run_rollout(state, jnp.array(start_obs_tokens), jnp.array(action_seq))
    
    # Compare tokens
    imagined_np = np.array(imagined_tokens[0])  # (10, 64)
    print(f"\n--- Token Comparison ---")
    print(f"  Imagined token range: [{imagined_np.min()}, {imagined_np.max()}]")
    print(f"  Imagined frame 0 (first 10): {imagined_np[0, :10]}")
    print(f"  Ground truth frame 0 (first 10): {gt_obs_tokens[0, :10]}")
    
    # Token accuracy (how many match exactly)
    matches = (imagined_np == gt_obs_tokens).sum()
    total = gt_obs_tokens.size
    print(f"  Token accuracy: {matches}/{total} = {100*matches/total:.1f}%")
    print(f"  (Random would be ~0.2% with vocab size 512)")
    
    # Visualize
    decode_and_viz(imagined_tokens, gt_pixels, "smoke_test_rollout.png")
    
    print("\nSmoke Test Complete!")
    print("NOTE: With only 15 gradient steps, the model is essentially random.")
    print("      For meaningful generation, train for many more epochs.")

if __name__ == "__main__":
    main()
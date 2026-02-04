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

# ============= TRAINING CONFIG =============
# Paper settings (Table 4 from the paper)
BATCH_SIZE = 16          # Reduced for 20GB GPU (paper uses larger with 8xH100)
SEQ_LEN = 20             # T_WM = 20 steps (paper: "largest that fits in memory")
EPOCHS = 500             # More epochs for better convergence
LR = 0.001               # Paper: 0.001

# Model architecture (EXACTLY from paper Table 4)
TOKENS_PER_BLOCK = 65    # 64 patches + 1 action
MAX_BLOCKS = SEQ_LEN     # 20 steps
EMBED_DIM = 128          # Paper: 128
NUM_LAYERS = 3           # Paper: 3
NUM_HEADS = 8            # Paper: 8
VOCAB_SIZE = 512
ACT_VOCAB_SIZE = 17      # Craftax actions

# Dropout rates (Paper Table 4)
EMBED_PDROP = 0.1
ATTN_PDROP = 0.1
RESID_PDROP = 0.1

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
    
    # Use gradient clipping for stability (Paper: max gradient norm 0.5)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
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
            config=TransformerConfig(TOKENS_PER_BLOCK, MAX_BLOCKS, 'causal', NUM_LAYERS, NUM_HEADS, EMBED_DIM, EMBED_PDROP, RESID_PDROP, ATTN_PDROP)
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

def teacher_force_test(state, obs_tokens, actions, test_idx, test_start, test_len):
    """
    Teacher forcing test: Feed ground truth tokens and check next-token prediction accuracy.
    This isolates model quality from autoregressive generation errors.
    """
    model = WorldModel(
        obs_vocab_size=VOCAB_SIZE, 
        act_vocab_size=ACT_VOCAB_SIZE, 
        config=TransformerConfig(TOKENS_PER_BLOCK, MAX_BLOCKS, 'causal', NUM_LAYERS, NUM_HEADS, EMBED_DIM, EMBED_PDROP, RESID_PDROP, ATTN_PDROP)
    )
    
    # Get a sequence of ground truth data
    gt_obs = obs_tokens[test_idx, test_start:test_start+test_len+1]  # (T+1, 64)
    gt_act = actions[test_idx, test_start:test_start+test_len]       # (T,)
    
    # Build the full input sequence (interleaved obs + actions)
    # [obs_0, act_0, obs_1, act_1, ..., obs_T-1, act_T-1]
    T = test_len
    tokens_list = []
    for t in range(T):
        tokens_list.append(gt_obs[t])  # 64 obs tokens
        tokens_list.append(np.array([gt_act[t]]))  # 1 action token
    
    input_tokens = np.concatenate(tokens_list)  # (T * 65,)
    input_tokens = jnp.array(input_tokens).reshape(1, -1)  # (1, T*65)
    
    # Forward pass (no cache, full sequence)
    output = model.apply(state.params, input_tokens, deterministic=True)
    
    # Get observation logits
    obs_logits = output.logits_observations  # (1, T*65, 512)
    
    # Build target sequence (shifted by 1)
    # Target for position i is the token at position i+1
    target_list = []
    for t in range(T):
        if t < T - 1:
            # Next obs tokens
            target_list.append(gt_obs[t+1])  # 64 tokens
        else:
            # Last timestep - we predict obs_T (which we have)
            target_list.append(gt_obs[t+1])
        # After action, we predict first obs token of next timestep
        # But this is complex with masking, so simplify:
    
    # Simpler: just check obs token predictions
    # For each obs position i in block, logits[i] predicts obs[i+1]
    
    # Check accuracy for first few blocks
    correct = 0
    total = 0
    
    print("  Checking next-token prediction accuracy (teacher forcing):")
    
    for t in range(min(3, T-1)):  # Check first 3 timesteps
        block_start = t * 65
        
        # For obs positions 0-62 in this block, check if we predict next obs token
        for i in range(63):  # positions 0-62
            pos = block_start + i
            pred = int(jnp.argmax(obs_logits[0, pos, :]))
            target = int(gt_obs[t, i+1])  # next obs token in same frame
            if pred == target:
                correct += 1
            total += 1
        
        # After action (position 64), we predict first obs of next frame
        action_pos = block_start + 64
        pred_next_frame = int(jnp.argmax(obs_logits[0, action_pos, :]))
        target_next_frame = int(gt_obs[t+1, 0])  # first obs token of next frame
        
        print(f"    Block {t}: After action, predict obs[0] of next frame: pred={pred_next_frame}, target={target_next_frame}, match={pred_next_frame==target_next_frame}")
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"  Within-frame next-token accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"  (If this is low, model needs more training or data)")
    print(f"  (If this is high but generation is bad, issue is compounding errors)")


def run_rollout_with_context(state, context_obs_tokens, context_actions, future_actions, 
                              debug=True, temperature=1.0, use_sampling=True):
    """
    Autoregressive generation WITH burn-in context (matches paper Algorithm 4).
    
    Paper quote: "we use burn-in to refresh the hidden state before rolling out the policy"
    
    Args:
        context_obs_tokens: (M+1, 64) - M burn-in frames + 1 starting frame
        context_actions: (M+1,) - Actions taken during context (including the last one before imagination)
        future_actions: (T,) - Actions to use for imagination rollout
        temperature: Sampling temperature
        use_sampling: Whether to sample or use argmax
    
    Returns:
        Generated observation tokens for T future timesteps
    """
    print(f"Running TWM Rollout with Context... [temp={temperature}, sampling={use_sampling}]")
    model = WorldModel(
        obs_vocab_size=VOCAB_SIZE, 
        act_vocab_size=ACT_VOCAB_SIZE, 
        config=TransformerConfig(TOKENS_PER_BLOCK, MAX_BLOCKS, 'causal', NUM_LAYERS, NUM_HEADS, EMBED_DIM, EMBED_PDROP, RESID_PDROP, ATTN_PDROP)
    )
    
    sample_rng = jax.random.PRNGKey(42)
    
    # 1. Setup KV Cache
    cache = KeysValues.init(n=1, num_heads=NUM_HEADS, max_tokens=SEQ_LEN*TOKENS_PER_BLOCK, 
                            embed_dim=EMBED_DIM, num_layers=NUM_LAYERS)
    
    M = len(context_actions)  # Number of burn-in steps (obs+action pairs)
    T_rollout = len(future_actions)
    
    # context_obs_tokens should have M+1 frames: M for burn-in + 1 starting frame
    assert len(context_obs_tokens) == M + 1, f"Expected {M+1} obs frames, got {len(context_obs_tokens)}"
    
    if debug:
        print(f"  Context length (burn-in): {M} steps")
        print(f"  Rollout length: {T_rollout} steps")
        print(f"  Context obs tokens sample (first frame, first 5): {context_obs_tokens[0, :5]}")
        print(f"  Context actions: {context_actions}")
        print(f"  Future actions: {future_actions}")
    
    # 2. Feed burn-in context: M (obs, action) pairs
    for m in range(M):
        # Feed observation frame
        obs_frame = jnp.array(context_obs_tokens[m]).reshape(1, -1)  # (1, 64)
        output, cache = model.apply(state.params, obs_frame, past_keys_values=cache, deterministic=True)
        
        # Feed action
        act_token = jnp.array([[context_actions[m]]])  # (1, 1)
        output, cache = model.apply(state.params, act_token, past_keys_values=cache, deterministic=True)
    
    if debug:
        print(f"  After burn-in, cache index: {cache[0].index}")
    
    # 3. Now we're at the starting point for imagination
    # The last context observation (index M) is our starting frame
    current_obs = jnp.array(context_obs_tokens[M]).reshape(1, -1)  # (1, 64)
    
    if debug:
        print(f"  Starting frame (context_obs[{M}], first 5): {context_obs_tokens[M, :5]}")
    
    generated_frames_tokens = []
    
    def sample_token(logits, rng):
        """Sample a token from logits with temperature."""
        if use_sampling and temperature > 0:
            scaled_logits = logits / temperature
            return jax.random.categorical(rng, scaled_logits, axis=-1).reshape(1, 1)
        else:
            return jnp.argmax(logits, axis=-1).reshape(1, 1)
    
    for t in range(T_rollout):
        # A. Process current observation (64 tokens)
        output, cache = model.apply(state.params, current_obs, past_keys_values=cache, deterministic=True)
        
        if debug and t == 0:
            print(f"  After first imagination obs, cache index: {cache[0].index}")
        
        # B. Process action (1 token)
        act_token = jnp.array([[future_actions[t]]])  # (1, 1)
        output, cache = model.apply(state.params, act_token, past_keys_values=cache, deterministic=True)
        
        if debug and t == 0:
            print(f"  After action, cache index: {cache[0].index}")
            action_logits = output.logits_observations[:, -1, :]
            print(f"  Logits stats: min={float(action_logits.min()):.2f}, max={float(action_logits.max()):.2f}")
            top5_idx = jnp.argsort(action_logits[0])[-5:][::-1]
            top5_probs = jax.nn.softmax(action_logits[0])[top5_idx]
            print(f"  First token - top5: {list(zip(np.array(top5_idx), np.array(top5_probs).round(3)))}")
        
        # C. Auto-regressive generation of the NEXT 64 observation tokens
        next_frame_tokens = []
        
        # Get first token prediction from action output
        logits = output.logits_observations[:, -1, :]
        sample_rng, rng = jax.random.split(sample_rng)
        next_token = sample_token(logits, rng)
        next_frame_tokens.append(next_token)
        
        if debug and t == 0:
            print(f"  First generated token: {int(next_token[0, 0])}")
        
        # Generate remaining 63 tokens
        for i in range(63):
            output, cache = model.apply(state.params, next_token, past_keys_values=cache, deterministic=True)
            logits = output.logits_observations[:, -1, :]
            
            if debug and t == 0 and i < 3:
                print(f"    Token {i+1}: logits sum={float(logits.sum()):.2f}, max={float(logits.max()):.2f}")
            
            sample_rng, rng = jax.random.split(sample_rng)
            next_token = sample_token(logits, rng)
            next_frame_tokens.append(next_token)
        
        # Stack this frame
        full_frame = jnp.concatenate(next_frame_tokens, axis=1)  # (1, 64)
        generated_frames_tokens.append(full_frame)
        
        if debug and t == 0:
            print(f"  Generated frame {t} tokens (first 10): {full_frame[0, :10]}")
            print(f"  Generated frame {t} unique values: {len(np.unique(np.array(full_frame)))}")
        
        # This generated frame becomes the current observation for the next step
        current_obs = full_frame

    result = jnp.stack(generated_frames_tokens, axis=1)  # (1, T, 64)
    
    if debug:
        print(f"  Final generated shape: {result.shape}")
        all_tokens = np.array(result).flatten()
        print(f"  Token diversity: {len(np.unique(all_tokens))} unique values out of {len(all_tokens)}")
    
    return result


# Keep old function for backwards compatibility but mark as deprecated
def run_rollout(state, initial_obs_tokens, action_sequence, debug=True, temperature=1.0, use_sampling=True):
    """DEPRECATED: Use run_rollout_with_context for paper-matching behavior."""
    print("WARNING: Using old rollout without burn-in context. Results may be poor.")
    # Convert to new format with no burn-in
    context_obs = initial_obs_tokens.reshape(1, 64)  # Just the initial frame
    context_actions = np.array([])  # No context actions
    future_actions = np.array(action_sequence).flatten()
    
    # Can't use the new function directly without context, so use simplified version
    print(f"Running TWM Rollout (No Context)... [temp={temperature}, sampling={use_sampling}]")
    model = WorldModel(
        obs_vocab_size=VOCAB_SIZE, 
        act_vocab_size=ACT_VOCAB_SIZE, 
        config=TransformerConfig(TOKENS_PER_BLOCK, MAX_BLOCKS, 'causal', NUM_LAYERS, NUM_HEADS, EMBED_DIM, EMBED_PDROP, RESID_PDROP, ATTN_PDROP)
    )
    
    sample_rng = jax.random.PRNGKey(42)
    cache = KeysValues.init(n=1, num_heads=NUM_HEADS, max_tokens=SEQ_LEN*TOKENS_PER_BLOCK, 
                            embed_dim=EMBED_DIM, num_layers=NUM_LAYERS)
    
    current_obs = initial_obs_tokens.reshape(1, -1)
    generated_frames_tokens = []
    T_rollout = len(future_actions)
    
    def sample_token(logits, rng):
        if use_sampling and temperature > 0:
            return jax.random.categorical(rng, logits / temperature, axis=-1).reshape(1, 1)
        return jnp.argmax(logits, axis=-1).reshape(1, 1)
    
    for t in range(T_rollout):
        output, cache = model.apply(state.params, current_obs, past_keys_values=cache, deterministic=True)
        act_token = jnp.array([[future_actions[t]]])
        output, cache = model.apply(state.params, act_token, past_keys_values=cache, deterministic=True)
        
        next_frame_tokens = []
        logits = output.logits_observations[:, -1, :]
        sample_rng, rng = jax.random.split(sample_rng)
        next_token = sample_token(logits, rng)
        next_frame_tokens.append(next_token)
        
        for _ in range(63):
            output, cache = model.apply(state.params, next_token, past_keys_values=cache, deterministic=True)
            logits = output.logits_observations[:, -1, :]
            sample_rng, rng = jax.random.split(sample_rng)
            next_token = sample_token(logits, rng)
            next_frame_tokens.append(next_token)
        
        full_frame = jnp.concatenate(next_frame_tokens, axis=1)
        generated_frames_tokens.append(full_frame)
        current_obs = full_frame

    return jnp.stack(generated_frames_tokens, axis=1)

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
        embed_pdrop=EMBED_PDROP, 
        resid_pdrop=RESID_PDROP, 
        attn_pdrop=ATTN_PDROP
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
    
    # 4. Rollout Smoke Test (Paper-style with burn-in context)
    print("--- Phase 4: Rollout Smoke Test (Paper-style) ---")
    
    # Paper settings (Table 5)
    BURNIN_M = 5  # Burn-in horizon M (paper: 5)
    ROLLOUT_LEN = 5  # Number of frames to generate
    
    # Pick a test sample - need enough room for burn-in + rollout
    test_idx = 0
    test_start = 10  # Start after some timesteps so we have diverse context
    
    # Ensure we have enough data
    total_needed = BURNIN_M + 1 + ROLLOUT_LEN  # context + starting frame + rollout
    assert test_start + total_needed < obs_tokens.shape[1], "Not enough data for test"
    
    # Context for burn-in (M frames with M actions, then 1 starting frame)
    # We need M+1 obs frames but only M actions for burn-in
    context_obs = obs_tokens[test_idx, test_start:test_start + BURNIN_M + 1]  # (M+1, 64) - includes starting frame
    context_act = actions[test_idx, test_start:test_start + BURNIN_M]         # (M,) - only M actions for burn-in
    
    # Future actions for rollout
    # Starting frame is at index (test_start + BURNIN_M)
    # Action at that index transitions to the next frame
    # So future_act starts at the same index as the starting frame
    future_act_start = test_start + BURNIN_M  # Action that takes us from starting frame to first generated frame
    future_act = actions[test_idx, future_act_start:future_act_start + ROLLOUT_LEN]   # (ROLLOUT_LEN,)
    
    # Ground truth: frames AFTER the starting frame (what we're trying to predict)
    gt_start = test_start + BURNIN_M + 1  # First frame to predict
    gt_obs_tokens = obs_tokens[test_idx, gt_start:gt_start + ROLLOUT_LEN]  # (ROLLOUT_LEN, 64)
    
    print(f"  Burn-in context: {BURNIN_M} steps (paper recommends M=5)")
    print(f"  Rollout length: {ROLLOUT_LEN} steps")
    print(f"  Context obs shape: {context_obs.shape} (M+1 frames)")
    print(f"  Context act shape: {context_act.shape} (M actions)")
    print(f"  Future act shape: {future_act.shape}")
    print(f"  Starting frame index: {test_start + BURNIN_M}")
    print(f"  Ground truth frames indices: {gt_start} to {gt_start + ROLLOUT_LEN - 1}")
    print(f"  Ground truth obs tokens (frame 0, first 10): {gt_obs_tokens[0, :10]}")
    print(f"  Ground truth token range: [{gt_obs_tokens.min()}, {gt_obs_tokens.max()}]")
    
    # Ground Truth Pixels
    data = np.load(DATA_PATH)
    gt_pixels = data['obs'][test_idx:test_idx+1, gt_start:gt_start + ROLLOUT_LEN]
    
    # === TEACHER FORCING TEST ===
    print("\n--- Teacher Forcing Test (Model Quality Check) ---")
    teacher_force_test(state, obs_tokens, actions, test_idx, test_start, ROLLOUT_LEN + BURNIN_M)
    
    # === PAPER-STYLE ROLLOUT WITH CONTEXT ===
    print("\n--- Paper-style Rollout (with burn-in context) ---")
    print("\n  [Greedy/Argmax]")
    imagined_greedy = run_rollout_with_context(
        state, context_obs, context_act, future_act,
        temperature=1.0, use_sampling=False
    )
    
    print("\n  [Temperature=0.7 Sampling]")
    imagined_sampled = run_rollout_with_context(
        state, context_obs, context_act, future_act,
        temperature=0.7, use_sampling=True
    )
    
    # Compare tokens for both methods
    for name, tokens in [("Greedy+Context", imagined_greedy), ("Temp0.7+Context", imagined_sampled)]:
        imagined_np = np.array(tokens[0])  # (ROLLOUT_LEN, 64)
        print(f"\n--- Token Comparison ({name}) ---")
        print(f"  Imagined token range: [{imagined_np.min()}, {imagined_np.max()}]")
        print(f"  Imagined frame 0 (first 10): {imagined_np[0, :10]}")
        print(f"  Ground truth frame 0 (first 10): {gt_obs_tokens[0, :10]}")
        
        # Token accuracy
        matches = (imagined_np == gt_obs_tokens).sum()
        total = gt_obs_tokens.size
        print(f"  Token accuracy: {matches}/{total} = {100*matches/total:.1f}%")
    
    # Visualize
    decode_and_viz(imagined_greedy, gt_pixels, "smoke_test_context_greedy.png")
    decode_and_viz(imagined_sampled, gt_pixels, "smoke_test_context_sampled.png")
    
    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE!")
    print("="*60)
    print(f"  Training loss: {np.mean(epoch_losses):.4f}")
    print(f"  Teacher forcing accuracy: See above")
    print(f"  Paper-style rollout with M={BURNIN_M} burn-in: See above")
    print("\nNext steps for production:")
    print("  1. Collect more data (aim for 50k+ transitions)")
    print("  2. Integrate with MBRL loop (Algorithm 1 from paper)")
    print("  3. Use flashbax for efficient replay buffer")

if __name__ == "__main__":
    main()
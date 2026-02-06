import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm

# Import the updated logic and classes
from train_twm_smoke import run_rollout_with_context, VQVAE, WorldModel, TransformerConfig

# --- Configuration ---
DATA_PATH = "../../replay_data/my_buffer/replay_data.npz"
VQVAE_PARAMS_PATH = "../tokenizer/vqvae_params.pkl"
TWM_PARAMS_PATH = "twm_params.pkl"
OUTPUT_DIR = "viz_gallery"
NUM_SAMPLES = 5   # Number of trajectories to visualize
BURNIN_M = 5      # Burn-in context (Paper recommends 5)
ROLLOUT_LEN = 10  # Future steps to imagine

def decode_tokens_to_pixels(tokens, vqvae, v_params):
    """Helper to turn (T, 64) tokens back into (T, 63, 63, 3) images."""
    codebook = v_params['params']['quantizer']['embedding']
    # Normalize codebook for consistent decoding
    codebook = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-10)
    
    # tokens shape is (1, T, 64) from rollout, we need (T, 8, 8)
    T = tokens.shape[1]
    indices = tokens.reshape(T, 8, 8)
    z_q = codebook[indices]
    
    pixels = vqvae.apply(v_params, z_q, method=lambda m, x: m.decoder(x, training=False))
    
    # CROP FIX: Ensure 64x64 decoder output matches Craftax 63x63
    pixels = pixels[:, :63, :63, :] 

    # Scale to uint8 for PIL
    return np.clip(pixels * 255, 0, 255).astype(np.uint8)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data & Params
    print("Loading data and model params...")
    data = np.load(DATA_PATH)
    all_obs = data['obs']
    all_acts = data['actions']
    
    with open(VQVAE_PARAMS_PATH, "rb") as f:
        v_params = pickle.load(f)
    with open(TWM_PARAMS_PATH, "rb") as f:
        t_params = pickle.load(f)

    # 2. Setup VQ-VAE and Mock State
    vqvae = VQVAE()
    class MockState:
        def __init__(self, params): self.params = params
    state = MockState(t_params)

    # 3. Process Random Trajectories
    indices = np.random.choice(len(all_obs), NUM_SAMPLES, replace=False)
    
    for i, traj_idx in enumerate(indices):
        print(f"\n[{i+1}/{NUM_SAMPLES}] Processing Trajectory {traj_idx}...")
        
        # Pick a random starting point
        max_start = all_obs.shape[1] - (BURNIN_M + ROLLOUT_LEN + 2)
        start_t = np.random.randint(5, max_start)
        
        # Prepare Context (M+1 frames)
        context_pixels = all_obs[traj_idx, start_t : start_t + BURNIN_M + 1]
        
        # Tokenize context
        flat_context = context_pixels.reshape(-1, 63, 63, 3)
        context_indices = vqvae.apply(v_params, flat_context, method=vqvae.encode_indices)
        context_tokens = context_indices.reshape(BURNIN_M + 1, 64)
        
        context_actions = all_acts[traj_idx, start_t : start_t + BURNIN_M]
        future_actions = all_acts[traj_idx, start_t + BURNIN_M : start_t + BURNIN_M + ROLLOUT_LEN]
        
        # Ground Truth Pixels
        gt_pixels = all_obs[traj_idx, start_t + BURNIN_M + 1 : start_t + BURNIN_M + 1 + ROLLOUT_LEN]

        # 4. Run the NEW Autoregressive Rollout
        # Higher temperature (0.8-1.0) allows for more variation in predictions
        img_tokens = run_rollout_with_context(
            state, context_tokens, context_actions, future_actions,
            temperature=0.8, use_sampling=True, debug=False
        )
        
        # 5. Decode and Visualize
        recon_pixels = decode_tokens_to_pixels(img_tokens, vqvae, v_params)
        
        # Construct the comparison grid
        gt_row = np.concatenate([gt_pixels[t] for t in range(ROLLOUT_LEN)], axis=1)
        pred_row = np.concatenate([recon_pixels[t] for t in range(ROLLOUT_LEN)], axis=1)
        
        # Normalize GT for concatenation
        if gt_row.max() <= 1.0: gt_row = (gt_row * 255).astype(np.uint8)
        
        # Add a 2-pixel black divider between rows
        divider = np.zeros((2, gt_row.shape[1], 3), dtype=np.uint8)
        combined = np.concatenate([gt_row, divider, pred_row], axis=0)
        
        save_path = os.path.join(OUTPUT_DIR, f"sample_traj_{traj_idx}.png")
        Image.fromarray(combined).save(save_path)
        print(f"   Saved viz to: {save_path}")

    print(f"\nSuccess! Check '{OUTPUT_DIR}' for the results.")

if __name__ == "__main__":
    main()
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm

# Import the logic you already wrote
from train_twm_smoke import run_rollout_with_context, VQVAE, WorldModel, TransformerConfig

# --- Configuration ---
DATA_PATH = "../../replay_data/my_buffer/replay_data.npz"
VQVAE_PARAMS_PATH = "../tokenizer/vqvae_params.pkl"
TWM_PARAMS_PATH = "twm_params.pkl"
OUTPUT_DIR = "viz_gallery"
NUM_SAMPLES = 5   # Number of different scenarios to visualize
BURNIN_M = 5      # How many real frames to show the model first
ROLLOUT_LEN = 10  # How many frames to predict into the future

def decode_tokens_to_pixels(tokens, vqvae, v_params):
    """Helper to turn (T, 64) tokens back into (T, 63, 63, 3) images."""
    codebook = v_params['params']['quantizer']['embedding']
    codebook = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-10)
    
    # tokens shape is (1, T, 64) from rollout, we need (T, 8, 8)
    T = tokens.shape[1]
    indices = tokens.reshape(T, 8, 8)
    z_q = codebook[indices]
    
    pixels = vqvae.apply(v_params, z_q, method=lambda m, x: m.decoder(x, training=False))
    # Scale to uint8 for PIL
    return np.clip(pixels * 255, 0, 255).astype(np.uint8)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data & Params
    print("Loading data and model params...")
    data = np.load(DATA_PATH)
    all_obs = data['obs']      # (N, T, 63, 63, 3)
    all_acts = data['actions']  # (N, T)
    
    with open(VQVAE_PARAMS_PATH, "rb") as f:
        v_params = pickle.load(f)
    with open(TWM_PARAMS_PATH, "rb") as f:
        t_params = pickle.load(f)

    # 2. Setup VQ-VAE for on-the-fly tokenization
    vqvae = VQVAE()
    
    # Wrapper for the "State" object the smoke test expects
    class MockState:
        def __init__(self, params): self.params = params
    state = MockState(t_params)

    # 3. Select Random Trajectories
    indices = np.random.choice(len(all_obs), NUM_SAMPLES, replace=False)
    
    for i, traj_idx in enumerate(indices):
        print(f"Generating Visualization {i+1}/{NUM_SAMPLES} (Traj {traj_idx})...")
        
        # Pick a random starting point that has enough room for rollout
        max_start = all_obs.shape[1] - (BURNIN_M + ROLLOUT_LEN + 2)
        start_t = np.random.randint(10, max_start)
        
        # Get Context
        context_pixels = all_obs[traj_idx, start_t : start_t + BURNIN_M + 1]
        
        # Tokenize context on the fly
        flat_context = context_pixels.reshape(-1, 63, 63, 3)
        context_indices = vqvae.apply(v_params, flat_context, method=vqvae.encode_indices)
        context_tokens = context_indices.reshape(BURNIN_M + 1, 64)
        
        context_actions = all_acts[traj_idx, start_t : start_t + BURNIN_M]
        future_actions = all_acts[traj_idx, start_t + BURNIN_M : start_t + BURNIN_M + ROLLOUT_LEN]
        
        # Ground Truth for comparison
        gt_pixels = all_obs[traj_idx, start_t + BURNIN_M + 1 : start_t + BURNIN_M + 1 + ROLLOUT_LEN]

        # 4. Run Rollout
        # We'll do one Greedy and one Sampled to see the difference
        img_sampled = run_rollout_with_context(
            state, context_tokens, context_actions, future_actions,
            temperature=0.8, use_sampling=True, debug=False
        )
        
        # 5. Decode
        recon_pixels = decode_tokens_to_pixels(img_sampled, vqvae, v_params)
        
        # 6. Create Row (GT on top, Prediction on bottom)
        gt_row = np.concatenate([gt_pixels[t] for t in range(ROLLOUT_LEN)], axis=1)
        pred_row = np.concatenate([recon_pixels[t] for t in range(ROLLOUT_LEN)], axis=1)
        
        # Convert GT to uint8 if it isn't already
        if gt_row.max() <= 1.0: gt_row = (gt_row * 255).astype(np.uint8)
        
        combined = np.concatenate([gt_row, pred_row], axis=0)
        
        # Add a small black border between frames for clarity if you want, 
        # but a simple concat usually works for smoke tests.
        Image.fromarray(combined).save(os.path.join(OUTPUT_DIR, f"sample_{traj_idx}.png"))

    print(f"\nDone! Check the '{OUTPUT_DIR}' folder for your visualizations.")

if __name__ == "__main__":
    main()
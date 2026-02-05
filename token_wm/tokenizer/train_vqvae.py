import os
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state
from PIL import Image
from tqdm import tqdm
import pickle

# Import your model
from vqvae import VQVAE

# --- Configuration ---
DATA_PATH = "../../replay_data/my_buffer/replay_data.npz"
BATCH_SIZE = 128
LEARNING_RATE = 1e-3  # Paper uses 0.001 (Source 694)
EPOCHS = 30  # Adjust as needed (5 is good for a quick test)
SEED = 42
OUTPUT_DIR = "vqvae_results"

# --- Data Loading ---
def load_and_prep_data(path):
    print(f"Loading data from {path}...")
    data = np.load(path)
    obs = data["obs"]  # Shape: (100, 700, 63, 63, 3)

    # Flatten episodes into a single batch of frames
    # New Shape: (70000, 63, 63, 3)
    obs = obs.reshape(-1, 63, 63, 3)

    # Ensure data is float32 and normalized to [0, 1] if it isn't already
    # (Assuming the saved float32 data is already 0-1, but clipping helps stability)
    obs = np.clip(obs, 0.0, 1.0)

    print(f"Data loaded. Total frames: {obs.shape[0]}, Shape: {obs.shape}")
    return obs


# --- Training Setup ---
def create_train_state(rng, model, sample_input):
    params = model.init(rng, sample_input)
    tx = optax.adam(LEARNING_RATE)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        # Forward pass
        # returns: recon, codebook_loss, commitment_loss (weighted), indices
        recon, l_code, l_commit, _ = state.apply_fn(params, batch, training=True)

        # Reconstruction Loss: Paper uses L1 loss (Lambda1=1, Lambda2=0)
        l_recon = jnp.mean(jnp.abs(batch - recon))

        # Total Loss
        total_loss = l_recon + l_code + l_commit

        return total_loss, (l_recon, l_code, l_commit, recon)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


# --- Visualization ---
def save_reconstruction_grid(originals, recons, step, save_dir):
    """Saves a grid of Original vs Reconstructed images."""
    # Take first 8 images
    n_images = min(8, originals.shape[0])
    orig = originals[:n_images]
    rec = recons[:n_images]

    # Clip to valid range and convert to uint8
    orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
    rec = np.clip(rec * 255, 0, 255).astype(np.uint8)

    # Concatenate vertically: Top row = Original, Bottom row = Recon
    # Then concatenate horizontally to make a row of pairs
    # Easier layout: Row of Originals on top of Row of Recons
    row_orig = np.concatenate([orig[i] for i in range(n_images)], axis=1)
    row_rec = np.concatenate([rec[i] for i in range(n_images)], axis=1)

    grid = np.concatenate([row_orig, row_rec], axis=0)

    os.makedirs(save_dir, exist_ok=True)
    img = Image.fromarray(grid)
    img.save(os.path.join(save_dir, f"recon_step_{step}.png"))


# --- Main Loop ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    all_obs = load_and_prep_data(DATA_PATH)
    n_samples = all_obs.shape[0]

    # 2. Init Model
    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)

    model = VQVAE()
    # Init with a small batch to set shapes
    dummy_batch = jnp.ones((8, 63, 63, 3))
    state = create_train_state(init_rng, model, dummy_batch)

    print("Model initialized. Starting training...")

    # 3. Training Loop
    steps_per_epoch = n_samples // BATCH_SIZE
    global_step = 0

    for epoch in range(EPOCHS):
        # Shuffle data each epoch
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, n_samples)
        shuffled_obs = all_obs[perms]

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_losses = []

        for i in pbar:
            batch_idx = perms[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            # Use numpy slicing for data loading, then convert to JAX array in step if needed
            # (passed automatically usually)
            batch = jnp.array(shuffled_obs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])

            state, loss, (l_rec, l_code, l_commit, recon) = train_step(state, batch)

            epoch_losses.append(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}", "rec": f"{l_rec:.4f}"})

            # Save reconstruction every 1000 steps
            if global_step % 500 == 0:
                # Need to act on the concrete numpy array for PIL
                # recon is currently on GPU/accelerator
                save_reconstruction_grid(batch, recon, global_step, OUTPUT_DIR)

            global_step += 1

        print(f"Epoch {epoch+1} Avg Loss: {np.mean(epoch_losses):.4f}")

    # Final Save
    print("Saving final reconstruction...")
    # Run one inference on a fixed batch
    test_batch = jnp.array(all_obs[:16])
    recon, _, _, _ = state.apply_fn(state.params, test_batch, training=False)
    save_reconstruction_grid(test_batch, recon, "final", OUTPUT_DIR)

    # After your training loop in train_vqvae.py
    with open("vqvae_params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("VQ-VAE weights saved to vqvae_params.pkl")

    print(f"Training complete. Check results in ./{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

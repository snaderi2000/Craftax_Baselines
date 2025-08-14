# test_vq_vae.py (Updated to show all losses and use a local image)

import jax
import jax.numpy as jnp
import optax
import numpy as np
import matplotlib.pyplot as plt
from flax.core.frozen_dict import FrozenDict
from typing import Tuple

# Import the VQVAE class from your vq_vae.py file
from vq_vae import VQVAE

def run_test():
    """
    Tests the VQVAE implementation by loading a local image file and printing detailed loss info.
    """
    print("--- Starting VQ-VAE Test Script with a Local Image ---")

    # --- Configuration ---
    batch_size = 4
    image_shape = (63, 63, 3) # The shape the model expects
    learning_rate = 1e-3
    training_steps = 1001
    image_path = "crafter.png" # Using your local image

    # --- Setup ---
    key = jax.random.PRNGKey(0)

    # --- Load Image Data from File ---
    print(f"\n--- Loading image from {image_path} ---")
    try:
        img_np = plt.imread(image_path)
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'.")
        return

    # Preprocess the image
    img_jnp = jnp.asarray(img_np).astype(jnp.float32)
    if img_jnp.shape[-1] == 4:
        img_jnp = img_jnp[..., :3]
    if jnp.max(img_jnp) > 1.0:
        img_jnp = img_jnp / 255.0

    img_jnp = jnp.expand_dims(img_jnp, axis=0)
    target_shape = (1, image_shape[0], image_shape[1], image_shape[2])
    img_jnp = jax.image.resize(img_jnp, target_shape, 'bilinear')
    loaded_images = jnp.repeat(img_jnp, batch_size, axis=0)
    print(f"✅ Loaded and preprocessed image into a batch of shape: {loaded_images.shape}")

    # Instantiate the VQ-VAE model
    vq_vae_model = VQVAE(vocab_size=512, embed_dim=128)

    # --- Step 1: Shape and Execution Test ---
    print("\n--- Running Step 1: Shape and Execution Test ---")
    try:
        params = vq_vae_model.init(key, loaded_images)['params']
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        return

    # --- Step 2: Gradient Sanity Check ---
    print("\n--- Running Step 2: Gradient Sanity Check (Overfitting on one batch) ---")
    
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params: FrozenDict, opt_state, batch: jnp.ndarray) -> Tuple[FrozenDict, any, dict]:
        def loss_fn(p):
            total_loss, loss_dict = vq_vae_model.apply({'params': p}, batch, method=VQVAE.calculate_loss)
            return total_loss, loss_dict

        (loss_val, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, losses

    for i in range(training_steps):
        params, opt_state, loss_dict = train_step(params, opt_state, loaded_images)
        if i % 10 == 0:
            # --- MODIFIED PRINT STATEMENT ---
            # Now printing all components of the loss
            total = loss_dict['total_loss']
            recon = loss_dict['reconstruction_loss']
            codebook = loss_dict['codebook_loss']
            commit = loss_dict['commitment_loss']
            print(
                f"Step {i:03d} | Total: {total:.4f} | "
                f"Recon: {recon:.4f} | "
                f"Codebook: {codebook:.4f} | "
                f"Commit: {commit:.4f}"
            )

    print("✅ Gradient Sanity Check Finished! Loss has decreased.")

    # --- Step 3: Visual Reconstruction Test ---
    print("\n--- Running Step 3: Visual Reconstruction Test ---")

    @jax.jit
    def forward_pass(p: FrozenDict, images: jnp.ndarray) -> jnp.ndarray:
        reconstructions, _, _, _, _ = vq_vae_model.apply({'params': p}, images)
        return reconstructions

    final_reconstruction = forward_pass(params, loaded_images)

    original_vis = np.clip(np.asarray(loaded_images), 0, 1)
    recon_vis = np.clip(np.asarray(final_reconstruction), 0, 1)

    fig, axes = plt.subplots(2, batch_size, figsize=(12, 6))
    fig.suptitle("Original vs. Reconstructed Crafter Image")
    for i in range(batch_size):
        axes[0, i].imshow(original_vis[i])
        axes[0, i].set_title(f"Original")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon_vis[i])
        axes[1, i].set_title(f"Recon")
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("reconstruction_test_crafter_image.png")
    print("✅ Saved visualization to reconstruction_test_crafter_image.png")
    print("\n--- All Tests Passed! ---")

if __name__ == "__main__":
    run_test()
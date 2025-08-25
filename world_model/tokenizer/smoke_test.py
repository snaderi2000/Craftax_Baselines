# smoke_test.py

import jax
import numpy as np
import optax
from PIL import Image
import matplotlib.pyplot as plt
from flax.training import train_state
from functools import partial

# Import the models and configs you created
from nets import EncoderDecoderConfig  # <<< THE MISSING LINE
from tokenizer import Tokenizer, compute_loss

def run_training():
    """
    Initializes and trains the Tokenizer model.
    """
    print("🚀 Starting VQ-VAE training process...")

    # --- 1. Configuration ---
    IMG_RESOLUTION = 64
    IMG_PATH = 'crafter.png'
    LEARNING_RATE = 1e-3
    TRAINING_STEPS = 1001  # Increase this for better results

    config = EncoderDecoderConfig(
        resolution=IMG_RESOLUTION, in_channels=3, z_channels=128, ch=64,
        ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[],
        out_ch=3, dropout=0.0
    )

    # --- 2. Load Data ---
    pil_img = Image.open(IMG_PATH).convert('RGB').resize((IMG_RESOLUTION, IMG_RESOLUTION))
    img_array = np.array(pil_img) / 255.0
    dummy_dataset_batch = np.expand_dims(img_array, axis=0)
    print(f"✅ Training data shape: {dummy_dataset_batch.shape}")

    # --- 3. Initialize Model and Optimizer ---
    key = jax.random.PRNGKey(0)
    init_key, train_key = jax.random.split(key)

    model = Tokenizer(
        vocab_size=512, embed_dim=128,
        encoder_config=config, decoder_config=config
    )

    params = model.init(init_key, dummy_dataset_batch, train=False)['params']
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    # --- 4. Define the Training Step ---
    @jax.jit
    def train_step(params, opt_state, batch, dropout_rng):
        rngs = {'dropout': dropout_rng}
        
        loss_fn = lambda p: compute_loss(model, p, batch, rngs).total_loss
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss_val

    # --- 5. Run the Training Loop ---
    print(f"🏃‍♂️ Starting training for {TRAINING_STEPS} steps...")
    for step in range(TRAINING_STEPS):
        train_key, dropout_key = jax.random.split(train_key)
        batch = {'observations': dummy_dataset_batch}
        
        params, opt_state, loss = train_step(params, opt_state, batch, dropout_key)
        
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    print("✅ Training finished.")

    # --- 6. Check the Result with Trained Parameters ---
    _, _, reconstructed_batch = model.apply({'params': params}, dummy_dataset_batch, train=False)
    reconstructed_img = np.clip(np.array(reconstructed_batch[0]), 0.0, 1.0)
    
    reconstructed_pil = Image.fromarray((reconstructed_img * 255).astype(np.uint8))
    reconstructed_pil.save("reconstructed_trained.png")
    print("💾 Trained reconstruction saved to 'reconstructed_trained.png'")
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pil_img)
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed_img)
    axes[1].set_title("Reconstructed (Trained)")
    plt.show()

if __name__ == "__main__":
    run_training()
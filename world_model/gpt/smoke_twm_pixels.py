import jax
import jax.numpy as jnp
import optax
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from typing import Tuple

# --- 1. Set up system paths ---
# Get the base directory (e.g., /.../world_model/) by going up one level from this script
base_dir = Path(__file__).resolve().parent.parent

# Define paths to modules relative to the base directory
nano_gpt_path = base_dir / "third_party" / "nanoGPT-jax"
tokenizer_path = base_dir / "tokenizer"

# Add these paths to Python's search path
sys.path.extend([str(nano_gpt_path), str(tokenizer_path)])

# --- 2. Imports after path modification ---
try:
    from vq_vae import VQVAE
    from model import GPT, GPTConfig
    print("✅ Modules imported successfully.")

except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please check your directory structure. Expected structure is:")
    print("world_model/\n"
          "├── third_party/nanoGPT-jax/model.py\n"
          "├── tokenizer/vq_vae.py\n"
          "└── gpt/smoke_twm_pixels.py")
    sys.exit(1)


def load_crafter_image(path: Path) -> jnp.ndarray:
    """Loads, resizes, and normalizes an image, returning a JAX array with a batch dim."""
    img = Image.open(path).convert("RGB").resize((63, 63))
    arr = np.array(img).astype(np.float32) / 255.0
    return jnp.expand_dims(arr, axis=0)


def main():
    # --- 3. Load Data & Setup Keys ---
    img_path = base_dir / "tokenizer" / "crafter.png"
    img_batch = load_crafter_image(img_path)
    print(f"Loaded image from: {img_path}")

    key = jax.random.PRNGKey(0)
    vqvae_init_key, gpt_init_key, gpt_train_key, gpt_gen_key = jax.random.split(key, 4)

    # --- 4. Initialize and Train VQ-VAE ---
    vqvae = VQVAE(vocab_size=512, embed_dim=128)
    vqvae_params = vqvae.init(vqvae_init_key, img_batch)['params']
    
    vqvae_optimizer = optax.adam(1e-3)
    vqvae_opt_state = vqvae_optimizer.init(vqvae_params)

    @jax.jit
    def vqvae_train_step(params, opt_state, batch):
        def loss_fn(p):
            total_loss, logs = vqvae.apply({'params': p}, batch, method=vqvae.calculate_loss)
            return total_loss, logs

        (loss_val, logs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = vqvae_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, logs
    
    print("\n--- Training VQ-VAE (smoke test) ---")
    for step in range(51):
        vqvae_params, vqvae_opt_state, logs = vqvae_train_step(
            vqvae_params, vqvae_opt_state, img_batch
        )
        if step % 10 == 0:
            print(f"[VQ-VAE] Step {step:02d} | Loss={logs['total_loss']:.4f}")

    # --- 5. Encode Image to Tokens ---
    _, _, _, _, tokens = vqvae.apply({'params': vqvae_params}, img_batch)
    tokens = tokens.reshape(1, -1)
    print(f"\nEncoded image to tokens with shape: {tokens.shape}")

    # --- 6. Initialize and Train GPT World Model ---
    gpt_config = GPTConfig(
        block_size=tokens.shape[1],
        vocab_size=512,
        n_layer=3, n_head=8, n_embd=128
    )
    gpt = GPT(gpt_config)
    
    init_rngs = {'params': gpt_init_key, 'dropout': gpt_init_key}
    gpt_params = gpt.init(init_rngs, tokens, train=True)['params']
    
    gpt_optimizer = optax.adam(1e-3)
    gpt_opt_state = gpt_optimizer.init(gpt_params)

    @jax.jit
    def gpt_train_step(params, opt_state, tokens, key):
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        
        def loss_fn(p):
            logits, _ = gpt.apply({"params": p}, inputs, train=True, rngs={"dropout": key})
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
            return loss

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = gpt_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    print("\n--- Training GPT (smoke test) ---")
    for step in range(51):
        gpt_train_key, _ = jax.random.split(gpt_train_key)
        gpt_params, gpt_opt_state, loss = gpt_train_step(
            gpt_params, gpt_opt_state, tokens, gpt_train_key
        )
        if step % 10 == 0:
            print(f"[GPT] Step {step:02d} | Loss={loss:.4f}")

    # --- 7. Generate & Decode ---
    print("\n--- Generating and Decoding ---")
    start_token = tokens[:, :1]
    
    generated_tokens = gpt.generate(
        params=gpt_params,
        input_tokens=start_token,
        max_new_tokens=tokens.shape[1] - 1,
        key=gpt_gen_key
    )[0]

    codebook = vqvae_params['quantizer']['codebook']
    gen_vecs_flat = codebook[generated_tokens.flatten()]
    
    h, w = 8, 8
    gen_vecs_spatial = gen_vecs_flat.reshape(1, h, w, -1)
    
    # Call the new 'decode' method by its string name
    reconstructed_image = vqvae.apply(
        {'params': vqvae_params},
        gen_vecs_spatial,
        method='decode'
    )
    
    # --- 9. Save the Final Image ---
    reconstructed_image_np = np.clip(np.array(reconstructed_image[0]) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(reconstructed_image_np).save("twm_reconstruction.png")
    print("\n✅ Successfully saved final reconstruction to 'twm_reconstruction.png'")

if __name__ == "__main__":
    main()
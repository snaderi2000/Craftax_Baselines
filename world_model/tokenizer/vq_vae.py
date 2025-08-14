# vq_vae.py (Corrected for shape mismatch)

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
from typing import Tuple

# --- 1. The Reusable Residual Block (Unchanged) ---
class ResnetBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        h = nn.relu(inputs)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        h = nn.relu(h)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        if inputs.shape[-1] != self.out_channels:
            inputs = nn.Conv(features=self.out_channels, kernel_size=(1, 1))(inputs)

        return inputs + h

# --- 2. The Corrected Encoder ---
# This version now more closely matches the paper's description of 5 ResBlocks
# with downsampling applied after specific blocks.
class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)

        # Block 1 (channels=64), then downsample
        x = ResnetBlock(out_channels=64)(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        
        # Block 2 (channels=64)
        x = ResnetBlock(out_channels=64)(x)
        
        # Block 3 (channels=128), then downsample
        x = ResnetBlock(out_channels=128)(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        
        # Block 4 (channels=128), then downsample
        x = ResnetBlock(out_channels=128)(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)

        # Block 5 (channels=256)
        x = ResnetBlock(out_channels=256)(x)
        
        # Final convolution to get to the embedding dimension
        x = nn.Conv(features=128, kernel_size=(1, 1))(x)
        return x

# --- 3. The Vector Quantizer (Unchanged) ---
class VectorQuantizer(nn.Module):
    vocab_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, z):
        codebook = self.param('codebook', nn.initializers.uniform(), (self.vocab_size, self.embed_dim))
        z_flat = jnp.reshape(z, (-1, self.embed_dim))
        
        d1 = jnp.sum(z_flat**2, axis=1, keepdims=True)
        d2 = jnp.sum(codebook**2, axis=1)
        d3 = 2 * jnp.dot(z_flat, codebook.T)
        distances = d1 + d2 - d3
        
        tokens = jnp.argmin(distances, axis=1)
        tokens = jnp.reshape(tokens, z.shape[:-1])
        
        z_quantized = codebook[tokens]
        return z_quantized, tokens, codebook

# --- 4. The Corrected Decoder ---
# This version carefully reverses the encoder's architecture and uses
# jax.image.resize for upsampling to ensure the output shape is exactly correct.
class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Initial convolution
        x = nn.Conv(features=256, kernel_size=(1, 1))(x)

        # Block 5
        x = ResnetBlock(out_channels=256)(x)
        
        # Upsample then Block 4
        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), 'nearest')
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = ResnetBlock(out_channels=128)(x)
        
        # Upsample then Block 3
        x = jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), 'nearest')
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(x)
        x = ResnetBlock(out_channels=128)(x)

        # Block 2
        x = ResnetBlock(out_channels=64)(x)
        
        # Upsample then Block 1
        x = jax.image.resize(x, (x.shape[0], 63, 63, x.shape[3]), 'nearest') # Upsample to final size
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = ResnetBlock(out_channels=64)(x)
        
        # Final convolution to get back to 3 channels (RGB)
        x = nn.Conv(features=3, kernel_size=(3, 3), padding='SAME')(x)
        return x

# --- 5. The Full VQ-VAE Model and Loss (Unchanged) ---
class VQVAE(nn.Module):
    vocab_size: int = 512
    embed_dim: int = 128

    def setup(self):
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(vocab_size=self.vocab_size, embed_dim=self.embed_dim)
        self.decoder = Decoder()

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z = self.encoder(x)
        z_quantized, tokens, codebook = self.quantizer(z)
        reconstructions = self.decoder(z_quantized)
        return reconstructions, z, z_quantized, codebook, tokens


    @nn.compact
    def decode(self, z_quantized: jnp.ndarray) -> jnp.ndarray:
        """A dedicated method to decode quantized vectors back into an image."""
        return self.decoder(z_quantized)


    def calculate_loss(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        reconstructions, z, z_quantized, _ , _ = self(x)
        
        lambda1 = 1.0
        lambda3 = 1.0
        lambda4 = 0.25

        recon_loss = lambda1 * jnp.mean(jnp.abs(x - reconstructions))
        codebook_loss = lambda3 * jnp.mean((jax.lax.stop_gradient(z) - z_quantized)**2)
        commitment_loss = lambda4 * jnp.mean((z - jax.lax.stop_gradient(z_quantized))**2)
        
        total_loss = recon_loss + codebook_loss + commitment_loss
        
        loss_dict = {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }
        
        return total_loss, loss_dict
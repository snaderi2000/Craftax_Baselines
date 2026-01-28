import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Tuple
import functools

# --- 1. Basic Layers (Matches nets.py logic) ---

class Swish(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.swish(x)

class ResnetBlock(nn.Module):
    """
    Standard ResNet block used in IRIS/Craftax VQ-VAE.
    Matches PyTorch implementation: GroupNorm -> Swish -> Conv -> ...
    """
    out_channels: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool = True):
        in_channels = x.shape[-1]
        
        # Residual path
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.swish(h)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(h)
        
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        if training and self.dropout_rate > 0:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=False)
        else:
            h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=True)
            
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(h)

        # Shortcut path
        if in_channels != self.out_channels:
            x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)

        return x + h

# --- 2. Encoder & Decoder (Matches Paper Sec A.2.1) ---

class Encoder(nn.Module):
    """
    Paper Sec A.2.1: 
    "The encoder uses a convolutional layer... then five residual blocks... 
    The channel sizes of the residual blocks are (64, 64, 128, 128, 256).
    A downsampling is applied on the first, third and fourth blocks." [cite: 685-687]
    """
    num_hiddens: int = 128  # Final output channels
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial Convolution
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        
        # 5 Residual Blocks with specific channel sizes and strides (Downsampling)
        # Block 1: 64 ch, Downsample
        x = ResnetBlock(out_channels=64)(x, training=training)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x) # Downsample 1
        
        # Block 2: 64 ch
        x = ResnetBlock(out_channels=64)(x, training=training)
        
        # Block 3: 128 ch, Downsample
        x = ResnetBlock(out_channels=128)(x, training=training)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x) # Downsample 2
        
        # Block 4: 128 ch, Downsample
        x = ResnetBlock(out_channels=128)(x, training=training)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x) # Downsample 3
        
        # Block 5: 256 ch
        x = ResnetBlock(out_channels=256)(x, training=training)
        
        # Final Convolution to align with z_channels
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.num_hiddens, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        
        return x

class Decoder(nn.Module):
    """
    Mirrors the Encoder logic.
    """
    out_channels: int = 3 # RGB
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial Conv
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        
        # Block 5 (Reverse)
        x = ResnetBlock(out_channels=256)(x, training=training)
        
        # Block 4 (Reverse + Upsample)
        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = ResnetBlock(out_channels=128)(x, training=training)

        # Block 3 (Reverse + Upsample)
        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = ResnetBlock(out_channels=128)(x, training=training)
        
        # Block 2 (Reverse)
        x = ResnetBlock(out_channels=64)(x, training=training)
        
        # Block 1 (Reverse + Upsample)
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = ResnetBlock(out_channels=64)(x, training=training)
        
        # Final Output
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME')(x)
        
        return x

# --- 3. Vector Quantizer (Matches Paper Normalization Logic) ---

class VectorQuantizer(nn.Module):
    """
    Implements VQ with Codebook Normalization and Latent Normalization.
    Paper: "We use codebook normalization... each latent embedding gets normalized before being quantized." [cite: 692-693]
    """
    num_embeddings: int = 512
    embedding_dim: int = 128 # Typically matches z_channels
    beta: float = 0.25 # Commitment loss weight [cite: 694]

    def setup(self):
        # Codebook init: Uniform as in tokenizer.py
        self.embedding = self.param('embedding', 
                                   nn.initializers.uniform(scale=1.0/self.num_embeddings), 
                                   (self.num_embeddings, self.embedding_dim))

    def __call__(self, z):
        # 1. Normalize Latents (z) [cite: 693]
        z_norm = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-10)
        
        # 2. Normalize Codebook [cite: 692]
        # We project the variables to the unit sphere
        embedding_norm = self.embedding / (jnp.linalg.norm(self.embedding, axis=-1, keepdims=True) + 1e-10)
        
        # 3. Calculate Distances
        # Reshape z to [Batch * Height * Width, Dim]
        b, h, w, c = z.shape
        z_flat = z_norm.reshape((-1, c))
        
        # Distance = ||z||^2 + ||e||^2 - 2*z*e
        # Since both are normalized, ||z||^2 = 1 and ||e||^2 = 1
        # So Distance = 2 - 2 * (z @ e.T)
        # We can just maximize dot product (cosine similarity)
        d = 2.0 - 2.0 * jnp.dot(z_flat, embedding_norm.T)
        
        # 4. Find Nearest Neighbors
        encoding_indices = jnp.argmin(d, axis=-1)
        z_q = embedding_norm[encoding_indices] # Quantized values
        
        # Reshape back to image
        z_q = z_q.reshape((b, h, w, c))
        
        # 5. Losses
        # Commitment loss: ||sg(z) - z_q||^2  (Not used in normalized version usually, but paper implies it)
        # Codebook loss:   ||z - sg(z_q)||^2
        
        # The paper uses:
        # L_VQ = ||sg(z) - z_hat||^2 (codebook) + beta * ||z - sg(z_hat)||^2 (commitment)
        # But since we use normalized codes, we apply the loss on the *normalized* z and z_q usually, 
        # or the unnormalized. The paper says "minimize Equation (5)" using the normalized codes.
        
        codebook_loss = jnp.mean((jax.lax.stop_gradient(z_norm) - z_q) ** 2)
        commitment_loss = jnp.mean((z_norm - jax.lax.stop_gradient(z_q)) ** 2)
        
        # 6. Straight Through Estimator
        # z_q = z + (z_q - z).detach()
        # We pass the gradients from z_q back to z_norm, and then back to z
        z_q = z_norm + jax.lax.stop_gradient(z_q - z_norm)
        
        return z_q, codebook_loss, commitment_loss * self.beta, encoding_indices

# --- 4. Main VQ-VAE Model ---

class VQVAE(nn.Module):
    num_embeddings: int = 512
    embedding_dim: int = 128
    
    def setup(self):
        self.encoder = Encoder(num_hiddens=self.embedding_dim)
        self.decoder = Decoder()
        self.quantizer = VectorQuantizer(num_embeddings=self.num_embeddings, 
                                         embedding_dim=self.embedding_dim)

    def __call__(self, x, training: bool = True):
        # Encode
        z = self.encoder(x, training=training)
        
        # Pre-quant convolution (map to embedding dim if needed, though we matched them here)
        # In tokenizer.py: self.pre_quant_conv
        # We built it into the end of Encoder for simplicity, but can add 1x1 conv here if strict matching needed.
        
        # Quantize
        z_q, loss_codebook, loss_commitment, indices = self.quantizer(z)
        
        # Decode
        x_recon = self.decoder(z_q, training=training)
        
        return x_recon, loss_codebook, loss_commitment, indices

    def encode_indices(self, x):
        """Helper for inference/rollouts"""
        z = self.encoder(x, training=False)
        _, _, _, indices = self.quantizer(z)
        return indices
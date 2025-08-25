# tokenizer.py

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass
from einops import rearrange

# Assuming nets.py is in the same directory
from .nets import Encoder, Decoder, EncoderDecoderConfig

@dataclass
class TokenizerEncoderOutput:
    """Output of the Tokenizer's encoder."""
    z: jnp.ndarray         # Latent features
    z_quantized: jnp.ndarray # Quantized latent features
    tokens: jnp.ndarray    # Discrete tokens

@dataclass
class LossWithIntermediateLosses:
    """Container for the total loss and its components."""
    total_loss: jnp.ndarray
    commitment_loss: jnp.ndarray
    reconstruction_loss: jnp.ndarray
    codebook_loss: jnp.ndarray

class Tokenizer(nn.Module):
    """VQ-VAE Tokenizer model."""
    vocab_size: int
    embed_dim: int
    encoder_config: EncoderDecoderConfig
    decoder_config: EncoderDecoderConfig
        
    def setup(self):
        self.encoder = Encoder(config=self.encoder_config)
        self.pre_quant_conv = nn.Conv(features=self.embed_dim, kernel_size=(1, 1))
        
        # Custom uniform initializer for the embedding table
        embedding_init = nn.initializers.uniform(scale=1.0 / self.vocab_size)
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_dim,
            embedding_init=embedding_init
        )
        
        self.post_quant_conv = nn.Conv(features=self.decoder_config.z_channels, kernel_size=(1, 1))
        self.decoder = Decoder(config=self.decoder_config)
        
    def __call__(self, x: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass: encode, quantize, and decode."""
        outputs = self.encode(x, train=train)
        
        # Straight-through estimator for gradients
        decoder_input = outputs.z + jax.lax.stop_gradient(outputs.z_quantized - outputs.z)
        
        reconstructions = self.decode(decoder_input, train=train)
        return outputs.z, outputs.z_quantized, reconstructions

    def encode(self, x: jnp.ndarray, *, train: bool) -> TokenizerEncoderOutput:
        """Encodes an image to discrete tokens."""
        # Pre-process from [0, 1] to [-1, 1]
        x = self.preprocess_input(x)
        
        # Encoder and pre-quantization conv
        z = self.encoder(x, train=train)
        z = self.pre_quant_conv(z)
        
        b, h, w, e = z.shape
        
        # Flatten for quantization
        z_flattened = rearrange(z, 'b h w e -> (b h w) e')
        
        # **MODIFICATION**: Normalize latents and codebook as per the paper
        z_normalized = z_flattened / (jnp.linalg.norm(z_flattened, axis=-1, keepdims=True) + 1e-6)
        codebook = self.embedding.embedding
        codebook_normalized = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-6)
        
        # Calculate squared L2 distance
        dist_sq = (
            jnp.sum(z_normalized**2, axis=1, keepdims=True) + 
            jnp.sum(codebook_normalized**2, axis=1) -
            2 * jnp.dot(z_normalized, codebook_normalized.T)
        )
        
        # Find closest codebook entry
        tokens = jnp.argmin(dist_sq, axis=-1)
        
        # Quantize by looking up the tokens in the original (un-normalized) codebook
        z_q_flattened = codebook[tokens]
        z_quantized = rearrange(z_q_flattened, '(b h w) e -> b h w e', b=b, h=h, w=w)
        
        tokens_reshaped = rearrange(tokens, '(b h w) -> b (h w)', b=b, h=h, w=w)
        
        return TokenizerEncoderOutput(z=z, z_quantized=z_quantized, tokens=tokens_reshaped)

    def decode(self, z_q: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """Decodes quantized latents back to an image."""
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q, train=train)
        
        # Post-process from [-1, 1] to [0, 1]
        rec = self.postprocess_output(rec)
        return rec


    def decode_from_tokens(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Looks up tokens in the embedding table and decodes to an image."""
        z_quantized_flat = self.embedding(tokens)

        # --- ADD THIS RESHAPE LOGIC ---
        # Infer batch size and embedding dim
        B, L, C = z_quantized_flat.shape
        # The latent grid size must be a square root of the number of tokens
        H = W = int(L**0.5)
        assert H * W == L, "Number of tokens must be a perfect square to reshape."

        # Reshape from (B, L, C) to (B, H, W, C) for the CNN decoder
        z_quantized = z_quantized_flat.reshape(B, H, W, C)
        # --- END OF CHANGE ---

        reconstructions = self.decode(z_quantized, train=False)
        return reconstructions



    def preprocess_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """Assumes input `x` is in [0, 1]. Scales to [-1, 1]."""
        return x * 2.0 - 1.0

    def postprocess_output(self, y: jnp.ndarray) -> jnp.ndarray:
        """Assumes output `y` is in [-1, 1]. Scales to [0, 1]."""
        return (y + 1.0) / 2.0

def compute_loss(
    model: Tokenizer,
    params: Dict,
    batch: Dict,
    rngs: Dict,
    lambda_rec: float = 1.0,
    lambda_codebook: float = 1.0,
    lambda_commitment: float = 0.25
) -> LossWithIntermediateLosses:
    """
    Computes the VQ-VAE loss based on Equation (5) from the paper.
    """
    observations = batch['observations'] # Expected shape: (B, H, W, C)
    
    # Get model outputs. We need `dropout` RNGs for training.
    z, z_quantized, reconstructions = model.apply(
        {'params': params},
        observations,
        train=True,
        rngs=rngs
    )
    # Crop the reconstructions to the same size as the observations
    h, w = observations.shape[1], observations.shape[2]
    reconstructions = reconstructions[:, :h, :w, :]

    # L1 Reconstruction Loss
    reconstruction_loss = lambda_rec * jnp.abs(observations - reconstructions).mean()
    
    # Codebook Loss (moves codebook vectors towards encoder outputs)
    codebook_loss = lambda_codebook * jnp.mean(jnp.square(jax.lax.stop_gradient(z) - z_quantized))

    # Commitment Loss (commits encoder to outputting vectors close to the codebook)
    commitment_loss = lambda_commitment * jnp.mean(jnp.square(z - jax.lax.stop_gradient(z_quantized)))

    total_loss = reconstruction_loss + codebook_loss + commitment_loss
    
    return LossWithIntermediateLosses(
        total_loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        codebook_loss=codebook_loss,
        commitment_loss=commitment_loss
    )
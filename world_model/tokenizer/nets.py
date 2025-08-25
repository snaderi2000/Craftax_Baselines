# nets.py

from functools import partial
from typing import List, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import dataclass

@dataclass
class EncoderDecoderConfig:
    """Configuration for the Encoder and Decoder."""
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: List[int]
    num_res_blocks: int
    attn_resolutions: List[int]
    out_ch: int
    dropout: float

def nonlinearity(x: jnp.ndarray) -> jnp.ndarray:
    """Swish activation function."""
    return jax.nn.silu(x)

class Normalize(nn.Module):
    """Group normalization layer."""
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.GroupNorm(num_groups=32, epsilon=1e-6)(x)

class Upsample(nn.Module):
    """Upsampling layer with optional convolution."""
    in_channels: int
    with_conv: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * 2, W * 2, C), method='nearest')
        if self.with_conv:
            x = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME'
            )(x)
        return x

class Downsample(nn.Module):
    """Downsampling layer with optional convolution."""
    in_channels: int
    with_conv: bool

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.with_conv:
            x = nn.Conv(
                features=self.in_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME'
            )(x)
        else:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x

class ResnetBlock(nn.Module):
    """Residual block for the autoencoder."""
    in_channels: int
    out_channels: int
    dropout: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        h = x
        h = Normalize()(h)
        h = nonlinearity(h)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding='SAME')(h)
        
        h = Normalize()(h)
        h = nonlinearity(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding='SAME')(h)

        if self.in_channels != self.out_channels:
            x = nn.Conv(features=self.out_channels, kernel_size=(1, 1), padding='VALID')(x)

        return x + h

class AttnBlock(nn.Module):
    """Self-attention block."""
    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h_ = x
        h_ = Normalize()(h_)
        q = nn.Conv(features=self.in_channels, kernel_size=(1, 1), padding='VALID')(h_)
        k = nn.Conv(features=self.in_channels, kernel_size=(1, 1), padding='VALID')(h_)
        v = nn.Conv(features=self.in_channels, kernel_size=(1, 1), padding='VALID')(h_)

        # Compute attention
        b, h, w, c = q.shape
        q = jnp.reshape(q, (b, h * w, c))
        k = jnp.reshape(k, (b, h * w, c))
        v = jnp.reshape(v, (b, h * w, c))

        w_ = jnp.einsum('bqc,bkc->bqk', q, k) * (c ** -0.5)
        w_ = jax.nn.softmax(w_, axis=-1)

        # Attend to values
        h_ = jnp.einsum('bqk,bkc->bqc', w_, v)
        h_ = jnp.reshape(h_, (b, h, w, c))

        h_ = nn.Conv(features=self.in_channels, kernel_size=(1, 1), padding='VALID')(h_)
        return x + h_

class Encoder(nn.Module):
    """Convolutional Encoder."""
    config: EncoderDecoderConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        # Note: Time embeddings (`temb`) are omitted as they are unused in the original code.
        
        # Downsampling
        h = nn.Conv(features=self.config.ch, kernel_size=(3, 3), padding='SAME')(x)
        
        curr_res = self.config.resolution
        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        
        # Downsampling blocks
        for i_level in range(len(self.config.ch_mult)):
            block_in = self.config.ch * in_ch_mult[i_level]
            block_out = self.config.ch * self.config.ch_mult[i_level]
            for i_block in range(self.config.num_res_blocks):
                h = ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=self.config.dropout
                )(h, train=train)
                block_in = block_out
                if curr_res in self.config.attn_resolutions:
                    h = AttnBlock(in_channels=block_in)(h)
            if i_level != len(self.config.ch_mult) - 1:
                h = Downsample(in_channels=block_in, with_conv=True)(h)
                curr_res //= 2
        
        # Middle
        h = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=self.config.dropout)(h, train=train)
        h = AttnBlock(in_channels=block_in)(h)
        h = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=self.config.dropout)(h, train=train)
        
        # End
        h = Normalize()(h)
        h = nonlinearity(h)
        h = nn.Conv(features=self.config.z_channels, kernel_size=(3, 3), padding='SAME')(h)
        return h


class Decoder(nn.Module):
    """Convolutional Decoder."""
    config: EncoderDecoderConfig

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        num_resolutions = len(self.config.ch_mult)
        block_in = self.config.ch * self.config.ch_mult[num_resolutions - 1]
        
        # z to block_in
        h = nn.Conv(features=block_in, kernel_size=(3, 3), padding='SAME')(z)
        
        # Middle
        h = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=self.config.dropout)(h, train=train)
        h = AttnBlock(in_channels=block_in)(h)
        h = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=self.config.dropout)(h, train=train)
        
        # Upsampling
        curr_res = self.config.resolution // 2**(num_resolutions - 1)
        for i_level in reversed(range(num_resolutions)):
            block_out = self.config.ch * self.config.ch_mult[i_level]
            for i_block in range(self.config.num_res_blocks + 1):
                h = ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=self.config.dropout
                )(h, train=train)
                block_in = block_out
                if curr_res in self.config.attn_resolutions:
                    h = AttnBlock(in_channels=block_in)(h)
            if i_level != 0:
                h = Upsample(in_channels=block_in, with_conv=True)(h)
                curr_res *= 2
                
        # End
        h = Normalize()(h)
        h = nonlinearity(h)
        h = nn.Conv(features=self.config.out_ch, kernel_size=(3, 3), padding='SAME')(h)
        return h
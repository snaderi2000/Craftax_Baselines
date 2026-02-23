import jax
import jax.numpy as jnp
from flax import linen as nn


def _safe_l2_normalize(x, axis=-1, eps=1e-6):
    denom = jnp.maximum(jnp.linalg.norm(x, axis=axis, keepdims=True), eps)
    return x / denom


class PatchVectorQuantizer(nn.Module):
    num_embeddings: int = 512
    embedding_dim: int = 128

    def setup(self):
        self.embedding = self.param(
            "embedding",
            nn.initializers.uniform(scale=1.0 / self.num_embeddings),
            (self.num_embeddings, self.embedding_dim),
        )

    def __call__(self, z):
        # z: (B, L, D), one embedding per patch token.
        # Use normalized vectors for nearest-neighbor lookup.
        z_lookup = _safe_l2_normalize(z)
        emb_norm = _safe_l2_normalize(self.embedding)

        b, l, d = z_lookup.shape
        z_flat = z_lookup.reshape(-1, d)

        # Nearest-neighbor lookup via cosine distance on normalized vectors.
        dist = 2.0 - 2.0 * jnp.dot(z_flat, emb_norm.T)
        indices = jnp.argmin(dist, axis=-1)
        z_q = emb_norm[indices].reshape(b, l, d)

        # Eq. (5)-style losses on latent space with stop-gradient.
        codebook_loss = jnp.mean((jax.lax.stop_gradient(z) - z_q) ** 2)
        commitment_loss = jnp.mean((z - jax.lax.stop_gradient(z_q)) ** 2)

        # Straight-through estimator: forward uses quantized embedding, backward
        # passes through the pre-quantized encoder output z.
        z_q_st = z + jax.lax.stop_gradient(z_q - z)
        return z_q_st, codebook_loss, commitment_loss, indices.reshape(b, l)


class PatchVQVAE(nn.Module):
    # Patch-factorized VQ-VAE used for M3.
    num_embeddings: int = 512
    embedding_dim: int = 128
    encoder_hidden_dim: int = 128
    patch_size: int = 7
    image_size: int = 63
    lambda_l1: float = 0.1
    lambda_l2: float = 1.0
    lambda_codebook: float = 1.0
    lambda_commitment: float = 0.02

    def setup(self):
        self.patch_dim = self.patch_size * self.patch_size * 3
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.encoder_fc1 = nn.Dense(self.encoder_hidden_dim)
        self.encoder_fc2 = nn.Dense(self.embedding_dim)
        self.decoder_fc = nn.Dense(self.patch_dim)
        self.quantizer = PatchVectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )

    def _patchify(self, x):
        b, h, w, c = x.shape
        p = self.patch_size
        g = self.grid_size
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"PatchVQVAE expects {self.image_size}x{self.image_size} inputs, got {h}x{w}")
        patches = x.reshape(b, g, p, g, p, c).transpose(0, 1, 3, 2, 4, 5)
        return patches.reshape(b, g * g, self.patch_dim)

    def _unpatchify(self, patch_flat):
        b, l, _ = patch_flat.shape
        g = self.grid_size
        p = self.patch_size
        c = 3
        if l != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {l}")
        x = patch_flat.reshape(b, g, g, p, p, c).transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(b, self.image_size, self.image_size, c)

    def _encode_embeddings(self, x):
        patches = self._patchify(x)
        h = self.encoder_fc1(patches)
        h = nn.relu(h)
        z = self.encoder_fc2(h)
        return z

    def __call__(self, x, training: bool = True):
        z = self._encode_embeddings(x)
        z_q, codebook_loss, commitment_loss, indices = self.quantizer(z)
        recon_patches = self.decoder_fc(z_q)
        recon = self._unpatchify(recon_patches)
        return recon, codebook_loss, commitment_loss, indices

    def encode(self, x):
        z = self._encode_embeddings(x)
        _, _, _, indices = self.quantizer(z)
        return indices.astype(jnp.int32)

    def decode_tokens(self, tokens):
        emb = self.quantizer.embedding
        emb_norm = _safe_l2_normalize(emb)
        z_q = emb_norm[tokens]
        recon_patches = self.decoder_fc(z_q)
        return self._unpatchify(recon_patches)

    def get_vq_loss(self, x):
        recon, codebook_loss, commitment_loss, indices = self.__call__(x, training=True)
        l1_loss = jnp.mean(jnp.abs(x - recon))
        l2_loss = jnp.mean(jnp.square(x - recon))
        total_loss = (
            self.lambda_l1 * l1_loss
            + self.lambda_l2 * l2_loss
            + self.lambda_codebook * codebook_loss
            + self.lambda_commitment * commitment_loss
        )
        metrics = {
            "l1": l1_loss,
            "l2": l2_loss,
            "codebook": codebook_loss,
            "commitment": commitment_loss,
            "total_loss": total_loss,
        }
        return recon, indices, total_loss, metrics

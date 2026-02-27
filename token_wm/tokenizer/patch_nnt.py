import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn


class PatchNNT(nn.Module):
    """Patch nearest-neighbor tokenizer (M4-style).

    Tokenization operates directly in raw patch space (default [0,1]) and
    decode is direct codebook lookup.
    """

    codebook_size: int = 4096
    patch_size: int = 7
    image_size: int = 63

    def setup(self):
        self.patch_dim = self.patch_size * self.patch_size * 3
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.codebook = self.param(
            "codebook",
            nn.initializers.zeros,
            (self.codebook_size, self.patch_dim),
        )
        self.codebook_count = self.param(
            "codebook_count",
            lambda *_: jnp.asarray(0, dtype=jnp.int32),
            (),
        )

    def _patchify(self, x):
        b, h, w, c = x.shape
        p = self.patch_size
        g = self.grid_size
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"PatchNNT expects {self.image_size}x{self.image_size} inputs, got {h}x{w}")
        if c != 3:
            raise ValueError(f"PatchNNT expects RGB inputs, got channels={c}")
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

    def encode(self, x):
        patches = self._patchify(x.astype(jnp.float32))  # (B, L, D)
        b, l, d = patches.shape
        flat = patches.reshape(-1, d)  # (N, D)

        cb = self.codebook.astype(jnp.float32)  # (K, D)
        count = jnp.clip(self.codebook_count.astype(jnp.int32), 1, self.codebook_size)

        # Squared L2 distances via ||x||^2 + ||c||^2 - 2x.c
        x2 = jnp.sum(flat * flat, axis=1, keepdims=True)  # (N, 1)
        c2 = jnp.sum(cb * cb, axis=1, keepdims=True).T    # (1, K)
        d2 = jnp.maximum(x2 + c2 - 2.0 * (flat @ cb.T), 0.0)

        # Mask entries >= codebook_count so they are never chosen.
        valid = (jnp.arange(self.codebook_size, dtype=jnp.int32) < count)[None, :]
        d2 = jnp.where(valid, d2, jnp.asarray(1e9, dtype=d2.dtype))
        tokens = jnp.argmin(d2, axis=1).astype(jnp.int32)
        return tokens.reshape(b, l)

    def decode_tokens(self, tokens):
        tok = tokens.astype(jnp.int32)
        tok = jnp.clip(tok, 0, self.codebook_size - 1)
        patch_flat = self.codebook[tok]
        return self._unpatchify(patch_flat)

    def __call__(self, x):
        tokens = self.encode(x)
        recon = self.decode_tokens(tokens)
        return recon, tokens


def _preprocess_obs_np(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max > 1.01:
        x = x / 255.0
    elif x_min < 0.0 and x_max <= 1.0:
        x = (x + 1.0) * 0.5
    return np.clip(x, 0.0, 1.0)


def _patchify_np(obs: np.ndarray, patch_size: int) -> np.ndarray:
    b, h, w, c = obs.shape
    if c != 3:
        raise ValueError(f"Expected RGB observations, got shape={obs.shape}")
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size {h}x{w} is not divisible by patch_size={patch_size}")
    gh = h // patch_size
    gw = w // patch_size
    patches = obs.reshape(b, gh, patch_size, gw, patch_size, c).transpose(0, 1, 3, 2, 4, 5)
    return patches.reshape(b, gh * gw, patch_size * patch_size * c)


def build_nnt_codebook_from_obs(
    obs: np.ndarray,
    patch_size: int = 7,
    codebook_size: int = 4096,
    tau_sq: float = 0.75,
):
    """Greedy Eq.(1)-style codebook construction in raw patch space."""
    x = _preprocess_obs_np(obs)
    patches = _patchify_np(x, patch_size).reshape(-1, patch_size * patch_size * 3)
    if patches.shape[0] == 0:
        codebook = np.zeros((codebook_size, patch_size * patch_size * 3), dtype=np.float32)
        return codebook, 0

    cb = np.zeros((codebook_size, patches.shape[1]), dtype=np.float32)
    cb_count = 0
    for i in range(patches.shape[0]):
        p = patches[i]
        if cb_count == 0:
            cb[0] = p
            cb_count = 1
            continue
        d2 = np.sum((cb[:cb_count] - p[None, :]) ** 2, axis=1)
        if float(np.min(d2)) > tau_sq:
            if cb_count < codebook_size:
                cb[cb_count] = p
                cb_count += 1
            else:
                break
    return cb, cb_count


def extend_nnt_codebook_from_obs(
    codebook: np.ndarray,
    codebook_count: int,
    obs: np.ndarray,
    patch_size: int = 7,
    tau_sq: float = 0.75,
):
    """Append-only update of existing NNT codebook from new observations.

    Existing entries are never changed; only new entries may be appended while
    `codebook_count < codebook.shape[0]`.
    """
    cb = np.asarray(codebook, dtype=np.float32).copy()
    kmax, patch_dim = cb.shape
    count = int(np.clip(codebook_count, 0, kmax))
    x = _preprocess_obs_np(obs)
    patches = _patchify_np(x, patch_size).reshape(-1, patch_dim)

    added = 0
    for i in range(patches.shape[0]):
        p = patches[i]
        if count == 0:
            cb[0] = p
            count = 1
            added += 1
            continue
        d2 = np.sum((cb[:count] - p[None, :]) ** 2, axis=1)
        if float(np.min(d2)) > tau_sq:
            if count < kmax:
                cb[count] = p
                count += 1
                added += 1
            else:
                break
    return cb, count, added


def set_nnt_codebook_params(params, codebook: np.ndarray, count: int):
    """Return params with updated codebook + codebook_count.

    Supports both plain dict and FrozenDict.
    """
    from flax.core.frozen_dict import freeze, unfreeze

    cb = jnp.asarray(codebook, dtype=jnp.float32)
    cnt = jnp.asarray(int(count), dtype=jnp.int32)

    if isinstance(params, dict):
        out = dict(params)
        out_params = dict(out.get("params", {}))
        out_params["codebook"] = cb
        out_params["codebook_count"] = cnt
        out["params"] = out_params
        return out

    mutable = unfreeze(params)
    mutable["params"]["codebook"] = cb
    mutable["params"]["codebook_count"] = cnt
    return freeze(mutable)

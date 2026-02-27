#!/usr/bin/env python3
"""
M4 NNT tokenizer preflight:
- Build nearest-neighbor patch codebook from replay-buffer observations.
- Encode/decode eval frames.
- Save recon metrics + a real/recon image sheet.

This is intentionally tokenizer-only (no TWM, no PPO) so you can validate
M4 encode/decode behavior before launching expensive training runs.
"""

import argparse
import gzip
import math
import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _load_pickle_maybe_gz(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_flat_buffer(checkpoint_dir: str):
    p1 = os.path.join(checkpoint_dir, "flat_buffer.pkl")
    p2 = p1 + ".gz"
    if os.path.exists(p2):
        return _load_pickle_maybe_gz(p2)
    if os.path.exists(p1):
        return _load_pickle_maybe_gz(p1)
    raise FileNotFoundError(f"Missing flat buffer in {checkpoint_dir} (expected flat_buffer.pkl[.gz])")


def _preprocess_obs(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max > 1.01:
        x = x / 255.0
    elif x_min < 0.0 and x_max <= 1.0:
        x = (x + 1.0) * 0.5
    return np.clip(x, 0.0, 1.0)


def _patchify(obs: np.ndarray, patch_size: int) -> np.ndarray:
    # obs: [B, H, W, 3]
    b, h, w, c = obs.shape
    if c != 3:
        raise ValueError(f"Expected RGB observations, got shape={obs.shape}")
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size {h}x{w} is not divisible by patch_size={patch_size}")
    gh = h // patch_size
    gw = w // patch_size
    patches = obs.reshape(b, gh, patch_size, gw, patch_size, c).transpose(0, 1, 3, 2, 4, 5)
    return patches.reshape(b, gh * gw, patch_size * patch_size * c)


def _unpatchify(patches: np.ndarray, patch_size: int, image_size: int) -> np.ndarray:
    # patches: [B, L, patch_dim]
    b, l, pdim = patches.shape
    c = 3
    expect = patch_size * patch_size * c
    if pdim != expect:
        raise ValueError(f"patch_dim mismatch: got {pdim}, expected {expect}")
    g = image_size // patch_size
    if g * g != l:
        raise ValueError(f"token length mismatch: got {l}, expected {g*g}")
    x = patches.reshape(b, g, g, patch_size, patch_size, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, image_size, image_size, c)


def _safe_l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


@dataclass
class NNTConfig:
    codebook_size: int = 4096
    distance_threshold: float = 0.75
    normalize_features: bool = False


class PatchNNTTokenizer:
    def __init__(self, patch_dim: int, config: NNTConfig):
        self.patch_dim = patch_dim
        self.cfg = config
        self.codebook_raw = np.zeros((0, patch_dim), dtype=np.float32)
        self.codebook_repr = np.zeros((0, patch_dim), dtype=np.float32)

    def _repr(self, patches: np.ndarray) -> np.ndarray:
        if self.cfg.normalize_features:
            return _safe_l2_normalize(patches, axis=-1)
        return patches

    def fit(self, patches: np.ndarray):
        if patches.ndim != 2 or patches.shape[1] != self.patch_dim:
            raise ValueError(f"Expected patches shape [N,{self.patch_dim}], got {patches.shape}")
        patch_repr = self._repr(patches.astype(np.float32))
        k = 0
        for i in range(patch_repr.shape[0]):
            p_repr = patch_repr[i]
            p_raw = patches[i]
            if k == 0:
                self.codebook_raw = np.expand_dims(p_raw, axis=0)
                self.codebook_repr = np.expand_dims(p_repr, axis=0)
                k = 1
                continue
            # Paper Eq. (1): compare squared Euclidean distance against tau.
            d2 = np.sum((self.codebook_repr - p_repr[None, :]) ** 2, axis=1)
            if float(np.min(d2)) > self.cfg.distance_threshold and k < self.cfg.codebook_size:
                self.codebook_raw = np.concatenate([self.codebook_raw, p_raw[None, :]], axis=0)
                self.codebook_repr = np.concatenate([self.codebook_repr, p_repr[None, :]], axis=0)
                k += 1
                if k >= self.cfg.codebook_size:
                    break

    def encode(self, patches: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        if self.codebook_repr.shape[0] == 0:
            raise RuntimeError("Codebook is empty; call fit() first.")
        p = patches.astype(np.float32)
        p_repr = self._repr(p)
        out = np.zeros((p_repr.shape[0],), dtype=np.int32)
        k = self.codebook_repr.shape[0]
        for s in range(0, p_repr.shape[0], batch_size):
            e = min(s + batch_size, p_repr.shape[0])
            x = p_repr[s:e]
            # Squared L2 distances via ||x||^2 + ||c||^2 - 2 x c^T
            x2 = np.sum(x * x, axis=1, keepdims=True)  # [B,1]
            c2 = np.sum(self.codebook_repr * self.codebook_repr, axis=1, keepdims=True).T  # [1,K]
            d2 = np.maximum(x2 + c2 - 2.0 * (x @ self.codebook_repr.T), 0.0)
            out[s:e] = np.argmin(d2, axis=1).astype(np.int32)
        return out

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        if self.codebook_raw.shape[0] == 0:
            raise RuntimeError("Codebook is empty; call fit() first.")
        if np.any(tokens < 0) or np.any(tokens >= self.codebook_raw.shape[0]):
            raise ValueError("Token index out of range for current codebook.")
        return self.codebook_raw[tokens]


def _save_sheet(path: str, real: np.ndarray, recon: np.ndarray, scale: int = 4):
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        print(f"Warning: PIL not available, skipping sheet save ({e})")
        return

    n, h, w, _ = real.shape
    gap = 2
    row_gap = 14
    canvas_h = (h * scale) * 2 + row_gap
    canvas_w = n * (w * scale + gap) - gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(12, 14, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, 0), "row0=real   row1=nnt_recon", fill=(220, 220, 220))

    def _to_img(x: np.ndarray) -> Image.Image:
        u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(u8, mode="RGB")
        if scale != 1:
            img = img.resize((w * scale, h * scale), resample=Image.NEAREST)
        return img

    y0 = row_gap
    y1 = row_gap + h * scale
    for i in range(n):
        x = i * (w * scale + gap)
        canvas.paste(_to_img(real[i]), (x, y0))
        canvas.paste(_to_img(recon[i]), (x, y1))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    canvas.save(path)


def _load_or_create_indices(path: str, count: int, n: int, rng: np.random.Generator) -> np.ndarray:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        idx = np.load(path)
        idx = idx.astype(np.int64).ravel()
        idx = idx[(idx >= 0) & (idx < count)]
        if idx.size > 0:
            return idx[:n]
    take = min(n, count)
    idx = rng.choice(count, size=take, replace=False).astype(np.int64)
    np.save(path, idx)
    return idx


def main():
    p = argparse.ArgumentParser(description="M4 NNT tokenizer encode/decode sanity from checkpoint buffer.")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patch_size", type=int, default=7)
    p.add_argument("--obs_image_size", type=int, default=63)
    p.add_argument("--codebook_size", type=int, default=4096)
    p.add_argument("--distance_threshold", type=float, default=0.75)
    p.add_argument(
        "--normalize_features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Non-paper option. Paper M4 uses raw [0,1] patch space for Euclidean thresholding.",
    )
    p.add_argument("--build_frames", type=int, default=4096, help="Number of frames sampled to build codebook.")
    p.add_argument("--eval_frames", type=int, default=64, help="Number of frames used for recon metrics.")
    p.add_argument("--viz_frames", type=int, default=16, help="Number of frames shown in sheet.")
    p.add_argument("--fixed_eval_indices_path", type=str, default="analysis/m4_nnt_eval_indices.npy")
    p.add_argument("--output_png", type=str, default="analysis/m4_nnt_recon_sanity.png")
    p.add_argument("--output_txt", type=str, default="analysis/m4_nnt_recon_sanity.txt")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    flat = _load_flat_buffer(args.checkpoint_dir)
    if not isinstance(flat, (tuple, list)) or len(flat) < 6:
        raise RuntimeError("Unexpected flat_buffer format.")
    obs = _preprocess_obs(np.asarray(flat[0]))
    count = int(np.asarray(flat[5]))
    if count <= 0:
        raise RuntimeError("Flat buffer count is zero.")
    obs = obs[:count]
    if obs.shape[1] != args.obs_image_size or obs.shape[2] != args.obs_image_size:
        raise ValueError(
            f"obs image size mismatch: buffer has {obs.shape[1]}x{obs.shape[2]}, "
            f"expected {args.obs_image_size}x{args.obs_image_size}"
        )

    build_take = min(args.build_frames, count)
    build_idx = rng.choice(count, size=build_take, replace=False)
    eval_idx = _load_or_create_indices(args.fixed_eval_indices_path, count, args.eval_frames, rng)

    build_obs = obs[build_idx]
    eval_obs = obs[eval_idx]

    build_patches = _patchify(build_obs, args.patch_size).reshape(-1, args.patch_size * args.patch_size * 3)
    tokenizer = PatchNNTTokenizer(
        patch_dim=build_patches.shape[1],
        config=NNTConfig(
            codebook_size=args.codebook_size,
            distance_threshold=args.distance_threshold,
            normalize_features=args.normalize_features,
        ),
    )
    tokenizer.fit(build_patches)
    k = tokenizer.codebook_raw.shape[0]
    print(
        f"Built NNT codebook: {k}/{args.codebook_size} entries "
        f"(tau_sq={args.distance_threshold}, normalize={args.normalize_features})"
    )

    eval_patch_seq = _patchify(eval_obs, args.patch_size)  # [B,L,D]
    b, l, d = eval_patch_seq.shape
    eval_flat = eval_patch_seq.reshape(-1, d)
    tok = tokenizer.encode(eval_flat)
    recon_flat = tokenizer.decode(tok)
    recon_obs = _unpatchify(recon_flat.reshape(b, l, d), args.patch_size, args.obs_image_size)

    mae = float(np.mean(np.abs(eval_obs - recon_obs)))
    mse = float(np.mean(np.square(eval_obs - recon_obs)))
    psnr = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))

    binc = np.bincount(tok, minlength=max(1, k)).astype(np.float64)
    probs = binc / max(1.0, float(np.sum(binc)))
    nz = probs[probs > 0]
    entropy = float(-np.sum(nz * np.log(nz + 1e-12)))
    perplexity = float(np.exp(entropy))
    used_codes = int(np.sum(binc > 0))
    top1 = float(np.max(probs)) if probs.size > 0 else 0.0

    viz_n = min(args.viz_frames, b)
    _save_sheet(args.output_png, eval_obs[:viz_n], recon_obs[:viz_n], scale=4)

    os.makedirs(os.path.dirname(args.output_txt) or ".", exist_ok=True)
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("M4 NNT Tokenizer Recon Sanity\n")
        f.write(f"checkpoint_dir={args.checkpoint_dir}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"count={count}\n")
        f.write(f"build_frames={build_take}\n")
        f.write(f"eval_frames={b}\n")
        f.write(f"patch_size={args.patch_size}\n")
        f.write(f"obs_image_size={args.obs_image_size}\n")
        f.write(f"codebook_size_limit={args.codebook_size}\n")
        f.write(f"codebook_size_built={k}\n")
        f.write(f"distance_threshold={args.distance_threshold}\n")
        f.write(f"normalize_features={args.normalize_features}\n")
        f.write(f"mae={mae:.8f}\n")
        f.write(f"mse={mse:.8f}\n")
        f.write(f"psnr={psnr:.4f}\n")
        f.write(f"used_codes={used_codes}\n")
        f.write(f"usage_frac={(used_codes / max(1, k)):.6f}\n")
        f.write(f"entropy={entropy:.8f}\n")
        f.write(f"perplexity={perplexity:.8f}\n")
        f.write(f"top1_code_frac={top1:.8f}\n")
        f.write(f"output_png={args.output_png}\n")

    print(f"Recon metrics: mae={mae:.6f} mse={mse:.6f} psnr={psnr:.2f}dB")
    print(f"Code usage: used={used_codes}/{k} perplexity={perplexity:.3f} top1={top1:.3f}")
    print(f"Saved: {args.output_png}")
    print(f"Saved: {args.output_txt}")


if __name__ == "__main__":
    main()

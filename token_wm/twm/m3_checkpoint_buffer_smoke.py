#!/usr/bin/env python3
"""
M3 checkpoint-buffer smoke test:
1) Load flat + flashbax buffers from an existing checkpoint.
2) Run tokenizer and TWM updates only (no env rollout, no policy updates).
3) Save a lightweight checkpoint-like directory (policy/vqvae/twm params).
4) Optionally run wm_m3_triptych_sheet.py on the updated params.
"""

import argparse
import csv
import gzip
import os
import pickle
import subprocess
import sys
import time

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = CURRENT_DIR
for _ in range(4):
    if os.path.exists(os.path.join(REPO_ROOT, "ppo_mbrl_m3.py")):
        break
    REPO_ROOT = os.path.dirname(REPO_ROOT)
if not os.path.exists(os.path.join(REPO_ROOT, "ppo_mbrl_m3.py")):
    raise RuntimeError("Could not locate repo root containing ppo_mbrl_m3.py")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ppo_mbrl_m3 import (
    _preprocess_obs_for_tokenizer,
    _project_patch_codebook_params,
    create_twm,
    create_vqvae,
    make_twm_update_fn,
)


def _image_grad_mag_mean(x):
    """Mean finite-difference gradient magnitude over H/W axes."""
    gx = jnp.abs(x[:, 1:, :, :] - x[:, :-1, :, :]).mean()
    gy = jnp.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return 0.5 * (gx + gy)


def make_vqvae_update_fn_debug(vqvae):
    """Tokenizer update with explicit finite diagnostics."""
    def _code_stats(indices, vocab_size):
        flat = indices.reshape(-1)
        counts = jnp.bincount(flat, length=vocab_size)
        total = jnp.maximum(counts.sum(), 1)
        probs = counts.astype(jnp.float32) / total.astype(jnp.float32)
        nz = probs > 0
        entropy = -jnp.sum(jnp.where(nz, probs * jnp.log(probs + 1e-12), 0.0))
        perplexity = jnp.exp(entropy)
        used = jnp.sum(counts > 0)
        dead = vocab_size - used
        usage_frac = used.astype(jnp.float32) / jnp.asarray(vocab_size, dtype=jnp.float32)
        top1_code_frac = jnp.max(counts).astype(jnp.float32) / total.astype(jnp.float32)
        return counts, used, dead, usage_frac, entropy, perplexity, top1_code_frac

    def _vqvae_loss_fn(params, obs_batch):
        obs_batch = _preprocess_obs_for_tokenizer(obs_batch)
        recon, indices, total_loss, metrics = vqvae.apply(params, obs_batch, method=vqvae.get_vq_loss)
        z = vqvae.apply(params, obs_batch, method=vqvae._encode_embeddings)
        z_norms = jnp.linalg.norm(z, axis=-1)
        metrics = {
            **metrics,
            "z_norm_min": jnp.min(z_norms),
            "z_norm_mean": jnp.mean(z_norms),
            "z_frac_tiny": jnp.mean((z_norms < 1e-3).astype(jnp.float32)),
            "recon_std_ratio": jnp.std(recon) / jnp.maximum(jnp.std(obs_batch), 1e-8),
            "recon_grad_ratio": _image_grad_mag_mean(recon) / jnp.maximum(_image_grad_mag_mean(obs_batch), 1e-8),
        }
        return total_loss, (metrics, indices)

    @jax.jit
    def vqvae_update_single(vqvae_state, obs_batch):
        grad_fn = jax.value_and_grad(_vqvae_loss_fn, has_aux=True)
        (_, (metrics, indices)), grads = grad_fn(vqvae_state.params, obs_batch)

        grad_is_finite = jnp.array(True, dtype=jnp.bool_)
        for g in jax.tree.leaves(grads):
            if jnp.issubdtype(g.dtype, jnp.floating):
                grad_is_finite = jnp.logical_and(grad_is_finite, jnp.all(jnp.isfinite(g)))

        metric_is_finite = jnp.array(True, dtype=jnp.bool_)
        for v in metrics.values():
            metric_is_finite = jnp.logical_and(metric_is_finite, jnp.all(jnp.isfinite(v)))

        emb_before = vqvae_state.params["params"]["quantizer"]["embedding"]
        emb_before_norms = jnp.linalg.norm(emb_before, axis=-1)

        update_is_finite = jnp.logical_and(grad_is_finite, metric_is_finite)
        vqvae_state = jax.lax.cond(
            update_is_finite,
            lambda s: s.replace(
                params=_project_patch_codebook_params(
                    s.apply_gradients(grads=grads).params
                )
            ),
            lambda s: s,
            vqvae_state,
        )
        emb_after = vqvae_state.params["params"]["quantizer"]["embedding"]
        emb_after_norms = jnp.linalg.norm(emb_after, axis=-1)

        token_counts, used, dead, usage_frac, entropy, perplexity, top1_code_frac = _code_stats(
            indices, vqvae.num_embeddings
        )

        metrics = {
            **metrics,
            "used_codes": used.astype(jnp.float32),
            "dead_codes": dead.astype(jnp.float32),
            "usage_frac": usage_frac,
            "entropy": entropy,
            "perplexity": perplexity,
            "top1_code_frac": top1_code_frac,
            "emb_norm_min_before": jnp.min(emb_before_norms),
            "emb_norm_mean_before": jnp.mean(emb_before_norms),
            "emb_norm_max_before": jnp.max(emb_before_norms),
            "emb_norm_min_after": jnp.min(emb_after_norms),
            "emb_norm_mean_after": jnp.mean(emb_after_norms),
            "emb_norm_max_after": jnp.max(emb_after_norms),
        }
        return (
            vqvae_state,
            metrics,
            token_counts,
            update_is_finite,
            grad_is_finite,
            metric_is_finite,
        )

    return vqvae_update_single


def _load_pickle_maybe_gz(base_path):
    if os.path.exists(base_path):
        with open(base_path, "rb") as f:
            return pickle.load(f)
    gz_path = f"{base_path}.gz"
    if os.path.exists(gz_path):
        with gzip.open(gz_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Could not find {base_path} or {gz_path}")


def _load_pickle_maybe_gz_optional(base_path):
    try:
        return _load_pickle_maybe_gz(base_path)
    except FileNotFoundError:
        return None


def _to_jax(tree):
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, (np.ndarray, np.generic)) else x,
        tree,
    )


def _build_config(args):
    tokens_per_obs = (args.obs_image_size // args.patch_size) ** 2
    return {
        "VQVAE_CODEBOOK_SIZE": args.vqvae_codebook_size,
        "VQVAE_EMBED_DIM": args.vqvae_embed_dim,
        "PATCH_ENCODER_HIDDEN_DIM": args.patch_encoder_hidden_dim,
        "PATCH_SIZE": args.patch_size,
        "OBS_IMAGE_SIZE": args.obs_image_size,
        "VQVAE_LAMBDA_L1": args.vqvae_lambda_l1,
        "VQVAE_LAMBDA_L2": args.vqvae_lambda_l2,
        "VQVAE_LAMBDA_CODEBOOK": args.vqvae_lambda_codebook,
        "VQVAE_LAMBDA_COMMITMENT": args.vqvae_lambda_commitment,
        "VQVAE_LR": args.vqvae_lr,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "TWM_SEQ_LEN": args.twm_seq_len,
        "TWM_NUM_LAYERS": args.twm_num_layers,
        "TWM_NUM_HEADS": args.twm_num_heads,
        "TWM_EMBED_DIM": args.twm_embed_dim,
        "TWM_DROPOUT": args.twm_dropout,
        "TWM_MAX_GRAD_NORM": args.twm_max_grad_norm,
        "TWM_LR": args.twm_lr,
        "NUM_ACTIONS": args.num_actions,
        "USE_BINARY_REWARD_TARGET": args.use_binary_reward_target,
        "BINARY_REWARD_THRESHOLD": args.binary_reward_threshold,
        "TOKENS_PER_OBS": tokens_per_obs,
        "TOKENS_PER_BLOCK": tokens_per_obs + 1,
        "TWM_MAX_BLOCKS": args.twm_seq_len + args.burnin_horizon,
    }


def _load_or_create_fixed_eval_indices(path, buffer_count, num_samples, rng):
    """Load fixed eval indices from disk, or create/save once."""
    n = int(max(1, min(buffer_count, num_samples)))
    if os.path.exists(path):
        loaded = np.load(path)
        loaded = np.asarray(loaded, dtype=np.int32).reshape(-1)
        loaded_max = int(loaded.max()) if loaded.size > 0 else -1
        if loaded.size >= n and loaded_max < buffer_count:
            return loaded[:n], rng, False
        print(
            f"Warning: fixed eval indices at {path} are invalid for buffer_count={buffer_count}; regenerating."
        )

    rng, sample_rng = jax.random.split(rng)
    perm = jax.random.permutation(sample_rng, buffer_count)
    idx = np.asarray(jax.device_get(perm[:n]), dtype=np.int32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, idx)
    return idx, rng, True


def _evaluate_tokenizer_diagnostics(vqvae, vq_params, buffer_obs, eval_indices, codebook_size, batch_size):
    """Evaluate recon quality and codebook usage on a fixed subset of flat buffer."""
    idx = jnp.asarray(eval_indices, dtype=jnp.int32)
    n = int(idx.shape[0])
    obs_eval = buffer_obs[idx]

    token_counts = np.zeros((codebook_size,), dtype=np.int64)
    mae_sum = 0.0
    mse_sum = 0.0
    z_norm_min = float("inf")
    z_norm_sum = 0.0
    z_norm_count = 0
    z_tiny_sum = 0.0
    recon_std_ratio_sum = 0.0
    recon_grad_ratio_sum = 0.0
    seen = 0

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = _preprocess_obs_for_tokenizer(obs_eval[start:end])
        tokens = vqvae.apply(vq_params, batch, method=vqvae.encode)
        recon = vqvae.apply(vq_params, tokens, method=vqvae.decode_tokens)
        z = vqvae.apply(vq_params, batch, method=vqvae._encode_embeddings)
        z_norms = jnp.linalg.norm(z, axis=-1)

        batch_mae = jnp.mean(jnp.abs(batch - recon))
        batch_mse = jnp.mean(jnp.square(batch - recon))
        batch_std_ratio = jnp.std(recon) / jnp.maximum(jnp.std(batch), 1e-8)
        batch_grad_ratio = _image_grad_mag_mean(recon) / jnp.maximum(_image_grad_mag_mean(batch), 1e-8)
        batch_tiny = jnp.mean((z_norms < 1e-3).astype(jnp.float32))
        bsz = end - start
        mae_sum += float(batch_mae) * bsz
        mse_sum += float(batch_mse) * bsz
        recon_std_ratio_sum += float(batch_std_ratio) * bsz
        recon_grad_ratio_sum += float(batch_grad_ratio) * bsz
        z_tiny_sum += float(batch_tiny) * bsz
        z_norm_min = min(z_norm_min, float(jnp.min(z_norms)))
        z_norm_sum += float(jnp.sum(z_norms))
        z_norm_count += int(z_norms.size)
        seen += bsz

        tok_np = np.asarray(jax.device_get(tokens)).reshape(-1)
        token_counts += np.bincount(tok_np, minlength=codebook_size)

    probs = token_counts.astype(np.float64) / max(1, int(token_counts.sum()))
    nz = probs > 0
    entropy = float(-(probs[nz] * np.log(probs[nz])).sum())
    perplexity = float(np.exp(entropy))
    used = int((token_counts > 0).sum())
    dead = int(codebook_size - used)
    usage_frac = float(used) / float(codebook_size)
    top1_code_frac = float(token_counts.max()) / float(max(1, token_counts.sum()))

    topk = min(10, codebook_size)
    top_ids = np.argsort(-token_counts)[:topk]
    top_pairs = [(int(i), int(token_counts[i])) for i in top_ids]

    stats = {
        "n_eval": int(seen),
        "recon_mae": float(mae_sum / max(1, seen)),
        "recon_mse": float(mse_sum / max(1, seen)),
        "recon_std_ratio": float(recon_std_ratio_sum / max(1, seen)),
        "recon_grad_ratio": float(recon_grad_ratio_sum / max(1, seen)),
        "z_norm_min": float(z_norm_min if z_norm_min != float("inf") else 0.0),
        "z_norm_mean": float(z_norm_sum / max(1, z_norm_count)),
        "z_frac_tiny": float(z_tiny_sum / max(1, seen)),
        "code_usage_frac": usage_frac,
        "used_codes": used,
        "dead_codes": dead,
        "token_entropy": entropy,
        "token_perplexity": perplexity,
        "top1_code_frac": top1_code_frac,
        "top_codes": top_pairs,
    }
    return stats


def main():
    p = argparse.ArgumentParser(description="Run tokenizer+TWM smoke updates from an M3 checkpoint buffer.")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints/smoke_m3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fresh_tokenizer", action=argparse.BooleanOptionalAction, default=False,
                   help="Initialize tokenizer params/optimizer from scratch instead of loading checkpoint state.")
    p.add_argument("--fresh_twm", action=argparse.BooleanOptionalAction, default=False,
                   help="Initialize TWM params/optimizer from scratch instead of loading checkpoint state.")

    p.add_argument("--n_iters_tok", type=int, default=100)
    p.add_argument("--n_iters_twm", type=int, default=100)
    p.add_argument("--n_mb_wm", type=int, default=3)
    p.add_argument("--print_each_iter", action=argparse.BooleanOptionalAction, default=True,
                   help="Print per-iteration tokenizer/TWM losses.")
    p.add_argument("--print_every", type=int, default=1,
                   help="Print cadence for iterations when --print_each_iter is enabled.")
    p.add_argument("--vqvae_batch_size", type=int, default=256)
    p.add_argument("--diag_eval_samples", type=int, default=4096,
                   help="Number of flat-buffer observations to sample for tokenizer diagnostics.")
    p.add_argument("--diag_eval_batch_size", type=int, default=512,
                   help="Batch size for tokenizer diagnostics.")
    p.add_argument("--diag_before_after", action=argparse.BooleanOptionalAction, default=True,
                   help="Run tokenizer diagnostics before and after smoke updates.")
    p.add_argument("--diag_log_every", type=int, default=10,
                   help="Run fixed-eval tokenizer diagnostics every N tokenizer iterations.")
    p.add_argument("--diag_fixed_eval_indices_path", type=str, default="",
                   help="Path to .npy of fixed eval indices. Defaults to <output_dir>/diag_eval_indices.npy")
    p.add_argument("--debug_nonfinite_limit", type=int, default=20,
                   help="Maximum number of non-finite tokenizer minibatch diagnostics to print.")
    p.add_argument("--failfast_enabled", action=argparse.BooleanOptionalAction, default=True,
                   help="Stop early when repeated tokenizer collapse/non-finite conditions are detected.")
    p.add_argument("--failfast_patience", type=int, default=5,
                   help="Consecutive bad-iteration threshold to trigger fail-fast.")
    p.add_argument("--failfast_perplexity_min", type=float, default=2.0,
                   help="Fail-fast when per-iteration token perplexity stays below this threshold.")
    p.add_argument("--failfast_top1_frac_max", type=float, default=0.5,
                   help="Fail-fast when per-iteration top-1 token fraction stays above this threshold.")
    p.add_argument("--failfast_z_tiny_frac_max", type=float, default=0.5,
                   help="Fail-fast when per-iteration z_frac_tiny stays above this threshold.")
    p.add_argument("--twm_batch_size", type=int, default=16)
    p.add_argument("--twm_seq_len", type=int, default=20)
    p.add_argument("--burnin_horizon", type=int, default=5)

    p.add_argument("--num_actions", type=int, default=17)
    p.add_argument("--patch_size", type=int, default=7)
    p.add_argument("--obs_image_size", type=int, default=63)
    p.add_argument("--vqvae_codebook_size", type=int, default=512)
    p.add_argument("--vqvae_embed_dim", type=int, default=128)
    p.add_argument("--patch_encoder_hidden_dim", type=int, default=128)
    p.add_argument("--vqvae_lambda_l1", type=float, default=0.1)
    p.add_argument("--vqvae_lambda_l2", type=float, default=1.0)
    p.add_argument("--vqvae_lambda_codebook", type=float, default=1.0)
    p.add_argument("--vqvae_lambda_commitment", type=float, default=0.02)
    p.add_argument("--vqvae_lr", type=float, default=1e-3)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    p.add_argument("--twm_num_layers", type=int, default=3)
    p.add_argument("--twm_num_heads", type=int, default=8)
    p.add_argument("--twm_embed_dim", type=int, default=128)
    p.add_argument("--twm_dropout", type=float, default=0.1)
    p.add_argument("--twm_lr", type=float, default=1e-3)
    p.add_argument("--twm_max_grad_norm", type=float, default=0.5)
    p.add_argument("--use_binary_reward_target", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--binary_reward_threshold", type=float, default=0.5)

    p.add_argument("--run_triptych", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--triptych_horizon", type=int, default=19)
    p.add_argument("--triptych_seed", type=int, default=0)
    p.add_argument("--triptych_output_png", type=str, default="analysis/wm_m3_smoke_triptych.png")
    p.add_argument("--triptych_policy_greedy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--triptych_wm_greedy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--triptych_wm_temperature", type=float, default=1.0)
    args = p.parse_args()

    ckpt = args.checkpoint_dir
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    cfg = _build_config(args)
    rng = jax.random.PRNGKey(args.seed)

    print(f"Loading checkpoint buffer state from: {ckpt}")
    policy_params = _to_jax(_load_pickle_maybe_gz(os.path.join(ckpt, "policy_params.pkl")))
    vqvae_params = None if args.fresh_tokenizer else _to_jax(_load_pickle_maybe_gz(os.path.join(ckpt, "vqvae_params.pkl")))
    twm_params = None if args.fresh_twm else _to_jax(_load_pickle_maybe_gz(os.path.join(ckpt, "twm_params.pkl")))
    vqvae_opt_state_raw = None if args.fresh_tokenizer else _load_pickle_maybe_gz_optional(os.path.join(ckpt, "vqvae_opt_state.pkl"))
    twm_opt_state_raw = None if args.fresh_twm else _load_pickle_maybe_gz_optional(os.path.join(ckpt, "twm_opt_state.pkl"))
    vqvae_opt_state = _to_jax(vqvae_opt_state_raw) if vqvae_opt_state_raw is not None else None
    twm_opt_state = _to_jax(twm_opt_state_raw) if twm_opt_state_raw is not None else None
    flat_buffer = _to_jax(_load_pickle_maybe_gz(os.path.join(ckpt, "flat_buffer.pkl")))
    fbx_state = _to_jax(_load_pickle_maybe_gz(os.path.join(ckpt, "fbx_buffer.pkl")))

    buffer_obs, _, _, _, _, buffer_count = flat_buffer
    buffer_count = int(np.asarray(buffer_count))
    if buffer_count <= 0:
        raise RuntimeError("Flat buffer is empty; cannot run tokenizer smoke updates.")

    print(f"Flat buffer count: {buffer_count}")

    os.makedirs(args.output_dir, exist_ok=True)
    tag = (
        f"smoke_{os.path.basename(os.path.normpath(ckpt))}"
        f"_tok{args.n_iters_tok}x{args.n_mb_wm}_twm{args.n_iters_twm}x{args.n_mb_wm}"
    )
    smoke_ckpt_dir = os.path.join(args.output_dir, tag)
    os.makedirs(smoke_ckpt_dir, exist_ok=True)

    eval_indices_path = args.diag_fixed_eval_indices_path.strip()
    if not eval_indices_path:
        eval_indices_path = os.path.join(args.output_dir, "diag_eval_indices.npy")
    eval_indices, rng, created_eval_idx = _load_or_create_fixed_eval_indices(
        eval_indices_path, buffer_count, args.diag_eval_samples, rng
    )
    if created_eval_idx:
        print(f"Created fixed eval indices: {eval_indices_path} ({eval_indices.shape[0]} samples)")
    else:
        print(f"Loaded fixed eval indices: {eval_indices_path} ({eval_indices.shape[0]} samples)")

    exp = fbx_state.experience
    add_batch_size = int(exp["obs"].shape[0])
    max_length_time_axis = int(exp["obs"].shape[1])
    has_done_after = "done_after" in exp
    print(
        f"Flashbax experience shape: add_batch_size={add_batch_size}, "
        f"max_length_time_axis={max_length_time_axis}, has_done_after={has_done_after}"
    )

    fbx_buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=args.twm_seq_len + 1,
        sample_batch_size=args.twm_batch_size,
        sample_sequence_length=args.twm_seq_len,
        period=1,
        add_batch_size=add_batch_size,
    )
    if not bool(fbx_buffer.can_sample(fbx_state)):
        raise RuntimeError("Flashbax buffer cannot sample with the current twm_seq_len/twm_batch_size.")

    vqvae = create_vqvae(cfg)
    twm = create_twm(cfg)
    rng, vq_init_rng = jax.random.split(rng)
    rng, twm_init_rng = jax.random.split(rng)
    vq_tx = optax.chain(
        optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
        optax.adam(cfg["VQVAE_LR"]),
    )
    twm_tx = optax.chain(
        optax.clip_by_global_norm(cfg["TWM_MAX_GRAD_NORM"]),
        optax.adam(cfg["TWM_LR"]),
    )
    if args.fresh_tokenizer:
        print("Using fresh tokenizer initialization.")
        vq_params_init = vqvae.init(vq_init_rng, jnp.zeros((1, cfg["OBS_IMAGE_SIZE"], cfg["OBS_IMAGE_SIZE"], 3), dtype=jnp.float32))
        vqvae_state = TrainState.create(apply_fn=vqvae.apply, params=vq_params_init, tx=vq_tx)
    else:
        vqvae_state = TrainState.create(apply_fn=vqvae.apply, params=vqvae_params, tx=vq_tx)
    if args.fresh_twm:
        print("Using fresh TWM initialization.")
        twm_params_init = twm.init(twm_init_rng, jnp.zeros((1, cfg["TOKENS_PER_BLOCK"]), dtype=jnp.int32))
        twm_state = TrainState.create(apply_fn=twm.apply, params=twm_params_init, tx=twm_tx)
    else:
        twm_state = TrainState.create(apply_fn=twm.apply, params=twm_params, tx=twm_tx)

    if (not args.fresh_tokenizer) and vqvae_opt_state is not None:
        vqvae_state = vqvae_state.replace(opt_state=vqvae_opt_state)
    else:
        print("Info: tokenizer optimizer state not loaded; using freshly initialized optimizer state.")
    if (not args.fresh_twm) and twm_opt_state is not None:
        twm_state = twm_state.replace(opt_state=twm_opt_state)
    else:
        print("Info: TWM optimizer state not loaded; using freshly initialized optimizer state.")

    if args.fresh_tokenizer and (not args.fresh_twm) and args.n_iters_twm > 0:
        print("Warning: fresh tokenizer + old TWM with twm updates can be unstable; prefer running with --n_iters_twm 0 first.")

    vq_update = make_vqvae_update_fn_debug(vqvae)
    twm_update = make_twm_update_fn(twm, vqvae, cfg)

    diag_before = None
    if args.diag_before_after:
        print("Running tokenizer diagnostics (before updates)...")
        diag_before = _evaluate_tokenizer_diagnostics(
            vqvae,
            vqvae_state.params,
            buffer_obs,
            eval_indices,
            cfg["VQVAE_CODEBOOK_SIZE"],
            args.diag_eval_batch_size,
        )
        print(
            "[diag before] "
            f"recon_mae={diag_before['recon_mae']:.6f} "
            f"recon_mse={diag_before['recon_mse']:.6f} "
            f"usage={diag_before['used_codes']}/{cfg['VQVAE_CODEBOOK_SIZE']} "
            f"({diag_before['code_usage_frac']:.3f}) "
            f"entropy={diag_before['token_entropy']:.3f} "
            f"perplexity={diag_before['token_perplexity']:.2f} "
            f"top1={diag_before['top1_code_frac']:.3f} "
            f"z_tiny={diag_before['z_frac_tiny']:.3f}"
        )

    print(
        f"Running smoke updates: tokenizer={args.n_iters_tok}x{args.n_mb_wm}, "
        f"twm={args.n_iters_twm}x{args.n_mb_wm}"
    )
    t0 = time.time()

    vq_loss_acc = 0.0
    vq_l1_acc = 0.0
    vq_l2_acc = 0.0
    vq_cb_acc = 0.0
    vq_cm_acc = 0.0
    vq_updates = 0
    vq_nonfinite = 0
    nonfinite_printed = 0
    diag_rows = []
    failfast_reason = ""
    failfast_bad_streak = 0
    stopped_early = False
    for tok_iter in range(args.n_iters_tok):
        vq_iter_loss = 0.0
        vq_iter_l1 = 0.0
        vq_iter_l2 = 0.0
        vq_iter_cb = 0.0
        vq_iter_cm = 0.0
        vq_iter_recon_std_ratio = 0.0
        vq_iter_recon_grad_ratio = 0.0
        vq_iter_z_norm_min = float("inf")
        vq_iter_z_norm_mean = 0.0
        vq_iter_z_frac_tiny = 0.0
        vq_iter_emb_norm_min_before = 0.0
        vq_iter_emb_norm_mean_before = 0.0
        vq_iter_emb_norm_max_before = 0.0
        vq_iter_emb_norm_min_after = 0.0
        vq_iter_emb_norm_mean_after = 0.0
        vq_iter_emb_norm_max_after = 0.0
        vq_iter_updates = 0
        iter_token_counts = np.zeros((cfg["VQVAE_CODEBOOK_SIZE"],), dtype=np.int64)
        for _ in range(args.n_mb_wm):
            rng, sample_rng = jax.random.split(rng)
            mb_size = min(args.vqvae_batch_size, buffer_count)
            mb_idx = jax.random.randint(sample_rng, (mb_size,), 0, buffer_count)
            obs_mb = buffer_obs[mb_idx]
            vqvae_state, vq_metrics, token_counts_mb, is_finite, grad_is_finite, metric_is_finite = vq_update(
                vqvae_state, obs_mb
            )
            iter_token_counts += np.asarray(jax.device_get(token_counts_mb))
            if bool(is_finite):
                loss_val = float(vq_metrics["total_loss"])
                l1_val = float(vq_metrics["l1"])
                l2_val = float(vq_metrics["l2"])
                cb_val = float(vq_metrics["codebook"])
                cm_val = float(vq_metrics["commitment"])
                recon_std_ratio_val = float(vq_metrics["recon_std_ratio"])
                recon_grad_ratio_val = float(vq_metrics["recon_grad_ratio"])
                z_norm_min_val = float(vq_metrics["z_norm_min"])
                z_norm_mean_val = float(vq_metrics["z_norm_mean"])
                z_frac_tiny_val = float(vq_metrics["z_frac_tiny"])
                emb_norm_min_before_val = float(vq_metrics["emb_norm_min_before"])
                emb_norm_mean_before_val = float(vq_metrics["emb_norm_mean_before"])
                emb_norm_max_before_val = float(vq_metrics["emb_norm_max_before"])
                emb_norm_min_after_val = float(vq_metrics["emb_norm_min_after"])
                emb_norm_mean_after_val = float(vq_metrics["emb_norm_mean_after"])
                emb_norm_max_after_val = float(vq_metrics["emb_norm_max_after"])
                vq_loss_acc += loss_val
                vq_l1_acc += l1_val
                vq_l2_acc += l2_val
                vq_cb_acc += cb_val
                vq_cm_acc += cm_val
                vq_updates += 1
                vq_iter_loss += loss_val
                vq_iter_l1 += l1_val
                vq_iter_l2 += l2_val
                vq_iter_cb += cb_val
                vq_iter_cm += cm_val
                vq_iter_recon_std_ratio += recon_std_ratio_val
                vq_iter_recon_grad_ratio += recon_grad_ratio_val
                vq_iter_z_norm_min = min(vq_iter_z_norm_min, z_norm_min_val)
                vq_iter_z_norm_mean += z_norm_mean_val
                vq_iter_z_frac_tiny += z_frac_tiny_val
                vq_iter_emb_norm_min_before += emb_norm_min_before_val
                vq_iter_emb_norm_mean_before += emb_norm_mean_before_val
                vq_iter_emb_norm_max_before += emb_norm_max_before_val
                vq_iter_emb_norm_min_after += emb_norm_min_after_val
                vq_iter_emb_norm_mean_after += emb_norm_mean_after_val
                vq_iter_emb_norm_max_after += emb_norm_max_after_val
                vq_iter_updates += 1
            else:
                vq_nonfinite += 1
                if nonfinite_printed < args.debug_nonfinite_limit:
                    obs_min = float(jnp.min(obs_mb))
                    obs_max = float(jnp.max(obs_mb))
                    obs_mean = float(jnp.mean(obs_mb))
                    obs_has_nan = bool(jnp.any(jnp.isnan(obs_mb)))
                    l1_val = float(vq_metrics["l1"])
                    l2_val = float(vq_metrics["l2"])
                    cb_val = float(vq_metrics["codebook"])
                    cm_val = float(vq_metrics["commitment"])
                    tot_val = float(vq_metrics["total_loss"])
                    print(
                        f"[tok nonfinite] iter={tok_iter + 1} "
                        f"grad_finite={bool(grad_is_finite)} metric_finite={bool(metric_is_finite)} "
                        f"loss={tot_val} l1={l1_val} l2={l2_val} cb={cb_val} cm={cm_val} "
                        f"obs[min,max,mean]=({obs_min:.6f},{obs_max:.6f},{obs_mean:.6f}) "
                        f"obs_has_nan={obs_has_nan}",
                        flush=True,
                    )
                    nonfinite_printed += 1

        iter_token_total = int(iter_token_counts.sum())
        if iter_token_total > 0:
            iter_probs = iter_token_counts.astype(np.float64) / float(iter_token_total)
            iter_nz = iter_probs > 0
            iter_entropy = float(-(iter_probs[iter_nz] * np.log(iter_probs[iter_nz])).sum())
            iter_perplexity = float(np.exp(iter_entropy))
            iter_used_codes = int((iter_token_counts > 0).sum())
            iter_dead_codes = int(cfg["VQVAE_CODEBOOK_SIZE"] - iter_used_codes)
            iter_usage_frac = float(iter_used_codes) / float(cfg["VQVAE_CODEBOOK_SIZE"])
            iter_top1_code_frac = float(iter_token_counts.max()) / float(iter_token_total)
        else:
            iter_entropy = float("nan")
            iter_perplexity = float("nan")
            iter_used_codes = 0
            iter_dead_codes = cfg["VQVAE_CODEBOOK_SIZE"]
            iter_usage_frac = 0.0
            iter_top1_code_frac = 1.0

        mean_iter = vq_iter_loss / float(max(1, vq_iter_updates))
        mean_l1 = vq_iter_l1 / float(max(1, vq_iter_updates))
        mean_l2 = vq_iter_l2 / float(max(1, vq_iter_updates))
        mean_cb = vq_iter_cb / float(max(1, vq_iter_updates))
        mean_cm = vq_iter_cm / float(max(1, vq_iter_updates))
        mean_recon_std_ratio = vq_iter_recon_std_ratio / float(max(1, vq_iter_updates))
        mean_recon_grad_ratio = vq_iter_recon_grad_ratio / float(max(1, vq_iter_updates))
        mean_z_norm_min = vq_iter_z_norm_min if vq_iter_z_norm_min != float("inf") else float("nan")
        mean_z_norm_mean = vq_iter_z_norm_mean / float(max(1, vq_iter_updates))
        mean_z_frac_tiny = vq_iter_z_frac_tiny / float(max(1, vq_iter_updates))
        mean_emb_norm_min_before = vq_iter_emb_norm_min_before / float(max(1, vq_iter_updates))
        mean_emb_norm_mean_before = vq_iter_emb_norm_mean_before / float(max(1, vq_iter_updates))
        mean_emb_norm_max_before = vq_iter_emb_norm_max_before / float(max(1, vq_iter_updates))
        mean_emb_norm_min_after = vq_iter_emb_norm_min_after / float(max(1, vq_iter_updates))
        mean_emb_norm_mean_after = vq_iter_emb_norm_mean_after / float(max(1, vq_iter_updates))
        mean_emb_norm_max_after = vq_iter_emb_norm_max_after / float(max(1, vq_iter_updates))

        should_run_diag = (
            (tok_iter == 0)
            or ((tok_iter + 1) % max(1, args.diag_log_every) == 0)
            or (tok_iter + 1 == args.n_iters_tok)
        )
        diag_iter_stats = None
        if should_run_diag:
            diag_iter_stats = _evaluate_tokenizer_diagnostics(
                vqvae,
                vqvae_state.params,
                buffer_obs,
                eval_indices,
                cfg["VQVAE_CODEBOOK_SIZE"],
                args.diag_eval_batch_size,
            )

        bad_reasons = []
        if vq_iter_updates == 0:
            bad_reasons.append("finite_mb==0")
        if not np.isfinite(iter_perplexity) or iter_perplexity < args.failfast_perplexity_min:
            bad_reasons.append(f"perplexity<{args.failfast_perplexity_min}")
        if iter_top1_code_frac > args.failfast_top1_frac_max:
            bad_reasons.append(f"top1_code_frac>{args.failfast_top1_frac_max}")
        if np.isfinite(mean_z_frac_tiny) and mean_z_frac_tiny > args.failfast_z_tiny_frac_max:
            bad_reasons.append(f"z_frac_tiny>{args.failfast_z_tiny_frac_max}")

        if bad_reasons:
            failfast_bad_streak += 1
        else:
            failfast_bad_streak = 0

        diag_rows.append(
            {
                "tok_iter": tok_iter + 1,
                "finite_mb": vq_iter_updates,
                "total_mb": args.n_mb_wm,
                "loss": mean_iter,
                "l1": mean_l1,
                "l2": mean_l2,
                "codebook": mean_cb,
                "commitment": mean_cm,
                "used_codes": iter_used_codes,
                "dead_codes": iter_dead_codes,
                "usage_frac": iter_usage_frac,
                "entropy": iter_entropy,
                "perplexity": iter_perplexity,
                "top1_code_frac": iter_top1_code_frac,
                "z_norm_min": mean_z_norm_min,
                "z_norm_mean": mean_z_norm_mean,
                "z_frac_tiny": mean_z_frac_tiny,
                "emb_norm_min_before": mean_emb_norm_min_before,
                "emb_norm_mean_before": mean_emb_norm_mean_before,
                "emb_norm_max_before": mean_emb_norm_max_before,
                "emb_norm_min_after": mean_emb_norm_min_after,
                "emb_norm_mean_after": mean_emb_norm_mean_after,
                "emb_norm_max_after": mean_emb_norm_max_after,
                "recon_std_ratio": mean_recon_std_ratio,
                "recon_grad_ratio": mean_recon_grad_ratio,
                "diag_recon_mae": float(diag_iter_stats["recon_mae"]) if diag_iter_stats else float("nan"),
                "diag_recon_mse": float(diag_iter_stats["recon_mse"]) if diag_iter_stats else float("nan"),
                "diag_recon_std_ratio": float(diag_iter_stats["recon_std_ratio"]) if diag_iter_stats else float("nan"),
                "diag_recon_grad_ratio": float(diag_iter_stats["recon_grad_ratio"]) if diag_iter_stats else float("nan"),
                "diag_used_codes": int(diag_iter_stats["used_codes"]) if diag_iter_stats else -1,
                "diag_dead_codes": int(diag_iter_stats["dead_codes"]) if diag_iter_stats else -1,
                "diag_usage_frac": float(diag_iter_stats["code_usage_frac"]) if diag_iter_stats else float("nan"),
                "diag_entropy": float(diag_iter_stats["token_entropy"]) if diag_iter_stats else float("nan"),
                "diag_perplexity": float(diag_iter_stats["token_perplexity"]) if diag_iter_stats else float("nan"),
                "diag_top1_code_frac": float(diag_iter_stats["top1_code_frac"]) if diag_iter_stats else float("nan"),
                "diag_z_norm_min": float(diag_iter_stats["z_norm_min"]) if diag_iter_stats else float("nan"),
                "diag_z_norm_mean": float(diag_iter_stats["z_norm_mean"]) if diag_iter_stats else float("nan"),
                "diag_z_frac_tiny": float(diag_iter_stats["z_frac_tiny"]) if diag_iter_stats else float("nan"),
                "failfast_bad_streak": failfast_bad_streak,
                "failfast_reasons": "|".join(bad_reasons),
            }
        )

        if args.print_each_iter and (
            ((tok_iter + 1) % max(1, args.print_every) == 0)
            or (tok_iter == 0)
            or (tok_iter + 1 == args.n_iters_tok)
        ):
            print(
                f"[tok {tok_iter + 1:04d}/{args.n_iters_tok}] "
                f"loss={mean_iter:.6f} l1={mean_l1:.6f} l2={mean_l2:.6f} "
                f"cb={mean_cb:.6f} cm={mean_cm:.6f} finite_mb={vq_iter_updates}/{args.n_mb_wm} "
                f"usage={iter_used_codes}/{cfg['VQVAE_CODEBOOK_SIZE']} perp={iter_perplexity:.3f} "
                f"top1={iter_top1_code_frac:.3f} z_tiny={mean_z_frac_tiny:.3f}",
                flush=True,
            )
            if diag_iter_stats is not None:
                print(
                    f"[diag {tok_iter + 1:04d}] recon_mae={diag_iter_stats['recon_mae']:.6f} "
                    f"recon_grad_ratio={diag_iter_stats['recon_grad_ratio']:.3f} "
                    f"usage={diag_iter_stats['used_codes']}/{cfg['VQVAE_CODEBOOK_SIZE']} "
                    f"perp={diag_iter_stats['token_perplexity']:.3f} "
                    f"top1={diag_iter_stats['top1_code_frac']:.3f} "
                    f"z_tiny={diag_iter_stats['z_frac_tiny']:.3f}",
                    flush=True,
                )

        if args.failfast_enabled and failfast_bad_streak >= max(1, args.failfast_patience):
            failfast_reason = (
                f"Triggered fail-fast at tok_iter={tok_iter + 1}: "
                f"bad conditions for {failfast_bad_streak} consecutive iterations "
                f"({'; '.join(bad_reasons) if bad_reasons else 'unknown'})"
            )
            print(failfast_reason, flush=True)
            stopped_early = True
            break

    twm_loss_acc = 0.0
    twm_rew_acc = 0.0
    twm_done_acc = 0.0
    twm_updates = 0
    for twm_iter in range(args.n_iters_twm if not stopped_early else 0):
        twm_iter_loss = 0.0
        twm_iter_rew = 0.0
        twm_iter_done = 0.0
        twm_iter_updates = 0
        for _ in range(args.n_mb_wm):
            rng, sample_rng = jax.random.split(rng)
            fbx_batch = fbx_buffer.sample(fbx_state, sample_rng)
            exp_batch = fbx_batch.experience
            batch_dones = exp_batch["done_after"] if ("done_after" in exp_batch) else exp_batch["done"]
            twm_state, twm_loss, (_, twm_loss_rew, twm_loss_done), rng = twm_update(
                twm_state,
                vqvae_state.params,
                exp_batch["obs"],
                exp_batch["action"],
                exp_batch["reward"],
                batch_dones,
                rng,
            )
            twm_loss_acc += float(twm_loss)
            twm_rew_acc += float(twm_loss_rew)
            twm_done_acc += float(twm_loss_done)
            twm_updates += 1
            twm_iter_loss += float(twm_loss)
            twm_iter_rew += float(twm_loss_rew)
            twm_iter_done += float(twm_loss_done)
            twm_iter_updates += 1
        if args.print_each_iter and (
            ((twm_iter + 1) % max(1, args.print_every) == 0)
            or (twm_iter == 0)
            or (twm_iter + 1 == args.n_iters_twm)
        ):
            mean_twm = twm_iter_loss / float(max(1, twm_iter_updates))
            mean_rew = twm_iter_rew / float(max(1, twm_iter_updates))
            mean_done = twm_iter_done / float(max(1, twm_iter_updates))
            print(
                f"[twm {twm_iter + 1:04d}/{args.n_iters_twm}] "
                f"loss={mean_twm:.6f} rew={mean_rew:.6f} ends={mean_done:.6f} "
                f"mb={twm_iter_updates}/{args.n_mb_wm}",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"Smoke updates complete in {elapsed:.2f}s")
    if vq_updates > 0:
        print(
            "Tokenizer mean: "
            f"loss={vq_loss_acc / vq_updates:.6f} "
            f"l1={vq_l1_acc / vq_updates:.6f} "
            f"l2={vq_l2_acc / vq_updates:.6f} "
            f"cb={vq_cb_acc / vq_updates:.6f} "
            f"cm={vq_cm_acc / vq_updates:.6f} "
            f"over {vq_updates} updates"
        )
    else:
        print("Tokenizer updates: 0")
    print(f"Tokenizer non-finite minibatches: {vq_nonfinite}")
    if twm_updates > 0:
        print(
            f"TWM mean loss: {twm_loss_acc / twm_updates:.6f} | "
            f"reward: {twm_rew_acc / twm_updates:.6f} | "
            f"ends: {twm_done_acc / twm_updates:.6f} over {twm_updates} updates"
        )
    else:
        print("TWM updates: 0")

    diag_after = None
    if args.diag_before_after:
        print("Running tokenizer diagnostics (after updates)...")
        diag_after = _evaluate_tokenizer_diagnostics(
            vqvae,
            vqvae_state.params,
            buffer_obs,
            eval_indices,
            cfg["VQVAE_CODEBOOK_SIZE"],
            args.diag_eval_batch_size,
        )
        print(
            "[diag after] "
            f"recon_mae={diag_after['recon_mae']:.6f} "
            f"recon_mse={diag_after['recon_mse']:.6f} "
            f"usage={diag_after['used_codes']}/{cfg['VQVAE_CODEBOOK_SIZE']} "
            f"({diag_after['code_usage_frac']:.3f}) "
            f"entropy={diag_after['token_entropy']:.3f} "
            f"perplexity={diag_after['token_perplexity']:.2f} "
            f"top1={diag_after['top1_code_frac']:.3f} "
            f"z_tiny={diag_after['z_frac_tiny']:.3f}"
        )

    with open(os.path.join(smoke_ckpt_dir, "policy_params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(policy_params), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(smoke_ckpt_dir, "vqvae_params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(vqvae_state.params), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(smoke_ckpt_dir, "twm_params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(twm_state.params), f, protocol=pickle.HIGHEST_PROTOCOL)

    diag_csv_path = os.path.join(smoke_ckpt_dir, "smoke_tokenizer_diag.csv")
    if diag_rows:
        with open(diag_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diag_rows)
    else:
        with open(diag_csv_path, "w", newline="") as f:
            f.write("tok_iter\n")

    with open(os.path.join(smoke_ckpt_dir, "smoke_metrics.txt"), "w") as f:
        f.write(f"source_checkpoint={ckpt}\n")
        f.write(f"fixed_eval_indices_path={eval_indices_path}\n")
        f.write(f"fixed_eval_indices_count={len(eval_indices)}\n")
        f.write(f"vq_updates={vq_updates}\n")
        f.write(f"vq_nonfinite_updates={vq_nonfinite}\n")
        f.write(f"twm_updates={twm_updates}\n")
        f.write(f"stopped_early={int(stopped_early)}\n")
        f.write(f"failfast_reason={failfast_reason}\n")
        f.write(f"vq_mean_loss={(vq_loss_acc / max(1, vq_updates)):.8f}\n")
        f.write(f"vq_mean_l1={(vq_l1_acc / max(1, vq_updates)):.8f}\n")
        f.write(f"vq_mean_l2={(vq_l2_acc / max(1, vq_updates)):.8f}\n")
        f.write(f"vq_mean_codebook={(vq_cb_acc / max(1, vq_updates)):.8f}\n")
        f.write(f"vq_mean_commitment={(vq_cm_acc / max(1, vq_updates)):.8f}\n")
        f.write(f"twm_mean_loss={(twm_loss_acc / max(1, twm_updates)):.8f}\n")
        f.write(f"twm_mean_loss_rew={(twm_rew_acc / max(1, twm_updates)):.8f}\n")
        f.write(f"twm_mean_loss_ends={(twm_done_acc / max(1, twm_updates)):.8f}\n")
        f.write(f"elapsed_sec={elapsed:.2f}\n")
        if diag_before is not None:
            f.write(f"diag_before_recon_mae={diag_before['recon_mae']:.8f}\n")
            f.write(f"diag_before_recon_mse={diag_before['recon_mse']:.8f}\n")
            f.write(f"diag_before_recon_std_ratio={diag_before['recon_std_ratio']:.8f}\n")
            f.write(f"diag_before_recon_grad_ratio={diag_before['recon_grad_ratio']:.8f}\n")
            f.write(f"diag_before_z_norm_min={diag_before['z_norm_min']:.8f}\n")
            f.write(f"diag_before_z_norm_mean={diag_before['z_norm_mean']:.8f}\n")
            f.write(f"diag_before_z_frac_tiny={diag_before['z_frac_tiny']:.8f}\n")
            f.write(f"diag_before_used_codes={diag_before['used_codes']}\n")
            f.write(f"diag_before_code_usage_frac={diag_before['code_usage_frac']:.8f}\n")
            f.write(f"diag_before_entropy={diag_before['token_entropy']:.8f}\n")
            f.write(f"diag_before_perplexity={diag_before['token_perplexity']:.8f}\n")
            f.write(f"diag_before_top1_code_frac={diag_before['top1_code_frac']:.8f}\n")
            f.write(f"diag_before_top_codes={diag_before['top_codes']}\n")
        if diag_after is not None:
            f.write(f"diag_after_recon_mae={diag_after['recon_mae']:.8f}\n")
            f.write(f"diag_after_recon_mse={diag_after['recon_mse']:.8f}\n")
            f.write(f"diag_after_recon_std_ratio={diag_after['recon_std_ratio']:.8f}\n")
            f.write(f"diag_after_recon_grad_ratio={diag_after['recon_grad_ratio']:.8f}\n")
            f.write(f"diag_after_z_norm_min={diag_after['z_norm_min']:.8f}\n")
            f.write(f"diag_after_z_norm_mean={diag_after['z_norm_mean']:.8f}\n")
            f.write(f"diag_after_z_frac_tiny={diag_after['z_frac_tiny']:.8f}\n")
            f.write(f"diag_after_used_codes={diag_after['used_codes']}\n")
            f.write(f"diag_after_code_usage_frac={diag_after['code_usage_frac']:.8f}\n")
            f.write(f"diag_after_entropy={diag_after['token_entropy']:.8f}\n")
            f.write(f"diag_after_perplexity={diag_after['token_perplexity']:.8f}\n")
            f.write(f"diag_after_top1_code_frac={diag_after['top1_code_frac']:.8f}\n")
            f.write(f"diag_after_top_codes={diag_after['top_codes']}\n")

    if failfast_reason:
        with open(os.path.join(smoke_ckpt_dir, "smoke_failfast_reason.txt"), "w") as f:
            f.write(f"{failfast_reason}\n")

    print(f"Saved smoke checkpoint-like params to: {smoke_ckpt_dir}")
    print(f"Saved tokenizer diag CSV: {diag_csv_path}")

    if args.run_triptych:
        os.makedirs(os.path.dirname(args.triptych_output_png) or ".", exist_ok=True)
        cmd = [
            sys.executable,
            "wm_m3_triptych_sheet.py",
            "--checkpoint_dir",
            smoke_ckpt_dir,
            "--horizon",
            str(args.triptych_horizon),
            "--seed",
            str(args.triptych_seed),
            "--wm_temperature",
            str(args.triptych_wm_temperature),
            "--output_png",
            args.triptych_output_png,
            "--include_tokenizer_recon_row",
        ]
        if args.triptych_policy_greedy:
            cmd.append("--policy_greedy")
        if args.triptych_wm_greedy:
            cmd.append("--wm_greedy")
        print("Running triptych visualization...")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Triptych written to: {args.triptych_output_png}")


if __name__ == "__main__":
    main()

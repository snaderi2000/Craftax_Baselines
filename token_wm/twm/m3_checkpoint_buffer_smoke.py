#!/usr/bin/env python3
"""
M3 checkpoint-buffer smoke test:
1) Load flat + flashbax buffers from an existing checkpoint.
2) Run tokenizer and TWM updates only (no env rollout, no policy updates).
3) Save a lightweight checkpoint-like directory (policy/vqvae/twm params).
4) Optionally run wm_m3_triptych_sheet.py on the updated params.
"""

import argparse
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

from ppo_mbrl_m3 import create_twm, create_vqvae, make_twm_update_fn, make_vqvae_update_fn


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

    vq_update = make_vqvae_update_fn(vqvae, cfg)
    twm_update = make_twm_update_fn(twm, vqvae, cfg)

    print(
        f"Running smoke updates: tokenizer={args.n_iters_tok}x{args.n_mb_wm}, "
        f"twm={args.n_iters_twm}x{args.n_mb_wm}"
    )
    t0 = time.time()

    vq_loss_acc = 0.0
    vq_updates = 0
    for tok_iter in range(args.n_iters_tok):
        vq_iter_loss = 0.0
        vq_iter_updates = 0
        for _ in range(args.n_mb_wm):
            rng, sample_rng = jax.random.split(rng)
            mb_size = min(args.vqvae_batch_size, buffer_count)
            mb_idx = jax.random.randint(sample_rng, (mb_size,), 0, buffer_count)
            obs_mb = buffer_obs[mb_idx]
            vqvae_state, vq_metrics, is_finite = vq_update(vqvae_state, obs_mb)
            if bool(is_finite):
                loss_val = float(vq_metrics["total_loss"])
                vq_loss_acc += loss_val
                vq_updates += 1
                vq_iter_loss += loss_val
                vq_iter_updates += 1
        if args.print_each_iter and (
            ((tok_iter + 1) % max(1, args.print_every) == 0)
            or (tok_iter == 0)
            or (tok_iter + 1 == args.n_iters_tok)
        ):
            mean_iter = vq_iter_loss / float(max(1, vq_iter_updates))
            print(
                f"[tok {tok_iter + 1:04d}/{args.n_iters_tok}] "
                f"loss={mean_iter:.6f} finite_mb={vq_iter_updates}/{args.n_mb_wm}",
                flush=True,
            )

    twm_loss_acc = 0.0
    twm_rew_acc = 0.0
    twm_done_acc = 0.0
    twm_updates = 0
    for twm_iter in range(args.n_iters_twm):
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
        print(f"Tokenizer mean loss: {vq_loss_acc / vq_updates:.6f} over {vq_updates} updates")
    else:
        print("Tokenizer updates: 0")
    if twm_updates > 0:
        print(
            f"TWM mean loss: {twm_loss_acc / twm_updates:.6f} | "
            f"reward: {twm_rew_acc / twm_updates:.6f} | "
            f"ends: {twm_done_acc / twm_updates:.6f} over {twm_updates} updates"
        )
    else:
        print("TWM updates: 0")

    os.makedirs(args.output_dir, exist_ok=True)
    tag = (
        f"smoke_{os.path.basename(os.path.normpath(ckpt))}"
        f"_tok{args.n_iters_tok}x{args.n_mb_wm}_twm{args.n_iters_twm}x{args.n_mb_wm}"
    )
    smoke_ckpt_dir = os.path.join(args.output_dir, tag)
    os.makedirs(smoke_ckpt_dir, exist_ok=True)

    with open(os.path.join(smoke_ckpt_dir, "policy_params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(policy_params), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(smoke_ckpt_dir, "vqvae_params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(vqvae_state.params), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(smoke_ckpt_dir, "twm_params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(twm_state.params), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(smoke_ckpt_dir, "smoke_metrics.txt"), "w") as f:
        f.write(f"source_checkpoint={ckpt}\n")
        f.write(f"vq_updates={vq_updates}\n")
        f.write(f"twm_updates={twm_updates}\n")
        f.write(f"vq_mean_loss={(vq_loss_acc / max(1, vq_updates)):.8f}\n")
        f.write(f"twm_mean_loss={(twm_loss_acc / max(1, twm_updates)):.8f}\n")
        f.write(f"twm_mean_loss_rew={(twm_rew_acc / max(1, twm_updates)):.8f}\n")
        f.write(f"twm_mean_loss_ends={(twm_done_acc / max(1, twm_updates)):.8f}\n")
        f.write(f"elapsed_sec={elapsed:.2f}\n")

    print(f"Saved smoke checkpoint-like params to: {smoke_ckpt_dir}")

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

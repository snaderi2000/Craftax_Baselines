#!/usr/bin/env python3
"""
Create a PNG contact sheet with three synchronized rollout rows:
1) Real environment rollout under policy actions
2) WM rollout teacher-forced by the same real actions
3) WM open-loop rollout with policy actions on WM observations
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from token_wm.tokenizer.patch_vqvae import PatchVQVAE
from token_wm.twm.world_model import WorldModel
from token_wm.twm.transformer import TransformerConfig
from token_wm.twm.kv_caching import KeysValues
from ppo_mbrl_m3 import ActorCriticRNN, ScannedRNN

try:
    import pygame
except ImportError as exc:
    raise SystemExit("pygame is required to save PNG sheets. Install with: pip install pygame") from exc


ACTION_ORDER = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]


def action_name(aid: int) -> str:
    if 0 <= aid < len(ACTION_ORDER):
        return ACTION_ORDER[aid]
    return f"action_{aid}"


def to_jax_tree(tree):
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, (np.ndarray, np.generic)) else x,
        tree,
    )


def ensure_apply_vars(params_or_vars):
    """
    Normalize checkpoint payload to Flax apply vars dict: {"params": ...}.
    Handles both saved formats:
      1) raw params pytree
      2) already-wrapped {"params": ...}
    """
    obj = params_or_vars
    if hasattr(obj, "keys"):
        keys = list(obj.keys())
        if keys == ["params"] or set(keys) == {"params"}:
            return obj
    return {"params": obj}


def to_uint8_frame(obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs)
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        # Decoder outputs are often slightly outside [0,1], e.g. [-0.06, 1.09].
        # Treat those as normalized images for rendering.
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max <= 2.0 and x_min >= -1.0:
            x = np.clip(x, 0.0, 1.0) * 255.0
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return x


def upsample_nearest(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def build_contact_sheet(rows: List[List[np.ndarray]], scale: int, gap: int, marker_w: int = 14) -> np.ndarray:
    # Convert all frames to uint8 and scale.
    proc_rows: List[List[np.ndarray]] = []
    for frames in rows:
        proc_rows.append([upsample_nearest(to_uint8_frame(f), scale) for f in frames])

    row_imgs = []
    for ridx, frames in enumerate(proc_rows):
        h, w, c = frames[0].shape
        row_w = len(frames) * w + (len(frames) - 1) * gap
        row = np.zeros((h, row_w, c), dtype=np.uint8)
        x = 0
        for fi, frame in enumerate(frames):
            row[:, x:x + w] = frame
            x += w
            if fi < len(frames) - 1:
                row[:, x:x + gap] = 30
                x += gap

        # Left marker strip so rows are visually distinct.
        marker = np.zeros((h, marker_w, 3), dtype=np.uint8)
        if ridx == 0:
            marker[:] = np.array([220, 60, 60], dtype=np.uint8)   # red-ish: real
        elif ridx == 1:
            marker[:] = np.array([60, 220, 120], dtype=np.uint8)  # green-ish: teacher WM
        else:
            marker[:] = np.array([70, 130, 240], dtype=np.uint8)  # blue-ish: open-loop WM
        row = np.concatenate([marker, np.full((h, gap, 3), 20, dtype=np.uint8), row], axis=1)
        row_imgs.append(row)

    sheet_h = sum(r.shape[0] for r in row_imgs) + gap * (len(row_imgs) - 1)
    sheet_w = max(r.shape[1] for r in row_imgs)
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    y = 0
    for i, r in enumerate(row_imgs):
        sheet[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]
        if i < len(row_imgs) - 1:
            sheet[y:y + gap] = 24
            y += gap
    return sheet


def summarize_frame(obs: np.ndarray, black_pixel_threshold: int) -> Dict[str, float]:
    f = np.asarray(obs, dtype=np.float32)
    u8 = to_uint8_frame(f)
    return {
        "min": float(np.min(f)),
        "max": float(np.max(f)),
        "mean": float(np.mean(f)),
        "std": float(np.std(f)),
        "black_frac": float(np.mean(u8 <= np.uint8(black_pixel_threshold))),
    }


def add_action_tracks(
    sheet: np.ndarray,
    teacher_actions: List[str],
    open_actions: List[str],
    frame_w: int,
    gap: int,
    marker_w: int,
    font_size: int,
) -> "pygame.Surface":
    # pygame must be initialized by caller.
    base = pygame.surfarray.make_surface(np.swapaxes(sheet, 0, 1))
    sw, sh = base.get_size()
    top_pad = max(22, font_size + 8)
    bot_pad = max(22, font_size + 8)
    canvas = pygame.Surface((sw, sh + top_pad + bot_pad))
    canvas.fill((10, 10, 14))
    canvas.blit(base, (0, top_pad))

    font = pygame.font.SysFont("monospace", font_size)
    hdr_font = pygame.font.SysFont("monospace", max(12, font_size - 1), bold=True)

    teacher_hdr = hdr_font.render("Teacher-forced actions (real env actions):", True, (140, 230, 160))
    open_hdr = hdr_font.render("Open-loop actions (policy on WM obs):", True, (120, 170, 250))
    canvas.blit(teacher_hdr, (6, 2))
    canvas.blit(open_hdr, (6, top_pad + sh + 2))

    x0 = marker_w + gap

    # Label the first frame as start.
    start_txt = font.render("start", True, (200, 200, 200))
    x_start = x0 + frame_w // 2 - start_txt.get_width() // 2
    canvas.blit(start_txt, (x_start, top_pad - start_txt.get_height()))
    canvas.blit(start_txt, (x_start, top_pad + sh + 2))

    for i, act in enumerate(teacher_actions, start=1):
        x_center = x0 + i * (frame_w + gap) + frame_w // 2
        txt = font.render(act, True, (220, 255, 230))
        canvas.blit(txt, (x_center - txt.get_width() // 2, top_pad - txt.get_height()))

    for i, act in enumerate(open_actions, start=1):
        x_center = x0 + i * (frame_w + gap) + frame_w // 2
        txt = font.render(act, True, (210, 230, 255))
        canvas.blit(txt, (x_center - txt.get_width() // 2, top_pad + sh + 2))

    return canvas


@dataclass
class WMState:
    cache: KeysValues
    obs: jnp.ndarray   # (63,63,3)
    done: jnp.ndarray  # scalar float
    rng: jnp.ndarray


def parse_args():
    p = argparse.ArgumentParser(description="Create M3 WM triptych PNG sheet.")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--output_png", type=str, default="wm_m3_triptych.png")
    p.add_argument("--output_meta", type=str, default=None)
    p.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--burnin_horizon", type=int, default=5)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--gap", type=int, default=2)
    p.add_argument("--annotate_actions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--action_font_size", type=int, default=14)
    p.add_argument("--debug_wm", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug_print_every", type=int, default=1)
    p.add_argument("--black_pixel_threshold", type=int, default=5)

    p.add_argument("--policy_greedy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wm_greedy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wm_temperature", type=float, default=1.0)

    p.add_argument("--layer_size", type=int, default=2048)
    p.add_argument("--play_max_blocks", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=7)
    p.add_argument("--obs_image_size", type=int, default=63)
    p.add_argument("--vqvae_codebook_size", type=int, default=512)
    p.add_argument("--vqvae_embed_dim", type=int, default=128)
    p.add_argument("--patch_encoder_hidden_dim", type=int, default=128)
    p.add_argument("--vqvae_lambda_l1", type=float, default=0.1)
    p.add_argument("--vqvae_lambda_l2", type=float, default=1.0)
    p.add_argument("--vqvae_lambda_codebook", type=float, default=1.0)
    p.add_argument("--vqvae_lambda_commitment", type=float, default=0.02)
    p.add_argument("--twm_num_layers", type=int, default=3)
    p.add_argument("--twm_num_heads", type=int, default=8)
    p.add_argument("--twm_embed_dim", type=int, default=128)
    p.add_argument("--use_binary_reward_target", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--binary_reward_threshold", type=float, default=0.5)
    return p.parse_args()


def load_params(ckpt_dir: str):
    req = ["policy_params.pkl", "vqvae_params.pkl", "twm_params.pkl"]
    for f in req:
        p = os.path.join(ckpt_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")
    with open(os.path.join(ckpt_dir, "policy_params.pkl"), "rb") as f:
        policy = pickle.load(f)
    with open(os.path.join(ckpt_dir, "vqvae_params.pkl"), "rb") as f:
        vq = pickle.load(f)
    with open(os.path.join(ckpt_dir, "twm_params.pkl"), "rb") as f:
        twm = pickle.load(f)
    return to_jax_tree(policy), to_jax_tree(vq), to_jax_tree(twm)


def select_action(network, policy_vars, hstate, obs_batched, done_batched, rng, greedy: bool):
    # obs_batched: (1,63,63,3), done_batched: (1,)
    hstate, pi, _ = network.apply(policy_vars, hstate, (obs_batched[jnp.newaxis, :], done_batched[jnp.newaxis, :]))
    if greedy:
        action = pi.mode().squeeze(0)
    else:
        action = pi.sample(seed=rng).squeeze(0)
    return hstate, int(np.asarray(action[0]))


def make_wm_step_fn(vqvae, twm, vqvae_vars, twm_vars, args, tokens_per_obs: int):
    temperature = float(args.wm_temperature)
    use_binary = bool(args.use_binary_reward_target)
    wm_greedy = bool(args.wm_greedy)

    def sample_tok(logits, rng):
        logits = jnp.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = jnp.clip(logits, -50.0, 50.0)
        if wm_greedy:
            tok = jnp.argmax(logits, axis=-1)
        else:
            tok = jax.random.categorical(rng, logits / temperature, axis=-1)
        return tok.astype(jnp.int32)

    @jax.jit
    def step_wm(cache: KeysValues, action_id: jnp.ndarray, rng: jnp.ndarray):
        rng, rew_rng, done_rng, tok_rng = jax.random.split(rng, 4)
        act = action_id.reshape(1, 1).astype(jnp.int32)
        output, cache = twm.apply(twm_vars, act, past_keys_values=cache, deterministic=True)

        rew_logits = output.logits_rewards[:, -1, :]
        rew_cls = sample_tok(rew_logits, rew_rng)
        reward = rew_cls.astype(jnp.float32) if use_binary else (rew_cls - 1).astype(jnp.float32)

        done_logits = jnp.nan_to_num(output.logits_ends[:, -1, :], nan=0.0, posinf=20.0, neginf=-20.0)
        done_prob = jax.nn.softmax(done_logits, axis=-1)[:, 1]
        if wm_greedy:
            done = (done_prob > 0.5).astype(jnp.float32)
        else:
            done = jax.random.bernoulli(done_rng, done_prob).astype(jnp.float32)

        first_obs_logits = output.logits_observations[:, -1, :]
        tok0 = sample_tok(first_obs_logits, tok_rng)

        if tokens_per_obs > 1:
            def _scan(carry, _):
                cache_scan, prev_tok, rng_scan = carry
                inp = prev_tok.reshape(1, 1)
                out, cache_scan = twm.apply(twm_vars, inp, past_keys_values=cache_scan, deterministic=True)
                rng_scan, tr = jax.random.split(rng_scan)
                nt = sample_tok(out.logits_observations[:, -1, :], tr)
                return (cache_scan, nt, rng_scan), nt

            (cache, _last, rng), tail = jax.lax.scan(_scan, (cache, tok0, rng), None, length=tokens_per_obs - 1)
            tail = jnp.swapaxes(tail, 0, 1)  # (1, L-1)
            obs_tokens = jnp.concatenate([tok0[:, None], tail], axis=1)
        else:
            obs_tokens = tok0[:, None]

        # Keep cache aligned with training-time rollout:
        # feed the final observation token so next step starts at block boundary.
        final_tok = obs_tokens[:, -1].reshape(1, 1)
        _, cache = twm.apply(twm_vars, final_tok, past_keys_values=cache, deterministic=True)

        next_obs = vqvae.apply(vqvae_vars, obs_tokens, method=vqvae.decode_tokens)[0]
        return cache, next_obs, reward[0], done[0], done_prob[0], rew_cls[0], rng

    return step_wm


def prime_cache_with_context(
    vqvae,
    twm,
    vqvae_vars,
    twm_vars,
    max_tokens: int,
    args,
    burnin_obs_seq: List[np.ndarray],
    burnin_action_seq: List[int],
    start_obs: jnp.ndarray,
):
    cache = KeysValues.init(
        n=1,
        num_heads=args.twm_num_heads,
        max_tokens=max_tokens,
        embed_dim=args.twm_embed_dim,
        num_layers=args.twm_num_layers,
    )

    # Burn-in cache with (obs_t, action_t) context, matching training rollout setup.
    for obs_np, act in zip(burnin_obs_seq, burnin_action_seq):
        obs_t = jnp.asarray(obs_np, dtype=jnp.float32)
        obs_tokens = vqvae.apply(vqvae_vars, obs_t[jnp.newaxis, ...], method=vqvae.encode)
        _, cache = twm.apply(twm_vars, obs_tokens, past_keys_values=cache, deterministic=True)
        act_tok = jnp.array([[int(act)]], dtype=jnp.int32)
        _, cache = twm.apply(twm_vars, act_tok, past_keys_values=cache, deterministic=True)

    # Feed rollout starting observation.
    obs_tokens = vqvae.apply(vqvae_vars, start_obs[jnp.newaxis, ...], method=vqvae.encode)
    _, cache = twm.apply(twm_vars, obs_tokens, past_keys_values=cache, deterministic=True)
    return cache


def main():
    args = parse_args()
    if args.obs_image_size % args.patch_size != 0:
        raise ValueError("obs_image_size must be divisible by patch_size")
    if args.horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if args.burnin_horizon < 0:
        raise ValueError("burnin_horizon must be >= 0")

    policy_params, vqvae_params, twm_params = load_params(args.checkpoint_dir)
    policy_vars = ensure_apply_vars(policy_params)
    vqvae_vars = ensure_apply_vars(vqvae_params)
    twm_vars = ensure_apply_vars(twm_params)

    env = make_craftax_env_from_name(args.env_name, True)
    env_params = env.default_params
    num_actions = int(env.action_space(env_params).n)
    print(f"Loaded env with {num_actions} actions.")

    network = ActorCriticRNN(num_actions, config={"LAYER_SIZE": args.layer_size})
    vqvae = PatchVQVAE(
        num_embeddings=args.vqvae_codebook_size,
        embedding_dim=args.vqvae_embed_dim,
        encoder_hidden_dim=args.patch_encoder_hidden_dim,
        patch_size=args.patch_size,
        image_size=args.obs_image_size,
        lambda_l1=args.vqvae_lambda_l1,
        lambda_l2=args.vqvae_lambda_l2,
        lambda_codebook=args.vqvae_lambda_codebook,
        lambda_commitment=args.vqvae_lambda_commitment,
    )
    tokens_per_obs = (args.obs_image_size // args.patch_size) ** 2
    tokens_per_block = tokens_per_obs + 1
    twm_cfg = TransformerConfig(
        tokens_per_block=tokens_per_block,
        max_blocks=args.play_max_blocks,
        attention="causal",
        num_layers=args.twm_num_layers,
        num_heads=args.twm_num_heads,
        embed_dim=args.twm_embed_dim,
        embed_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    twm = WorldModel(
        obs_vocab_size=args.vqvae_codebook_size,
        act_vocab_size=num_actions,
        config=twm_cfg,
        reward_num_classes=2 if args.use_binary_reward_target else 3,
        binary_reward_threshold=args.binary_reward_threshold,
    )

    max_tokens = args.play_max_blocks * tokens_per_block
    wm_step = make_wm_step_fn(vqvae, twm, vqvae_vars, twm_vars, args, tokens_per_obs)

    # Initial real observation.
    rng = jax.random.PRNGKey(args.seed)
    rng, env_rng = jax.random.split(rng)
    real_obs, env_state = env.reset(env_rng, env_params)
    real_obs = jnp.asarray(real_obs, dtype=jnp.float32)

    # Initial states.
    h_real = ScannedRNN.initialize_carry(1, 256)
    real_done = jnp.array([False])

    # Burn-in phase from real environment to match training-time context usage.
    burnin_obs_seq: List[np.ndarray] = []
    burnin_action_seq: List[int] = []
    burnin_done_seq: List[bool] = []
    for _ in range(args.burnin_horizon):
        burnin_obs_seq.append(np.asarray(real_obs))
        burnin_done_seq.append(bool(np.asarray(real_done[0])))

        rng, pol_rng = jax.random.split(rng)
        h_real, action_burn = select_action(
            network,
            policy_vars,
            h_real,
            real_obs[jnp.newaxis, ...],
            real_done,
            pol_rng,
            args.policy_greedy,
        )
        action_burn = int(np.clip(action_burn, 0, num_actions - 1))
        burnin_action_seq.append(action_burn)

        rng, step_rng = jax.random.split(rng)
        next_obs, env_state, _r, d, _info = env.step(step_rng, env_state, jnp.array(action_burn, dtype=jnp.int32), env_params)
        if bool(d):
            rng, r = jax.random.split(rng)
            next_obs, env_state = env.reset(r, env_params)
        real_obs = jnp.asarray(next_obs, dtype=jnp.float32)
        real_done = jnp.array([bool(d)])

    # Open-loop policy hidden state receives same burn-in observation/done context.
    h_open = ScannedRNN.initialize_carry(1, 256)
    if args.burnin_horizon > 0:
        burn_obs_t = jnp.asarray(np.stack(burnin_obs_seq, axis=0), dtype=jnp.float32)[:, jnp.newaxis, ...]
        burn_done_t = jnp.asarray(np.array(burnin_done_seq), dtype=bool)[:, jnp.newaxis]
        h_open, _, _ = network.apply(policy_vars, h_open, (burn_obs_t, burn_done_t))
    open_done = real_done

    wm_teacher = WMState(
        cache=prime_cache_with_context(
            vqvae, twm, vqvae_vars, twm_vars, max_tokens, args, burnin_obs_seq, burnin_action_seq, real_obs
        ),
        obs=real_obs,
        done=jnp.array(0.0),
        rng=jax.random.PRNGKey(args.seed + 11),
    )
    wm_open = WMState(
        cache=prime_cache_with_context(
            vqvae, twm, vqvae_vars, twm_vars, max_tokens, args, burnin_obs_seq, burnin_action_seq, real_obs
        ),
        obs=real_obs,
        done=jnp.array(0.0),
        rng=jax.random.PRNGKey(args.seed + 29),
    )

    real_frames = [np.asarray(real_obs)]
    wm_teacher_frames = [np.asarray(real_obs)]
    wm_open_frames = [np.asarray(real_obs)]
    teacher_action_names: List[str] = []
    open_action_names: List[str] = []
    warned_tf_black = False
    warned_open_black = False

    log_lines = []
    log_lines.append("row0=real_env, row1=wm_teacher_forced, row2=wm_open_loop")
    log_lines.append(f"burnin_horizon={args.burnin_horizon}")
    log_lines.append(
        "t,real_action,real_reward,real_done,wm_tf_reward,wm_tf_done_prob,wm_tf_done,wm_tf_obs_min,wm_tf_obs_max,wm_tf_obs_mean,wm_tf_obs_std,wm_tf_black_frac,"
        "wm_open_action,wm_open_reward,wm_open_done_prob,wm_open_done,wm_open_obs_min,wm_open_obs_max,wm_open_obs_mean,wm_open_obs_std,wm_open_black_frac"
    )

    for t in range(args.horizon):
        # Real policy action on real observation.
        rng, pol_rng = jax.random.split(rng)
        h_real, action_real = select_action(network, policy_vars, h_real, real_obs[jnp.newaxis, ...], real_done, pol_rng, args.policy_greedy)
        action_real = int(np.clip(action_real, 0, num_actions - 1))
        action_real_name = action_name(action_real)
        teacher_action_names.append(action_real_name)

        # Real env transition.
        rng, step_rng = jax.random.split(rng)
        next_real_obs, env_state, real_reward, real_done_scalar, _info = env.step(step_rng, env_state, jnp.array(action_real, dtype=jnp.int32), env_params)
        real_frames.append(np.asarray(next_real_obs))
        real_reward_f = float(np.asarray(real_reward))
        real_done_f = float(np.asarray(real_done_scalar))

        if bool(real_done_scalar):
            rng, r = jax.random.split(rng)
            next_real_obs, env_state = env.reset(r, env_params)
        real_obs = jnp.asarray(next_real_obs, dtype=jnp.float32)
        real_done = jnp.array([bool(real_done_scalar)])

        # WM teacher-forced with the same action.
        aid_tf = jnp.array(action_real, dtype=jnp.int32)
        wm_teacher.cache, obs_tf, rew_tf, done_tf, done_prob_tf, _rc_tf, wm_teacher.rng = wm_step(
            wm_teacher.cache, aid_tf, wm_teacher.rng
        )
        wm_teacher.obs = obs_tf
        wm_teacher.done = done_tf
        wm_teacher_frames.append(np.asarray(obs_tf))

        # WM open-loop: policy action on WM observation.
        rng, pol_open_rng = jax.random.split(rng)
        h_open, action_open = select_action(
            network,
            policy_vars,
            h_open,
            wm_open.obs[jnp.newaxis, ...],
            jnp.array([bool(float(np.asarray(wm_open.done)) >= 0.5)]),
            pol_open_rng,
            args.policy_greedy,
        )
        action_open = int(np.clip(action_open, 0, num_actions - 1))
        action_open_name = action_name(action_open)
        open_action_names.append(action_open_name)
        aid_open = jnp.array(action_open, dtype=jnp.int32)
        wm_open.cache, obs_open, rew_open, done_open, done_prob_open, _rc_open, wm_open.rng = wm_step(
            wm_open.cache, aid_open, wm_open.rng
        )
        wm_open.obs = obs_open
        wm_open.done = done_open
        wm_open_frames.append(np.asarray(obs_open))

        tf_stats = summarize_frame(np.asarray(obs_tf), args.black_pixel_threshold)
        open_stats = summarize_frame(np.asarray(obs_open), args.black_pixel_threshold)

        if args.debug_wm:
            should_print = (t % max(1, args.debug_print_every) == 0)
            if should_print:
                print(
                    f"[t={t:02d}] tf_black={tf_stats['black_frac']:.3f} tf_minmax=({tf_stats['min']:.4f},{tf_stats['max']:.4f}) "
                    f"open_black={open_stats['black_frac']:.3f} open_minmax=({open_stats['min']:.4f},{open_stats['max']:.4f}) "
                    f"tf_done_p={float(np.asarray(done_prob_tf)):.3f} open_done_p={float(np.asarray(done_prob_open)):.3f}"
                )
            if (not warned_tf_black) and tf_stats["black_frac"] > 0.98:
                print(f"[warn] teacher-forced frame nearly black at t={t}")
                warned_tf_black = True
            if (not warned_open_black) and open_stats["black_frac"] > 0.98:
                print(f"[warn] open-loop frame nearly black at t={t}")
                warned_open_black = True

        log_lines.append(
            f"{t},{action_real_name}({action_real}),{real_reward_f:.3f},{int(real_done_f)},"
            f"{float(np.asarray(rew_tf)):.3f},{float(np.asarray(done_prob_tf)):.3f},{int(float(np.asarray(done_tf)) >= 0.5)},"
            f"{tf_stats['min']:.6f},{tf_stats['max']:.6f},{tf_stats['mean']:.6f},{tf_stats['std']:.6f},{tf_stats['black_frac']:.6f},"
            f"{action_open_name}({action_open}),{float(np.asarray(rew_open)):.3f},{float(np.asarray(done_prob_open)):.3f},{int(float(np.asarray(done_open)) >= 0.5)},"
            f"{open_stats['min']:.6f},{open_stats['max']:.6f},{open_stats['mean']:.6f},{open_stats['std']:.6f},{open_stats['black_frac']:.6f}"
        )

    marker_w = 14
    sheet = build_contact_sheet(
        [real_frames, wm_teacher_frames, wm_open_frames],
        scale=args.scale,
        gap=args.gap,
        marker_w=marker_w,
    )

    if "DISPLAY" not in os.environ:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        if args.annotate_actions:
            frame_w = int(real_frames[0].shape[1]) * int(args.scale)
            surf = add_action_tracks(
                sheet,
                teacher_actions=teacher_action_names,
                open_actions=open_action_names,
                frame_w=frame_w,
                gap=args.gap,
                marker_w=marker_w,
                font_size=args.action_font_size,
            )
        else:
            surf = pygame.surfarray.make_surface(np.swapaxes(sheet, 0, 1))
        out_dir = os.path.dirname(args.output_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pygame.image.save(surf, args.output_png)
    finally:
        pygame.quit()

    out_meta = args.output_meta
    if out_meta is None:
        root, _ = os.path.splitext(args.output_png)
        out_meta = root + ".txt"
    with open(out_meta, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"Saved sheet: {args.output_png}")
    print(f"Saved metadata: {out_meta}")
    print("Row legend: red=real env, green=WM teacher-forced, blue=WM open-loop")


if __name__ == "__main__":
    main()

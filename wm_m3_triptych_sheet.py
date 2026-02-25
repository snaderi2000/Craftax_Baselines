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
import flax.linen as nn
import distrax
from flax.linen.initializers import orthogonal

from craftax.craftax_env import make_craftax_env_from_name
from token_wm.tokenizer.patch_vqvae import PatchVQVAE
from token_wm.twm.world_model import WorldModel
from token_wm.twm.transformer import TransformerConfig
from token_wm.twm.kv_caching import KeysValues

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
        if float(np.nanmax(x)) <= 1.0 + 1e-6:
            x = x * 255.0
        x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
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


class ImpalaResBlock(nn.Module):
    channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.relu(x)
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        return x + residual


class ImpalaStack(nn.Module):
    channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ImpalaResBlock(self.channels)(x)
        x = ImpalaResBlock(self.channels)(x)
        return x


class DenseResBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.width, kernel_init=orthogonal(2))(x)
        return x + residual


class ScannedRNN(nn.Module):
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            carry,
        )
        new_carry, y = nn.GRUCell(features=ins.shape[1])(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        T, B, H, W, C = obs.shape
        x_enc = obs.astype(jnp.float32).reshape((T * B, H, W, C))

        x_enc = ImpalaStack(16)(x_enc)
        x_enc = ImpalaStack(32)(x_enc)
        x_enc = ImpalaStack(32)(x_enc)
        x_enc = nn.relu(x_enc)
        x_enc = x_enc.reshape((T * B, -1))
        x_enc = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)))(x_enc)
        x_enc = nn.relu(x_enc)

        actor_feat, critic_feat = jnp.split(x_enc, 2, axis=-1)
        actor_feat = actor_feat.reshape((T, B, -1))
        critic_feat = critic_feat.reshape((T, B, -1))

        rnn_in = (jnp.concatenate([actor_feat, critic_feat], axis=-1), dones)
        h, rnn_out = ScannedRNN()(hidden, rnn_in)
        shared = rnn_out.reshape((T * B, -1))

        h_actor = nn.LayerNorm()(shared)
        h_actor = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h_actor)
        h_actor = nn.relu(h_actor)
        h_actor = DenseResBlock(self.config["LAYER_SIZE"])(h_actor)
        h_actor = DenseResBlock(self.config["LAYER_SIZE"])(h_actor)
        h_actor = nn.relu(h_actor)
        h_actor = nn.LayerNorm()(h_actor)
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h_actor)
        pi = distrax.Categorical(logits=actor_logits)

        h_critic = nn.LayerNorm()(shared)
        h_critic = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h_critic)
        h_critic = nn.relu(h_critic)
        h_critic = DenseResBlock(self.config["LAYER_SIZE"])(h_critic)
        h_critic = DenseResBlock(self.config["LAYER_SIZE"])(h_critic)
        h_critic = nn.relu(h_critic)
        h_critic = nn.LayerNorm()(h_critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(h_critic)
        return h, pi, jnp.squeeze(value, axis=-1)


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
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--gap", type=int, default=2)

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

        next_obs = vqvae.apply(vqvae_vars, obs_tokens, method=vqvae.decode_tokens)[0]
        return cache, next_obs, reward[0], done[0], done_prob[0], rew_cls[0], rng

    return step_wm


def prime_cache(vqvae, twm, vqvae_vars, twm_vars, max_tokens: int, args, start_obs: jnp.ndarray):
    cache = KeysValues.init(
        n=1,
        num_heads=args.twm_num_heads,
        max_tokens=max_tokens,
        embed_dim=args.twm_embed_dim,
        num_layers=args.twm_num_layers,
    )
    obs_tokens = vqvae.apply(vqvae_vars, start_obs[jnp.newaxis, ...], method=vqvae.encode)
    _, cache = twm.apply(twm_vars, obs_tokens, past_keys_values=cache, deterministic=True)
    return cache


def main():
    args = parse_args()
    if args.obs_image_size % args.patch_size != 0:
        raise ValueError("obs_image_size must be divisible by patch_size")
    if args.horizon <= 0:
        raise ValueError("horizon must be >= 1")

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
    h_open = ScannedRNN.initialize_carry(1, 256)
    real_done = jnp.array([False])
    open_done = jnp.array([False])

    wm_teacher = WMState(
        cache=prime_cache(vqvae, twm, vqvae_vars, twm_vars, max_tokens, args, real_obs),
        obs=real_obs,
        done=jnp.array(0.0),
        rng=jax.random.PRNGKey(args.seed + 11),
    )
    wm_open = WMState(
        cache=prime_cache(vqvae, twm, vqvae_vars, twm_vars, max_tokens, args, real_obs),
        obs=real_obs,
        done=jnp.array(0.0),
        rng=jax.random.PRNGKey(args.seed + 29),
    )

    real_frames = [np.asarray(real_obs)]
    wm_teacher_frames = [np.asarray(real_obs)]
    wm_open_frames = [np.asarray(real_obs)]

    log_lines = []
    log_lines.append("row0=real_env, row1=wm_teacher_forced, row2=wm_open_loop")
    log_lines.append("t,real_action,real_reward,real_done,wm_tf_reward,wm_tf_done_prob,wm_tf_done,wm_open_action,wm_open_reward,wm_open_done_prob,wm_open_done")

    for t in range(args.horizon):
        # Real policy action on real observation.
        rng, pol_rng = jax.random.split(rng)
        h_real, action_real = select_action(network, policy_vars, h_real, real_obs[jnp.newaxis, ...], real_done, pol_rng, args.policy_greedy)
        action_real = int(np.clip(action_real, 0, num_actions - 1))

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
        aid_open = jnp.array(action_open, dtype=jnp.int32)
        wm_open.cache, obs_open, rew_open, done_open, done_prob_open, _rc_open, wm_open.rng = wm_step(
            wm_open.cache, aid_open, wm_open.rng
        )
        wm_open.obs = obs_open
        wm_open.done = done_open
        wm_open_frames.append(np.asarray(obs_open))

        log_lines.append(
            f"{t},{action_name(action_real)}({action_real}),{real_reward_f:.3f},{int(real_done_f)},"
            f"{float(np.asarray(rew_tf)):.3f},{float(np.asarray(done_prob_tf)):.3f},{int(float(np.asarray(done_tf)) >= 0.5)},"
            f"{action_name(action_open)}({action_open}),{float(np.asarray(rew_open)):.3f},{float(np.asarray(done_prob_open)):.3f},{int(float(np.asarray(done_open)) >= 0.5)}"
        )

    sheet = build_contact_sheet([real_frames, wm_teacher_frames, wm_open_frames], scale=args.scale, gap=args.gap)

    if "DISPLAY" not in os.environ:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
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

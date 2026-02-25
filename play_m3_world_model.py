#!/usr/bin/env python3
"""
Interactive player for M3 world model rollouts (Craftax-Classic).

Loads a trained checkpoint (twm/vqvae params), resets from a real environment
observation, and lets you control actions in the imagined world model using
keyboard input.
"""

import argparse
import os
import pickle
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name

from token_wm.tokenizer.patch_vqvae import PatchVQVAE
from token_wm.twm.world_model import WorldModel
from token_wm.twm.transformer import TransformerConfig
from token_wm.twm.kv_caching import KeysValues

try:
    import pygame
except ImportError as exc:
    raise SystemExit(
        "pygame is required for GUI play. Install with: pip install pygame"
    ) from exc


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

KEY_TO_ACTION_NAME = {
    pygame.K_a: "move_left",
    pygame.K_d: "move_right",
    pygame.K_w: "move_up",
    pygame.K_s: "move_down",
    pygame.K_SPACE: "do",
    pygame.K_TAB: "sleep",
    pygame.K_r: "place_stone",
    pygame.K_t: "place_table",
    pygame.K_f: "place_furnace",
    pygame.K_p: "place_plant",
    pygame.K_1: "make_wood_pickaxe",
    pygame.K_2: "make_stone_pickaxe",
    pygame.K_3: "make_iron_pickaxe",
    pygame.K_4: "make_wood_sword",
    pygame.K_5: "make_stone_sword",
    pygame.K_6: "make_iron_sword",
}

ACTION_NAME_TO_ID = {name: idx for idx, name in enumerate(ACTION_ORDER)}
KEY_TO_ACTION_ID = {k: ACTION_NAME_TO_ID[v] for k, v in KEY_TO_ACTION_NAME.items()}


def to_uint8_frame(obs: np.ndarray, autocontrast: bool = False) -> np.ndarray:
    x = np.asarray(obs)
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        # Handle either [0, 1] or [0, 255]-style ranges.
        if float(np.nanmax(x)) <= 1.0 + 1e-6:
            x = x * 255.0
        if autocontrast:
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                x = (x - x_min) * (255.0 / (x_max - x_min))
        x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return x


def load_checkpoint_params(checkpoint_dir: str):
    req_files = ["vqvae_params.pkl", "twm_params.pkl"]
    for fname in req_files:
        path = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    with open(os.path.join(checkpoint_dir, "vqvae_params.pkl"), "rb") as f:
        vqvae_params = pickle.load(f)
    with open(os.path.join(checkpoint_dir, "twm_params.pkl"), "rb") as f:
        twm_params = pickle.load(f)
    return vqvae_params, twm_params


class WorldModelPlayer:
    def __init__(self, args, num_actions: int, vqvae_params, twm_params):
        self.args = args
        self.num_actions = num_actions
        self.vqvae_params = vqvae_params
        self.twm_params = twm_params
        self.tokens_per_obs = (args.obs_image_size // args.patch_size) ** 2
        self.tokens_per_block = self.tokens_per_obs + 1
        self.max_tokens = args.play_max_blocks * self.tokens_per_block

        self.vqvae = PatchVQVAE(
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

        twm_config = TransformerConfig(
            tokens_per_block=self.tokens_per_block,
            max_blocks=args.play_max_blocks,
            attention="causal",
            num_layers=args.twm_num_layers,
            num_heads=args.twm_num_heads,
            embed_dim=args.twm_embed_dim,
            embed_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        self.twm = WorldModel(
            obs_vocab_size=args.vqvae_codebook_size,
            act_vocab_size=num_actions,
            config=twm_config,
            reward_num_classes=2 if args.use_binary_reward_target else 3,
            binary_reward_threshold=args.binary_reward_threshold,
        )

        self.cache = None
        self.current_obs = None
        self.pred_return = 0.0
        self.episode_steps = 0
        self.total_steps = 0
        self.rng = jax.random.PRNGKey(args.seed)

        temp = float(args.temperature)
        greedy = bool(args.greedy)
        use_binary_reward_target = bool(args.use_binary_reward_target)
        tokens_per_obs = int(self.tokens_per_obs)

        def _prime_cache_fn(cache: KeysValues, obs: jnp.ndarray) -> KeysValues:
            obs_b = obs[jnp.newaxis, ...].astype(jnp.float32)
            obs_tokens = self.vqvae.apply(self.vqvae_params, obs_b, method=self.vqvae.encode)
            _, new_cache = self.twm.apply(
                self.twm_params,
                obs_tokens,
                past_keys_values=cache,
                deterministic=True,
            )
            return new_cache

        def _sample_from_logits(logits: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
            logits = jnp.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            logits = jnp.clip(logits, -50.0, 50.0)
            if greedy:
                tok = jnp.argmax(logits, axis=-1)
            else:
                tok = jax.random.categorical(rng, logits / temp, axis=-1)
            return tok.astype(jnp.int32)

        def _step_fn(
            cache: KeysValues,
            action_token: jnp.ndarray,
            rng: jnp.ndarray,
        ) -> Tuple[KeysValues, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            rng, rew_rng, done_rng, tok_rng = jax.random.split(rng, 4)

            output, cache = self.twm.apply(
                self.twm_params,
                action_token,
                past_keys_values=cache,
                deterministic=True,
            )

            rew_logits = output.logits_rewards[:, -1, :]
            rew_cls = _sample_from_logits(rew_logits, rew_rng)
            if use_binary_reward_target:
                reward = rew_cls.astype(jnp.float32)
            else:
                reward = (rew_cls - 1).astype(jnp.float32)

            done_logits = jnp.nan_to_num(output.logits_ends[:, -1, :], nan=0.0, posinf=20.0, neginf=-20.0)
            done_probs = jax.nn.softmax(done_logits, axis=-1)[:, 1]
            if greedy:
                done = (done_probs > 0.5).astype(jnp.float32)
            else:
                done = jax.random.bernoulli(done_rng, done_probs).astype(jnp.float32)

            # First observation token from action-step logits.
            obs_logits0 = output.logits_observations[:, -1, :]
            tok0 = _sample_from_logits(obs_logits0, tok_rng)

            if tokens_per_obs > 1:
                def _gen_scan(carry, _):
                    cache_scan, prev_tok, rng_scan = carry
                    prev_tok_in = prev_tok.reshape(1, 1)
                    out_scan, cache_scan = self.twm.apply(
                        self.twm_params,
                        prev_tok_in,
                        past_keys_values=cache_scan,
                        deterministic=True,
                    )
                    rng_scan, scan_rng = jax.random.split(rng_scan)
                    obs_logits = out_scan.logits_observations[:, -1, :]
                    next_tok = _sample_from_logits(obs_logits, scan_rng)
                    return (cache_scan, next_tok, rng_scan), next_tok

                (cache, _last_tok, rng), tail_tokens = jax.lax.scan(
                    _gen_scan,
                    (cache, tok0, rng),
                    None,
                    length=tokens_per_obs - 1,
                )
                tail_tokens = jnp.swapaxes(tail_tokens, 0, 1)  # (1, L-1)
                obs_tokens = jnp.concatenate([tok0[:, None], tail_tokens], axis=1)
            else:
                obs_tokens = tok0[:, None]

            next_obs = self.vqvae.apply(self.vqvae_params, obs_tokens, method=self.vqvae.decode_tokens)
            next_obs = next_obs[0]

            return cache, next_obs, reward[0], done[0], done_probs[0], rew_cls[0], rng

        self._prime_cache_fn = jax.jit(_prime_cache_fn)
        self._step_fn = jax.jit(_step_fn)

    def _new_cache(self) -> KeysValues:
        return KeysValues.init(
            n=1,
            num_heads=self.args.twm_num_heads,
            max_tokens=self.max_tokens,
            embed_dim=self.args.twm_embed_dim,
            num_layers=self.args.twm_num_layers,
        )

    def reset(self, obs: np.ndarray):
        self.current_obs = jnp.asarray(obs, dtype=jnp.float32)
        self.cache = self._new_cache()
        self.cache = self._prime_cache_fn(self.cache, self.current_obs)
        self.pred_return = 0.0
        self.episode_steps = 0

    def _refresh_cache_if_needed(self):
        cache_index = int(np.asarray(self.cache[0].index))
        # Keep enough room for one action token plus a full observation generation.
        headroom = self.tokens_per_obs + 2
        if cache_index + headroom >= self.max_tokens:
            self.cache = self._new_cache()
            self.cache = self._prime_cache_fn(self.cache, self.current_obs)

    def step(self, action_id: int):
        if action_id < 0 or action_id >= self.num_actions:
            raise ValueError(f"Action id {action_id} out of range [0, {self.num_actions - 1}]")

        self._refresh_cache_if_needed()
        action_tok = jnp.array([[action_id]], dtype=jnp.int32)
        self.cache, next_obs, reward, done, done_prob, rew_cls, self.rng = self._step_fn(
            self.cache, action_tok, self.rng
        )

        self.current_obs = next_obs
        reward_f = float(np.asarray(reward))
        done_f = float(np.asarray(done))
        done_prob_f = float(np.asarray(done_prob))
        rew_cls_i = int(np.asarray(rew_cls))

        self.pred_return += reward_f
        self.episode_steps += 1
        self.total_steps += 1
        return np.asarray(next_obs), reward_f, done_f, done_prob_f, rew_cls_i


def draw(screen, font, frame, lines, scale: int, autocontrast: bool = False):
    img = to_uint8_frame(frame, autocontrast=autocontrast)
    h, w = img.shape[:2]
    surf = pygame.surfarray.make_surface(np.swapaxes(img, 0, 1))
    surf = pygame.transform.scale(surf, (w * scale, h * scale))
    screen.fill((15, 15, 20))
    screen.blit(surf, (0, 0))

    y = h * scale + 6
    for line in lines:
        txt = font.render(line, True, (230, 230, 230))
        screen.blit(txt, (8, y))
        y += 18

    pygame.display.flip()


def parse_args():
    parser = argparse.ArgumentParser(description="Play Craftax in trained M3 world model.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="Use argmax instead of sampling.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--play_max_blocks", type=int, default=512)
    parser.add_argument(
        "--render_autocontrast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For display only: stretch each frame to [0,255] to reveal low-contrast outputs",
    )
    parser.add_argument(
        "--auto_reset_on_done",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically reset to a fresh real observation when predicted done=1",
    )
    parser.add_argument("--key_repeat_delay_ms", type=int, default=160)
    parser.add_argument("--key_repeat_interval_ms", type=int, default=60)

    # Tokenizer/world-model architecture (must match training).
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--obs_image_size", type=int, default=63)
    parser.add_argument("--vqvae_codebook_size", type=int, default=512)
    parser.add_argument("--vqvae_embed_dim", type=int, default=128)
    parser.add_argument("--patch_encoder_hidden_dim", type=int, default=128)
    parser.add_argument("--vqvae_lambda_l1", type=float, default=0.1)
    parser.add_argument("--vqvae_lambda_l2", type=float, default=1.0)
    parser.add_argument("--vqvae_lambda_codebook", type=float, default=1.0)
    parser.add_argument("--vqvae_lambda_commitment", type=float, default=0.02)
    parser.add_argument("--twm_num_layers", type=int, default=3)
    parser.add_argument("--twm_num_heads", type=int, default=8)
    parser.add_argument("--twm_embed_dim", type=int, default=128)
    parser.add_argument("--use_binary_reward_target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary_reward_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.obs_image_size % args.patch_size != 0:
        raise ValueError("obs_image_size must be divisible by patch_size")

    vqvae_params, twm_params = load_checkpoint_params(args.checkpoint_dir)

    env = make_craftax_env_from_name(args.env_name, True)
    env_params = env.default_params
    num_actions = int(env.action_space(env_params).n)

    max_bound = max(KEY_TO_ACTION_ID.values())
    if max_bound >= num_actions:
        raise ValueError(
            f"Key mapping expects action id up to {max_bound}, but env has only {num_actions} actions."
        )

    print("\nKey bindings:")
    for key, action_name in KEY_TO_ACTION_NAME.items():
        print(f"  {pygame.key.name(key):>5} -> {action_name} ({ACTION_NAME_TO_ID[action_name]})")
    print("  backspace -> reset imagined episode")
    print("  esc/q     -> quit\n")

    player = WorldModelPlayer(args, num_actions, vqvae_params, twm_params)

    env_rng = jax.random.PRNGKey(args.seed + 1000)

    def reset_from_env():
        nonlocal env_rng
        env_rng, r = jax.random.split(env_rng)
        obs, _ = env.reset(r, env_params)
        player.reset(np.asarray(obs))
        return np.asarray(obs)

    frame = reset_from_env()

    pygame.init()
    try:
        if args.key_repeat_delay_ms > 0 and args.key_repeat_interval_ms > 0:
            pygame.key.set_repeat(args.key_repeat_delay_ms, args.key_repeat_interval_ms)
        font = pygame.font.SysFont("monospace", 16)
        h, w = frame.shape[:2]
        screen = pygame.display.set_mode((w * args.scale, h * args.scale + 120))
        pygame.display.set_caption("Craftax M3 World Model Player")
        clock = pygame.time.Clock()

        last_action_name = "none"
        last_reward = 0.0
        last_done = 0.0
        last_done_prob = 0.0
        last_reward_cls = 0
        frame_min = float(np.asarray(frame).min())
        frame_max = float(np.asarray(frame).max())

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        continue

                    if event.key == pygame.K_BACKSPACE:
                        frame = reset_from_env()
                        frame_min = float(np.asarray(frame).min())
                        frame_max = float(np.asarray(frame).max())
                        last_action_name = "reset"
                        last_reward = 0.0
                        last_done = 0.0
                        last_done_prob = 0.0
                        last_reward_cls = 0
                        continue

                    if event.key in KEY_TO_ACTION_ID:
                        action_id = KEY_TO_ACTION_ID[event.key]
                        action_name = KEY_TO_ACTION_NAME[event.key]
                        frame, last_reward, last_done, last_done_prob, last_reward_cls = player.step(action_id)
                        frame_min = float(np.asarray(frame).min())
                        frame_max = float(np.asarray(frame).max())
                        last_action_name = action_name
                        if args.auto_reset_on_done and last_done >= 0.5:
                            frame = reset_from_env()
                            frame_min = float(np.asarray(frame).min())
                            frame_max = float(np.asarray(frame).max())
                            last_action_name = f"{action_name} (done->reset)"

            lines = [
                f"episode_step={player.episode_steps}  total_step={player.total_steps}  pred_return={player.pred_return:.2f}",
                f"last_action={last_action_name}",
                f"pred_reward={last_reward:.3f} (class={last_reward_cls})",
                f"pred_done={last_done:.0f}  done_prob={last_done_prob:.3f}",
                f"frame_min={frame_min:.2f}  frame_max={frame_max:.2f}",
                f"temp={args.temperature:.2f}  greedy={args.greedy}  max_blocks={args.play_max_blocks}  auto_reset={args.auto_reset_on_done}",
            ]
            draw(screen, font, frame, lines, args.scale, autocontrast=args.render_autocontrast)
            clock.tick(args.fps)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

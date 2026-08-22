import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax

if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)

    jax.tree = _JaxTreeCompat()

import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import render_local_view


def _render_labeled_step(
    obs: np.ndarray,
    episode: int,
    step: int,
    value: float,
    action: int,
    reward: float,
    done: bool,
    block_size: int,
    display_scale: int,
) -> Image.Image:
    textures = load_all_textures(block_size)
    panel_textures = (
        textures
        if display_scale == 1
        else load_all_textures(block_size * display_scale)
    )
    view = render_local_view(
        obs,
        block_size=block_size,
        include_inventory=True,
        display_scale=display_scale,
        textures=textures,
        panel_textures=panel_textures,
    )
    label_h = max(82, 18 * display_scale)
    out = Image.new("RGB", (view.width, view.height + label_h), "white")
    out.paste(view, (0, 0))
    draw = ImageDraw.Draw(out)
    font_size = max(22, 5 * display_scale)
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            break
    if font is None:
        font = ImageFont.load_default()
    line_1 = f"step: {step}"
    line_2 = f"value: {value:.4f}"
    y0 = view.height + max(4, (label_h - 2 * font_size - 8) // 2)
    draw.text((10, y0), line_1, fill=(0, 0, 0), font=font)
    draw.text((10, y0 + font_size + 8), line_2, fill=(0, 0, 0), font=font)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one reproduced PPO episode step.")
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/episode_step_renders")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=19)
    parser.add_argument("--step", type=int, default=610)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=6)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name

    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )

    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    @jax.jit
    def policy_apply(params, obs, rng):
        pi, value = network.apply(params, obs[None, ...])
        if args.greedy:
            action = jnp.argmax(pi.logits[0])
            return action, value[0], rng
        rng, action_rng = jax.random.split(rng)
        return pi.sample(seed=action_rng)[0], value[0], rng

    rng = jax.random.PRNGKey(args.seed)
    target = None
    for episode in range(args.episode + 1):
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, env_state = reset_jit(reset_rng, env_params)
        for step in range(args.max_steps):
            action, value, policy_rng = policy_apply(train_state.params, obs, policy_rng)
            action_i = int(np.asarray(action))
            value_f = float(np.asarray(value))
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, _info = step_jit(
                step_rng, env_state, action, env_params
            )
            reward_f = float(np.asarray(reward))
            done_b = bool(np.asarray(done))
            if episode == args.episode and step == args.step:
                target = {
                    "obs": np.asarray(obs),
                    "action": action_i,
                    "value": value_f,
                    "reward": reward_f,
                    "done": done_b,
                }
                break
            obs = next_obs
            if done_b:
                break
        if target is not None:
            break

    if target is None:
        raise RuntimeError(f"Could not reach episode={args.episode} step={args.step}")

    img = _render_labeled_step(
        target["obs"],
        args.episode,
        args.step,
        target["value"],
        target["action"],
        target["reward"],
        target["done"],
        args.block_size,
        args.display_scale,
    )
    out_path = os.path.join(
        args.out_dir,
        f"episode_{args.episode:02d}_step_{args.step:04d}.png",
    )
    img.save(out_path)
    print(f"checkpoint_step={checkpoint_step}")
    print(
        f"step={args.step} value={target['value']:.6f} "
        f"action={target['action']}:{_action_name(target['action'])} "
        f"reward={target['reward']:.6f} done={target['done']}"
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

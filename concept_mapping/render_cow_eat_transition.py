"""Roll out a frozen PPO policy and render its first cow-eating transition."""

import argparse
import csv
import os
import sys
from pathlib import Path

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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from craftax.craftax_classic.constants import Achievement, load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import (
    _action_name, _init_network, _load_wandb_config, _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import render_local_view


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def write_figure(before, after, output_dir, metadata, block_size, display_scale):
    textures = load_all_textures(block_size)
    panel_textures = load_all_textures(block_size * display_scale)
    views = [
        render_local_view(obs, block_size=block_size, include_inventory=True,
                          display_scale=display_scale, textures=textures,
                          panel_textures=panel_textures)
        for obs in (before, after)
    ]
    frames = []
    for view, title, subtitle in zip(
        views,
        ("Before eating cow", "After eating cow"),
        ("cow present", "food restored"),
    ):
        footer = Image.new("RGB", (view.width, 76), "white")
        draw = ImageDraw.Draw(footer)
        draw.text((5, 5), title, fill="black", font=font(24, bold=True))
        draw.text((5, 39), subtitle, fill="black", font=font(23))
        frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
        frame.paste(view, (0, 0)); frame.paste(footer, (0, view.height))
        frames.append(frame)

    margin, gap = 24, 32
    width, height = frames[0].size
    figure = Image.new("RGB", (2 * width + gap + 2 * margin, height + 2 * margin), "white")
    figure.paste(frames[0], (margin, margin))
    figure.paste(frames[1], (margin + width + gap, margin))
    figure.save(output_dir / "cow_eat_transition.png")
    views[0].save(output_dir / "before_eating_cow.png")
    views[1].save(output_dir / "after_eating_cow.png")
    with (output_dir / "transition_metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata))
        writer.writeheader(); writer.writerow(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--timestep", type=int, default=9999941632)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--out_dir", default="concept_mapping/runs/cow_eat_transition_m0mw4end")
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    output_dir = Path(args.out_dir); output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )
    reset, step_env = jax.jit(env.reset), jax.jit(env.step)

    @jax.jit
    def policy(obs, rng):
        pi, _value = network.apply(train_state.params, obs[None, ...])
        if args.greedy:
            return rng, pi.logits[0].argmax()
        rng, action_rng = jax.random.split(rng)
        return rng, pi.sample(seed=action_rng)[0]

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng, policy_rng = jax.random.split(rng, 3)
    obs, state = reset(reset_rng, env_params)
    achievement = Achievement.EAT_COW.value
    for step in range(args.max_steps):
        policy_rng, action = policy(obs, policy_rng)
        before = np.asarray(obs, dtype=np.float32)
        before_achievement = bool(np.asarray(state.achievements[achievement]))
        rng, step_rng = jax.random.split(rng)
        next_obs, next_state, reward, done, _info = step_env(step_rng, state, action, env_params)
        ate_cow = not before_achievement and bool(np.asarray(next_state.achievements[achievement]))
        if ate_cow:
            metadata = {
                "checkpoint": checkpoint_step, "seed": args.seed, "episode_step": step,
                "action": int(np.asarray(action)), "action_name": _action_name(int(np.asarray(action))),
                "reward": float(np.asarray(reward)),
                "food_before": float(np.asarray(state.player_food)),
                "food_after": float(np.asarray(next_state.player_food)),
            }
            write_figure(before, np.asarray(next_obs, dtype=np.float32), output_dir, metadata,
                         args.block_size, args.display_scale)
            print(f"Saved cow-eating transition at step {step}: {metadata}", flush=True)
            return
        obs, state = next_obs, next_state
        if bool(np.asarray(done)):
            break
    raise RuntimeError(f"No cow-eating transition found in this episode ({step + 1} steps).")


if __name__ == "__main__":
    main()

"""Roll out a frozen PPO policy and render its first stone-pickaxe transition."""

import argparse
import json
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

import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from PIL import Image, ImageDraw, ImageFont

from concept_mapping.render_ppo_episodes import _action_name, _init_network, _load_wandb_config, _restore_train_state
from concept_mapping.visualize_symbolic_trajectory import MAP_DIM, render_local_view


STONE_PICKAXE_OFFSET = MAP_DIM + 7


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{filename}"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def _write_figure(before, after, output_dir: Path, metadata: dict, block_size: int, display_scale: int):
    textures = load_all_textures(block_size)
    panel_textures = load_all_textures(block_size * display_scale)
    views = [
        render_local_view(obs, block_size, include_inventory=True, display_scale=display_scale, textures=textures, panel_textures=panel_textures)
        for obs in (before, after)
    ]
    labels = ("Before crafting", "After crafting")
    frames = []
    for view, label in zip(views, labels):
        footer = Image.new("RGB", (view.width, 74), "white")
        draw = ImageDraw.Draw(footer)
        draw.text((5, 5), label, fill="black", font=_font(27, bold=True))
        draw.text((5, 38), "stone pickaxe" + (": absent" if label.startswith("Before") else ": obtained"), fill="black", font=_font(23))
        frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
        frame.paste(view, (0, 0))
        frame.paste(footer, (0, view.height))
        frames.append(frame)
    gap, margin = 32, 24
    width, height = frames[0].size
    figure = Image.new("RGB", (2 * width + gap + 2 * margin, height + 2 * margin), "white")
    figure.paste(frames[0], (margin, margin))
    figure.paste(frames[1], (margin + width + gap, margin))
    figure.save(output_dir / "stone_pickaxe_transition.png")
    views[0].save(output_dir / "before_stone_pickaxe.png")
    views[1].save(output_dir / "after_stone_pickaxe.png")
    with (output_dir / "transition_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--timestep", type=int, default=9999941632)
    parser.add_argument("--out_dir", default="concept_mapping/runs/stone_pickaxe_transition_m0mw4end")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--block_size", type=int, default=10)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, restored_step = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)
    reset = jax.jit(env.reset)

    @jax.jit
    def rollout(initial_obs, initial_state, rng, policy_rng):
        """One full episode trace without host-device synchronisation per step."""
        def body(carry, _unused):
            obs, state, rng, policy_rng, already_done = carry
            rng, step_rng = jax.random.split(rng)
            if args.greedy:
                pi, _value = network.apply(train_state.params, obs[None, ...])
                action, next_policy_rng = jnp.argmax(pi.logits[0]), policy_rng
            else:
                next_policy_rng, action_rng = jax.random.split(policy_rng)
                pi, _value = network.apply(train_state.params, obs[None, ...])
                action = pi.sample(seed=action_rng)[0]

            def active(_):
                next_obs, next_state, reward, done, _info = env.step(
                    step_rng, state, action, env_params
                )
                return next_obs, next_state, reward, done

            def inactive(_):
                return obs, state, jnp.asarray(0.0), jnp.asarray(True)

            next_obs, next_state, reward, done = jax.lax.cond(
                already_done, inactive, active, operand=None
            )
            next_done = jnp.logical_or(already_done, done)
            return (next_obs, next_state, rng, next_policy_rng, next_done), (obs, next_obs, action, reward, done)

        initial = (initial_obs, initial_state, rng, policy_rng, jnp.asarray(False))
        return jax.lax.scan(body, initial, xs=None, length=args.max_steps)

    rng = jax.random.PRNGKey(args.seed)
    for episode in range(args.max_episodes):
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, state = reset(reset_rng, env_params)
        (next_obs, next_state, rng, policy_rng, _done), trace = rollout(obs, state, rng, policy_rng)
        before_trace, after_trace, actions, rewards, dones = (np.asarray(value) for value in trace)
        transitions = np.flatnonzero(
            (before_trace[:, STONE_PICKAXE_OFFSET] < 0.05)
            & (after_trace[:, STONE_PICKAXE_OFFSET] >= 0.05)
        )
        if len(transitions):
            episode_step = int(transitions[0])
            before, after = before_trace[episode_step], after_trace[episode_step]
            action_i, reward = int(actions[episode_step]), float(rewards[episode_step])
            metadata = {
                "checkpoint": restored_step,
                "episode": episode,
                "episode_step": episode_step,
                "action": action_i,
                "action_name": _action_name(action_i),
                "reward": reward,
                "stone_pickaxe_before": float(before[STONE_PICKAXE_OFFSET] * 10),
                "stone_pickaxe_after": float(after[STONE_PICKAXE_OFFSET] * 10),
            }
            _write_figure(before, after, output_dir, metadata, args.block_size, args.display_scale)
            print(f"Saved transition at episode={episode}, step={episode_step}: {metadata['action_name']}", flush=True)
            return
        print(f"Scanned episode {episode + 1}/{args.max_episodes}", flush=True)
    raise RuntimeError(f"No stone-pickaxe transition found in {args.max_episodes} episodes")


if __name__ == "__main__":
    main()

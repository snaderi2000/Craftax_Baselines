"""Render the final steps of a frozen PPO rollout ending after diamond collection."""

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

import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import _action_name, _init_network, _load_wandb_config, _restore_train_state
from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_DIM, render_local_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--last_steps", type=int, default=5)
    parser.add_argument("--out_dir", default="concept_mapping/runs/lava_suicide_episode00_last5")
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint_step = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)
    reset = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    @jax.jit
    def policy(params, obs, rng):
        pi, value = network.apply(params, obs[None, ...])
        if args.greedy:
            return jnp.argmax(pi.logits[0]), rng, value[0]
        rng, action_rng = jax.random.split(rng)
        return pi.sample(seed=action_rng)[0], rng, value[0]

    rng = jax.random.PRNGKey(args.seed)
    target = None
    for episode_index in range(args.episode + 1):
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, state = reset(reset_rng, env_params)
        trace, diamond_first_collected = [], None
        for step in range(args.max_steps):
            action, policy_rng, value = policy(train_state.params, obs, policy_rng)
            action_int = int(np.asarray(action))
            obs_np = np.asarray(obs, dtype=np.float32)
            rng, step_rng = jax.random.split(rng)
            next_obs, state, reward, done, _info = step_env(step_rng, state, action, env_params)
            next_obs_np = np.asarray(next_obs, dtype=np.float32)
            if diamond_first_collected is None and obs_np[MAP_DIM + INV_NAMES.index("diamond")] <= .05 and next_obs_np[MAP_DIM + INV_NAMES.index("diamond")] > .05:
                diamond_first_collected = step
            trace.append({
                "step": step, "obs": obs_np, "action": action_int,
                "action_name": _action_name(action_int), "reward": float(np.asarray(reward)),
                "done": bool(np.asarray(done)), "value": float(np.asarray(value)),
                "diamond_count": float(obs_np[MAP_DIM + INV_NAMES.index("diamond")] * 10),
            })
            obs = next_obs
            if bool(np.asarray(done)):
                break
        if episode_index == args.episode:
            target = (trace, diamond_first_collected)
    if target is None:
        raise RuntimeError("Target episode was not collected")
    trace, diamond_first_collected = target
    if not trace[-1]["done"]:
        raise RuntimeError(f"Episode did not terminate within {args.max_steps} steps")
    selected = trace[-args.last_steps:]

    textures = load_all_textures(args.block_size)
    panel_textures = textures if args.display_scale == 1 else load_all_textures(args.block_size * args.display_scale)
    rendered = []
    bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    action_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    for record in selected:
        view = render_local_view(record["obs"], block_size=args.block_size, include_inventory=True,
                                 display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
        footer_h = 73
        cell = Image.new("RGB", (view.width, view.height + footer_h), "white")
        draw = ImageDraw.Draw(cell)
        cell.paste(view, (0, 0))
        action_name = record["action_name"].replace("_", " ").title()
        title = f"Step {record['step']}" + (" (terminal)" if record["done"] else "")
        draw.text((5, view.height + 4), title, fill="black", font=bold_font)
        draw.text((5, view.height + 28), f"Action: {action_name}", fill="black", font=action_font)
        draw.text((5, view.height + 48), f"Value: {record['value']:.4f}", fill="black", font=bold_font)
        rendered.append(cell)

    gap = 18
    sheet = Image.new("RGB", (sum(cell.width for cell in rendered) + gap * (len(rendered) - 1), max(cell.height for cell in rendered)), "white")
    x = 0
    for cell in rendered:
        sheet.paste(cell, (x, 0)); x += cell.width + gap
    sheet.save(out_dir / "last_5_steps.png")
    with (out_dir / "last_5_steps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "action", "action_name", "reward", "done", "value", "diamond_count"])
        writer.writeheader(); writer.writerows([{key: record[key] for key in writer.fieldnames} for record in selected])
    (out_dir / "summary.txt").write_text(
        f"checkpoint_step={checkpoint_step}\nseed={args.seed}\nepisode={args.episode}\n"
        f"diamond_first_collected_step={diamond_first_collected}\nterminal_step={trace[-1]['step']}\n"
        f"terminal_action={trace[-1]['action_name']}\n"
    )
    print(f"Saved {out_dir / 'last_5_steps.png'}")
    print(f"Diamond first collected at step {diamond_first_collected}; terminal DOWN action at step {trace[-1]['step']}")


if __name__ == "__main__":
    main()

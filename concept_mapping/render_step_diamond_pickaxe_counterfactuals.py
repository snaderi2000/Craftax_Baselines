import argparse
import csv
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

from craftax.craftax_classic.constants import BlockType, OBS_DIM, load_all_textures
from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import (
    MAP_CHANNELS,
    MAP_DIM,
    render_local_view,
)


IRON_PICKAXE_INDEX = 8
BASIC_RESOURCE_INDICES = (0, 1, 2, 3)


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _capture_episode_step(args, network, train_state, env, env_params):
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
                return np.asarray(obs, dtype=np.float32), action_i, value_f, reward_f, done_b
            obs = next_obs
            if done_b:
                break
    raise RuntimeError(f"Could not reach episode={args.episode} step={args.step}")


def _set_visible_diamonds(obs: np.ndarray, block: BlockType) -> np.ndarray:
    out = np.array(obs, copy=True)
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    blocks = np.argmax(map_view[..., : len(BlockType)], axis=-1)
    for row, col in np.argwhere(blocks == BlockType.DIAMOND.value):
        map_view[row, col, : len(BlockType)] = 0.0
        map_view[row, col, block.value] = 1.0
    return out


def _set_iron_pickaxe(obs: np.ndarray, present: bool) -> np.ndarray:
    out = np.array(obs, copy=True)
    out[MAP_DIM + IRON_PICKAXE_INDEX] = 0.1 if present else 0.0
    return out


def _set_full_basic_resources(obs: np.ndarray) -> np.ndarray:
    out = np.array(obs, copy=True)
    for idx in BASIC_RESOURCE_INDICES:
        out[MAP_DIM + idx] = 0.9
    return out


def _evaluate_values(network, params, obs_by_key: dict[str, np.ndarray]) -> dict[str, float]:
    keys = list(obs_by_key.keys())
    obs_batch = jnp.asarray(np.stack([obs_by_key[k] for k in keys], axis=0), dtype=jnp.float32)
    _pi, values = network.apply(params, obs_batch)
    values = np.asarray(values).reshape(-1)
    return {key: float(value) for key, value in zip(keys, values)}


def _make_panel(
    obs: np.ndarray,
    value: float,
    block_size: int,
    display_scale: int,
    textures: dict,
    panel_textures: dict,
) -> Image.Image:
    view = render_local_view(
        obs,
        block_size=block_size,
        include_inventory=True,
        display_scale=display_scale,
        textures=textures,
        panel_textures=panel_textures,
    )
    label_h = max(74, 15 * display_scale)
    out = Image.new("RGB", (view.width, view.height + label_h), "white")
    out.paste(view, (0, 0))
    draw = ImageDraw.Draw(out)
    font = _font(max(36, 7 * display_scale))
    draw.text((0, view.height + 2), f"value: {value:.4f}", fill=(0, 0, 0), font=font)
    return out


def _compose_2x2(panels: dict[str, Image.Image], out_path: str, gap_x: int, gap_y: int) -> None:
    order = {
        "no_pickaxe_no_diamond": (0, 0),
        "pickaxe_no_diamond": (1, 0),
        "no_pickaxe_diamond": (0, 1),
        "pickaxe_diamond": (1, 1),
    }
    cell_w = max(panel.width for panel in panels.values())
    cell_h = max(panel.height for panel in panels.values())
    width = 2 * cell_w + gap_x
    height = 2 * cell_h + gap_y
    out = Image.new("RGB", (width, height), "white")
    for key, (col, row) in order.items():
        x = col * (cell_w + gap_x)
        y = row * (cell_h + gap_y)
        out.paste(panels[key], (x, y))
    out.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 2x2 diamond/iron-pickaxe counterfactuals for one PPO state."
    )
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/episode19_step610_diamond_pickaxe_counterfactuals")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=19)
    parser.add_argument("--step", type=int, default=610)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=5)
    parser.add_argument("--gap_x", type=int, default=160)
    parser.add_argument("--gap_y", type=int, default=52)
    parser.add_argument(
        "--full_basic_resources",
        action="store_true",
        help="Set wood, stone, coal, and iron inventory to 9 in every counterfactual.",
    )
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

    base_obs, action, base_value, reward, done = _capture_episode_step(
        args, network, train_state, env, env_params
    )
    diamond_visible = np.array(base_obs, copy=True)
    if args.full_basic_resources:
        diamond_visible = _set_full_basic_resources(diamond_visible)
    diamond_covered = _set_visible_diamonds(base_obs, BlockType.STONE)
    if args.full_basic_resources:
        diamond_covered = _set_full_basic_resources(diamond_covered)

    obs_by_key = {
        "no_pickaxe_no_diamond": _set_iron_pickaxe(diamond_covered, False),
        "pickaxe_no_diamond": _set_iron_pickaxe(diamond_covered, True),
        "no_pickaxe_diamond": _set_iron_pickaxe(diamond_visible, False),
        "pickaxe_diamond": _set_iron_pickaxe(diamond_visible, True),
    }
    values = _evaluate_values(network, train_state.params, obs_by_key)

    csv_path = os.path.join(args.out_dir, f"episode_{args.episode:02d}_step_{args.step:04d}_diamond_pickaxe_values.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "diamond_visible", "iron_pickaxe_present", "value"])
        writer.writerow(["top_left", False, False, values["no_pickaxe_no_diamond"]])
        writer.writerow(["top_right", False, True, values["pickaxe_no_diamond"]])
        writer.writerow(["bottom_left", True, False, values["no_pickaxe_diamond"]])
        writer.writerow(["bottom_right", True, True, values["pickaxe_diamond"]])
        writer.writerow([])
        writer.writerow(["base_step_value", base_value])
        writer.writerow(["base_action", action])
        writer.writerow(["base_action_name", _action_name(action)])
        writer.writerow(["reward_after_action", reward])
        writer.writerow(["done_after_action", done])
        writer.writerow(["checkpoint_step", checkpoint_step])

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    panels = {
        key: _make_panel(
            obs,
            values[key],
            args.block_size,
            args.display_scale,
            textures,
            panel_textures,
        )
        for key, obs in obs_by_key.items()
    }
    fig_path = os.path.join(args.out_dir, f"episode_{args.episode:02d}_step_{args.step:04d}_diamond_pickaxe_grid.png")
    _compose_2x2(panels, fig_path, args.gap_x, args.gap_y)

    print(f"checkpoint_step={checkpoint_step}")
    print(f"base step value={base_value:.6f} action={action}:{_action_name(action)} reward={reward:.6f} done={done}")
    for key in ("no_pickaxe_no_diamond", "pickaxe_no_diamond", "no_pickaxe_diamond", "pickaxe_diamond"):
        print(f"{key}: V={values[key]:.6f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()

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
from PIL import Image, ImageDraw

from craftax.craftax_classic.constants import BlockType, OBS_DIM, load_all_textures
from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.render_ppo_episodes import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import (
    INV_NAMES,
    MAP_CHANNELS,
    MAP_DIM,
    render_local_view,
)


def _make_all_grass_obs(inventory_counts: np.ndarray) -> np.ndarray:
    obs = np.zeros((MAP_DIM + 22,), dtype=np.float32)
    map_view = obs[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    map_view[..., BlockType.GRASS.value] = 1.0
    stats = obs[MAP_DIM:]
    stats[:12] = np.asarray(inventory_counts, dtype=np.float32) / 10.0
    stats[12:16] = np.array([9, 9, 9, 9], dtype=np.float32) / 10.0
    stats[16 + 2] = 1.0
    stats[20] = 1.0
    return obs


def _profiles() -> list[tuple[str, str, np.ndarray]]:
    return [
        ("empty", "No inventory", np.zeros(12, dtype=np.float32)),
        (
            "wood_pickaxe",
            "Wood pickaxe",
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        ),
        (
            "wood_sword",
            "Wood sword",
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.float32),
        ),
        (
            "late_no_diamond",
            "Late game, no diamond",
            np.array([9, 9, 6, 9, 0, 6, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        ),
    ]


def _evaluate_values(network, params, obs_by_key: dict[str, np.ndarray]) -> dict[str, float]:
    keys = list(obs_by_key.keys())
    obs_batch = jnp.asarray(np.stack([obs_by_key[k] for k in keys], axis=0), dtype=jnp.float32)
    _pi, values = network.apply(params, obs_batch)
    values = np.asarray(values).reshape(-1)
    return {key: float(value) for key, value in zip(keys, values)}


def _make_cell(
    obs: np.ndarray,
    title: str,
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
    header_h = 48
    out = Image.new("RGB", (view.width, header_h + view.height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), title, fill=(20, 40, 90))
    draw.text((8, 25), f"V(s) = {value:.4f}", fill=(0, 0, 0))
    out.paste(view, (0, header_h))
    return out


def _compose(cells: dict[str, Image.Image], values: dict[str, float], out_path: str, checkpoint_step: int) -> None:
    keys = ["empty", "wood_pickaxe", "wood_sword", "late_no_diamond"]
    gap = 16
    foot_h = 84
    cell_w = max(cells[k].width for k in keys)
    cell_h = max(cells[k].height for k in keys)
    width = 2 * cell_w + gap
    height = 2 * cell_h + gap + foot_h
    sheet = Image.new("RGB", (width, height), "white")
    positions = {
        "empty": (0, 0),
        "wood_pickaxe": (cell_w + gap, 0),
        "wood_sword": (0, cell_h + gap),
        "late_no_diamond": (cell_w + gap, cell_h + gap),
    }
    for key, pos in positions.items():
        sheet.paste(cells[key], pos)

    draw = ImageDraw.Draw(sheet)
    y = 2 * cell_h + gap + 10
    drop_empty_to_pickaxe = values["wood_pickaxe"] - values["empty"]
    drop_pickaxe_to_sword = values["wood_sword"] - values["wood_pickaxe"]
    drop_sword_to_late = values["late_no_diamond"] - values["wood_sword"]
    draw.text((8, y), f"10B PPO symbolic critic; checkpoint={checkpoint_step}; all grass; full intrinsics", fill=(0, 0, 0))
    draw.text((8, y + 20), f"empty -> wood pickaxe: {drop_empty_to_pickaxe:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 38), f"wood pickaxe -> wood sword: {drop_pickaxe_to_sword:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 56), f"wood sword -> late/no diamond: {drop_sword_to_late:.4f}", fill=(0, 0, 0))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a 2x2 inventory progression value probe in an all-grass scene."
    )
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/inventory_progress_quadrants")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
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

    profiles = _profiles()
    obs_by_key = {key: _make_all_grass_obs(inv) for key, _title, inv in profiles}
    values = _evaluate_values(network, train_state.params, obs_by_key)

    csv_path = os.path.join(args.out_dir, "inventory_progress_quadrants.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "title", "value", "inventory_order", "inventory_counts"])
        for key, title, inv in profiles:
            writer.writerow([
                key,
                title,
                values[key],
                ",".join(INV_NAMES),
                ",".join(str(int(x)) for x in inv),
            ])
        writer.writerow([])
        writer.writerow(["checkpoint_step", checkpoint_step])

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    cells = {
        key: _make_cell(
            obs_by_key[key],
            title,
            values[key],
            args.block_size,
            args.display_scale,
            textures,
            panel_textures,
        )
        for key, title, _inv in profiles
    }
    fig_path = os.path.join(args.out_dir, "inventory_progress_quadrants_grid.png")
    _compose(cells, values, fig_path, checkpoint_step)

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved figure: {fig_path}", flush=True)
    for key, title, _inv in profiles:
        print(f"{title}: V={values[key]:.4f}", flush=True)


if __name__ == "__main__":
    main()

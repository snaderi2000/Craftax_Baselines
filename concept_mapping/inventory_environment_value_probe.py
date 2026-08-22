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
    MAP_CHANNELS,
    MAP_DIM,
    render_local_view,
)


def _set_block(map_view: np.ndarray, row: int, col: int, block: BlockType) -> None:
    map_view[row, col, : len(BlockType)] = 0.0
    map_view[row, col, block.value] = 1.0


def _make_grass_scene(
    stone_in_front: bool,
    wood_pickaxe: bool,
    stone_top_right_quadrant: bool,
) -> np.ndarray:
    obs = np.zeros((MAP_DIM + 22,), dtype=np.float32)
    map_view = obs[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    map_view[..., BlockType.GRASS.value] = 1.0

    center_r, center_c = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    if stone_in_front:
        if stone_top_right_quadrant:
            for row in range(0, center_r + 1):
                for col in range(center_c, OBS_DIM[1]):
                    _set_block(map_view, row, col, BlockType.STONE)
        else:
            _set_block(map_view, center_r - 1, center_c, BlockType.STONE)

    stats = obs[MAP_DIM:]
    if wood_pickaxe:
        stats[6] = 0.1
    stats[12:16] = np.array([9, 9, 9, 9], dtype=np.float32) / 10.0
    stats[16 + 2] = 1.0  # facing up, so "front" is the cell above.
    stats[20] = 1.0
    return obs


def _make_grass_tree_wood_scene(tree_in_front: bool, wood_count: int) -> np.ndarray:
    obs = np.zeros((MAP_DIM + 22,), dtype=np.float32)
    map_view = obs[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    map_view[..., BlockType.GRASS.value] = 1.0

    center_r, center_c = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    if tree_in_front:
        _set_block(map_view, center_r - 1, center_c, BlockType.TREE)

    stats = obs[MAP_DIM:]
    stats[0] = float(wood_count) / 10.0
    stats[12:16] = np.array([9, 9, 9, 9], dtype=np.float32) / 10.0
    stats[16 + 2] = 1.0
    stats[20] = 1.0
    return obs


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
    header_h = 46
    out = Image.new("RGB", (view.width, header_h + view.height), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 6), title, fill=(20, 40, 90))
    draw.text((8, 24), f"V(s) = {value:.4f}", fill=(0, 0, 0))
    out.paste(view, (0, header_h))
    return out


def _compose_grid(
    cells: dict[str, Image.Image],
    values: dict[str, float],
    out_path: str,
    checkpoint_step: int,
    stone_label: str,
    bottom_row_label: str = "Wood pickaxe",
    stone_effect_label: str = "Stone effect",
    inventory_effect_label: str = "Pickaxe effect",
) -> None:
    row_label_w = 130
    col_header_h = 44
    foot_h = 128
    gap = 14
    keys = ("empty_grass", "empty_stone", "pickaxe_grass", "pickaxe_stone")
    cell_w = max(cells[k].width for k in keys)
    cell_h = max(cells[k].height for k in keys)
    width = row_label_w + 2 * cell_w + gap
    height = col_header_h + 2 * cell_h + gap + foot_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((row_label_w + 8, 12), "All grass", fill=(0, 0, 0))
    draw.text((row_label_w + cell_w + gap + 8, 12), stone_label, fill=(0, 0, 0))
    draw.text((10, col_header_h + cell_h // 2 - 8), "Empty inventory", fill=(0, 0, 0))
    draw.text((10, col_header_h + cell_h + gap + cell_h // 2 - 8), bottom_row_label, fill=(0, 0, 0))

    positions = {
        "empty_grass": (row_label_w, col_header_h),
        "empty_stone": (row_label_w + cell_w + gap, col_header_h),
        "pickaxe_grass": (row_label_w, col_header_h + cell_h + gap),
        "pickaxe_stone": (row_label_w + cell_w + gap, col_header_h + cell_h + gap),
    }
    for key, pos in positions.items():
        sheet.paste(cells[key], pos)

    empty_stone_effect = values["empty_stone"] - values["empty_grass"]
    pickaxe_stone_effect = values["pickaxe_stone"] - values["pickaxe_grass"]
    pickaxe_no_stone_effect = values["pickaxe_grass"] - values["empty_grass"]
    pickaxe_with_stone_effect = values["pickaxe_stone"] - values["empty_stone"]
    interaction = pickaxe_stone_effect - empty_stone_effect

    y = col_header_h + 2 * cell_h + gap + 10
    draw.text((8, y), f"10B PPO symbolic critic; checkpoint={checkpoint_step}; full health/food/drink/energy", fill=(0, 0, 0))
    draw.text((8, y + 20), f"{stone_effect_label} without inventory: {empty_stone_effect:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 38), f"{stone_effect_label} with inventory:    {pickaxe_stone_effect:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 56), f"{inventory_effect_label} without feature: {pickaxe_no_stone_effect:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 74), f"{inventory_effect_label} with feature:    {pickaxe_with_stone_effect:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 92), f"Interaction: {interaction:.4f}", fill=(0, 0, 0))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe value-function interactions between inventory and local environment."
    )
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/inventory_environment_value_probe")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument(
        "--stone_top_right_quadrant",
        action="store_true",
        help="Use a top-right quadrant of stone in the stone condition instead of one stone tile.",
    )
    parser.add_argument(
        "--tree_wood_grid",
        action="store_true",
        help="Render a 2x2 probe: no/5 wood inventory crossed with grass/tree in front.",
    )
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

    if args.tree_wood_grid:
        obs_by_key = {
            "empty_grass": _make_grass_tree_wood_scene(tree_in_front=False, wood_count=0),
            "empty_tree": _make_grass_tree_wood_scene(tree_in_front=True, wood_count=0),
            "wood_grass": _make_grass_tree_wood_scene(tree_in_front=False, wood_count=5),
            "wood_tree": _make_grass_tree_wood_scene(tree_in_front=True, wood_count=5),
        }
        values = _evaluate_values(network, train_state.params, obs_by_key)

        csv_path = os.path.join(args.out_dir, "tree_wood_value_probe.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cell", "wood_count", "tree_in_front", "value"])
            writer.writerow(["empty_grass", 0, False, values["empty_grass"]])
            writer.writerow(["empty_tree", 0, True, values["empty_tree"]])
            writer.writerow(["wood_grass", 5, False, values["wood_grass"]])
            writer.writerow(["wood_tree", 5, True, values["wood_tree"]])
            writer.writerow([])
            writer.writerow(["tree_effect_no_wood", values["empty_tree"] - values["empty_grass"]])
            writer.writerow(["tree_effect_with_wood", values["wood_tree"] - values["wood_grass"]])
            writer.writerow(["wood_effect_no_tree", values["wood_grass"] - values["empty_grass"]])
            writer.writerow(["wood_effect_with_tree", values["wood_tree"] - values["empty_tree"]])
            writer.writerow([
                "interaction",
                (values["wood_tree"] - values["wood_grass"])
                - (values["empty_tree"] - values["empty_grass"]),
            ])
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
                key.replace("_", " + "),
                values[key],
                args.block_size,
                args.display_scale,
                textures,
                panel_textures,
            )
            for key in obs_by_key
        }
        # Reuse the same compositor by mapping the semantic keys onto its expected 2x2 names.
        remapped_cells = {
            "empty_grass": cells["empty_grass"],
            "empty_stone": cells["empty_tree"],
            "pickaxe_grass": cells["wood_grass"],
            "pickaxe_stone": cells["wood_tree"],
        }
        remapped_values = {
            "empty_grass": values["empty_grass"],
            "empty_stone": values["empty_tree"],
            "pickaxe_grass": values["wood_grass"],
            "pickaxe_stone": values["wood_tree"],
        }
        fig_path = os.path.join(args.out_dir, "tree_wood_value_probe_grid.png")
        _compose_grid(
            remapped_cells,
            remapped_values,
            fig_path,
            checkpoint_step,
            "Tree in front",
            bottom_row_label="5 wood",
            stone_effect_label="Tree effect",
            inventory_effect_label="Wood effect",
        )
        print(f"Saved CSV: {csv_path}", flush=True)
        print(f"Saved figure: {fig_path}", flush=True)
        for key in ("empty_grass", "empty_tree", "wood_grass", "wood_tree"):
            print(f"{key}: V={values[key]:.4f}", flush=True)
        print(
            "interaction="
            f"{(values['wood_tree'] - values['wood_grass']) - (values['empty_tree'] - values['empty_grass']):.4f}",
            flush=True,
        )
        return

    obs_by_key = {
        "empty_grass": _make_grass_scene(
            stone_in_front=False,
            wood_pickaxe=False,
            stone_top_right_quadrant=args.stone_top_right_quadrant,
        ),
        "empty_stone": _make_grass_scene(
            stone_in_front=True,
            wood_pickaxe=False,
            stone_top_right_quadrant=args.stone_top_right_quadrant,
        ),
        "pickaxe_grass": _make_grass_scene(
            stone_in_front=False,
            wood_pickaxe=True,
            stone_top_right_quadrant=args.stone_top_right_quadrant,
        ),
        "pickaxe_stone": _make_grass_scene(
            stone_in_front=True,
            wood_pickaxe=True,
            stone_top_right_quadrant=args.stone_top_right_quadrant,
        ),
    }
    values = _evaluate_values(network, train_state.params, obs_by_key)

    csv_path = os.path.join(args.out_dir, "inventory_environment_value_probe.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "wood_pickaxe", "stone_in_front", "value"])
        writer.writerow(["empty_grass", False, False, values["empty_grass"]])
        writer.writerow(["empty_stone", False, True, values["empty_stone"]])
        writer.writerow(["pickaxe_grass", True, False, values["pickaxe_grass"]])
        writer.writerow(["pickaxe_stone", True, True, values["pickaxe_stone"]])
        writer.writerow([])
        writer.writerow(["stone_effect_without_pickaxe", values["empty_stone"] - values["empty_grass"]])
        writer.writerow(["stone_effect_with_pickaxe", values["pickaxe_stone"] - values["pickaxe_grass"]])
        writer.writerow(["pickaxe_effect_without_stone", values["pickaxe_grass"] - values["empty_grass"]])
        writer.writerow(["pickaxe_effect_with_stone", values["pickaxe_stone"] - values["empty_stone"]])
        writer.writerow([
            "interaction",
            (values["pickaxe_stone"] - values["pickaxe_grass"])
            - (values["empty_stone"] - values["empty_grass"]),
        ])
        writer.writerow(["checkpoint_step", checkpoint_step])
        writer.writerow(["stone_top_right_quadrant", args.stone_top_right_quadrant])

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    cells = {
        key: _make_cell(
            obs_by_key[key],
            key.replace("_", " + "),
            values[key],
            args.block_size,
            args.display_scale,
            textures,
            panel_textures,
        )
        for key in obs_by_key
    }
    fig_path = os.path.join(args.out_dir, "inventory_environment_value_probe_grid.png")
    _compose_grid(
        cells,
        values,
        fig_path,
        checkpoint_step,
        "Stone top-right" if args.stone_top_right_quadrant else "Stone in front",
    )

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved figure: {fig_path}", flush=True)
    for key in ("empty_grass", "empty_stone", "pickaxe_grass", "pickaxe_stone"):
        print(f"{key}: V={values[key]:.4f}", flush=True)
    print(
        "interaction="
        f"{(values['pickaxe_stone'] - values['pickaxe_grass']) - (values['empty_stone'] - values['empty_grass']):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

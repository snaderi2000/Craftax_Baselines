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

from concept_mapping.inspect_concepts import load_transition_subset
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


def _decode_blocks(obs: np.ndarray) -> np.ndarray:
    map_view = obs[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    return np.argmax(map_view[..., : len(BlockType)], axis=-1)


def _set_block(map_view: np.ndarray, row: int, col: int, block: BlockType) -> None:
    map_view[row, col, : len(BlockType)] = 0.0
    map_view[row, col, block.value] = 1.0


def _replace_block(obs: np.ndarray, source: BlockType, replacement: BlockType) -> np.ndarray:
    out = obs.copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    blocks = np.argmax(map_view[..., : len(BlockType)], axis=-1)
    for row, col in np.argwhere(blocks == source.value):
        _set_block(map_view, int(row), int(col), replacement)
    return out


def _replace_lava(obs: np.ndarray, replacement: BlockType) -> np.ndarray:
    return _replace_block(obs, BlockType.LAVA, replacement)


def _set_inventory_profile(
    obs: np.ndarray,
    profile: str,
    drink_level: float,
    health_level: float,
) -> np.ndarray:
    out = obs.copy()
    stats = out[MAP_DIM:]
    if profile == "low_items":
        inv = np.zeros(12, dtype=np.float32)
    elif profile == "high_items":
        # Classic inventory order:
        # wood, stone, coal, iron, diamond, sapling, wood/stone/iron pickaxe, wood/stone/iron sword.
        inv = np.array([9, 9, 6, 4, 1, 6, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown profile {profile}")
    stats[:12] = inv / 10.0
    stats[12:16] = np.array([health_level, 9, drink_level, 9], dtype=np.float32) / 10.0
    stats[12:16] = np.clip(stats[12:16], 0.0, 1.0)
    return out


def _set_inventory_counts(
    obs: np.ndarray,
    inv: np.ndarray,
    drink_level: float,
    health_level: float,
) -> np.ndarray:
    out = obs.copy()
    stats = out[MAP_DIM:]
    stats[:12] = np.asarray(inv, dtype=np.float32) / 10.0
    stats[12:16] = np.array([health_level, 9, drink_level, 9], dtype=np.float32) / 10.0
    stats[12:16] = np.clip(stats[12:16], 0.0, 1.0)
    return out


def _inventory_progress_profiles() -> list[tuple[str, np.ndarray]]:
    return [
        ("empty", np.zeros(12, dtype=np.float32)),
        ("wood", np.array([9, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
        ("stone", np.array([9, 9, 2, 0, 0, 3, 1, 0, 0, 0, 0, 0], dtype=np.float32)),
        ("tools", np.array([9, 9, 5, 3, 0, 5, 1, 1, 0, 1, 1, 0], dtype=np.float32)),
        ("iron", np.array([9, 9, 6, 9, 0, 6, 1, 1, 1, 1, 1, 1], dtype=np.float32)),
        ("diamond", np.array([9, 9, 6, 9, 3, 6, 1, 1, 1, 1, 1, 1], dtype=np.float32)),
    ]


def _add_reachable_water(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    # Put water near the bottom of the local view but not under the player.
    _set_block(map_view, OBS_DIM[0] - 1, OBS_DIM[1] // 2, BlockType.WATER)
    return out


def _add_bottom_right_water(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    _set_block(map_view, OBS_DIM[0] - 1, OBS_DIM[1] - 2, BlockType.WATER)
    _set_block(map_view, OBS_DIM[0] - 1, OBS_DIM[1] - 1, BlockType.WATER)
    return out


def _add_bottom_row_water(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    for col in range(OBS_DIM[1]):
        _set_block(map_view, OBS_DIM[0] - 1, col, BlockType.WATER)
    return out


def _add_lava_wall_above(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    center_r = OBS_DIM[0] // 2
    for row in range(max(0, center_r - 3), center_r):
        for col in range(OBS_DIM[1]):
            _set_block(map_view, row, col, BlockType.LAVA)
    return out


def _add_lava_patch_above(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    center_r, center_c = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    for col in range(center_c - 1, center_c + 2):
        _set_block(map_view, center_r - 1, col, BlockType.LAVA)
    return out


def _find_lava_base_obs(data: dict[str, np.ndarray], lava_radius: int) -> tuple[np.ndarray, int, int]:
    center = np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
    best = None
    for idx, obs in enumerate(data["obs"]):
        lava_cells = np.argwhere(_decode_blocks(obs) == BlockType.LAVA.value)
        if lava_cells.size == 0:
            continue
        dists = np.abs(lava_cells - center).sum(axis=1)
        min_dist = int(dists.min())
        if min_dist <= lava_radius:
            return obs.astype(np.float32), idx, min_dist
        if best is None or min_dist < best[2]:
            best = (obs.astype(np.float32), idx, min_dist)
    if best is None:
        raise RuntimeError("Could not find any observation with visible lava.")
    print(
        f"No lava within radius {lava_radius}; using closest visible lava at manhattan distance {best[2]}.",
        flush=True,
    )
    return best


def _make_grass_base_obs(template_obs: np.ndarray, lava: bool) -> np.ndarray:
    out = template_obs.astype(np.float32).copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    map_view[...] = 0.0
    map_view[..., BlockType.GRASS.value] = 1.0
    if lava:
        center_r, center_c = OBS_DIM[0] // 2, OBS_DIM[1] // 2
        _set_block(map_view, center_r - 1, center_c - 1, BlockType.LAVA)
        _set_block(map_view, center_r - 1, center_c + 1, BlockType.LAVA)
    stats = out[MAP_DIM:]
    stats[:] = 0.0
    stats[16 + 2] = 1.0  # facing up
    stats[20] = 1.0
    return out


def _evaluate_values(network, params, obs_list: list[np.ndarray]) -> np.ndarray:
    obs_batch = jnp.asarray(np.stack(obs_list, axis=0), dtype=jnp.float32)
    _pi, values = network.apply(params, obs_batch)
    return np.asarray(values).reshape(-1)


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(0, 0, 0)) -> None:
    draw.text(xy, text, fill=fill)


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
    header_h = 44
    out = Image.new("RGB", (view.width, header_h + view.height), "white")
    draw = ImageDraw.Draw(out)
    _draw_text(draw, (8, 6), title, fill=(20, 40, 90))
    _draw_text(draw, (8, 24), f"V(s) = {value:.4f}")
    out.paste(view, (0, header_h))
    return out


def _compose_grid(
    cells: dict[str, Image.Image],
    values: dict[str, float],
    out_path: str,
    base_note: str,
) -> None:
    row_label_w = 120
    col_header_h = 44
    foot_h = 112
    gap = 14
    cell_w = max(cells["B"].width, cells["A"].width, cells["D"].width, cells["C"].width)
    cell_h = max(cells["B"].height, cells["A"].height, cells["D"].height, cells["C"].height)
    width = row_label_w + 2 * cell_w + gap
    height = col_header_h + 2 * cell_h + gap + foot_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    _draw_text(draw, (row_label_w + 8, 12), "Grass nearby")
    _draw_text(draw, (row_label_w + cell_w + gap + 8, 12), "Lava nearby")
    _draw_text(draw, (12, col_header_h + cell_h // 2 - 8), "Low items")
    _draw_text(draw, (12, col_header_h + cell_h + gap + cell_h // 2 - 8), "High items")

    positions = {
        "B": (row_label_w, col_header_h),
        "A": (row_label_w + cell_w + gap, col_header_h),
        "D": (row_label_w, col_header_h + cell_h + gap),
        "C": (row_label_w + cell_w + gap, col_header_h + cell_h + gap),
    }
    for key, pos in positions.items():
        sheet.paste(cells[key], pos)

    low_penalty = values["B"] - values["A"]
    high_penalty = values["D"] - values["C"]
    y = col_header_h + 2 * cell_h + gap + 12
    note_lines = base_note.split("\n")
    _draw_text(draw, (8, y), note_lines[0])
    if len(note_lines) > 1:
        _draw_text(draw, (8, y + 16), note_lines[1])
    _draw_text(draw, (8, y + 38), f"Low-items lava penalty:  V_B - V_A = {low_penalty:.4f}")
    _draw_text(draw, (8, y + 58), f"High-items lava penalty: V_D - V_C = {high_penalty:.4f}")
    _draw_text(draw, (8, y + 78), "Interpretation: larger penalty means lava lowers V more.")
    sheet.save(out_path)


def _compose_reachable_water_grid(
    cells: dict[str, Image.Image],
    values: dict[str, float],
    out_path: str,
    base_note: str,
) -> None:
    row_label_w = 120
    col_header_h = 44
    foot_h = 136
    gap = 14
    keys = ("B0", "A0", "B1", "A1", "D0", "C0", "D1", "C1")
    cell_w = max(cells[k].width for k in keys)
    cell_h = max(cells[k].height for k in keys)
    width = row_label_w + 4 * cell_w + 3 * gap
    height = col_header_h + 2 * cell_h + gap + foot_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    headers = ["Grass, no water", "Lava, no water", "Grass, water", "Lava, water"]
    for i, header in enumerate(headers):
        _draw_text(draw, (row_label_w + i * (cell_w + gap) + 8, 12), header)
    _draw_text(draw, (12, col_header_h + cell_h // 2 - 8), "Low items")
    _draw_text(draw, (12, col_header_h + cell_h + gap + cell_h // 2 - 8), "High items")

    layout = {
        "B0": (0, 0),
        "A0": (1, 0),
        "B1": (2, 0),
        "A1": (3, 0),
        "D0": (0, 1),
        "C0": (1, 1),
        "D1": (2, 1),
        "C1": (3, 1),
    }
    for key, (col, row) in layout.items():
        x = row_label_w + col * (cell_w + gap)
        y = col_header_h + row * (cell_h + gap)
        sheet.paste(cells[key], (x, y))

    penalties = {
        "Low, no water": values["B0"] - values["A0"],
        "High, no water": values["D0"] - values["C0"],
        "Low, reachable water": values["B1"] - values["A1"],
        "High, reachable water": values["D1"] - values["C1"],
    }
    y = col_header_h + 2 * cell_h + gap + 12
    note_lines = base_note.split("\n")
    _draw_text(draw, (8, y), note_lines[0])
    if len(note_lines) > 1:
        _draw_text(draw, (8, y + 16), note_lines[1])
    _draw_text(draw, (8, y + 40), f"Low no-water penalty:        {penalties['Low, no water']:.4f}")
    _draw_text(draw, (8, y + 58), f"High no-water penalty:       {penalties['High, no water']:.4f}")
    _draw_text(draw, (8, y + 76), f"Low reachable-water penalty: {penalties['Low, reachable water']:.4f}")
    _draw_text(draw, (8, y + 94), f"High reachable-water penalty:{penalties['High, reachable water']:.4f}")
    _draw_text(draw, (8, y + 114), "Penalty = V(grass) - V(lava); larger means lava lowers V more.")
    sheet.save(out_path)


def _compose_low_inventory_recovery_grid(
    cells: dict[str, Image.Image],
    values: dict[str, float],
    out_path: str,
    base_note: str,
    recovery_label: str,
) -> None:
    row_label_w = 128
    col_header_h = 44
    foot_h = 104
    gap = 14
    keys = ("G0", "L0", "G1", "L1")
    cell_w = max(cells[k].width for k in keys)
    cell_h = max(cells[k].height for k in keys)
    width = row_label_w + 2 * cell_w + gap
    height = col_header_h + 2 * cell_h + gap + foot_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    _draw_text(draw, (row_label_w + 8, 12), "All grass")
    _draw_text(draw, (row_label_w + cell_w + gap + 8, 12), "Lava wall")
    _draw_text(draw, (12, col_header_h + cell_h // 2 - 8), "No water")
    _draw_text(draw, (12, col_header_h + cell_h + gap + cell_h // 2 - 8), recovery_label)

    layout = {"G0": (0, 0), "L0": (1, 0), "G1": (0, 1), "L1": (1, 1)}
    for key, (col, row) in layout.items():
        x = row_label_w + col * (cell_w + gap)
        y = col_header_h + row * (cell_h + gap)
        sheet.paste(cells[key], (x, y))

    no_water_penalty = values["G0"] - values["L0"]
    water_penalty = values["G1"] - values["L1"]
    y = col_header_h + 2 * cell_h + gap + 12
    note_lines = base_note.split("\n")
    _draw_text(draw, (8, y), note_lines[0])
    if len(note_lines) > 1:
        _draw_text(draw, (8, y + 16), note_lines[1])
    _draw_text(draw, (8, y + 40), f"No-water lava-wall penalty:       {no_water_penalty:.4f}")
    _draw_text(draw, (8, y + 60), f"Reachable-water lava-wall penalty:{water_penalty:.4f}")
    _draw_text(draw, (8, y + 80), "Penalty = V(grass) - V(lava wall); larger means lava lowers V more.")
    sheet.save(out_path)


def _compose_inventory_progress_grid(
    cells: dict[str, Image.Image],
    rows: list[str],
    values: dict[str, float],
    out_path: str,
    base_note: str,
) -> None:
    row_label_w = 108
    col_header_h = 44
    foot_h = 76
    gap = 12
    cell_w = max(cells[f"{row}_grass"].width for row in rows)
    cell_w = max(cell_w, max(cells[f"{row}_lava"].width for row in rows))
    cell_h = max(cells[f"{row}_grass"].height for row in rows)
    cell_h = max(cell_h, max(cells[f"{row}_lava"].height for row in rows))
    width = row_label_w + 2 * cell_w + gap
    height = col_header_h + len(rows) * cell_h + (len(rows) - 1) * gap + foot_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    _draw_text(draw, (row_label_w + 8, 12), "All grass")
    _draw_text(draw, (row_label_w + cell_w + gap + 8, 12), "Lava patch")
    for i, row in enumerate(rows):
        y = col_header_h + i * (cell_h + gap)
        _draw_text(draw, (10, y + cell_h // 2 - 8), row)
        sheet.paste(cells[f"{row}_grass"], (row_label_w, y))
        sheet.paste(cells[f"{row}_lava"], (row_label_w + cell_w + gap, y))

    y = col_header_h + len(rows) * cell_h + (len(rows) - 1) * gap + 10
    _draw_text(draw, (8, y), base_note)
    _draw_text(draw, (8, y + 20), "Penalty = V(grass) - V(lava); negative means lava increases predicted value.")
    penalties = [values[f"{row}_grass"] - values[f"{row}_lava"] for row in rows]
    _draw_text(
        draw,
        (8, y + 40),
        f"Penalty range: min={min(penalties):.4f}, max={max(penalties):.4f}",
    )
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO value counterfactuals for lava versus grass in symbolic Craftax."
    )
    parser.add_argument("--data", default="concept_mapping/data/ppo_10b_symbolic")
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/lava_counterfactual_10b")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--max_shards", type=int, default=8)
    parser.add_argument("--max_transitions", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lava_radius", type=int, default=3)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument(
        "--drink_level",
        type=float,
        default=9.0,
        help="Set the symbolic drink/thirst intrinsic for all four counterfactuals on the 0-10 scale.",
    )
    parser.add_argument(
        "--health_level",
        type=float,
        default=9.0,
        help="Set the symbolic health intrinsic for all counterfactuals on the 0-10 scale.",
    )
    parser.add_argument(
        "--remove_water",
        action="store_true",
        help="Replace visible WATER tiles with grass before evaluating/rendering.",
    )
    parser.add_argument(
        "--reachable_water_grid",
        action="store_true",
        help="Render a 2x4 grid: no-water grass/lava and reachable-water grass/lava.",
    )
    parser.add_argument(
        "--synthetic_grass_lava",
        action="store_true",
        help="Use an all-grass empty scene and add two lava tiles for the lava condition.",
    )
    parser.add_argument(
        "--low_inventory_lava_wall_grid",
        action="store_true",
        help="Render a low-inventory 2x2 recovery grid: all grass vs three lava rows, no water vs bottom-right water.",
    )
    parser.add_argument(
        "--inventory_progress_grid",
        action="store_true",
        help="Render a value-only inventory progression grid: all grass vs lava patch.",
    )
    parser.add_argument(
        "--lava_patch_above",
        action="store_true",
        help="For --low_inventory_lava_wall_grid, use three lava cells directly above the agent instead of three full lava rows.",
    )
    parser.add_argument(
        "--bottom_row_water",
        action="store_true",
        help="For --low_inventory_lava_wall_grid, make the recovery condition use a full bottom row of water.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_transition_subset(
        args.data,
        max_transitions=args.max_transitions,
        seed=args.seed,
        max_shards=args.max_shards,
    )
    base_obs, base_idx, lava_dist = _find_lava_base_obs(data, args.lava_radius)

    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )

    base_obs = (
        _replace_block(base_obs, BlockType.WATER, BlockType.GRASS)
        if args.remove_water
        else base_obs
    )

    if args.synthetic_grass_lava:
        grass_base = _make_grass_base_obs(base_obs, lava=False)
        lava_base = _make_grass_base_obs(base_obs, lava=True)
        low_grass = _set_inventory_profile(
            grass_base, "low_items", args.drink_level, args.health_level
        )
        low_lava = _set_inventory_profile(
            lava_base, "low_items", args.drink_level, args.health_level
        )
        high_grass = _set_inventory_profile(
            grass_base, "high_items", args.drink_level, args.health_level
        )
        high_lava = _set_inventory_profile(
            lava_base, "high_items", args.drink_level, args.health_level
        )
    elif args.inventory_progress_grid:
        grass_base = _make_grass_base_obs(base_obs, lava=False)
        lava_base = _add_lava_patch_above(grass_base)
        low_grass = grass_base
        low_lava = lava_base
        high_grass = grass_base
        high_lava = lava_base
    elif args.low_inventory_lava_wall_grid:
        grass_base = _make_grass_base_obs(base_obs, lava=False)
        lava_base = (
            _add_lava_patch_above(grass_base)
            if args.lava_patch_above
            else _add_lava_wall_above(grass_base)
        )
        low_grass = _set_inventory_profile(
            grass_base, "low_items", args.drink_level, args.health_level
        )
        low_lava = _set_inventory_profile(
            lava_base, "low_items", args.drink_level, args.health_level
        )
        high_grass = low_grass
        high_lava = low_lava
    else:
        low_lava = _set_inventory_profile(base_obs, "low_items", args.drink_level, args.health_level)
        low_grass = _replace_lava(low_lava, BlockType.GRASS)
        high_lava = _set_inventory_profile(base_obs, "high_items", args.drink_level, args.health_level)
        high_grass = _replace_lava(high_lava, BlockType.GRASS)

    if args.inventory_progress_grid:
        profiles = _inventory_progress_profiles()
        obs_by_key = {}
        rows = []
        for name, inv in profiles:
            rows.append(name)
            obs_by_key[f"{name}_grass"] = _set_inventory_counts(
                low_grass, inv, args.drink_level, args.health_level
            )
            obs_by_key[f"{name}_lava"] = _set_inventory_counts(
                low_lava, inv, args.drink_level, args.health_level
            )
        order = [key for row in rows for key in (f"{row}_grass", f"{row}_lava")]
        values_arr = _evaluate_values(network, train_state.params, [obs_by_key[k] for k in order])
        values = {k: float(v) for k, v in zip(order, values_arr)}

        csv_path = os.path.join(args.out_dir, "lava_inventory_progress.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "grass_value", "lava_value", "lava_penalty", "inventory_order", "inventory_counts"])
            for name, inv in profiles:
                grass_v = values[f"{name}_grass"]
                lava_v = values[f"{name}_lava"]
                writer.writerow([
                    name,
                    grass_v,
                    lava_v,
                    grass_v - lava_v,
                    ",".join(INV_NAMES),
                    ",".join(str(int(x)) for x in inv),
                ])
            writer.writerow([])
            writer.writerow(["checkpoint_step", step])
            writer.writerow(["health_level", args.health_level])
            writer.writerow(["drink_level", args.drink_level])
            writer.writerow(["scene", "synthetic_all_grass_vs_three_lava_cells_above"])

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
            for key in order
        }
        fig_path = os.path.join(args.out_dir, "lava_inventory_progress_grid.png")
        note = (
            f"10B PPO checkpoint={step}; synthetic grass scene; lava=three cells above agent; "
            f"health={args.health_level:g}; drink={args.drink_level:g}"
        )
        _compose_inventory_progress_grid(cells, rows, values, fig_path, note)
        print(f"Saved CSV: {csv_path}", flush=True)
        print(f"Saved figure: {fig_path}", flush=True)
        for name in rows:
            grass_v = values[f"{name}_grass"]
            lava_v = values[f"{name}_lava"]
            print(
                f"{name:>7s}: grass={grass_v:.4f} lava={lava_v:.4f} penalty={grass_v - lava_v:.4f}",
                flush=True,
            )
        return

    if args.low_inventory_lava_wall_grid:
        obs_by_key = {
            "G0": low_grass,
            "L0": low_lava,
            "G1": (_add_bottom_row_water if args.bottom_row_water else _add_bottom_right_water)(low_grass),
            "L1": (_add_bottom_row_water if args.bottom_row_water else _add_bottom_right_water)(low_lava),
        }
        order = ("G0", "L0", "G1", "L1")
        values_arr = _evaluate_values(network, train_state.params, [obs_by_key[k] for k in order])
        values = {k: float(v) for k, v in zip(order, values_arr)}

        csv_path = os.path.join(args.out_dir, "lava_wall_low_inventory_recovery.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cell", "scene", "recovery_water", "value"])
            writer.writerow(["G0", "all_grass", False, values["G0"]])
            writer.writerow(["L0", "three_lava_rows_above", False, values["L0"]])
            writer.writerow(["G1", "all_grass", True, values["G1"]])
            writer.writerow(["L1", "three_lava_rows_above", True, values["L1"]])
            writer.writerow([])
            writer.writerow(["no_water_lava_wall_penalty", values["G0"] - values["L0"]])
            writer.writerow(["reachable_water_lava_wall_penalty", values["G1"] - values["L1"]])
            writer.writerow(["checkpoint_step", step])
            writer.writerow(["drink_level", args.drink_level])
            writer.writerow(["health_level", args.health_level])
            writer.writerow(["inventory_profile", "low_items"])
            writer.writerow(["lava_patch_above", args.lava_patch_above])
            writer.writerow(["bottom_row_water", args.bottom_row_water])

        textures = load_all_textures(args.block_size)
        panel_textures = (
            textures
            if args.display_scale == 1
            else load_all_textures(args.block_size * args.display_scale)
        )
        cells = {
            "G0": _make_cell(obs_by_key["G0"], "G0: grass, no water", values["G0"], args.block_size, args.display_scale, textures, panel_textures),
            "L0": _make_cell(obs_by_key["L0"], "L0: lava wall, no water", values["L0"], args.block_size, args.display_scale, textures, panel_textures),
            "G1": _make_cell(obs_by_key["G1"], "G1: grass + water", values["G1"], args.block_size, args.display_scale, textures, panel_textures),
            "L1": _make_cell(obs_by_key["L1"], "L1: lava wall + water", values["L1"], args.block_size, args.display_scale, textures, panel_textures),
        }
        fig_path = os.path.join(args.out_dir, "lava_wall_low_inventory_recovery_grid.png")
        note = (
            f"10B PPO checkpoint={step}; low inventory; synthetic grass scene\n"
            f"health={args.health_level:g}; drink={args.drink_level:g}; "
            f"lava={'three cells above agent' if args.lava_patch_above else 'three rows above agent'}; "
            f"water={'bottom row' if args.bottom_row_water else 'bottom-right'}"
        )
        _compose_low_inventory_recovery_grid(
            cells,
            values,
            fig_path,
            note,
            "Water bottom row" if args.bottom_row_water else "Water bottom-right",
        )
        print(f"Saved CSV: {csv_path}", flush=True)
        print(f"Saved figure: {fig_path}", flush=True)
        print(f"No-water lava-wall penalty        = {values['G0'] - values['L0']:.4f}", flush=True)
        print(f"Reachable-water lava-wall penalty = {values['G1'] - values['L1']:.4f}", flush=True)
        return

    if args.reachable_water_grid:
        obs_by_key = {
            "A0": low_lava,
            "B0": low_grass,
            "C0": high_lava,
            "D0": high_grass,
            "A1": _add_reachable_water(low_lava),
            "B1": _add_reachable_water(low_grass),
            "C1": _add_reachable_water(high_lava),
            "D1": _add_reachable_water(high_grass),
        }
        order = ("A0", "B0", "C0", "D0", "A1", "B1", "C1", "D1")
        values_arr = _evaluate_values(network, train_state.params, [obs_by_key[k] for k in order])
        values = {k: float(v) for k, v in zip(order, values_arr)}

        csv_path = os.path.join(args.out_dir, "lava_value_counterfactual_2x4.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cell", "inventory_profile", "nearby_tile", "reachable_water", "value"])
            writer.writerow(["A0", "low_items", "lava", False, values["A0"]])
            writer.writerow(["B0", "low_items", "grass", False, values["B0"]])
            writer.writerow(["C0", "high_items", "lava", False, values["C0"]])
            writer.writerow(["D0", "high_items", "grass", False, values["D0"]])
            writer.writerow(["A1", "low_items", "lava", True, values["A1"]])
            writer.writerow(["B1", "low_items", "grass", True, values["B1"]])
            writer.writerow(["C1", "high_items", "lava", True, values["C1"]])
            writer.writerow(["D1", "high_items", "grass", True, values["D1"]])
            writer.writerow([])
            writer.writerow(["low_items_no_water_lava_penalty", values["B0"] - values["A0"]])
            writer.writerow(["high_items_no_water_lava_penalty", values["D0"] - values["C0"]])
            writer.writerow(["low_items_reachable_water_lava_penalty", values["B1"] - values["A1"]])
            writer.writerow(["high_items_reachable_water_lava_penalty", values["D1"] - values["C1"]])
            writer.writerow(["base_transition_index", base_idx])
            writer.writerow(["base_lava_manhattan_distance", lava_dist])
            writer.writerow(["checkpoint_step", step])
            writer.writerow(["drink_level", args.drink_level])
            writer.writerow(["health_level", args.health_level])
            writer.writerow(["removed_visible_water", args.remove_water])
            writer.writerow(["synthetic_grass_lava", args.synthetic_grass_lava])
            writer.writerow(["high_inventory_order", ",".join(INV_NAMES)])

        textures = load_all_textures(args.block_size)
        panel_textures = (
            textures
            if args.display_scale == 1
            else load_all_textures(args.block_size * args.display_scale)
        )
        cells = {
            "A0": _make_cell(obs_by_key["A0"], "A0: low + lava", values["A0"], args.block_size, args.display_scale, textures, panel_textures),
            "B0": _make_cell(obs_by_key["B0"], "B0: low + grass", values["B0"], args.block_size, args.display_scale, textures, panel_textures),
            "C0": _make_cell(obs_by_key["C0"], "C0: high + lava", values["C0"], args.block_size, args.display_scale, textures, panel_textures),
            "D0": _make_cell(obs_by_key["D0"], "D0: high + grass", values["D0"], args.block_size, args.display_scale, textures, panel_textures),
            "A1": _make_cell(obs_by_key["A1"], "A1: low + lava + water", values["A1"], args.block_size, args.display_scale, textures, panel_textures),
            "B1": _make_cell(obs_by_key["B1"], "B1: low + grass + water", values["B1"], args.block_size, args.display_scale, textures, panel_textures),
            "C1": _make_cell(obs_by_key["C1"], "C1: high + lava + water", values["C1"], args.block_size, args.display_scale, textures, panel_textures),
            "D1": _make_cell(obs_by_key["D1"], "D1: high + grass + water", values["D1"], args.block_size, args.display_scale, textures, panel_textures),
        }
        fig_path = os.path.join(args.out_dir, "lava_value_counterfactual_2x4_grid.png")
        if args.synthetic_grass_lava:
            note = (
                f"10B PPO checkpoint={step}; synthetic all-grass scene with two lava tiles\n"
                f"health={args.health_level:g}; drink={args.drink_level:g}; original water removed={args.remove_water}"
            )
        else:
            note = (
                f"10B PPO checkpoint={step}; base transition index={base_idx}; visible lava distance={lava_dist}\n"
                f"health={args.health_level:g}; drink={args.drink_level:g}; original water removed={args.remove_water}"
            )
        _compose_reachable_water_grid(cells, values, fig_path, note)
        print(f"Saved CSV: {csv_path}", flush=True)
        print(f"Saved figure: {fig_path}", flush=True)
        print(f"Low no-water penalty         = {values['B0'] - values['A0']:.4f}", flush=True)
        print(f"High no-water penalty        = {values['D0'] - values['C0']:.4f}", flush=True)
        print(f"Low reachable-water penalty  = {values['B1'] - values['A1']:.4f}", flush=True)
        print(f"High reachable-water penalty = {values['D1'] - values['C1']:.4f}", flush=True)
        return

    obs_by_key = {
        "A": low_lava,
        "B": low_grass,
        "C": high_lava,
        "D": high_grass,
    }
    values_arr = _evaluate_values(network, train_state.params, [obs_by_key[k] for k in ("A", "B", "C", "D")])
    values = {k: float(v) for k, v in zip(("A", "B", "C", "D"), values_arr)}

    csv_path = os.path.join(args.out_dir, "lava_value_counterfactual.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "inventory_profile", "nearby_tile", "value"])
        writer.writerow(["A", "low_items", "lava", values["A"]])
        writer.writerow(["B", "low_items", "grass", values["B"]])
        writer.writerow(["C", "high_items", "lava", values["C"]])
        writer.writerow(["D", "high_items", "grass", values["D"]])
        writer.writerow([])
        writer.writerow(["low_items_lava_penalty", values["B"] - values["A"]])
        writer.writerow(["high_items_lava_penalty", values["D"] - values["C"]])
        writer.writerow(["base_transition_index", base_idx])
        writer.writerow(["base_lava_manhattan_distance", lava_dist])
        writer.writerow(["checkpoint_step", step])
        writer.writerow(["drink_level", args.drink_level])
        writer.writerow(["health_level", args.health_level])
        writer.writerow(["removed_visible_water", args.remove_water])
        writer.writerow(["synthetic_grass_lava", args.synthetic_grass_lava])
        writer.writerow(["high_inventory_order", ",".join(INV_NAMES)])

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    cells = {
        "A": _make_cell(low_lava, "A: low items + lava", values["A"], args.block_size, args.display_scale, textures, panel_textures),
        "B": _make_cell(low_grass, "B: low items + grass", values["B"], args.block_size, args.display_scale, textures, panel_textures),
        "C": _make_cell(high_lava, "C: high items + lava", values["C"], args.block_size, args.display_scale, textures, panel_textures),
        "D": _make_cell(high_grass, "D: high items + grass", values["D"], args.block_size, args.display_scale, textures, panel_textures),
    }
    fig_path = os.path.join(args.out_dir, "lava_value_counterfactual_grid.png")
    if args.synthetic_grass_lava:
        note = (
            f"10B PPO checkpoint={step}; synthetic all-grass scene with two lava tiles\n"
            f"health={args.health_level:g}; drink={args.drink_level:g}; visible water removed={args.remove_water}"
        )
    else:
        note = (
            f"10B PPO checkpoint={step}; base transition index={base_idx}; visible lava distance={lava_dist}\n"
            f"health={args.health_level:g}; drink={args.drink_level:g}; visible water removed={args.remove_water}"
        )
    _compose_grid(cells, values, fig_path, note)

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved figure: {fig_path}", flush=True)
    print(f"Low-items lava penalty  V_B - V_A = {values['B'] - values['A']:.4f}", flush=True)
    print(f"High-items lava penalty V_D - V_C = {values['D'] - values['C']:.4f}", flush=True)


if __name__ == "__main__":
    main()

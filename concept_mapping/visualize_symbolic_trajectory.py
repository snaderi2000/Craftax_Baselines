import argparse
import glob
import os
from typing import NamedTuple

import numpy as np
import yaml
from PIL import Image, ImageDraw

from craftax.craftax_classic.constants import (
    Action,
    BlockType,
    OBS_DIM,
    load_all_textures,
)


MAP_CHANNELS = len(BlockType) + 4
MAP_DIM = OBS_DIM[0] * OBS_DIM[1] * MAP_CHANNELS
INV_NAMES = [
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
]
INTRINSIC_NAMES = ["health", "food", "drink", "energy"]
DIR_NAMES = ["left", "right", "up", "down"]
MOB_NAMES = ["zombie", "cow", "skeleton", "arrow"]


class DecodedObs(NamedTuple):
    blocks: np.ndarray
    mobs: np.ndarray
    inventory: np.ndarray
    intrinsics: np.ndarray
    direction: int
    light: float
    sleeping: bool


def decode_obs(obs: np.ndarray) -> DecodedObs:
    if obs.shape[-1] != MAP_DIM + 22:
        raise ValueError(f"Expected obs dim {MAP_DIM + 22}, got {obs.shape[-1]}")
    map_view = obs[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    blocks = np.argmax(map_view[..., : len(BlockType)], axis=-1)
    mobs = map_view[..., len(BlockType) :] > 0.5
    stats = obs[MAP_DIM:]
    inventory = stats[:12] * 10.0
    intrinsics = stats[12:16] * 10.0
    direction = int(np.argmax(stats[16:20]))
    light = float(stats[20])
    sleeping = bool(stats[21] > 0.5)
    return DecodedObs(blocks, mobs, inventory, intrinsics, direction, light, sleeping)


def _overlay(base: np.ndarray, texture: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return base * (1.0 - alpha) + texture * alpha


def _paste_number(panel: np.ndarray, textures: dict, amount: float, col: int, row: int, block_size: int) -> None:
    number = int(np.clip(round(float(amount)), 0, 9))
    if number <= 0:
        return
    number_size = int(block_size * 0.6)
    number_offset = block_size - number_size
    y0 = row * block_size + number_offset - 1
    x0 = col * block_size + number_offset - 1
    y1 = y0 + number_size
    x1 = x0 + number_size
    number_tex = np.asarray(textures["number_textures"][number], dtype=np.float32)
    number_alpha = np.asarray(textures["number_textures_alpha"][number], dtype=np.float32)
    panel[y0:y1, x0:x1] = panel[y0:y1, x0:x1] * (1.0 - number_alpha) + number_tex


def render_inventory_panel(decoded: DecodedObs, textures: dict, block_size: int) -> Image.Image:
    cols = OBS_DIM[1]
    panel = np.zeros((2 * block_size, cols * block_size, 3), dtype=np.float32)
    left = (block_size - int(0.8 * block_size)) // 2 - 1
    right = block_size - int(0.8 * block_size) - left

    smaller_block_textures = np.asarray(textures["smaller_block_textures"], dtype=np.float32)
    empty = np.asarray(textures["smaller_empty_texture"], dtype=np.float32)

    def slot(texture_ref, amount):
        if amount <= 0:
            return empty, amount
        if isinstance(texture_ref, str):
            return np.asarray(textures[texture_ref], dtype=np.float32), amount
        return texture_ref, amount

    first_row = [
        ("health_texture", decoded.intrinsics[0]),
        ("hunger_texture", decoded.intrinsics[1]),
        ("thirst_texture", decoded.intrinsics[2]),
        ("energy_texture", decoded.intrinsics[3]),
        ("sapling_texture", decoded.inventory[5]),
        (smaller_block_textures[BlockType.WOOD.value, :, :, :3], decoded.inventory[0]),
        (smaller_block_textures[BlockType.STONE.value, :, :, :3], decoded.inventory[1]),
        (smaller_block_textures[BlockType.COAL.value, :, :, :3], decoded.inventory[2]),
        (smaller_block_textures[BlockType.IRON.value, :, :, :3], decoded.inventory[3]),
    ]
    second_row = [
        (smaller_block_textures[BlockType.DIAMOND.value, :, :, :3], decoded.inventory[4]),
        ("wood_pickaxe_texture", decoded.inventory[6]),
        ("stone_pickaxe_texture", decoded.inventory[7]),
        ("iron_pickaxe_texture", decoded.inventory[8]),
        ("wood_sword_texture", decoded.inventory[9]),
        ("stone_sword_texture", decoded.inventory[10]),
        ("iron_sword_texture", decoded.inventory[11]),
    ]

    def paste_slot(row: int, col: int, texture_ref, amount: float) -> None:
        icon, amount = slot(texture_ref, amount)
        y0 = row * block_size + left
        y1 = (row + 1) * block_size - right
        x0 = col * block_size + left
        x1 = (col + 1) * block_size - right
        panel[y0:y1, x0:x1] = icon[:, :, :3]
        _paste_number(panel, textures, amount, col, row, block_size)

    for row, slots in enumerate((first_row, second_row)):
        for col, (texture_ref, amount) in enumerate(slots):
            paste_slot(row, col, texture_ref, float(amount))

    return Image.fromarray(np.clip(panel, 0, 255).astype(np.uint8))


def render_local_view(
    obs: np.ndarray,
    block_size: int,
    include_inventory: bool = True,
    display_scale: int = 1,
    textures: dict | None = None,
    panel_textures: dict | None = None,
) -> Image.Image:
    decoded = decode_obs(obs)
    textures = textures if textures is not None else load_all_textures(block_size)
    h, w = OBS_DIM
    canvas = np.zeros((h * block_size, w * block_size, 3), dtype=np.float32)

    block_textures = np.asarray(textures["block_textures"], dtype=np.float32)
    for r in range(h):
        for c in range(w):
            block_id = int(decoded.blocks[r, c])
            tile = block_textures[block_id, :, :, :3]
            canvas[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ] = tile

    mob_texture_keys = [
        ("zombie_texture", "zombie_texture_alpha"),
        ("cow_texture", "cow_texture_alpha"),
        ("skeleton_texture", "skeleton_texture_alpha"),
        ("arrow_texture", "arrow_texture_alpha"),
    ]
    for mob_idx, (tex_key, alpha_key) in enumerate(mob_texture_keys):
        tex = np.asarray(textures[tex_key], dtype=np.float32)
        alpha = np.asarray(textures[alpha_key], dtype=np.float32)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        for r in range(h):
            for c in range(w):
                if decoded.mobs[r, c, mob_idx]:
                    region = canvas[
                        r * block_size : (r + 1) * block_size,
                        c * block_size : (c + 1) * block_size,
                    ]
                    canvas[
                        r * block_size : (r + 1) * block_size,
                        c * block_size : (c + 1) * block_size,
                    ] = _overlay(region, tex, alpha)

    player_idx = 4 if decoded.sleeping else decoded.direction
    player_rgba = np.asarray(textures["player_textures"][player_idx], dtype=np.float32)
    player_tex = player_rgba[:, :, :3]
    player_alpha = np.repeat(player_rgba[:, :, 3:4] / 255.0, repeats=3, axis=2)
    cr, cc = h // 2, w // 2
    region = canvas[
        cr * block_size : (cr + 1) * block_size,
        cc * block_size : (cc + 1) * block_size,
    ]
    canvas[
        cr * block_size : (cr + 1) * block_size,
        cc * block_size : (cc + 1) * block_size,
    ] = _overlay(region, player_tex, player_alpha)

    # Approximate lighting from symbolic state.
    canvas = decoded.light * canvas + (1.0 - decoded.light) * (0.45 * canvas)
    if decoded.sleeping:
        gray = canvas.mean(axis=-1, keepdims=True)
        canvas = 0.6 * gray + np.array([0.0, 0.0, 24.0], dtype=np.float32)

    view = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    if display_scale != 1:
        view = view.resize(
            (view.width * display_scale, view.height * display_scale),
            Image.Resampling.NEAREST,
        )
    if not include_inventory:
        return view

    panel_block_size = block_size * display_scale
    if panel_textures is None:
        panel_textures = textures if display_scale == 1 else load_all_textures(panel_block_size)
    panel = render_inventory_panel(decoded, panel_textures, panel_block_size)
    out = Image.new("RGB", (view.width, view.height + panel.height), "black")
    out.paste(view, (0, 0))
    out.paste(panel, (0, view.height))
    return out


def _action_name(action: int) -> str:
    try:
        return Action(action).name
    except ValueError:
        return str(action)


def _summary_text(obs: np.ndarray, action: int, reward: float, done: bool, t: int, env_id: int) -> list[str]:
    decoded = decode_obs(obs)
    inv_parts = [
        f"{name}:{decoded.inventory[i]:.0f}"
        for i, name in enumerate(INV_NAMES)
        if decoded.inventory[i] >= 0.5
    ]
    mob_counts = decoded.mobs.sum(axis=(0, 1))
    mob_parts = [
        f"{name}:{int(mob_counts[i])}"
        for i, name in enumerate(MOB_NAMES)
        if mob_counts[i] > 0
    ]
    intr = ", ".join(
        f"{name}:{decoded.intrinsics[i]:.0f}" for i, name in enumerate(INTRINSIC_NAMES)
    )
    return [
        f"t={t} env={env_id} action={int(action)}:{_action_name(int(action))}",
        f"reward={reward:.3f} done={bool(done)} dir={DIR_NAMES[decoded.direction]} light={decoded.light:.2f}",
        intr,
        "inv " + (", ".join(inv_parts) if inv_parts else "empty"),
        "mobs " + (", ".join(mob_parts) if mob_parts else "none"),
    ]


def make_labeled_frame(
    obs: np.ndarray,
    action: int,
    reward: float,
    done: bool,
    t: int,
    env_id: int,
    block_size: int,
    display_scale: int = 1,
) -> Image.Image:
    view = render_local_view(obs, block_size, display_scale=display_scale)
    text_lines = _summary_text(obs, action, reward, done, t, env_id)
    pad = 8
    line_h = 15
    panel_h = pad * 2 + line_h * len(text_lines)
    out = Image.new("RGB", (view.width, view.height + panel_h), "white")
    out.paste(view, (0, 0))
    draw = ImageDraw.Draw(out)
    y = view.height + pad
    for line in text_lines:
        draw.text((pad, y), line, fill=(0, 0, 0))
        y += line_h
    return out


def load_shard(data_dir: str, shard: int):
    path = os.path.join(data_dir, f"transitions_{shard:05d}.npz")
    if not os.path.exists(path):
        paths = sorted(glob.glob(os.path.join(data_dir, "transitions_*.npz")))
        if not paths:
            raise FileNotFoundError(f"No transition shards found in {data_dir}")
        path = paths[shard]
    return path, np.load(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="concept_mapping/data/ppo_10b_symbolic")
    parser.add_argument("--out_dir", default="concept_mapping/viz/ppo_10b_symbolic")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--display_scale", type=int, default=1)
    parser.add_argument("--make_gif", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    metadata_path = os.path.join(args.data_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata.yaml in {args.data_dir}")
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    num_envs = int(metadata["num_envs"])
    steps_per_shard = int(metadata["steps_per_shard"])
    if args.env_id >= num_envs:
        raise ValueError(f"env_id {args.env_id} >= num_envs {num_envs}")

    path, data = load_shard(args.data_dir, args.shard)
    obs = data["obs"].reshape(steps_per_shard, num_envs, -1)
    actions = data["action"].reshape(steps_per_shard, num_envs)
    rewards = data["reward"].reshape(steps_per_shard, num_envs)
    dones = data["done"].reshape(steps_per_shard, num_envs)

    end = min(args.start + args.length, steps_per_shard)
    os.makedirs(args.out_dir, exist_ok=True)
    frames = []
    print(f"Rendering {path}, env_id={args.env_id}, steps={args.start}:{end}")
    for t in range(args.start, end):
        frame = make_labeled_frame(
            obs[t, args.env_id],
            int(actions[t, args.env_id]),
            float(rewards[t, args.env_id]),
            bool(dones[t, args.env_id]),
            t,
            args.env_id,
            args.block_size,
            args.display_scale,
        )
        frame_path = os.path.join(args.out_dir, f"shard{args.shard:05d}_env{args.env_id:04d}_t{t:04d}.png")
        frame.save(frame_path)
        frames.append(frame)

    cols = min(4, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * frames[0].width, rows * frames[0].height), "white")
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * frame.width, (i // cols) * frame.height))
    sheet_path = os.path.join(args.out_dir, f"sheet_shard{args.shard:05d}_env{args.env_id:04d}.png")
    sheet.save(sheet_path)
    print(f"Saved sheet: {sheet_path}")

    if args.make_gif:
        gif_path = os.path.join(args.out_dir, f"traj_shard{args.shard:05d}_env{args.env_id:04d}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
        print(f"Saved gif: {gif_path}")


if __name__ == "__main__":
    main()

"""Render 2x2 wood-pickaxe/resource interactions for the fixed ten-state subset."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import BlockType, load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from PIL import Image, ImageDraw, ImageFont

from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state
from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_CHANNELS, MAP_DIM, render_local_view


DEFAULT_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/wood_pickaxe_harvesting_examples"
DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"


def _font(size: int):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def _set_block(obs: np.ndarray, row: int, col: int, block: BlockType) -> np.ndarray:
    result = np.array(obs, copy=True)
    map_view = result[:MAP_DIM].reshape(7, 9, MAP_CHANNELS)
    map_view[row, col, :len(BlockType)] = 0.0
    map_view[row, col, block.value] = 1.0
    return result


def _remove_wood_pickaxe(obs: np.ndarray) -> np.ndarray:
    result = np.array(obs, copy=True)
    result[MAP_DIM + INV_NAMES.index("wood_pickaxe")] = 0.0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples_dir", default=DEFAULT_DIR)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--timestep", type=int, default=1499987968)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    examples_dir = Path(args.examples_dir)
    out_dir = examples_dir / "resource_pickaxe_interaction"
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(examples_dir / "wood_pickaxe_harvesting_5stone_5coal.npz") as data:
        base = np.asarray(data["base_obs"], dtype=np.float32)
        resource = np.asarray(data["source_concept"])
        target_rows = np.asarray(data["target_row"], dtype=int)
        target_cols = np.asarray(data["target_col"], dtype=int)
        episodes = np.asarray(data["episode_id"], dtype=int)
        timesteps = np.asarray(data["timestep"], dtype=int)
    # A: absent/no pick; B: absent/pick; C: present/no pick; D: present/pick.
    d = base
    c = np.stack([_remove_wood_pickaxe(obs) for obs in d])
    b = np.stack([_set_block(obs, row, col, BlockType.PATH) for obs, row, col in zip(d, target_rows, target_cols)])
    a = np.stack([_remove_wood_pickaxe(obs) for obs in b])
    conditions = {"A": a, "B": b, "C": c, "D": d}
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)
    @jax.jit
    def value_apply(params, observations):
        _pi, values = network.apply(params, observations)
        return values
    values = {key: np.asarray(value_apply(train_state.params, jnp.asarray(obs))) for key, obs in conditions.items()}
    interaction = (values["D"] - values["C"]) - (values["B"] - values["A"])
    with (out_dir / "interaction_values.csv").open("w", newline="") as handle:
        fields = ["sample_index", "resource", "episode", "timestep", "A_absent_no_pick", "B_absent_pick", "C_present_no_pick", "D_present_pick", "tool_effect_resource_absent", "tool_effect_resource_present", "interaction", "interaction_positive"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i in range(len(base)):
            writer.writerow({"sample_index": i, "resource": str(resource[i]), "episode": int(episodes[i]), "timestep": int(timesteps[i]),
                             "A_absent_no_pick": float(values["A"][i]), "B_absent_pick": float(values["B"][i]),
                             "C_present_no_pick": float(values["C"][i]), "D_present_pick": float(values["D"][i]),
                             "tool_effect_resource_absent": float(values["B"][i] - values["A"][i]),
                             "tool_effect_resource_present": float(values["D"][i] - values["C"][i]),
                             "interaction": float(interaction[i]), "interaction_positive": bool(interaction[i] > 0)})
    textures = load_all_textures(args.block_size)
    panel_textures = textures if args.display_scale == 1 else load_all_textures(args.block_size * args.display_scale)
    labels = {
        "A": "A: resource absent\nno wood pickaxe", "B": "B: resource absent\nwood pickaxe",
        "C": "C: resource present\nno wood pickaxe", "D": "D: resource present\nwood pickaxe",
    }
    sheets = []
    for i in range(len(base)):
        frames = []
        for key in ("A", "B", "C", "D"):
            view = render_local_view(conditions[key][i], block_size=args.block_size, include_inventory=True,
                                     display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
            footer = Image.new("RGB", (view.width, 72), "white")
            draw = ImageDraw.Draw(footer)
            draw.multiline_text((4, 2), labels[key], fill="black", font=_font(14), spacing=0)
            draw.text((4, 42), f"value: {values[key][i]:.4f}", fill="black", font=_font(15))
            frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
            frame.paste(view, (0, 0)); frame.paste(footer, (0, view.height)); frames.append(frame)
        width, height = frames[0].size
        footer_h = 38
        sheet = Image.new("RGB", (2 * width + 16, 2 * height + 16 + footer_h), "white")
        for k, frame in enumerate(frames):
            sheet.paste(frame, ((k % 2) * (width + 16), (k // 2) * (height + 16)))
        ImageDraw.Draw(sheet).text((4, 2 * height + 20), f"{str(resource[i]).title()} | interaction: {interaction[i]:+.4f}", fill="black", font=_font(16))
        sheet.save(out_dir / f"example_{i:02d}_{str(resource[i])}.png")
        sheets.append(sheet)
    width, height = sheets[0].size
    contact = Image.new("RGB", (2 * width + 48, 5 * height + 6 * 16), "white")
    for i, sheet in enumerate(sheets):
        contact.paste(sheet, (16 + (i % 2) * (width + 16), 16 + (i // 2) * (height + 16)))
    contact.save(out_dir / "interaction_examples_10.png")
    print(f"Saved {len(base)} four-condition examples with checkpoint {checkpoint:,}; positive interactions={(interaction > 0).sum()}/{len(base)}")


if __name__ == "__main__":
    main()

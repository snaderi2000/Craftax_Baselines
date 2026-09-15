"""Create a ten-state illustrative wood-pickaxe harvesting counterfactual set.

Five states are sampled from the frozen stone bank and five from the coal bank.
Each positive state is immediately before its first successful resource-collection
action; its paired negative changes only ``wood_pickaxe`` from one to zero.
"""

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
from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from PIL import Image, ImageDraw, ImageFont

from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state
from concept_mapping.visualize_symbolic_trajectory import render_local_view


DEFAULT_BANK_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks"
DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"


def _font(size: int):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def _load_selected(bank_dir: Path, concept: str, count: int, seed: int):
    with np.load(bank_dir / f"{concept}_bank.npz") as bank:
        indices = np.sort(np.random.default_rng(seed).choice(len(bank["base_obs"]), count, replace=False))
        return indices, {key: np.asarray(bank[key])[indices] for key in bank.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank_dir", default=DEFAULT_BANK_DIR)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--per_resource", type=int, default=5)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()
    bank_dir = Path(args.bank_dir)
    out_dir = bank_dir / "wood_pickaxe_harvesting_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for offset, concept in enumerate(("stone", "coal")):
        indices, payload = _load_selected(bank_dir, concept, args.per_resource, args.seed + offset)
        selected.extend((concept, int(index), {key: payload[key][rank] for key in payload}) for rank, index in enumerate(indices))
    base = np.stack([row[2]["base_obs"] for row in selected]).astype(np.float32)
    counterfactual = np.stack([row[2]["counterfactual_obs"] for row in selected]).astype(np.float32)
    source_concept = np.asarray([row[0] for row in selected])
    source_index = np.asarray([row[1] for row in selected], dtype=np.int32)
    np.savez_compressed(
        out_dir / "wood_pickaxe_harvesting_5stone_5coal.npz",
        base_obs=base, counterfactual_obs=counterfactual, source_concept=source_concept,
        source_index=source_index,
        episode_id=np.asarray([row[2]["episode_id"] for row in selected], dtype=np.int32),
        timestep=np.asarray([row[2]["timestep"] for row in selected], dtype=np.int32),
        source_npz=np.asarray([row[2]["source_npz"] for row in selected]),
        target_row=np.asarray([row[2]["target_row"] for row in selected], dtype=np.int8),
        target_col=np.asarray([row[2]["target_col"] for row in selected], dtype=np.int8),
    )
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)
    @jax.jit
    def values(params, observations):
        _policy, value = network.apply(params, observations)
        return value
    plus_values = np.asarray(values(train_state.params, jnp.asarray(base)))
    minus_values = np.asarray(values(train_state.params, jnp.asarray(counterfactual)))
    with (out_dir / "values.csv").open("w", newline="") as handle:
        fields = ["sample_index", "resource", "source_index", "episode_id", "timestep", "value_collectible", "value_no_wood_pickaxe", "delta_value", "positive"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(selected):
            writer.writerow({"sample_index": i, "resource": row[0], "source_index": row[1],
                             "episode_id": int(row[2]["episode_id"]), "timestep": int(row[2]["timestep"]),
                             "value_collectible": float(plus_values[i]), "value_no_wood_pickaxe": float(minus_values[i]),
                             "delta_value": float(plus_values[i] - minus_values[i]), "positive": bool(plus_values[i] > minus_values[i])})
    textures = load_all_textures(args.block_size)
    panel_textures = textures if args.display_scale == 1 else load_all_textures(args.block_size * args.display_scale)
    pairs = []
    for i, row in enumerate(selected):
        subframes = []
        for observation, title, value in ((base[i], f"{row[0].title()} collectible", plus_values[i]),
                                          (counterfactual[i], "No wood pickaxe", minus_values[i])):
            view = render_local_view(observation, block_size=args.block_size, include_inventory=True,
                                     display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
            footer = Image.new("RGB", (view.width, 80), "white")
            draw = ImageDraw.Draw(footer)
            draw.text((5, 4), title, fill="black", font=_font(16))
            draw.text((5, 31), f"value: {value:.4f}", fill="black", font=_font(16))
            draw.text((5, 55), f"episode {int(row[2]['episode_id'])}, step {int(row[2]['timestep'])}", fill="black", font=_font(13))
            frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
            frame.paste(view, (0, 0)); frame.paste(footer, (0, view.height))
            subframes.append(frame)
        pair = Image.new("RGB", (2 * subframes[0].width + 14, subframes[0].height), "white")
        pair.paste(subframes[0], (0, 0)); pair.paste(subframes[1], (subframes[0].width + 14, 0))
        pairs.append(pair)
    width, height = pairs[0].size
    sheet = Image.new("RGB", (2 * width + 48, 5 * height + 6 * 16), "white")
    for i, pair in enumerate(pairs):
        sheet.paste(pair, (16 + (i % 2) * (width + 16), 16 + (i // 2) * (height + 16)))
    # Ten pairs means five rows and two columns.
    sheet.save(out_dir / "wood_pickaxe_harvesting_5stone_5coal_values.png")
    print(f"Saved 10 state subset and value figure at checkpoint {checkpoint:,} to {out_dir}")


if __name__ == "__main__":
    main()

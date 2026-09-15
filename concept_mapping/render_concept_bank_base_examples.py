"""Render natural positive/base states from a frozen hierarchy concept bank."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from PIL import Image, ImageDraw, ImageFont

from concept_mapping.visualize_symbolic_trajectory import render_local_view
from concept_mapping.visualize_symbolic_trajectory import INV_NAMES, MAP_DIM
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


def _font(size: int):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if os.path.isfile(path) else ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", default="wood_pickaxe")
    parser.add_argument("--bank_dir", default="concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--remove_wood", action="store_true",
                        help="Render each base state beside a counterfactual with only wood set to zero.")
    parser.add_argument("--value_run_path", default=None,
                        help="Optional PPO run path used to print critic values under each panel.")
    parser.add_argument("--value_timestep", type=int, default=None,
                        help="Optional checkpoint; defaults to the final checkpoint in --value_run_path.")
    args = parser.parse_args()
    bank_path = Path(args.bank_dir) / f"{args.concept}_bank.npz"
    out_dir = Path(args.bank_dir) / f"{args.concept}_base_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(bank_path) as bank:
        base_obs = np.asarray(bank["base_obs"], dtype=np.float32)
        episode_ids = np.asarray(bank["episode_id"], dtype=int)
        timesteps = np.asarray(bank["timestep"], dtype=int)
    chosen = np.sort(np.random.default_rng(args.seed).choice(len(base_obs), min(args.n_examples, len(base_obs)), replace=False))
    base_values = no_wood_values = None
    if args.value_run_path:
        if not args.remove_wood:
            raise ValueError("--value_run_path currently requires --remove_wood so both conditions can be evaluated.")
        no_wood_observations = np.array(base_obs[chosen], copy=True)
        no_wood_observations[:, MAP_DIM + INV_NAMES.index("wood")] = 0.0
        config = _load_wandb_config(args.value_run_path)
        config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
        env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
        env_params = env.default_params
        network = _init_network(config, env, env_params)
        train_state, restored_step = _restore_train_state(
            args.value_run_path, config, network, env, env_params, args.value_timestep
        )
        @jax.jit
        def value_apply(parameters, observations):
            _pi, values = network.apply(parameters, observations)
            return values
        base_values = np.asarray(value_apply(train_state.params, jnp.asarray(base_obs[chosen])))
        no_wood_values = np.asarray(value_apply(train_state.params, jnp.asarray(no_wood_observations)))
        print(f"Using critic checkpoint {restored_step:,}", flush=True)
    textures = load_all_textures(args.block_size)
    panel_textures = textures if args.display_scale == 1 else load_all_textures(args.block_size * args.display_scale)
    frames = []
    for rank, index in enumerate(chosen):
        view = render_local_view(base_obs[index], block_size=args.block_size, include_inventory=True,
                                 display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
        footer_h = 78 if base_values is not None else 52
        footer = Image.new("RGB", (view.width, footer_h), "white")
        draw = ImageDraw.Draw(footer)
        draw.text((5, 5), "Base state" if args.remove_wood else f"episode {episode_ids[index]}, step {timesteps[index]}", fill="black", font=_font(17))
        if base_values is not None:
            draw.text((5, 31), f"value: {base_values[rank]:.4f}", fill="black", font=_font(17))
        frame = Image.new("RGB", (view.width, view.height + footer.height), "white")
        frame.paste(view, (0, 0))
        frame.paste(footer, (0, view.height))
        if args.remove_wood:
            counterfactual = np.array(base_obs[index], copy=True)
            counterfactual[MAP_DIM + INV_NAMES.index("wood")] = 0.0
            cf_view = render_local_view(counterfactual, block_size=args.block_size, include_inventory=True,
                                        display_scale=args.display_scale, textures=textures, panel_textures=panel_textures)
            cf_footer = Image.new("RGB", (cf_view.width, footer_h), "white")
            cf_draw = ImageDraw.Draw(cf_footer)
            cf_draw.text((5, 5), "No wood", fill="black", font=_font(17))
            if no_wood_values is not None:
                cf_draw.text((5, 31), f"value: {no_wood_values[rank]:.4f}", fill="black", font=_font(17))
            cf_frame = Image.new("RGB", (cf_view.width, cf_view.height + cf_footer.height), "white")
            cf_frame.paste(cf_view, (0, 0))
            cf_frame.paste(cf_footer, (0, cf_view.height))
            pair = Image.new("RGB", (frame.width + 16 + cf_frame.width, frame.height), "white")
            pair.paste(frame, (0, 0))
            pair.paste(cf_frame, (frame.width + 16, 0))
            frames.append(pair)
        else:
            frames.append(frame)
    columns = 2 if args.remove_wood else 5
    width, height = frames[0].size
    gap, margin = 16, 18
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (2 * margin + columns * width + (columns - 1) * gap,
                                2 * margin + rows * height + (rows - 1) * gap), "white")
    for rank, frame in enumerate(frames):
        sheet.paste(frame, (margin + (rank % columns) * (width + gap), margin + (rank // columns) * (height + gap)))
    output = out_dir / (
        "base_vs_no_wood_values_10.png" if base_values is not None
        else "base_vs_no_wood_examples_10.png" if args.remove_wood else "base_examples_10.png"
    )
    sheet.save(output)
    print(f"Saved {len(chosen)} {args.concept} base states to {output}")


if __name__ == "__main__":
    main()

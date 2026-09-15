"""Render fixed stone/pickaxe counterfactual pairs at peak and final checkpoints."""

from __future__ import annotations

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

from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state
from concept_mapping.visualize_symbolic_trajectory import render_local_view


RUN = "wandb/run-20260830_231509-616wuss8"
BANK = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "wood_pickaxe_functional/wood_pickaxe_functional_bank.npz"
)
OUT = Path(
    "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/"
    "progression_vs_pickaxe_value/direct_pickaxe_effect_stride4"
)
CHECKPOINTS = ((18_939_904, "peak_19m"), (1_499_987_968, "final_1p5b"))


def _font(size: int):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if os.path.isfile(path) else ImageFont.load_default()


def _panel(obs, value, title, textures, panel_textures):
    view = render_local_view(obs, block_size=7, include_inventory=True, display_scale=4,
                             textures=textures, panel_textures=panel_textures)
    footer_h = 74
    panel = Image.new("RGB", (view.width, view.height + footer_h), "white"); panel.paste(view)
    draw = ImageDraw.Draw(panel)
    draw.text((6, view.height + 5), title, fill="black", font=_font(16))
    draw.text((6, view.height + 35), f"value: {value:.4f}", fill="black", font=_font(18))
    return panel


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(BANK) as data:
        stone_indices = np.flatnonzero(data["resource"].astype(str) == "stone")
        chosen = np.sort(np.random.default_rng(50).choice(stone_indices, 4, replace=False))
        with_pickaxe = np.asarray(data["D"])[chosen]
        without_pickaxe = np.asarray(data["C"])[chosen]
        episode_ids = np.asarray(data["episode_id"])[chosen]
        timesteps = np.asarray(data["timestep"])[chosen]

    config = _load_wandb_config(RUN); config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False); env_params = env.default_params
    network = _init_network(config, env, env_params)
    textures, panel_textures = load_all_textures(7), load_all_textures(28)

    @jax.jit
    def values(params, observations):
        _policy, value = network.apply(params, observations)
        return value

    for requested, name in CHECKPOINTS:
        train_state, restored = _restore_train_state(RUN, config, network, env, env_params, requested)
        packed = np.concatenate([with_pickaxe, without_pickaxe])
        predicted = np.asarray(values(train_state.params, jnp.asarray(packed)))
        plus_values, minus_values = np.split(predicted, 2)
        example_dir = OUT / "counterfactual_examples" / name
        example_dir.mkdir(parents=True, exist_ok=True)
        pairs = []
        for rank in range(len(chosen)):
            plus = _panel(with_pickaxe[rank], plus_values[rank], "Stone + wood pickaxe", textures, panel_textures)
            minus = _panel(without_pickaxe[rank], minus_values[rank], "Stone, no wood pickaxe", textures, panel_textures)
            gap = 18
            pair = Image.new("RGB", (plus.width + gap + minus.width, plus.height + 32), "white")
            pair.paste(plus, (0, 0)); pair.paste(minus, (plus.width + gap, 0))
            ImageDraw.Draw(pair).text(
                (5, plus.height + 4),
                f"episode {episode_ids[rank]}, step {timesteps[rank]}; delta value: {plus_values[rank]-minus_values[rank]:.4f}",
                fill="black", font=_font(13),
            )
            pair.save(example_dir / f"example_{rank:02d}.png"); pairs.append(pair)
        columns, gap, margin = 2, 22, 20
        width = 2 * margin + columns * pairs[0].width + gap
        height = 72 + 2 * margin + 2 * pairs[0].height + gap
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, 12), f"Wood-pickaxe counterfactuals at {restored:,} training steps",
                  fill="black", font=_font(22))
        for rank, pair in enumerate(pairs):
            x = margin + (rank % columns) * (pair.width + gap)
            y = 72 + margin + (rank // columns) * (pair.height + gap)
            sheet.paste(pair, (x, y))
        sheet.save(OUT / f"wood_pickaxe_examples_{name}.png")
        print(f"Saved checkpoint {restored:,} examples to {example_dir}", flush=True)


if __name__ == "__main__":
    main()

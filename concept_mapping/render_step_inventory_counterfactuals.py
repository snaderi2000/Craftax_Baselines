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

from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.render_ppo_episodes import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import (
    INV_NAMES,
    MAP_DIM,
    render_local_view,
)


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _capture_episode_step(args, network, train_state, env, env_params) -> np.ndarray:
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
            action, _value, policy_rng = policy_apply(train_state.params, obs, policy_rng)
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, _reward, done, _info = step_jit(
                step_rng, env_state, action, env_params
            )
            if episode == args.episode and step == args.step:
                return np.asarray(obs, dtype=np.float32)
            obs = next_obs
            if bool(np.asarray(done)):
                break
    raise RuntimeError(f"Could not reach episode={args.episode} step={args.step}")


def _set_inventory(base_obs: np.ndarray, additions: dict[str, float]) -> np.ndarray:
    obs = np.array(base_obs, copy=True)
    stats = obs[MAP_DIM:]
    inv = stats[:12] * 10.0
    for name, amount in additions.items():
        idx = INV_NAMES.index(name)
        inv[idx] = max(float(inv[idx]), float(amount))
    stats[:12] = inv / 10.0
    return obs


def _counterfactuals(base_obs: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    return [
        ("unchanged", "unchanged", np.array(base_obs, copy=True)),
        ("wood", "+ wood", _set_inventory(base_obs, {"wood": 1})),
        (
            "wood_pickaxe",
            "+ wood + wood pickaxe",
            _set_inventory(base_obs, {"wood": 1, "wood_pickaxe": 1}),
        ),
        (
            "stone_wood_pickaxe",
            "+ wood + stone + wood pickaxe",
            _set_inventory(base_obs, {"wood": 1, "stone": 1, "wood_pickaxe": 1}),
        ),
        (
            "stone_pickaxe",
            "+ wood + stone + wood pickaxe + stone pickaxe",
            _set_inventory(
                base_obs,
                {"wood": 1, "stone": 1, "wood_pickaxe": 1, "stone_pickaxe": 1},
            ),
        ),
    ]


def _evaluate_values(network, params, obs_list: list[np.ndarray]) -> np.ndarray:
    obs_batch = jnp.asarray(np.stack(obs_list, axis=0), dtype=jnp.float32)
    _pi, values = network.apply(params, obs_batch)
    return np.asarray(values).reshape(-1)


def _make_panel(
    obs: np.ndarray,
    title: str,
    step: int,
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
    label_h = max(112, 20 * display_scale)
    out = Image.new("RGB", (view.width, view.height + label_h), "white")
    out.paste(view, (0, 0))
    draw = ImageDraw.Draw(out)
    value_font = _font(max(22, 5 * display_scale))
    y = view.height + max(6, (label_h - 2 * max(30, 5 * display_scale + 8)) // 2)
    draw.text((10, y), f"step: {step}", fill=(0, 0, 0), font=value_font)
    draw.text((10, y + max(30, 5 * display_scale + 8)), f"value: {value:.4f}", fill=(0, 0, 0), font=value_font)
    return out


def _compose_row(panels: list[Image.Image], gap: int) -> Image.Image:
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render inventory-only counterfactuals for one reproduced PPO state."
    )
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/episode19_inventory_counterfactuals")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=19)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=5)
    parser.add_argument("--gap", type=int, default=18)
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

    base_obs = _capture_episode_step(args, network, train_state, env, env_params)
    profiles = _counterfactuals(base_obs)
    values = _evaluate_values(network, train_state.params, [obs for _key, _title, obs in profiles])

    csv_path = os.path.join(args.out_dir, f"episode_{args.episode:02d}_step_{args.step:04d}_inventory_counterfactuals.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "title", "value", "inventory_order", "inventory_counts"])
        for (key, title, obs), value in zip(profiles, values):
            inv = obs[MAP_DIM : MAP_DIM + 12] * 10.0
            writer.writerow([
                key,
                title,
                float(value),
                ",".join(INV_NAMES),
                ",".join(str(int(round(x))) for x in inv),
            ])
        writer.writerow([])
        writer.writerow(["checkpoint_step", checkpoint_step])

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    panels = [
        _make_panel(
            obs,
            title,
            args.step,
            float(value),
            args.block_size,
            args.display_scale,
            textures,
            panel_textures,
        )
        for (_key, title, obs), value in zip(profiles, values)
    ]
    fig = _compose_row(panels, args.gap)
    fig_path = os.path.join(args.out_dir, f"episode_{args.episode:02d}_step_{args.step:04d}_inventory_counterfactuals.png")
    fig.save(fig_path)

    print(f"checkpoint_step={checkpoint_step}")
    for (key, title, _obs), value in zip(profiles, values):
        print(f"{key}: {title}: V={float(value):.6f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()

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
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import (
    INV_NAMES,
    MAP_CHANNELS,
    MAP_DIM,
    decode_obs,
    render_local_view,
)


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


def _remove_diamond(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    out[MAP_DIM + 4] = 0.0
    return out


def _block_type_from_name(name: str) -> BlockType:
    try:
        return BlockType[name.upper()]
    except KeyError as exc:
        valid = ", ".join(block.name.lower() for block in BlockType)
        raise ValueError(f"Unknown replacement block {name!r}. Valid options: {valid}") from exc


def _value(network, params, obs: np.ndarray) -> float:
    _pi, value = network.apply(params, jnp.asarray(obs[None, :], dtype=jnp.float32))
    return float(np.asarray(value).reshape(-1)[0])


def _rollout_to_target(args, network, train_state, env, env_params):
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    @jax.jit
    def policy_apply(params, obs, rng):
        pi, _ = network.apply(params, obs[None, ...])
        if args.greedy:
            action = jnp.argmax(pi.logits[0])
            return action, rng
        rng, action_rng = jax.random.split(rng)
        return pi.sample(seed=action_rng)[0], rng

    rng = jax.random.PRNGKey(args.seed)
    for episode in range(args.episode + 1):
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, env_state = reset_jit(reset_rng, env_params)
        for step in range(args.max_steps):
            action, policy_rng = policy_apply(train_state.params, obs, policy_rng)
            action_i = int(np.asarray(action))
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, _info = step_jit(
                step_rng, env_state, action, env_params
            )
            reward_f = float(np.asarray(reward))
            done_b = bool(np.asarray(done))
            if episode == args.episode and step == args.step:
                return np.asarray(obs), action_i, reward_f, done_b
            obs = next_obs
            if done_b:
                break
    raise RuntimeError(
        f"Could not reach episode={args.episode} step={args.step}; rollout ended earlier."
    )


def _summary(obs: np.ndarray) -> dict[str, object]:
    decoded = decode_obs(obs)
    blocks = decoded.blocks
    block_counts = {
        block.name: int((blocks == block.value).sum())
        for block in BlockType
        if int((blocks == block.value).sum()) > 0
    }
    return {
        "block_counts": block_counts,
        "inventory": {
            name: float(decoded.inventory[i])
            for i, name in enumerate(INV_NAMES)
            if decoded.inventory[i] > 0
        },
        "health": float(decoded.intrinsics[0]),
        "food": float(decoded.intrinsics[1]),
        "drink": float(decoded.intrinsics[2]),
        "energy": float(decoded.intrinsics[3]),
        "direction": int(decoded.direction),
        "light": float(decoded.light),
        "sleeping": bool(decoded.sleeping),
    }


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


def _compose(
    cells: dict[str, Image.Image],
    values: dict[str, float],
    out_path: str,
    note: str,
    replacement_name: str,
) -> None:
    row_label_w = 138
    col_header_h = 44
    foot_h = 108
    gap = 14
    cell_w = max(cell.width for cell in cells.values())
    cell_h = max(cell.height for cell in cells.values())
    width = row_label_w + 2 * cell_w + gap
    height = col_header_h + 2 * cell_h + gap + foot_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((row_label_w + 8, 12), "Lava present", fill=(0, 0, 0))
    draw.text((row_label_w + cell_w + gap + 8, 12), f"Lava -> {replacement_name}", fill=(0, 0, 0))
    draw.text((10, col_header_h + cell_h // 2 - 8), "Actual inventory", fill=(0, 0, 0))
    draw.text((10, col_header_h + cell_h + gap + cell_h // 2 - 8), "Diamond removed", fill=(0, 0, 0))

    positions = {
        "actual_lava": (row_label_w, col_header_h),
        "actual_replaced": (row_label_w + cell_w + gap, col_header_h),
        "no_diamond_lava": (row_label_w, col_header_h + cell_h + gap),
        "no_diamond_replaced": (row_label_w + cell_w + gap, col_header_h + cell_h + gap),
    }
    for key, pos in positions.items():
        sheet.paste(cells[key], pos)

    actual_penalty = values["actual_replaced"] - values["actual_lava"]
    no_diamond_penalty = values["no_diamond_replaced"] - values["no_diamond_lava"]
    y = col_header_h + 2 * cell_h + gap + 10
    for i, line in enumerate(note.split("\n")):
        draw.text((8, y + 16 * i), line, fill=(0, 0, 0))
    draw.text((8, y + 42), f"Actual-inventory lava penalty:     {actual_penalty:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 62), f"No-diamond lava penalty:           {no_diamond_penalty:.4f}", fill=(0, 0, 0))
    draw.text((8, y + 82), f"Penalty = V(lava->{replacement_name}) - V(lava present).", fill=(0, 0, 0))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-enact a real lava-death rollout state and evaluate value counterfactuals."
    )
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/lava_suicide_reenact_episode00_step239")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=239)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument(
        "--replacement_block",
        default="grass",
        help="Block used to replace visible lava in counterfactuals, e.g. grass or stone.",
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
    train_state, restored_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )

    obs, action, reward, done = _rollout_to_target(args, network, train_state, env, env_params)
    replacement_block = _block_type_from_name(args.replacement_block)
    replacement_name = replacement_block.name.lower()
    actual_lava = obs.astype(np.float32)
    actual_replaced = _replace_block(actual_lava, BlockType.LAVA, replacement_block)
    no_diamond_lava = _remove_diamond(actual_lava)
    no_diamond_replaced = _replace_block(no_diamond_lava, BlockType.LAVA, replacement_block)

    obs_by_key = {
        "actual_lava": actual_lava,
        "actual_replaced": actual_replaced,
        "no_diamond_lava": no_diamond_lava,
        "no_diamond_replaced": no_diamond_replaced,
    }
    values = {key: _value(network, train_state.params, value) for key, value in obs_by_key.items()}

    csv_path = os.path.join(args.out_dir, "lava_suicide_reenact_values.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "lava_present", "diamond_present", "replacement_block", "value"])
        writer.writerow(["actual_lava", True, True, "", values["actual_lava"]])
        writer.writerow(["actual_replaced", False, True, replacement_name, values["actual_replaced"]])
        writer.writerow(["no_diamond_lava", True, False, "", values["no_diamond_lava"]])
        writer.writerow(["no_diamond_replaced", False, False, replacement_name, values["no_diamond_replaced"]])
        writer.writerow([])
        writer.writerow(["actual_inventory_lava_penalty", values["actual_replaced"] - values["actual_lava"]])
        writer.writerow(["no_diamond_lava_penalty", values["no_diamond_replaced"] - values["no_diamond_lava"]])
        writer.writerow(["seed", args.seed])
        writer.writerow(["episode", args.episode])
        writer.writerow(["step", args.step])
        writer.writerow(["action", action])
        writer.writerow(["action_name", _action_name(action)])
        writer.writerow(["reward_after_action", reward])
        writer.writerow(["done_after_action", done])
        writer.writerow(["checkpoint_step", restored_step])
        writer.writerow(["actual_summary", repr(_summary(actual_lava))])

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    cells = {
        key: _make_cell(
            obs_by_key[key],
            key.replace("replaced", replacement_name).replace("_", " "),
            values[key],
            args.block_size,
            args.display_scale,
            textures,
            panel_textures,
        )
        for key in obs_by_key
    }
    fig_path = os.path.join(args.out_dir, "lava_suicide_reenact_grid.png")
    note = (
        f"10B PPO checkpoint={restored_step}; seed={args.seed}; episode={args.episode}; step={args.step}\n"
        f"sampled action={action}:{_action_name(action)} reward={reward:.3f} done={done}"
    )
    _compose(cells, values, fig_path, note, replacement_name)

    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved figure: {fig_path}", flush=True)
    print(f"action={action}:{_action_name(action)} reward={reward:.3f} done={done}", flush=True)
    for key in ("actual_lava", "actual_replaced", "no_diamond_lava", "no_diamond_replaced"):
        print(f"{key}: V={values[key]:.4f}", flush=True)
    print(
        f"actual lava penalty={values['actual_replaced'] - values['actual_lava']:.4f}",
        flush=True,
    )
    print(
        f"no-diamond lava penalty={values['no_diamond_replaced'] - values['no_diamond_lava']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

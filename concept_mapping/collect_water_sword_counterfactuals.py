"""Collect and evaluate reusable water and sword value-function probes.

The collector uses one frozen PPO policy rollout stream.  An episode may supply
up to ``max_per_episode_per_task`` samples for water and independently for
swords, subject to ``min_spacing`` within each task.  Raw base observations are
saved before counterfactual construction so new analyses can be run later
without recollecting trajectories.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import deque

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
from craftax.craftax_classic.constants import BlockType, OBS_DIM
from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from PIL import Image, ImageDraw

from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import (
    MAP_CHANNELS,
    MAP_DIM,
    MOB_NAMES,
    decode_obs,
    render_local_view,
)


DRINK_OFFSET = MAP_DIM + 12 + 2
SWORD_OFFSETS = {
    "wood": MAP_DIM + 9,
    "stone": MAP_DIM + 10,
    "iron": MAP_DIM + 11,
}
HOSTILE_MOBS = ("zombie", "skeleton")
PLAYER_ROW = OBS_DIM[0] // 2
PLAYER_COL = OBS_DIM[1] // 2


def _obs_key(obs: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()


def _visible_water(obs: np.ndarray) -> list[tuple[int, int]]:
    blocks = decode_obs(obs).blocks
    return [tuple(map(int, rc)) for rc in np.argwhere(blocks == BlockType.WATER.value)]


def _hostile_summary(obs: np.ndarray) -> tuple[tuple[str, ...], int]:
    mobs = decode_obs(obs).mobs
    names = []
    nearest = None
    for name in HOSTILE_MOBS:
        index = MOB_NAMES.index(name)
        positions = np.argwhere(mobs[..., index])
        if len(positions):
            names.append(name)
            distances = np.abs(positions[:, 0] - PLAYER_ROW) + np.abs(positions[:, 1] - PLAYER_COL)
            distance = int(distances.min())
            nearest = distance if nearest is None else min(nearest, distance)
    return tuple(names), (-1 if nearest is None else nearest)


def _replace_visible_water(obs: np.ndarray) -> np.ndarray:
    out = np.array(obs, dtype=np.float32, copy=True)
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    blocks = np.argmax(map_view[..., : len(BlockType)], axis=-1)
    for row, col in np.argwhere(blocks == BlockType.WATER.value):
        map_view[row, col, : len(BlockType)] = 0.0
        map_view[row, col, BlockType.GRASS.value] = 1.0
    return out


def _set_drink(obs: np.ndarray, level: int) -> np.ndarray:
    if level not in (0, 9):
        raise ValueError(f"drink level must be 0 or 9, got {level}")
    out = np.array(obs, dtype=np.float32, copy=True)
    out[DRINK_OFFSET] = level / 10.0
    return out


def _set_only_sword(obs: np.ndarray, sword: str | None) -> np.ndarray:
    out = np.array(obs, dtype=np.float32, copy=True)
    for offset in SWORD_OFFSETS.values():
        out[offset] = 0.0
    if sword is not None:
        out[SWORD_OFFSETS[sword]] = 0.1
    return out


def _water_counterfactuals(obs: np.ndarray) -> dict[str, np.ndarray]:
    no_water = _replace_visible_water(obs)
    return {
        "empty_no_water": _set_drink(no_water, 0),
        "empty_water": _set_drink(obs, 0),
        "full_no_water": _set_drink(no_water, 9),
        "full_water": _set_drink(obs, 9),
        "factual": np.asarray(obs, dtype=np.float32),
    }


def _sword_counterfactuals(obs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "wood_only": _set_only_sword(obs, "wood"),
        "stone_only": _set_only_sword(obs, "stone"),
        "iron_only": _set_only_sword(obs, "iron"),
        "no_sword": _set_only_sword(obs, None),
        "factual": np.asarray(obs, dtype=np.float32),
    }


def _collect(args, network, train_state, env, env_params):
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    @jax.jit
    def policy_apply(params, obs, rng):
        pi, _value = network.apply(params, obs[None, ...])
        if args.greedy:
            return jnp.argmax(pi.logits[0]), rng
        rng, action_rng = jax.random.split(rng)
        return pi.sample(seed=action_rng)[0], rng

    water_samples, sword_samples = [], []
    seen_water, seen_sword = set(), set()
    rng = jax.random.PRNGKey(args.seed)
    for episode in range(args.max_episodes):
        if (
            not args.run_all_episodes
            and len(water_samples) >= args.water_target
            and len(sword_samples) >= args.sword_target
        ):
            break
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, env_state = reset_jit(reset_rng, env_params)
        history = deque(maxlen=args.lookback + 1)
        water_in_episode = sword_in_episode = 0
        last_water_step = last_sword_step = -args.min_spacing

        for step in range(args.max_steps):
            obs_np = np.asarray(obs, dtype=np.float32)
            history.append((step, obs_np))

            hostile_names, hostile_distance = _hostile_summary(obs_np)
            sword_eligible = (
                hostile_names
                and len(sword_samples) < args.sword_target
                and sword_in_episode < args.max_per_episode_per_task
                and step - last_sword_step >= args.min_spacing
            )
            if sword_eligible:
                key = _obs_key(obs_np)
                if key not in seen_sword:
                    decoded = decode_obs(obs_np)
                    sword_samples.append(
                        {
                            "episode": episode,
                            "base_step": step,
                            "event_step": step,
                            "hostiles": "+".join(hostile_names),
                            "nearest_hostile_distance": hostile_distance,
                            "health": float(decoded.intrinsics[0]),
                            "drink": float(decoded.intrinsics[2]),
                            "base_obs": obs_np,
                        }
                    )
                    seen_sword.add(key)
                    sword_in_episode += 1
                    last_sword_step = step
                    print(f"sword collected {len(sword_samples)}/{args.sword_target}: episode={episode} step={step} hostiles={'+'.join(hostile_names)}", flush=True)

            drink_before = float(decode_obs(obs_np).intrinsics[2])
            action, policy_rng = policy_apply(train_state.params, obs, policy_rng)
            action_i = int(np.asarray(action))
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, _info = step_jit(step_rng, env_state, action, env_params)
            next_obs_np = np.asarray(next_obs, dtype=np.float32)
            drink_after = float(decode_obs(next_obs_np).intrinsics[2])

            water_eligible = (
                drink_after > drink_before + 0.05
                and len(history) > args.lookback
                and len(water_samples) < args.water_target
                and water_in_episode < args.max_per_episode_per_task
            )
            if water_eligible:
                base_step, base_obs = history[0]
                if base_step - last_water_step >= args.min_spacing and _visible_water(base_obs):
                    key = _obs_key(base_obs)
                    if key not in seen_water:
                        water_samples.append(
                            {
                                "episode": episode,
                                "base_step": base_step,
                                "event_step": step,
                                "action": action_i,
                                "action_name": _action_name(action_i),
                                "reward": float(np.asarray(reward)),
                                "drink_before": drink_before,
                                "drink_after": drink_after,
                                "visible_water_count": len(_visible_water(base_obs)),
                                "base_obs": base_obs,
                            }
                        )
                        seen_water.add(key)
                        water_in_episode += 1
                        last_water_step = base_step
                        print(f"water collected {len(water_samples)}/{args.water_target}: episode={episode} base_step={base_step} drink_event_step={step}", flush=True)

            obs = next_obs
            if bool(np.asarray(done)):
                break
        if episode % 25 == 0:
            print(f"scanned episode={episode} water={len(water_samples)}/{args.water_target} sword={len(sword_samples)}/{args.sword_target}", flush=True)

    return water_samples, sword_samples


def _evaluate_values(network, params, counterfactuals: dict[str, np.ndarray], batch_size: int):
    @jax.jit
    def apply_batch(obs):
        _pi, value = network.apply(params, obs)
        return value

    values = {}
    for name, array in counterfactuals.items():
        batches = []
        for start in range(0, len(array), batch_size):
            batches.append(np.asarray(apply_batch(jnp.asarray(array[start : start + batch_size]))))
        values[name] = np.concatenate(batches)
    return values


def _write_samples(out_dir: str, task: str, samples: list[dict]):
    base_obs = np.stack([sample["base_obs"] for sample in samples])
    np.savez_compressed(
        os.path.join(out_dir, f"{task}_base_states.npz"),
        base_obs=base_obs,
        episodes=np.asarray([sample["episode"] for sample in samples], dtype=np.int32),
        base_steps=np.asarray([sample["base_step"] for sample in samples], dtype=np.int32),
        event_steps=np.asarray([sample["event_step"] for sample in samples], dtype=np.int32),
    )
    fields = [key for key in samples[0] if key != "base_obs"]
    with open(os.path.join(out_dir, f"{task}_metadata.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: sample[key] for key in fields} for sample in samples])


def _write_evaluation(out_dir: str, task: str, samples: list[dict], counterfactuals, values):
    np.savez_compressed(os.path.join(out_dir, f"{task}_counterfactuals.npz"), **counterfactuals)
    with open(os.path.join(out_dir, f"{task}_counterfactual_values.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "episode", "base_step", *values.keys()])
        for index, sample in enumerate(samples):
            writer.writerow([index, sample["episode"], sample["base_step"], *[float(values[key][index]) for key in values]])


def _write_summary(out_dir: str, water_values, sword_values, checkpoint_step: int):
    empty_effect = water_values["empty_water"] - water_values["empty_no_water"]
    full_effect = water_values["full_water"] - water_values["full_no_water"]
    interaction = empty_effect - full_effect
    iron, stone, wood = (sword_values[key] for key in ("iron_only", "stone_only", "wood_only"))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ("checkpoint", checkpoint_step),
            ("water_n", len(empty_effect)),
            ("water_empty_effect_mean", float(empty_effect.mean())),
            ("water_full_effect_mean", float(full_effect.mean())),
            ("water_empty_effect_positive_rate", float((empty_effect > 0).mean())),
            ("water_interaction_mean", float(interaction.mean())),
            ("water_interaction_positive_rate", float((interaction > 0).mean())),
            ("sword_n", len(iron)),
            ("sword_iron_gt_stone_rate", float((iron > stone).mean())),
            ("sword_stone_gt_wood_rate", float((stone > wood).mean())),
            ("sword_full_ranking_rate", float(((iron > stone) & (stone > wood)).mean())),
        ])


def _render_examples(args, out_dir: str, task: str, samples, counterfactuals, values, titles):
    if args.render_examples <= 0:
        return
    inspection_dir = os.path.join(out_dir, f"{task}_inspection")
    os.makedirs(inspection_dir, exist_ok=True)
    textures = load_all_textures(args.block_size)
    panel_textures = textures if args.display_scale == 1 else load_all_textures(args.block_size * args.display_scale)
    rng = np.random.default_rng(args.seed + (0 if task == "water" else 1))
    indices = rng.choice(len(samples), size=min(args.render_examples, len(samples)), replace=False)
    columns = 3
    for rank, index in enumerate(indices):
        frames = []
        for key, title in titles:
            view = render_local_view(
                counterfactuals[key][index],
                block_size=args.block_size,
                include_inventory=True,
                display_scale=args.display_scale,
                textures=textures,
                panel_textures=panel_textures,
            )
            header = Image.new("RGB", (view.width, 38), "white")
            draw = ImageDraw.Draw(header)
            draw.text((5, 4), title, fill="black")
            draw.text((5, 20), f"V={values[key][index]:.4f}", fill="black")
            frame = Image.new("RGB", (view.width, view.height + header.height), "white")
            frame.paste(header, (0, 0))
            frame.paste(view, (0, header.height))
            frames.append(frame)
        width, height = frames[0].size
        rows = (len(frames) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * width, rows * height), "white")
        for frame_index, frame in enumerate(frames):
            sheet.paste(frame, ((frame_index % columns) * width, (frame_index // columns) * height))
        sheet.save(os.path.join(inspection_dir, f"sample_{rank:03d}_idx_{index:03d}.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect joint water and sword counterfactual datasets.")
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--out_dir", default="concept_mapping/runs/water_sword_500_m0mw4end")
    parser.add_argument("--timestep", type=int, default=9999941632)
    parser.add_argument("--env_name", default=None)
    parser.add_argument("--water_target", type=int, default=500)
    parser.add_argument("--sword_target", type=int, default=500)
    parser.add_argument("--max_per_episode_per_task", type=int, default=3)
    parser.add_argument("--min_spacing", type=int, default=256)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=20000)
    parser.add_argument(
        "--run_all_episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not stop early when both state targets have been reached.",
    )
    parser.add_argument(
        "--allow_partial",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save collected states even when a target was not reached.",
    )
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--render_examples", type=int, default=24)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint_step = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)

    water_samples, sword_samples = _collect(args, network, train_state, env, env_params)
    if not water_samples or not sword_samples:
        raise RuntimeError(
            f"No usable samples for at least one task: water={len(water_samples)}, sword={len(sword_samples)}"
        )
    if (
        (len(water_samples) < args.water_target or len(sword_samples) < args.sword_target)
        and not args.allow_partial
    ):
        raise RuntimeError(f"Collected water={len(water_samples)}/{args.water_target}, sword={len(sword_samples)}/{args.sword_target}")
    if len(water_samples) < args.water_target or len(sword_samples) < args.sword_target:
        print(
            f"Saving partial collection: water={len(water_samples)}/{args.water_target}, "
            f"sword={len(sword_samples)}/{args.sword_target}",
            flush=True,
        )
    _write_samples(args.out_dir, "water", water_samples)
    _write_samples(args.out_dir, "sword", sword_samples)

    water_cfs = {key: np.stack([_water_counterfactuals(s["base_obs"])[key] for s in water_samples]) for key in _water_counterfactuals(water_samples[0]["base_obs"])}
    sword_cfs = {key: np.stack([_sword_counterfactuals(s["base_obs"])[key] for s in sword_samples]) for key in _sword_counterfactuals(sword_samples[0]["base_obs"])}
    water_values = _evaluate_values(network, train_state.params, water_cfs, args.batch_size)
    sword_values = _evaluate_values(network, train_state.params, sword_cfs, args.batch_size)
    _write_evaluation(args.out_dir, "water", water_samples, water_cfs, water_values)
    _write_evaluation(args.out_dir, "sword", sword_samples, sword_cfs, sword_values)
    _write_summary(args.out_dir, water_values, sword_values, checkpoint_step)
    _render_examples(
        args,
        args.out_dir,
        "water",
        water_samples,
        water_cfs,
        water_values,
        (
            ("factual", "Factual base"),
            ("empty_no_water", "Empty drink, no water"),
            ("empty_water", "Empty drink, water"),
            ("full_no_water", "Full drink, no water"),
            ("full_water", "Full drink, water"),
        ),
    )
    _render_examples(
        args,
        args.out_dir,
        "sword",
        sword_samples,
        sword_cfs,
        sword_values,
        (
            ("factual", "Factual base"),
            ("no_sword", "No sword"),
            ("wood_only", "Wood sword only"),
            ("stone_only", "Stone sword only"),
            ("iron_only", "Iron sword only"),
        ),
    )

    with open(os.path.join(args.out_dir, "collection_manifest.json"), "w") as handle:
        json.dump({**vars(args), "checkpoint_step": checkpoint_step, "env_name": env_name}, handle, indent=2, sort_keys=True)
    print(f"Saved water and sword datasets to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

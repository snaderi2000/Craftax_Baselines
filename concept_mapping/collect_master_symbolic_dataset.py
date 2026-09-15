"""Collect a resumable master dataset of full frozen-policy Craftax episodes.

Each episode is stored independently as a compressed NPZ file.  This avoids
holding a potentially multi-gigabyte 1,000-episode rollout in memory and lets
an interrupted collection resume without repeating completed episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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

from craftax.craftax_classic.constants import Achievement, BlockType, OBS_DIM
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import (
    INTRINSIC_NAMES,
    INV_NAMES,
    MAP_CHANNELS,
    MAP_DIM,
    MOB_NAMES,
)


DEFAULT_RUN_PATH = "wandb/run-20260830_231509-616wuss8"
DEFAULT_OUT_DIR = "concept_mapping/runs/master_symbolic_1p5b_1000episodes"


def _derived_observation_fields(observations: np.ndarray) -> dict[str, np.ndarray]:
    """Create convenient metadata while retaining observations exactly as stored.

    Args:
        observations: exact PPO observations, shape [T, 1345], dtype float32.

    Returns:
        Inventory and intrinsic counts, visible block IDs and mob masks, and
        direction/light/sleep information derived deterministically from obs.
    """
    map_view = observations[:, :MAP_DIM].reshape((-1, *OBS_DIM, MAP_CHANNELS))
    # [T, 7, 9]: one categorical block ID for every visible map cell.
    visible_blocks = np.argmax(map_view[..., :len(BlockType)], axis=-1).astype(np.uint8)
    # [T, 7, 9, 4]: visible zombie, cow, skeleton, and arrow masks.
    visible_mobs = (map_view[..., len(BlockType):] > 0.5).astype(np.uint8)
    stats = observations[:, MAP_DIM:]
    return {
        # Counts are stored in game units (0--9), not PPO-normalized units.
        "inventory": np.rint(stats[:, :len(INV_NAMES)] * 10).astype(np.uint8),  # [T, 12]
        "intrinsics": np.rint(
            stats[:, len(INV_NAMES):len(INV_NAMES) + len(INTRINSIC_NAMES)] * 10
        ).astype(np.uint8),  # [T, 4]: health, food, drink, energy
        "visible_blocks": visible_blocks,
        "visible_mobs": visible_mobs,
        "direction": np.argmax(stats[:, 16:20], axis=-1).astype(np.uint8),  # [T]
        "light": stats[:, 20].astype(np.float32),  # [T]
        "sleeping": (stats[:, 21] > 0.5).astype(np.uint8),  # [T]
    }


def _write_metadata(out_dir: Path, args, checkpoint: int, observation_dim: int, max_steps: int):
    metadata_path = out_dir / "dataset_metadata.json"
    if metadata_path.exists():
        return
    metadata = {
        "description": "Master frozen-policy Craftax-Classic symbolic rollout dataset.",
        "run_path": args.run_path,
        "checkpoint": checkpoint,
        "policy_sampling": "greedy argmax" if args.greedy else "categorical sampling with saved seed",
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "max_steps_per_episode": max_steps,
        "observation_dtype": "float32",
        "observation_shape_per_state": [observation_dim],
        "observation_semantics": "Exact symbolic observation returned by the environment before each action.",
        "terminal_observation_semantics": "Exact observation returned after the terminal action.",
        "inventory_names": INV_NAMES,
        "intrinsic_names": INTRINSIC_NAMES,
        "mob_names": MOB_NAMES,
        "block_type_names": [block.name for block in BlockType],
        "achievement_names": [achievement.name for achievement in Achievement],
        "episode_file_format": {
            "observations": "[T, 1345] float32 exact symbolic pre-action states",
            "step": "[T] int32 within-episode state index",
            "achievements": "[T, 22] uint8 achievements already completed at each pre-action state",
            "actions": "[T] int16 sampled/greedy action taken from each state",
            "rewards": "[T] float32 reward returned by the action",
            "dones": "[T] uint8 whether that action terminated the episode",
            "inventory": "[T, 12] uint8 game-unit inventory counts",
            "visible_blocks": "[T, 7, 9] uint8 block IDs",
            "visible_mobs": "[T, 7, 9, 4] uint8 visible mob masks",
            "terminal_observation": "[1345] float32 post-terminal observation",
            "terminal_achievements": "[22] uint8 achievements after the terminal action",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def _append_manifest_row(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_progress(out_dir: Path, completed_episodes: int, requested_episodes: int,
                    latest_episode_id: int, latest_state_count: int, elapsed_seconds: float):
    """Write progress independently of stdout for detached/server execution."""
    payload = {
        "completed_episodes": completed_episodes,
        "requested_episodes": requested_episodes,
        "latest_episode_id": latest_episode_id,
        "latest_episode_pre_action_states": latest_state_count,
        "elapsed_seconds": elapsed_seconds,
        "updated_unix_time": time.time(),
    }
    temporary = out_dir / ".collection_progress.tmp.json"
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(out_dir / "collection_progress.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN_PATH)
    parser.add_argument("--timestep", type=int, default=None,
                        help="Defaults to the final checkpoint in --run_path.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="0 uses Craftax's full 10,000-step episode limit.")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("Require 0 <= worker_index < num_workers")

    out_dir = Path(args.out_dir)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = (out_dir if args.num_workers == 1 else
                    out_dir / "workers" / f"worker_{args.worker_index}")
    progress_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = progress_dir / "episode_manifest.csv"

    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    max_steps = args.max_steps or int(env_params.max_timesteps)
    network = _init_network(config, env, env_params)
    train_state, checkpoint = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )
    observation_dim = int(env.observation_space(env_params).shape[0])
    _write_metadata(out_dir, args, checkpoint, observation_dim, max_steps)

    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    @jax.jit
    def rollout(initial_obs, initial_state, env_rng, policy_rng):
        """Run until terminal or max_steps, storing a preallocated trace buffer.

        A JAX ``while_loop`` is important here: unlike a fixed ``scan``, it
        stops network and environment computation immediately after death or
        episode termination rather than evaluating padded no-op rows up to
        Craftax's 10,000-step horizon.
        """
        achievement_dim = initial_state.achievements.shape[0]
        observation_buffer = jnp.zeros((max_steps, observation_dim), dtype=initial_obs.dtype)
        achievement_buffer = jnp.zeros((max_steps, achievement_dim), dtype=initial_state.achievements.dtype)
        action_buffer = jnp.zeros((max_steps,), dtype=jnp.int32)
        reward_buffer = jnp.zeros((max_steps,), dtype=jnp.float32)
        done_buffer = jnp.zeros((max_steps,), dtype=jnp.bool_)

        def condition(carry):
            step, _obs, _state, _env_rng, _policy_rng, done, *_buffers = carry
            return jnp.logical_and(step < max_steps, jnp.logical_not(done))

        def body(carry):
            (
                step, obs, state, env_rng, policy_rng, _done,
                observation_buffer, achievement_buffer, action_buffer,
                reward_buffer, done_buffer,
            ) = carry
            env_rng, step_rng = jax.random.split(env_rng)
            pi, _value = network.apply(train_state.params, obs[None, ...])
            if args.greedy:
                action, next_policy_rng = jnp.argmax(pi.logits[0]), policy_rng
            else:
                next_policy_rng, action_rng = jax.random.split(policy_rng)
                action = pi.sample(seed=action_rng)[0]
            next_obs, next_state, reward, done, _info = env.step(
                step_rng, state, action, env_params
            )
            return (
                step + 1, next_obs, next_state, env_rng, next_policy_rng, done,
                observation_buffer.at[step].set(obs),
                achievement_buffer.at[step].set(state.achievements),
                action_buffer.at[step].set(action),
                reward_buffer.at[step].set(reward),
                done_buffer.at[step].set(done),
            )

        initial = (
            jnp.asarray(0, dtype=jnp.int32), initial_obs, initial_state,
            env_rng, policy_rng, jnp.asarray(False), observation_buffer,
            achievement_buffer, action_buffer, reward_buffer, done_buffer,
        )
        return jax.lax.while_loop(condition, body, initial)

    completed = {
        int(path.stem.split("_")[-1])
        for path in episodes_dir.glob("episode_*.npz")
        if path.stem.split("_")[-1].isdigit()
    }
    print(
        f"Collecting up to {args.episodes} complete episodes from checkpoint {checkpoint:,}; "
        f"{len(completed)} episode files already exist and will be skipped.",
        flush=True,
    )
    rng = jax.random.PRNGKey(args.seed)
    started = time.time()
    worker_completed = sum(i % args.num_workers == args.worker_index for i in completed)
    worker_requested = len(range(args.worker_index, args.episodes, args.num_workers))
    for episode_id in range(args.episodes):
        rng, reset_rng, env_rng, policy_rng = jax.random.split(rng, 4)
        # Advance the global seed sequence even for other workers' episodes.
        if episode_id % args.num_workers != args.worker_index:
            continue
        if episode_id in completed:
            continue
        obs, state = reset_jit(reset_rng, env_params)
        final_carry = rollout(obs, state, env_rng, policy_rng)
        active_count = int(np.asarray(final_carry[0]))
        if active_count == 0:
            raise RuntimeError(f"Episode {episode_id} produced no active states")
        observations = np.asarray(final_carry[6])
        achievements = np.asarray(final_carry[7])
        actions = np.asarray(final_carry[8])
        rewards = np.asarray(final_carry[9])
        dones = np.asarray(final_carry[10])

        # Keep precisely the T real pre-action states; discard fixed-scan padding.
        observations = observations[:active_count].astype(np.float32, copy=False)  # [T, 1345]
        achievements = achievements[:active_count].astype(np.uint8, copy=False)  # [T, 22]
        actions = actions[:active_count].astype(np.int16, copy=False)  # [T]
        rewards = rewards[:active_count].astype(np.float32, copy=False)  # [T]
        dones = dones[:active_count].astype(np.uint8, copy=False)  # [T]
        derived = _derived_observation_fields(observations)
        final_obs, final_state = final_carry[1], final_carry[2]
        terminated = bool(dones[-1])
        output_path = episodes_dir / f"episode_{episode_id:04d}.npz"
        temporary_path = episodes_dir / f".episode_{episode_id:04d}.tmp.npz"
        np.savez_compressed(
            temporary_path,
            episode_id=np.int32(episode_id),
            checkpoint=np.int64(checkpoint),
            step=np.arange(active_count, dtype=np.int32),
            observations=observations,
            achievements=achievements,
            actions=actions,
            rewards=rewards,
            dones=dones,
            terminal_observation=np.asarray(final_obs, dtype=np.float32),
            terminal_achievements=np.asarray(final_state.achievements, dtype=np.uint8),
            **derived,
        )
        temporary_path.replace(output_path)
        row = {
            "episode_id": episode_id,
            "checkpoint": checkpoint,
            "n_pre_action_states": active_count,
            "terminated": int(terminated),
            "final_reward": float(rewards[-1]),
            "final_action": int(actions[-1]),
            "final_action_name": _action_name(int(actions[-1])),
            "file": str(output_path.relative_to(out_dir)),
        }
        _append_manifest_row(manifest_path, row)
        elapsed_seconds = time.time() - started
        worker_completed += 1
        _write_progress(progress_dir, worker_completed, worker_requested, episode_id,
                        active_count, elapsed_seconds)
        elapsed_minutes = elapsed_seconds / 60
        print(
            f"saved episode {episode_id + 1}/{args.episodes}: "
            f"states={active_count:,} terminated={terminated} "
            f"elapsed={elapsed_minutes:.1f} min",
            flush=True,
        )

    print(f"Master dataset collection complete: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

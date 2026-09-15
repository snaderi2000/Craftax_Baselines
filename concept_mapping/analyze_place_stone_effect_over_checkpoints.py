"""Evaluate first-Place-Stone transitions at the existing stride-3 checkpoints."""

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
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)
    jax.tree = _JaxTreeCompat()
import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


MASTER = Path("concept_mapping/runs/master_symbolic_1p5b_1000episodes")
BASE_DIR = MASTER / "achievement_stabilized_effect_over_checkpoints_stride3"
DEFAULT_OUT = BASE_DIR / "place_stone_checkpoint_sweep_stride3"
DEFAULT_BANK = MASTER / "place_stone_transition_fixed_bank.npz"
DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"


def checkpoints() -> list[int]:
    with (BASE_DIR / "achievement_stabilized_effect_over_checkpoints.csv").open(newline="") as handle:
        return [int(row["checkpoint"]) for row in csv.DictReader(handle)]


def build_or_load_bank(path: Path) -> dict[str, np.ndarray]:
    if path.exists():
        with np.load(path) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    names = [achievement.name.lower() for achievement in Achievement]
    achievement_index = names.index("place_stone")
    before, after, episode_ids, completion_steps = [], [], [], []
    for episode_path in sorted((MASTER / "episodes").glob("episode_*.npz")):
        with np.load(episode_path) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            achievements = np.concatenate([
                np.asarray(data["achievements"]),
                np.asarray(data["terminal_achievements"])[None],
            ])
            terminal_observation = np.asarray(data["terminal_observation"], dtype=np.float32)
            episode_id = int(data["episode_id"])
        hits = np.flatnonzero(achievements[:, achievement_index] > 0)
        if not len(hits) or hits[0] == 0:
            continue
        after_index = int(hits[0])
        before.append(observations[after_index - 1])
        after.append(observations[after_index] if after_index < len(observations)
                     else terminal_observation)
        episode_ids.append(episode_id); completion_steps.append(after_index)
    payload = {
        "before": np.asarray(before, dtype=np.float32),
        "after": np.asarray(after, dtype=np.float32),
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "completion_step": np.asarray(completion_steps, dtype=np.int32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)
    return payload


def output_path(out_dir: Path, step: int) -> Path:
    return out_dir / "per_checkpoint" / f"checkpoint_{step}.npz"


def run_worker(args, steps: list[int], bank: dict[str, np.ndarray]) -> None:
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    packed = np.concatenate([bank["before"], bank["after"]]).astype(np.float32)
    n_states = len(bank["before"])

    @jax.jit
    def value_batch(params, observations):
        _policy, value = network.apply(params, observations)
        return value

    worker_steps = [step for index, step in enumerate(steps)
                    if index % args.num_workers == args.worker_index]
    print(f"worker {args.worker_index}/{args.num_workers}: {len(worker_steps)} checkpoints", flush=True)
    for step in worker_steps:
        destination = output_path(Path(args.out_dir), step)
        if destination.exists():
            continue
        print(f"worker {args.worker_index}: evaluating {step:,}", flush=True)
        train_state, restored = _restore_train_state(
            args.run_path, config, network, env, env_params, step
        )
        values = []
        for start in range(0, len(packed), args.batch_size):
            values.append(np.asarray(value_batch(
                train_state.params, jnp.asarray(packed[start:start + args.batch_size])
            )))
        before_values, after_values = np.split(np.concatenate(values), [n_states])
        delta = after_values - before_values
        stabilized = delta / (1.0 + np.abs(before_values))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, checkpoint=np.int64(restored),
                median_raw=np.float64(np.median(delta)),
                median_stabilized=np.float64(np.median(stabilized)),
                raw_effect=delta.astype(np.float32),
                stabilized_effect=stabilized.astype(np.float32),
            )
        temporary.replace(destination)


def aggregate(args, steps: list[int], bank: dict[str, np.ndarray]) -> None:
    out_dir = Path(args.out_dir)
    missing = [step for step in steps if not output_path(out_dir, step).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} outputs; first={missing[:5]}")
    rows = []
    for step in steps:
        with np.load(output_path(out_dir, step)) as data:
            rows.append({
                "checkpoint": step,
                "n_states": len(bank["before"]),
                "median_raw_delta_place_stone": float(data["median_raw"]),
                "median_stabilized_place_stone": float(data["median_stabilized"]),
            })
    with (out_dir / "place_stone_effect_over_checkpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved {len(rows)} checkpoints; n_states={len(bank['before'])}; "
          f"median completion step={np.median(bank['completion_step']):g}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    steps = checkpoints(); bank = build_or_load_bank(args.bank)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args, steps, bank)
    else:
        run_worker(args, steps, bank)


if __name__ == "__main__":
    main()

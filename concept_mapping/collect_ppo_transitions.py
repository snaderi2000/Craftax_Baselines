import argparse
import os
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
from flax.training.train_state import TrainState

# Compatibility for newer Optax/Flax code running on JAX versions before
# jax.tree was introduced, such as jax==0.4.23.
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)

    jax.tree = _JaxTreeCompat()

from craftax.craftax_env import make_craftax_env_from_name
from models.actor_critic import ActorCritic, ActorCriticConv


def _load_wandb_config(run_path: str) -> dict[str, Any]:
    files_dir = run_path if os.path.basename(run_path) == "files" else os.path.join(run_path, "files")
    config_path = os.path.join(files_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find W&B config at {config_path}")
    with open(config_path) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else v
        for k, v in raw_config.items()
    }


def _files_dir(run_path: str) -> str:
    return run_path if os.path.basename(run_path) == "files" else os.path.join(run_path, "files")


def _init_network(config: dict[str, Any], env, env_params):
    action_dim = env.action_space(env_params).n
    layer_size = int(config.get("LAYER_SIZE", 512))
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    if "Symbolic" in env_name:
        network = ActorCritic(action_dim, layer_size)
    else:
        network = ActorCriticConv(action_dim, layer_size)
    return network


def _restore_train_state(run_path: str, config: dict[str, Any], network, env, env_params, timestep: int | None):
    init_obs = jnp.zeros((1, *env.observation_space(env_params).shape), dtype=jnp.float32)
    params = network.init(jax.random.PRNGKey(0), init_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(float(config.get("MAX_GRAD_NORM", 1.0))),
        optax.adam(float(config.get("LR", 2e-4)), eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    ckpt_dir = os.path.join(_files_dir(run_path), "policies")
    manager = ocp.CheckpointManager(ckpt_dir, ocp.PyTreeCheckpointer())
    step = manager.latest_step() if timestep is None else timestep
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    print(f"Restoring policy checkpoint step={step} from {ckpt_dir}")
    restored = manager.restore(step, items=train_state)
    return restored, int(step)


def _tree_to_np(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _save_shard(output_dir: str, shard_idx: int, shard: dict[str, np.ndarray]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"transitions_{shard_idx:05d}.npz")
    np.savez_compressed(path, **shard)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved {path} ({size_mb:.1f} MB)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect symbolic transition shards from a trained ppo.py policy."
    )
    parser.add_argument("--run_path", required=True, help="W&B run dir or its files/ subdir")
    parser.add_argument("--output_dir", default="concept_mapping/data/ppo_transitions")
    parser.add_argument("--timestep", type=int, default=None, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--env_name", type=str, default=None, help="Override env from config.")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps_per_shard", type=int, default=256)
    parser.add_argument("--num_shards", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax actions instead of sampling from the policy.",
    )
    parser.add_argument(
        "--save_info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save scalar info fields as info/<key> arrays when possible.",
    )
    args = parser.parse_args()

    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    if "Symbolic" not in env_name:
        raise ValueError(f"This collector is intended for symbolic policies, got {env_name}")
    config["ENV_NAME"] = env_name

    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    obs_shape = env.observation_space(env_params).shape
    print(f"env={env_name} obs_shape={obs_shape} num_envs={args.num_envs}")

    network = _init_network(config, env, env_params)
    train_state, restored_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )

    reset_vmap = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    @jax.jit
    def policy_step(params, obs, env_state, rng):
        pi, value = network.apply(params, obs)
        if args.greedy:
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            rng, action_rng = jax.random.split(rng)
            action = pi.sample(seed=action_rng)
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, args.num_envs)
        next_obs, next_env_state, reward, done, info = step_vmap(
            step_rngs, env_state, action, env_params
        )
        transition = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": reward,
            "done": done,
            "value": value,
        }
        return (next_obs, next_env_state, rng), transition, info

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, args.num_envs)
    obs, env_state = reset_vmap(reset_rngs, env_params)

    metadata = {
        "run_path": args.run_path,
        "checkpoint_step": restored_step,
        "env_name": env_name,
        "num_envs": args.num_envs,
        "steps_per_shard": args.steps_per_shard,
        "greedy": args.greedy,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(metadata, f)

    total = 0
    for shard_idx in range(args.num_shards):
        arrays = []
        info_arrays = []
        for _ in range(args.steps_per_shard):
            (obs, env_state, rng), transition, info = policy_step(
                train_state.params, obs, env_state, rng
            )
            arrays.append(_tree_to_np(transition))
            if args.save_info:
                info_arrays.append(_tree_to_np(info))

        shard = {}
        for key in arrays[0].keys():
            stacked = np.stack([x[key] for x in arrays], axis=0)
            shard[key] = stacked.reshape(-1, *stacked.shape[2:])

        if args.save_info and info_arrays:
            for key, value in info_arrays[0].items():
                if np.asarray(value).dtype == object:
                    continue
                try:
                    stacked = np.stack([x[key] for x in info_arrays], axis=0)
                except Exception:
                    continue
                if stacked.ndim >= 2 and stacked.shape[1] == args.num_envs:
                    shard[f"info/{key}"] = stacked.reshape(-1, *stacked.shape[2:])

        _save_shard(args.output_dir, shard_idx, shard)
        total += shard["action"].shape[0]
        print(f"Collected {total:,} transitions")


if __name__ == "__main__":
    main()

"""Evaluate Craftax-Classic achievement completion for saved PPO checkpoints."""

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


ACHIEVEMENTS = tuple(achievement.name.lower() for achievement in Achievement)


def _evaluate(env, env_params, network, params, episodes: int, max_steps: int, seed: int, greedy: bool) -> np.ndarray:
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    @jax.jit
    def policy(obs, key):
        pi, _value = network.apply(params, obs[None, ...])
        if greedy:
            return jnp.argmax(pi.logits[0]), key
        key, action_key = jax.random.split(key)
        return pi.sample(seed=action_key)[0], key

    rng = jax.random.PRNGKey(seed)
    completed = np.zeros(len(ACHIEVEMENTS), dtype=np.int32)
    for episode in range(episodes):
        rng, reset_key, policy_key = jax.random.split(rng, 3)
        obs, state = reset(reset_key, env_params)
        for _step in range(max_steps):
            action, policy_key = policy(obs, policy_key)
            rng, step_key = jax.random.split(rng)
            obs, state, _reward, done, _info = step(step_key, state, action, env_params)
            if bool(np.asarray(done)):
                break
        completed += np.asarray(state.achievements, dtype=np.int32)
        if (episode + 1) % 10 == 0 or episode + 1 == episodes:
            print(f"  episodes={episode + 1}/{episodes}", flush=True)
    return completed / episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated saved checkpoint steps.")
    parser.add_argument("--episodes", type=int, default=59)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    checkpoints = [int(value) for value in args.checkpoints.split(",") if value.strip()]
    rows = []
    for checkpoint in checkpoints:
        print(f"Evaluating checkpoint {checkpoint:,}", flush=True)
        train_state, restored = _restore_train_state(args.run_path, config, network, env, env_params, checkpoint)
        rates = _evaluate(env, env_params, network, train_state.params, args.episodes, args.max_steps, args.seed, args.greedy)
        rows.append({"checkpoint": restored, "episodes": args.episodes, **{name: float(rate) for name, rate in zip(ACHIEVEMENTS, rates)}})
    with (out_dir / "achievement_success.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_dir / 'achievement_success.csv'}", flush=True)


if __name__ == "__main__":
    main()

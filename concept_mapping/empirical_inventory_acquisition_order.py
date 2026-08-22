"""Estimate the frozen policy's empirical inventory-acquisition order."""

import argparse
import csv
import json
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state
from concept_mapping.visualize_symbolic_trajectory import MAP_DIM


INVENTORY = (
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--timestep", type=int, default=9999941632)
    parser.add_argument("--out_dir", default="concept_mapping/runs/empirical_inventory_order_50episodes_m0mw4end")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, restored = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)
    reset_many = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))

    @jax.jit
    def rollout(initial_obs, initial_state, rng):
        def step_one(obs, state, key, already_done):
            pi, _value = network.apply(train_state.params, obs[None, ...])
            if args.greedy:
                action = jnp.argmax(pi.logits[0])
            else:
                key, action_key = jax.random.split(key)
                action = pi.sample(seed=action_key)[0]
            key, step_key = jax.random.split(key)
            def active(_):
                next_obs, next_state, _reward, done, _info = env.step(step_key, state, action, env_params)
                return next_obs, next_state, done
            def inactive(_):
                return obs, state, jnp.asarray(True)
            next_obs, next_state, done = jax.lax.cond(already_done, inactive, active, None)
            return next_obs, next_state, key, done

        def body(carry, _):
            obs, state, keys, dones = carry
            next_obs, next_state, next_keys, next_dones = jax.vmap(step_one)(obs, state, keys, dones)
            return (next_obs, next_state, next_keys, jnp.logical_or(dones, next_dones)), obs

        keys = jax.random.split(rng, args.episodes)
        carry = (initial_obs, initial_state, keys, jnp.zeros(args.episodes, dtype=bool))
        return jax.lax.scan(body, carry, xs=None, length=args.max_steps)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key, rollout_key = jax.random.split(rng, 3)
    initial_obs, initial_state = reset_many(jax.random.split(reset_key, args.episodes), env_params)
    (_obs, _state, _keys, done), obs_trace = rollout(initial_obs, initial_state, rollout_key)
    trace = np.asarray(obs_trace, dtype=np.float32)  # [steps, episodes, observation]
    inventory = trace[:, :, MAP_DIM:MAP_DIM + len(INVENTORY)] * 10.0
    present = inventory > .05
    acquired = present.any(axis=0)
    first_step = np.where(acquired, present.argmax(axis=0), -1).astype(np.int32)
    final_done = np.asarray(done)
    np.savez_compressed(out_dir / "first_acquisition_steps.npz", first_step=first_step, acquired=acquired, episode_done=final_done)
    rows = []
    for item_index, item in enumerate(INVENTORY):
        steps = first_step[:, item_index]
        observed = steps >= 0
        rows.append({
            "item": item,
            "episodes_acquired": int(observed.sum()),
            "acquisition_rate": float(observed.mean()),
            "median_first_step_among_acquired": float(np.median(steps[observed])) if observed.any() else None,
            "mean_first_step_among_acquired": float(steps[observed].mean()) if observed.any() else None,
        })
    rows.sort(key=lambda row: (float("inf") if row["median_first_step_among_acquired"] is None else row["median_first_step_among_acquired"], row["item"]))
    with (out_dir / "acquisition_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Pairwise empirical precedence: P(row item appears before column item),
    # conditional on both being acquired in an episode.
    precedence = np.full((len(INVENTORY), len(INVENTORY)), np.nan)
    for i in range(len(INVENTORY)):
        for j in range(len(INVENTORY)):
            both = (first_step[:, i] >= 0) & (first_step[:, j] >= 0)
            if both.any():
                precedence[i, j] = np.mean(first_step[both, i] < first_step[both, j])
    np.save(out_dir / "pairwise_precedence.npy", precedence)
    fig, ax = plt.subplots(figsize=(7.4, 5.3), constrained_layout=True)
    label_order = [row["item"] for row in rows]
    indices = [INVENTORY.index(item) for item in label_order]
    medians = [row["median_first_step_among_acquired"] for row in rows]
    rates = [100 * row["acquisition_rate"] for row in rows]
    ax.scatter(medians, range(len(rows)), s=np.asarray(rates) * 2.2 + 15, color="#2563eb", alpha=.85)
    ax.set_yticks(range(len(rows)), [item.replace("_", " ").title() for item in label_order])
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.25)
    ax.set_xlabel("Median first-acquisition step (episodes where acquired)")
    ax.set_title("Empirical frozen-policy inventory acquisition order")
    for y, median, rate in zip(range(len(rows)), medians, rates):
        if median is not None:
            ax.annotate(f"{rate:.0f}%", (median, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    fig.savefig(out_dir / "empirical_acquisition_order.png", dpi=300)
    fig.savefig(out_dir / "empirical_acquisition_order.pdf")
    (out_dir / "manifest.json").write_text(json.dumps({**vars(args), "checkpoint_step": restored, "inventory": INVENTORY}, indent=2, sort_keys=True))
    print(f"Saved empirical first-acquisition order for {args.episodes} episodes to {out_dir}")


if __name__ == "__main__":
    main()

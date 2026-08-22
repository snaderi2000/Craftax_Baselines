"""Test whether PPO critic value decreases along a progression-ordered inventory chain."""

import argparse
import csv
import itertools
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
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state
from concept_mapping.visualize_symbolic_trajectory import MAP_DIM


INVENTORY = (
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)
OFFSETS = {name: MAP_DIM + index for index, name in enumerate(INVENTORY)}
# Every ordering respects the achievement hierarchy; only members of a braced
# group may exchange positions.
TIED_GROUPS = (
    ("wood",),
    ("wood_pickaxe", "wood_sword"),
    ("stone", "coal"),
    ("stone_pickaxe", "stone_sword"),
    ("iron",),
    ("iron_pickaxe", "iron_sword"),
    ("diamond",),
)


def _parse_order(order_spec: str | None) -> list[tuple[str, ...]]:
    """Return either all hierarchy-consistent ties or one explicit ordering."""
    if order_spec is None:
        return _orders()
    order = tuple(part.strip() for part in order_spec.split(",") if part.strip())
    if len(order) != len(INVENTORY) or set(order) != set(INVENTORY):
        raise ValueError(
            "--order must list every inventory component exactly once: "
            + ", ".join(INVENTORY)
        )
    return [order]


def _orders():
    return [tuple(item for group in groups for item in group)
            for groups in itertools.product(*[tuple(itertools.permutations(group)) for group in TIED_GROUPS])]


def _counterfactual_chain(obs: np.ndarray, order: tuple[str, ...]) -> np.ndarray:
    """Empty inventory then add one present/absent progression component per rung."""
    base = np.asarray(obs, dtype=np.float32).copy()
    base[MAP_DIM : MAP_DIM + len(INVENTORY)] = 0.0
    chain = [base.copy()]
    current = base.copy()
    for item in order:
        current = current.copy()
        # One unit for resources and one copy for tools: presence, not quantity.
        current[OFFSETS[item]] = 0.1
        chain.append(current)
    return np.stack(chain)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--timestep", type=int, default=9999941632)
    parser.add_argument("--target_states", type=int, default=500)
    parser.add_argument(
        "--base_states",
        default=None,
        help="Optional existing base_states.npz file; reuse it instead of recollecting rollout states.",
    )
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--sample_every", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--order", default=None,
        help="Comma-separated explicit inventory order. Defaults to all valid tied hierarchy orderings.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.rollout_steps % args.sample_every:
        raise ValueError("rollout_steps must be divisible by sample_every")

    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, restored = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)
    reset_vmap = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    @jax.jit
    def rollout_step(obs, state, rng):
        pi, _value = network.apply(train_state.params, obs)
        if args.greedy:
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            rng, action_rng = jax.random.split(rng)
            action = pi.sample(seed=action_rng)
        rng, step_rng = jax.random.split(rng)
        next_obs, next_state, _reward, _done, _info = step_vmap(
            jax.random.split(step_rng, args.num_envs), state, action, env_params
        )
        return next_obs, next_state, rng

    @jax.jit
    def value_batch(obs):
        _pi, value = network.apply(train_state.params, obs)
        return value

    if args.base_states:
        base_obs = np.asarray(np.load(args.base_states)["base_obs"], dtype=np.float32)
        print(f"Reusing {len(base_obs)} saved base states from {args.base_states}", flush=True)
        # Copy the exact states into this run directory so every output is
        # self-contained and can be merged without referring to another run.
        np.savez_compressed(out_dir / "base_states.npz", base_obs=base_obs)
    else:
        rng = jax.random.PRNGKey(args.seed)
        rng, reset_rng = jax.random.split(rng)
        obs, state = reset_vmap(jax.random.split(reset_rng, args.num_envs), env_params)
        candidates, candidate_meta = [], []
        for step in range(args.rollout_steps):
            if step % args.sample_every == 0:
                candidates.append(np.asarray(obs, dtype=np.float32))
                candidate_meta.extend((step, env_index) for env_index in range(args.num_envs))
            obs, state, rng = rollout_step(obs, state, rng)
        candidates = np.concatenate(candidates, axis=0)
        if args.target_states > len(candidates):
            raise ValueError(f"Requested {args.target_states} states but only have {len(candidates)} candidates")
        chosen = np.random.default_rng(args.seed).choice(len(candidates), args.target_states, replace=False)
        base_obs = candidates[chosen]
        np.savez_compressed(
            out_dir / "base_states.npz", base_obs=base_obs,
            rollout_step=np.asarray([candidate_meta[index][0] for index in chosen], dtype=np.int32),
            env_index=np.asarray([candidate_meta[index][1] for index in chosen], dtype=np.int32),
        )

    orders = _parse_order(args.order)
    levels_per_order = len(orders[0]) + 1
    rows = []
    for base_index, base in enumerate(base_obs):
        all_obs = np.concatenate([_counterfactual_chain(base, order) for order in orders], axis=0)
        values = []
        for start in range(0, len(all_obs), args.batch_size):
            values.append(np.asarray(value_batch(jnp.asarray(all_obs[start:start + args.batch_size]))))
        values = np.concatenate(values)
        for order_index, order in enumerate(orders):
            for level, value in enumerate(values[order_index * levels_per_order:(order_index + 1) * levels_per_order]):
                rows.append((base_index, order_index, level, "none" if level == 0 else order[level - 1], float(value)))
        if (base_index + 1) % 25 == 0 or base_index + 1 == len(base_obs):
            print(f"evaluated states={base_index + 1}/{len(base_obs)}", flush=True)
    with (out_dir / "counterfactual_values.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["base_index", "ordering_index", "inventory_level", "item_added", "value"])
        writer.writerows(rows)
    manifest = {**vars(args), "checkpoint_step": restored, "inventory_components": list(INVENTORY), "hierarchy": [list(group) for group in TIED_GROUPS], "orders": [list(order) for order in orders], "n_orderings": len(orders)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved {len(base_obs)} base states and {len(rows)} counterfactual values to {out_dir}")


if __name__ == "__main__":
    main()

"""Plot emergence of the diamond/pickaxe value ordering on a fixed probe set."""

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
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from concept_mapping.diamond_affordance_over_time import _make_counterfactuals
from concept_mapping.render_ppo_episodes import (
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from craftax.craftax_env import make_craftax_env_from_name


def _checkpoint_steps(run_path: str) -> list[int]:
    files_dir = run_path if os.path.basename(run_path) == "files" else os.path.join(run_path, "files")
    policies = Path(files_dir) / "policies"
    steps = []
    for path in policies.iterdir():
        if path.is_dir():
            try:
                steps.append(int(path.name))
            except ValueError:
                pass
    if not steps:
        raise FileNotFoundError(f"No numeric checkpoint directories found in {policies}")
    return sorted(steps)


def _evaluate_values(network, params, observations: dict[str, np.ndarray], batch_size: int):
    @jax.jit
    def apply_batch(obs):
        _pi, value = network.apply(params, obs)
        return value

    values = {}
    for name, array in observations.items():
        chunks = []
        for start in range(0, len(array), batch_size):
            chunks.append(np.asarray(apply_batch(jnp.asarray(array[start : start + batch_size]))))
        values[name] = np.concatenate(chunks)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed diamond counterfactual states at every saved PPO checkpoint."
    )
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument(
        "--states_npz",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/clean_diamond_tminus_states.npz",
    )
    parser.add_argument(
        "--out_dir",
        default="concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/ordering_over_checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--checkpoints",
        default="",
        help="Optional comma-separated checkpoint steps; default evaluates every saved checkpoint.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    states = np.load(args.states_npz)
    base_obs = np.asarray(states["base_obs"], dtype=np.float32)
    counterfactuals = [_make_counterfactuals(obs) for obs in base_obs]
    observations = {
        key: np.stack([cf[key] for cf in counterfactuals])
        for key in counterfactuals[0]
    }

    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    steps = (
        [int(value) for value in args.checkpoints.split(",") if value.strip()]
        if args.checkpoints
        else _checkpoint_steps(args.run_path)
    )

    rows = []
    for step in steps:
        print(f"Evaluating checkpoint {step:,}", flush=True)
        train_state, restored_step = _restore_train_state(
            args.run_path, config, network, env, env_params, step
        )
        values = _evaluate_values(network, train_state.params, observations, args.batch_size)
        a = values["A_no_diamond_no_pickaxe"]
        b = values["B_no_diamond_pickaxe"]
        c = values["C_diamond_no_pickaxe"]
        d = values["D_diamond_pickaxe"]
        full = (d > a) & (d > c) & (a > b) & (c > b)
        rows.append(
            {
                "checkpoint": restored_step,
                "n_states": len(a),
                "full_ordering_satisfied": float(full.mean()),
                "D_gt_A": float((d > a).mean()),
                "D_gt_C": float((d > c).mean()),
                "A_gt_B": float((a > b).mean()),
                "C_gt_B": float((c > b).mean()),
            }
        )

    csv_path = os.path.join(args.out_dir, "ordering_over_checkpoints.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.asarray([row["checkpoint"] for row in rows])
    y = 100 * np.asarray([row["full_ordering_satisfied"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(x, y, marker="o", markersize=4, linewidth=2, color="#0f766e")
    ax.set_xscale("log")
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Training environment steps")
    ax.set_ylabel("Full ordering satisfied (%)")
    ax.set_title("Emergence of diamond-pickaxe value ordering")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(os.path.join(args.out_dir, "ordering_over_checkpoints.png"), dpi=300)
    fig.savefig(os.path.join(args.out_dir, "ordering_over_checkpoints.pdf"))
    print(f"Saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()

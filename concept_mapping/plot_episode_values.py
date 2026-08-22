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
import matplotlib.pyplot as plt
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)


def _rollout_episode(args, network, train_state, env, env_params):
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
    target_records = None
    for episode in range(args.episode + 1):
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, env_state = reset_jit(reset_rng, env_params)
        records = []
        for step in range(args.max_steps):
            action, value, policy_rng = policy_apply(train_state.params, obs, policy_rng)
            action_i = int(np.asarray(action))
            value_f = float(np.asarray(value))
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, _info = step_jit(
                step_rng, env_state, action, env_params
            )
            reward_f = float(np.asarray(reward))
            done_b = bool(np.asarray(done))
            records.append(
                {
                    "step": step,
                    "value": value_f,
                    "action": action_i,
                    "action_name": _action_name(action_i),
                    "reward": reward_f,
                    "done": done_b,
                }
            )
            obs = next_obs
            if done_b:
                break
        if episode == args.episode:
            target_records = records
            break
    if target_records is None:
        raise RuntimeError(f"Could not roll out episode {args.episode}")
    return target_records


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.shape[0] < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _plot(records: list[dict], out_path: str, title: str, smooth_window: int) -> None:
    steps = np.asarray([r["step"] for r in records])
    values = np.asarray([r["value"] for r in records], dtype=np.float32)
    rewards = np.asarray([r["reward"] for r in records], dtype=np.float32)
    done = np.asarray([r["done"] for r in records], dtype=bool)
    smooth_values = _moving_average(values, smooth_window)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D7DCE2",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    ax.plot(steps, values, color="#93A4B8", linewidth=1.0, alpha=0.45, label="raw V(s)")
    ax.plot(steps, smooth_values, color="#175CD3", linewidth=2.6, label=f"V(s), smoothed")
    ax.fill_between(steps, smooth_values, np.minimum(values.min(), smooth_values.min()), color="#175CD3", alpha=0.08)

    positive_idx = np.where(rewards > 0)[0]
    negative_idx = np.where(rewards < 0)[0]
    if positive_idx.size:
        ax.scatter(
            steps[positive_idx],
            smooth_values[positive_idx],
            s=34,
            color="#12B76A",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="positive reward",
        )
    if negative_idx.size:
        ax.scatter(
            steps[negative_idx],
            smooth_values[negative_idx],
            s=34,
            color="#F04438",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="negative reward",
        )
    done_idx = np.where(done)[0]
    if done_idx.size:
        ax.axvline(steps[done_idx[0]], color="#111827", linewidth=1.2, linestyle="--", alpha=0.65)
        ax.text(
            steps[done_idx[0]],
            ax.get_ylim()[1],
            " done",
            va="top",
            ha="left",
            color="#111827",
            fontsize=10,
        )

    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Critic value V(s)")
    ax.grid(axis="y", color="#E8ECF2", linewidth=0.9)
    ax.grid(axis="x", color="#F3F5F8", linewidth=0.6)
    ax.legend(loc="upper right", frameon=False, ncol=3)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO critic values over one reproduced episode.")
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/episode_value_plots")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=19)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--title", default="Critic Value Over Episode")
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
    records = _rollout_episode(args, network, train_state, env, env_params)

    csv_path = os.path.join(args.out_dir, f"episode_{args.episode:02d}_values.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "value", "action", "action_name", "reward", "done"],
        )
        writer.writeheader()
        writer.writerows(records)

    png_path = os.path.join(args.out_dir, f"episode_{args.episode:02d}_values.png")
    _plot(records, png_path, args.title, args.smooth_window)

    values = np.asarray([r["value"] for r in records], dtype=np.float32)
    rewards = np.asarray([r["reward"] for r in records], dtype=np.float32)
    print(f"checkpoint_step={checkpoint_step}", flush=True)
    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved figure: {png_path}", flush=True)
    print(f"steps={len(records)} min_value={values.min():.4f} max_value={values.max():.4f}", flush=True)
    print(f"total_reward={rewards.sum():.4f}", flush=True)


if __name__ == "__main__":
    main()

"""Fast vanilla PPO on slightly delayed CartPole.

Dense CartPole pays +1 on every step, so GAE has almost nothing to do and a
richer advantage will not show up. This script keeps the CartPole dynamics and
holds those +1 rewards, releasing them as one lump every ``--reward-delay``
steps and always on termination. Undiscounted episode return is unchanged, so
the solve bar stays the Gymnasium CartPole-v1 bar: mean return of the last 100
episodes at or above 475.

Delay 1 is ordinary dense CartPole. Larger delays make credit assignment the
thing that has to work, which is the regime these advantage experiments are for.

The default batch is 4 environments by 128 steps, the usual CartPole PPO setup.
A much larger batch takes fewer updates before the learning rate and the value
clip run out, and the return stalls well below the solve bar.

Efficiency is written to the run directory: env steps and wall time at the
moment the solve bar is crossed, plus the learning curve.

    conda activate craftax
    python advantage_experiments/ppo_delayed_cartpole.py
    python advantage_experiments/ppo_delayed_cartpole.py --reward-delay 1
    python advantage_experiments/watch_cartpole.py --run advantage_experiments/runs/gae_delay20_seed0
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


def _configure_runtime_from_argv(argv: list[str]) -> None:
    """Pick CPU or one GPU before JAX initializes and grabs every device."""
    device = "cpu"
    gpu = "0"
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        elif arg == "--gpu" and i + 1 < len(argv):
            gpu = argv[i + 1]
            i += 2
            continue
        elif arg.startswith("--gpu="):
            gpu = arg.split("=", 1)[1]
        i += 1
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
    else:
        raise SystemExit(f"Unknown --device {device!r}. Use cpu or gpu.")


_configure_runtime_from_argv(sys.argv[1:])

import jax
import jax.numpy as jnp
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flax.linen as nn
import distrax
import gymnax

from advantage import ADVANTAGE_FNS, Rollout


class ActorCritic(nn.Module):
    action_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        actor = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor)
        actor = nn.tanh(actor)
        logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor)

        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.tanh(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)


def release_delayed_reward(true_reward, done, acc, steps, delay: int):
    """Hold +1 rewards and release the sum every ``delay`` steps, or on done.

    The sum of released rewards over a finished episode equals the sum of the
    raw rewards, so the task score does not change. Only the timing does.
    """
    acc = acc + true_reward
    steps = steps + 1
    release = jnp.logical_or(steps >= delay, done)
    reward = jnp.where(release, acc, jnp.zeros_like(acc))
    acc = jnp.where(release, jnp.zeros_like(acc), acc)
    steps = jnp.where(release, jnp.zeros_like(steps), steps)
    return reward, acc, steps


def _check_delay_conserves_return() -> None:
    true = jnp.ones((6,), dtype=jnp.float32)
    done = jnp.array([False, False, False, True, False, True])
    acc = jnp.array(0.0, dtype=jnp.float32)
    steps = jnp.array(0, dtype=jnp.int32)
    released = []
    for t in range(true.shape[0]):
        reward, acc, steps = release_delayed_reward(true[t], done[t], acc, steps, 2)
        released.append(reward)
    released = jnp.stack(released)
    if not jnp.allclose(released.sum(), true.sum()):
        raise RuntimeError("Reward delay dropped return mass.")
    if not jnp.allclose(released, jnp.array([0, 2, 0, 2, 0, 2], dtype=jnp.float32)):
        raise RuntimeError(f"Unexpected delay schedule: {released}")


def save_policy(path: Path, params, args) -> None:
    """Write the actor-critic weights next to the run metrics."""

    def _plain(tree):
        if hasattr(tree, "unfreeze"):
            tree = tree.unfreeze()
        if isinstance(tree, dict):
            return {str(key): _plain(value) for key, value in tree.items()}
        return np.asarray(tree)

    payload = {
        "params": _plain(params),
        "hidden_size": args.hidden_size,
        "action_dim": 2,
        "reward_delay": args.reward_delay,
        "seed": args.seed,
        "advantage": args.advantage,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--reward-delay", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lookahead-horizon", type=int, default=80)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--advantage", choices=sorted(ADVANTAGE_FNS), default="gae")
    parser.add_argument("--solve-threshold", type=float, default=475.0)
    parser.add_argument("--solve-window", type=int, default=100)
    parser.add_argument(
        "--anneal-lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--stop-on-solve", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    batch = args.num_envs * args.num_steps
    if args.reward_delay < 1:
        parser.error("--reward-delay must be >= 1")
    if args.lookahead_horizon < 1:
        parser.error("--lookahead-horizon must be >= 1")
    if args.group_size < 1:
        parser.error("--group-size must be >= 1")
    if args.total_timesteps < batch:
        parser.error("--total-timesteps must cover at least one rollout")
    if batch % args.num_minibatches != 0:
        parser.error("num-envs * num-steps must be divisible by num-minibatches")
    return args


def make_train_step(args, env, env_params, vmap_step):
    network = ActorCritic(env.num_actions, args.hidden_size)

    def advantage_fn(rollout, last_values, *, gamma, gae_lambda, params, rng):
        return ADVANTAGE_FNS[args.advantage](
            rollout,
            last_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
            params=params,
            rng=rng,
            network=network,
            env=env,
            env_params=env_params,
            horizon=args.lookahead_horizon,
            group_size=args.group_size,
        )

    batch_size = args.num_steps * args.num_envs
    total_updates = args.total_timesteps // batch_size
    grad_steps_per_update = args.num_minibatches * args.update_epochs

    def linear_schedule(count):
        update = count // grad_steps_per_update
        frac = 1.0 - update / total_updates
        return args.lr * jnp.maximum(frac, 0.0)

    def init_state(rng):
        rng, net_rng, reset_rng = jax.random.split(rng, 3)
        params = network.init(net_rng, jnp.zeros((1, 4)))
        if args.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        reset_keys = jax.random.split(reset_rng, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        acc = jnp.zeros((args.num_envs,), dtype=jnp.float32)
        steps_since_release = jnp.zeros((args.num_envs,), dtype=jnp.int32)
        return train_state, env_state, obs, acc, steps_since_release, rng

    def _loss(params, batch, advantages, targets):
        pi, value = network.apply(params, batch.obs)
        log_prob = pi.log_prob(batch.actions)
        logratio = log_prob - batch.log_probs
        ratio = jnp.exp(logratio)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
        actor_loss = -jnp.minimum(unclipped, clipped).mean()

        # The lookahead target is an H-step return, often much larger than the
        # 0.2 clip. Clipping would keep the critic from using that target.
        if args.advantage == "group_lookahead":
            value_loss = 0.5 * jnp.square(value - targets).mean()
        else:
            value_clipped = batch.values + (value - batch.values).clip(
                -args.clip_eps, args.clip_eps
            )
            value_loss = (
                0.5
                * jnp.maximum(
                    jnp.square(value - targets), jnp.square(value_clipped - targets)
                ).mean()
            )
        entropy = pi.entropy().mean()
        loss = actor_loss + args.vf_coef * value_loss - args.ent_coef * entropy
        approx_kl = ((ratio - 1.0) - logratio).mean()
        return loss, (actor_loss, value_loss, entropy, approx_kl)

    def train_step(train_state, env_state, obs, acc, steps_since_release, rng):
        def _env_step(carry, _):
            train_state, env_state, obs, acc, steps_since_release, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            pi, value = network.apply(train_state.params, obs)
            action = pi.sample(seed=action_rng)
            log_prob = pi.log_prob(action)
            step_keys = jax.random.split(step_rng, args.num_envs)
            next_obs, env_state, true_reward, done, _info = vmap_step(
                step_keys, env_state, action, env_params
            )
            true_reward = true_reward.astype(jnp.float32)
            done_f = done.astype(jnp.float32)
            reward, acc, steps_since_release = release_delayed_reward(
                true_reward, done, acc, steps_since_release, args.reward_delay
            )
            transition = Rollout(
                obs=obs,
                actions=action,
                rewards=reward,
                true_rewards=true_reward,
                dones=done_f,
                values=jax.lax.stop_gradient(value),
                log_probs=jax.lax.stop_gradient(log_prob),
            )
            return (
                train_state,
                env_state,
                next_obs,
                acc,
                steps_since_release,
                rng,
            ), transition

        (
            train_state,
            env_state,
            obs,
            acc,
            steps_since_release,
            rng,
        ), rollout = jax.lax.scan(
            _env_step,
            (train_state, env_state, obs, acc, steps_since_release, rng),
            None,
            args.num_steps,
        )

        _, last_values = network.apply(train_state.params, obs)
        last_values = jax.lax.stop_gradient(last_values)
        rng, adv_rng = jax.random.split(rng)
        advantages, targets, rollout = advantage_fn(
            rollout,
            last_values,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            params=jax.lax.stop_gradient(train_state.params),
            rng=adv_rng,
        )
        advantages = jax.lax.stop_gradient(advantages)
        targets = jax.lax.stop_gradient(targets)

        def _flatten(tree):
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), tree
            )

        flat_rollout = _flatten(rollout)
        flat_adv = advantages.reshape((batch_size,))
        flat_targets = targets.reshape((batch_size,))

        def _minibatch(train_state, minibatch):
            batch, adv, tgt = minibatch

            def loss_fn(params):
                return _loss(params, batch, adv, tgt)

            (_loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, aux

        def _epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)
            shuffled = (
                jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), flat_rollout
                ),
                jnp.take(flat_adv, permutation, axis=0),
                jnp.take(flat_targets, permutation, axis=0),
            )

            def _split(x):
                return x.reshape((args.num_minibatches, -1) + x.shape[1:])

            minibatches = jax.tree_util.tree_map(_split, shuffled)
            train_state, aux = jax.lax.scan(_minibatch, train_state, minibatches)
            return (train_state, rng), aux

        (train_state, rng), aux = jax.lax.scan(
            _epoch, (train_state, rng), None, args.update_epochs
        )
        actor_loss, value_loss, entropy, approx_kl = aux
        metrics = {
            "actor_loss": actor_loss.mean(),
            "value_loss": value_loss.mean(),
            "entropy": entropy.mean(),
            "approx_kl": approx_kl.mean(),
            "nonzero_reward_frac": jnp.mean(rollout.rewards != 0),
        }
        return (
            train_state,
            env_state,
            obs,
            acc,
            steps_since_release,
            rng,
            metrics,
            rollout.true_rewards,
            rollout.dones,
        )

    return init_state, jax.jit(train_step), total_updates


def _window_mean(window: deque) -> float:
    if not window:
        return float("nan")
    return float(sum(window) / len(window))


def _record_episodes(
    true_rewards,
    dones,
    partial,
    window,
    episodes,
    env_steps_before,
    wall_s,
    solve_window,
    solve_threshold,
    already_solved,
):
    """Fold a rollout into completed-episode returns. ``partial`` spans updates.

    Returns the first episode that pushes the solve window over the threshold,
    or None. Wall time is the end of this update, since that is when the
    returns become available to the loop.
    """
    crossed = None
    num_steps, num_envs = true_rewards.shape
    for t in range(num_steps):
        partial += true_rewards[t]
        finished = np.flatnonzero(dones[t] > 0.5)
        step_at = env_steps_before + (t + 1) * num_envs
        for index in finished:
            episode_return = float(partial[index])
            episodes.append(
                {
                    "episode": len(episodes) + 1,
                    "env_steps": int(step_at),
                    "return": episode_return,
                    "wall_s": wall_s,
                }
            )
            window.append(episode_return)
            partial[index] = 0.0
            if (
                crossed is None
                and not already_solved
                and len(window) == solve_window
                and _window_mean(window) >= solve_threshold
            ):
                crossed = {
                    "env_steps_to_solve": int(step_at),
                    "episodes_to_solve": len(episodes),
                    "wall_seconds_to_solve": wall_s,
                    "mean_return_100": _window_mean(window),
                }
    return crossed


def _write_curve(rows, path: Path, threshold: float, solved_steps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["env_steps"] for row in rows]
    means = [row["mean_return_100"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(steps, means, color="#1f4e79", lw=1.8)
    ax.axhline(
        threshold, color="#444444", ls="--", lw=1.0, label=f"solve at {threshold:g}"
    )
    if solved_steps is not None:
        ax.axvline(
            solved_steps,
            color="#b45309",
            ls=":",
            lw=1.2,
            label=f"solved at {solved_steps:,} steps",
        )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean return, last 100 episodes")
    ax.set_ylim(0, 520)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    _check_delay_conserves_return()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        out_dir = (
            Path(__file__).resolve().parent
            / "runs"
            / (f"{stamp}_delay{args.reward_delay}_seed{args.seed}")
        )
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env, env_params = gymnax.make("CartPole-v1")
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    init_state, train_step, total_updates = make_train_step(
        args, env, env_params, vmap_step
    )

    print(
        f"PPO CartPole  advantage={args.advantage}  reward_delay={args.reward_delay}  "
        f"device={jax.devices()[0]}  budget={args.total_timesteps:,} steps"
    )
    print(
        f"Solve when the mean of {args.solve_window} episodes reaches {args.solve_threshold:g}."
    )
    print(f"Writing {out_dir}")

    rng = jax.random.PRNGKey(args.seed)
    train_state, env_state, obs, acc, steps_since_release, rng = init_state(rng)
    # Touch the params so init is finished before the training clock starts.
    jax.block_until_ready(train_state.params)

    batch = args.num_envs * args.num_steps
    window = deque(maxlen=args.solve_window)
    partial = np.zeros((args.num_envs,), dtype=np.float32)
    episodes = []
    rows = []
    solved = False
    solve_record = None

    loop_start = time.perf_counter()
    first_update_seconds = None

    for update in range(1, total_updates + 1):
        (
            train_state,
            env_state,
            obs,
            acc,
            steps_since_release,
            rng,
            metrics,
            true_rewards,
            dones,
        ) = train_step(train_state, env_state, obs, acc, steps_since_release, rng)
        jax.block_until_ready(metrics["actor_loss"])
        wall_s = time.perf_counter() - loop_start
        if first_update_seconds is None:
            first_update_seconds = wall_s

        true_rewards = np.asarray(true_rewards)
        dones = np.asarray(dones)
        env_steps = update * batch
        crossed = _record_episodes(
            true_rewards,
            dones,
            partial,
            window,
            episodes,
            env_steps - batch,
            wall_s,
            args.solve_window,
            args.solve_threshold,
            solved,
        )
        mean_return = _window_mean(window)
        if crossed is not None:
            solved = True
            solve_record = {
                **crossed,
                "updates_to_solve": update,
                "train_seconds_to_solve_excluding_first_update": wall_s
                - first_update_seconds,
            }
            print(
                f"SOLVED  update {update}  steps {crossed['env_steps_to_solve']:,}  "
                f"episodes {crossed['episodes_to_solve']}  "
                f"mean{args.solve_window} {crossed['mean_return_100']:.1f}  "
                f"wall {wall_s:.2f}s"
            )
        row = {
            "update": update,
            "env_steps": env_steps,
            "wall_s": wall_s,
            "episodes": len(episodes),
            "window_episodes": len(window),
            "mean_return_100": mean_return,
            "nonzero_reward_frac": float(metrics["nonzero_reward_frac"]),
            "actor_loss": float(metrics["actor_loss"]),
            "value_loss": float(metrics["value_loss"]),
            "entropy": float(metrics["entropy"]),
            "approx_kl": float(metrics["approx_kl"]),
            "solved": int(solved),
        }
        rows.append(row)
        if update == 1 or update % args.log_interval == 0 or crossed is not None:
            mean_text = "n/a" if np.isnan(mean_return) else f"{mean_return:.1f}"
            print(
                f"update {update:4d}/{total_updates}  steps {env_steps:7d}  "
                f"wall {wall_s:7.2f}s  episodes {len(episodes):5d}  "
                f"mean{args.solve_window} {mean_text:>7}  "
                f"kl {row['approx_kl']:.4f}  reward_nz {row['nonzero_reward_frac']:.3f}"
            )
        if solved and args.stop_on_solve:
            break

    wall_total = time.perf_counter() - loop_start
    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    episodes_path = out_dir / "episodes.csv"
    with episodes_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["episode", "env_steps", "return", "wall_s"]
        )
        writer.writeheader()
        writer.writerows(episodes)

    policy_path = out_dir / "policy.pkl"
    save_policy(policy_path, train_state.params, args)

    summary = {
        "env": "CartPole-v1",
        "policy_path": str(policy_path),
        "advantage": args.advantage,
        "reward_delay": args.reward_delay,
        "seed": args.seed,
        "device": str(jax.devices()[0]),
        "solve_threshold": args.solve_threshold,
        "solve_window": args.solve_window,
        "solved": solved,
        "first_update_seconds": first_update_seconds,
        "total_env_steps": rows[-1]["env_steps"],
        "total_episodes": len(episodes),
        "total_wall_seconds": wall_total,
        "env_steps_per_second": rows[-1]["env_steps"] / wall_total,
        "final_mean_return_100": rows[-1]["mean_return_100"],
        "config": vars(args),
        **(solve_record or {}),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        _write_curve(
            rows,
            out_dir / "return_curve.png",
            args.solve_threshold,
            solve_record["env_steps_to_solve"] if solve_record else None,
        )
    except Exception as exc:
        print(f"Skipping plot: {exc}")

    print(
        f"Done. solved={solved}  steps={summary['total_env_steps']:,}  "
        f"wall={wall_total:.2f}s  first_update={first_update_seconds:.2f}s  "
        f"final_mean{args.solve_window}={summary['final_mean_return_100']:.1f}"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

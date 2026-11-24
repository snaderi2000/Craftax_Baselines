"""Render a saved CartPole policy to an mp4.

The trainer does not keep weights unless it has been run with the current
script, which writes ``policy.pkl`` into the run directory.

    conda activate craftax
    python advantage_experiments/watch_cartpole.py --run advantage_experiments/runs/gae_delay20_seed0
"""

import argparse
import pickle
import sys
from pathlib import Path

import ppo_delayed_cartpole  # noqa: F401  sets the JAX platform before jax imports

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

import gymnasium as gym
import gymnax

from ppo_delayed_cartpole import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "gae_delay20_seed0"),
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def load_policy(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    network = ActorCritic(int(payload["action_dim"]), int(payload["hidden_size"]))
    return network, params, payload


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(255, 255, 255))
    draw.text((10, 6), text, fill=(0, 0, 0))
    return np.asarray(image)


def _render(gym_env, state) -> np.ndarray:
    gym_env.unwrapped.state = np.array(
        [state.x, state.x_dot, state.theta, state.theta_dot], dtype=np.float64
    )
    return np.asarray(gym_env.render())


def play_episode(network, params, env, env_params, gym_env, rng, episode_index: int):
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset_env(reset_rng, env_params)
    frames = []
    episode_return = 0.0
    step = 0
    while True:
        step += 1
        label = f"episode {episode_index}   step {step}   return {episode_return:.0f}"
        frames.append(_caption(_render(gym_env, state), label))
        pi, _value = network.apply(params, obs[None, :])
        action = jnp.argmax(pi.logits)
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, _info = env.step_env(
            step_rng, state, action, env_params
        )
        episode_return += float(reward)
        if bool(done):
            end = (
                "survived"
                if int(state.time) >= int(env_params.max_steps_in_episode)
                else "fell"
            )
            frames.append(
                _caption(
                    _render(gym_env, state),
                    f"episode {episode_index}   {end}   return {episode_return:.0f}",
                )
            )
            break
    return frames, episode_return, rng


def main():
    args = parse_args()
    run_dir = Path(args.run)
    policy_path = run_dir / "policy.pkl"
    if not policy_path.exists():
        sys.exit(
            f"No policy at {policy_path}. Train first:\n"
            "  python advantage_experiments/ppo_delayed_cartpole.py "
            f"--out-dir {run_dir}"
        )
    out_path = Path(args.out) if args.out else run_dir / "cartpole_play.mp4"

    network, params, payload = load_policy(policy_path)
    env, env_params = gymnax.make("CartPole-v1")
    gym_env = gym.make("CartPole-v1", render_mode="rgb_array")
    gym_env.reset(seed=args.seed)

    rng = jax.random.PRNGKey(args.seed)
    frames = []
    returns = []
    try:
        for episode in range(1, args.episodes + 1):
            episode_frames, episode_return, rng = play_episode(
                network, params, env, env_params, gym_env, rng, episode
            )
            returns.append(episode_return)
            if frames:
                hold = episode_frames[0].copy()
                frames.extend([hold] * args.fps)
            frames.extend(episode_frames)
            print(
                f"episode {episode}: return {episode_return:.0f}  steps {len(episode_frames) - 1}"
            )
    finally:
        gym_env.close()

    imageio.mimwrite(
        out_path,
        frames,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    print(
        f"Wrote {out_path}  "
        f"({len(frames)} frames, {len(frames) / args.fps:.1f}s, "
        f"advantage={payload.get('advantage')} delay={payload.get('reward_delay')})"
    )


if __name__ == "__main__":
    main()

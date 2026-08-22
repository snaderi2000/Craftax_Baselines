import argparse
import os
import sys
from typing import Any

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
import optax
import orbax.checkpoint as ocp
import yaml
from flax.training.train_state import TrainState
from PIL import Image, ImageDraw

from craftax.craftax_classic.constants import load_all_textures
from craftax.craftax_env import make_craftax_env_from_name
from ppo_rnn_og import ActorCriticRNN, ScannedRNN

from concept_mapping.visualize_symbolic_trajectory import (
    _action_name,
    render_local_view,
)


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


def _init_rnn_train_state(config: dict[str, Any], network, env, env_params) -> TrainState:
    obs_shape = env.observation_space(env_params).shape
    init_x = (
        jnp.zeros((1, 1, *obs_shape), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=bool),
    )
    init_hstate = ScannedRNN.initialize_carry(1, int(config["LAYER_SIZE"]))
    params = network.init(jax.random.PRNGKey(0), init_hstate, init_x)
    tx = optax.chain(
        optax.clip_by_global_norm(float(config.get("MAX_GRAD_NORM", 1.0))),
        optax.adam(float(config.get("LR", 2e-4)), eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def _restore_train_state(
    run_path: str,
    config: dict[str, Any],
    network,
    env,
    env_params,
    timestep: int | None,
) -> tuple[TrainState, int]:
    train_state = _init_rnn_train_state(config, network, env, env_params)
    ckpt_dir = os.path.join(_files_dir(run_path), "policies")
    manager = ocp.CheckpointManager(ckpt_dir, ocp.PyTreeCheckpointer())
    step = manager.latest_step() if timestep is None else timestep
    if step is None:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    print(f"Restoring RNN policy checkpoint step={step} from {ckpt_dir}", flush=True)
    return manager.restore(step, items=train_state), int(step)


def _label_frame(
    obs: np.ndarray,
    action: int,
    reward: float,
    done: bool,
    value: float,
    step: int,
    episode: int,
    block_size: int,
    display_scale: int,
    textures: dict,
    panel_textures: dict,
) -> Image.Image:
    view = render_local_view(
        obs,
        block_size=block_size,
        include_inventory=True,
        display_scale=display_scale,
        textures=textures,
        panel_textures=panel_textures,
    )
    label_h = max(30, 6 * display_scale + 24)
    out = Image.new("RGB", (view.width, view.height + label_h), "white")
    out.paste(view, (0, 0))
    draw = ImageDraw.Draw(out)
    label = (
        f"episode={episode} step={step} action={action}:{_action_name(action)} "
        f"reward={reward:.3f} V={value:.3f} done={done}"
    )
    draw.text((6, view.height + 6), label, fill=(0, 0, 0))
    return out


def _write_mp4(frames: list[Image.Image], path: str, fps: int, ffmpeg_path: str | None) -> None:
    import matplotlib as mpl
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    if ffmpeg_path:
        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
    height, width = arrays[0].shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    image = ax.imshow(arrays[0])

    def update(i):
        image.set_data(arrays[i])
        return (image,)

    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=-1)
    ani = animation.FuncAnimation(fig, update, frames=len(arrays), interval=1000 / fps, blit=True)
    ani.save(path, writer=writer)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render trained PPO-RNN symbolic policy episodes to MP4."
    )
    parser.add_argument("--run_path", default="wandb/run-20260606_170458-1b1neu4j")
    parser.add_argument("--out_dir", default="concept_mapping/runs/ppo_rnn_10b_episode_videos")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--ffmpeg_path", default="/home/shawheen/miniconda3/bin/ffmpeg")
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax actions instead of sampling from the policy.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = _load_wandb_config(args.run_path)
    config = {k.upper(): v for k, v in config.items()}
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    if "Symbolic" not in env_name:
        raise ValueError(f"This renderer expects a symbolic policy, got {env_name}")
    config["ENV_NAME"] = env_name

    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = ActorCriticRNN(env.action_space(env_params).n, config=config)
    train_state, restored_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )
    print(f"env={env_name} checkpoint_step={restored_step}", flush=True)

    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )

    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    @jax.jit
    def policy_apply(params, hstate, obs, done, rng):
        ac_in = (obs[None, None, ...], done.reshape(1, 1))
        hstate, pi, value = network.apply(params, hstate, ac_in)
        if args.greedy:
            action = jnp.argmax(pi.logits[0, 0])
            return hstate, action, value[0, 0], rng
        rng, action_rng = jax.random.split(rng)
        return hstate, pi.sample(seed=action_rng)[0, 0], value[0, 0], rng

    rng = jax.random.PRNGKey(args.seed)
    summary_path = os.path.join(args.out_dir, "episode_summary.csv")
    with open(summary_path, "w") as summary:
        summary.write("episode,frames,total_reward,done,last_step,mp4,actions\n")
        for episode in range(args.episodes):
            print(f"Rolling out episode {episode}/{args.episodes - 1}", flush=True)
            rng, reset_rng, policy_rng = jax.random.split(rng, 3)
            obs, env_state = reset_jit(reset_rng, env_params)
            hstate = ScannedRNN.initialize_carry(1, int(config["LAYER_SIZE"]))
            last_done = jnp.zeros((), dtype=bool)
            frames = []
            total_reward = 0.0
            done_b = False
            last_step = -1

            actions_path = os.path.join(args.out_dir, f"episode_{episode:02d}_actions.txt")
            with open(actions_path, "w") as f:
                f.write("step,action,action_name,reward,value,done\n")
                for step in range(args.max_steps):
                    hstate, action, value, policy_rng = policy_apply(
                        train_state.params, hstate, obs, last_done, policy_rng
                    )
                    action_i = int(np.asarray(action))
                    value_f = float(np.asarray(value))
                    rng, step_rng = jax.random.split(rng)
                    next_obs, env_state, reward, done, _info = step_jit(
                        step_rng, env_state, action, env_params
                    )
                    reward_f = float(np.asarray(reward))
                    done_b = bool(np.asarray(done))
                    total_reward += reward_f
                    last_step = step
                    frames.append(
                        _label_frame(
                            np.asarray(obs),
                            action_i,
                            reward_f,
                            done_b,
                            value_f,
                            step,
                            episode,
                            args.block_size,
                            args.display_scale,
                            textures,
                            panel_textures,
                        )
                    )
                    f.write(
                        f"{step},{action_i},{_action_name(action_i)},"
                        f"{reward_f:.6f},{value_f:.6f},{done_b}\n"
                    )
                    obs = next_obs
                    last_done = done
                    if done_b:
                        break

            mp4_path = os.path.join(args.out_dir, f"episode_{episode:02d}.mp4")
            print(f"Encoding {mp4_path}", flush=True)
            _write_mp4(frames, mp4_path, args.fps, args.ffmpeg_path)
            summary.write(
                f"{episode},{len(frames)},{total_reward:.6f},{done_b},{last_step},"
                f"{mp4_path},{actions_path}\n"
            )
            summary.flush()
            print(f"Saved {mp4_path} ({len(frames)} frames, reward={total_reward:.3f})")
            print(f"Saved {actions_path}")

    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

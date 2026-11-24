import argparse
import hashlib
import json
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

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import load_all_textures
from models.actor_critic import ActorCritic, ActorCriticConv

from concept_mapping.visualize_symbolic_trajectory import (
    _action_name,
    render_local_view,
)


def _load_wandb_config(run_path: str) -> dict[str, Any]:
    files_dir = run_path if os.path.basename(run_path) == "files" else os.path.join(run_path, "files")
    config_path = os.path.join(files_dir, "config.yaml")
    if not os.path.exists(config_path):
        explicit_path = os.path.join(files_dir, 'study_config.json')
        if os.path.exists(explicit_path):
            with open(explicit_path) as f:
                return json.load(f)
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
        if config.get('SYMBOLIC_ENCODER')=='map_mlp':
            from models.actor_critic_map_mlp import map_mlp_network
            return map_mlp_network(config,action_dim)
        if config.get('SYMBOLIC_ENCODER', 'mlp') == 'spatial':
            from models.conditioned_symbolic import ConditionedSymbolicActorCritic
            return ConditionedSymbolicActorCritic(action_dim)
        return ActorCritic(action_dim, layer_size)
    return ActorCriticConv(action_dim, layer_size)


def _restore_train_state(
    run_path: str,
    config: dict[str, Any],
    network,
    env,
    env_params,
    timestep: int | None,
):
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
    print(f"Restoring policy checkpoint step={step} from {ckpt_dir}", flush=True)
    return manager.restore(step, items=train_state), int(step)


def _label_frame(
    obs: np.ndarray,
    action: int,
    reward: float,
    done: bool,
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
    label_h = max(48, 5 * display_scale + 22)
    out = Image.new("RGB", (view.width, view.height + label_h), "white")
    out.paste(view, (0, 0))
    draw = ImageDraw.Draw(out)
    label = (
        f"episode={episode} step={step}\n"
        f"action={_action_name(action)} reward={reward:.3f}\n"
        f"done={done}"
    )
    draw.multiline_text((6, view.height + 6), label, fill=(0, 0, 0), spacing=3)
    return out


def _write_mp4(frames: list[Image.Image], path: str, fps: int) -> None:
    try:
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "MP4 writing needs matplotlib with ffmpeg available in this environment."
        ) from exc

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
        description="Render trained PPO policy episodes from symbolic observations to MP4."
    )
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/ppo_10b_episode_videos")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--independent_episode_seeds", action="store_true",
                        help="Fold episode index into seed so policies get matched starts regardless of prior episode lengths.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--death_penalty",type=float,default=None,
                        help="Death cost magnitude; defaults to the saved training configuration.")
    parser.add_argument(
        "--greedy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax actions instead of sampling from the policy.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    if "Symbolic" not in env_name:
        raise ValueError(f"This renderer expects a symbolic policy, got {env_name}")
    config["ENV_NAME"] = env_name

    env = make_craftax_env_from_name(env_name, auto_reset=False)
    death_penalty=float(config.get('DEATH_PENALTY',0.) if args.death_penalty is None else args.death_penalty)
    if death_penalty:
        from wrappers import ClassicDeathPenaltyWrapper
        env=ClassicDeathPenaltyWrapper(env,death_penalty)
    if config.get('LAVA_PENALTY',0.):
        from wrappers import ClassicLavaPenaltyWrapper
        env=ClassicLavaPenaltyWrapper(env,config['LAVA_PENALTY'])
    if config.get('DIAMOND_REWARD', 1.) != 1. or config.get('PLANT_REWARD', 1.) != 1.:
        from wrappers import ClassicAchievementRewardWrapper
        env=ClassicAchievementRewardWrapper(env, config.get('DIAMOND_REWARD', 1.), config.get('PLANT_REWARD', 1.))
    env_params = env.default_params
    network = _init_network(config, env, env_params)
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
    def policy_apply(params, obs, rng):
        pi, _ = network.apply(params, obs[None, ...])
        if args.greedy:
            action = jnp.argmax(pi.logits[0])
            return action, rng
        rng, action_rng = jax.random.split(rng)
        return pi.sample(seed=action_rng)[0], rng

    rng = jax.random.PRNGKey(args.seed)
    summaries=[]
    for episode in range(args.episodes):
        print(f"Rolling out episode {episode}/{args.episodes - 1}", flush=True)
        if args.independent_episode_seeds:
            rng = jax.random.fold_in(jax.random.PRNGKey(args.seed),episode)
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, env_state = reset_jit(reset_rng, env_params)
        initial_map_sha256 = hashlib.sha256(np.asarray(env_state.map).tobytes()).hexdigest()
        initial_position = np.asarray(env_state.player_position).tolist()
        frames = []
        total_reward=0.
        actions_path = os.path.join(args.out_dir, f"episode_{episode:02d}_actions.txt")
        with open(actions_path, "w") as f:
            f.write("step,action,action_name,reward,done\n")
            for step in range(args.max_steps):
                action, policy_rng = policy_apply(train_state.params, obs, policy_rng)
                action_i = int(np.asarray(action))
                rng, step_rng = jax.random.split(rng)
                next_obs, env_state, reward, done, _info = step_jit(
                    step_rng, env_state, action, env_params
                )
                reward_f = float(np.asarray(reward))
                total_reward+=reward_f
                done_b = bool(np.asarray(done))
                frames.append(
                    _label_frame(
                        np.asarray(obs),
                        action_i,
                        reward_f,
                        done_b,
                        step,
                        episode,
                        args.block_size,
                        args.display_scale,
                        textures,
                        panel_textures,
                    )
                )
                f.write(f"{step},{action_i},{_action_name(action_i)},{reward_f:.6f},{done_b}\n")
                obs = next_obs
                if done_b:
                    break

        # Include the final post-action state as well as all pre-action frames.
        frames.append(_label_frame(np.asarray(obs),action_i,reward_f,done_b,step+1,episode,
                                   args.block_size,args.display_scale,textures,panel_textures))

        mp4_path = os.path.join(args.out_dir, f"episode_{episode:02d}.mp4")
        print(f"Encoding {mp4_path}", flush=True)
        _write_mp4(frames, mp4_path, args.fps)
        print(f"Saved {mp4_path} ({len(frames)} frames)")
        print(f"Saved {actions_path}")
        from craftax.craftax_classic.constants import BlockType,Achievement
        position=np.asarray(env_state.player_position)
        summaries.append(dict(episode=episode,steps=step+1,frames=len(frames),done=done_b,
            initial_map_sha256=initial_map_sha256,initial_position=initial_position,
            truncated_by_renderer=not done_b,total_reward=total_reward,
            collected_diamond=bool(np.asarray(env_state.achievements[Achievement.COLLECT_DIAMOND.value])),
            woke_up=bool(np.asarray(env_state.achievements[Achievement.WAKE_UP.value])),
            ended_in_lava=bool(np.asarray(env_state.map)[tuple(position)]==BlockType.LAVA.value),
            final_health=int(np.asarray(env_state.player_health)),
            mp4=mp4_path))
        with open(os.path.join(args.out_dir,'summary.json'),'w') as f:
            json.dump(dict(run=args.run_path,checkpoint=restored_step,seed=args.seed,
                           independent_episode_seeds=args.independent_episode_seeds,
                           death_penalty_magnitude=death_penalty,death_reward_cost=-death_penalty,
                           fps=args.fps,max_steps=args.max_steps,greedy=args.greedy,episodes=summaries),f,indent=2)


if __name__ == "__main__":
    main()

# Save as: analysis/record_one_episode_mp4_gpu_headless.py
#
# Headless (no monitor) policy rollout -> MP4, using your Orbax checkpoint.
# Designed to run on a GPU node under xvfb-run (virtual display).
#
# Usage example:
#   xvfb-run -s "-screen 0 1280x720x24" \
#   python analysis/record_one_episode_mp4_gpu_headless.py \
#     --path ~/craftax_runs/kp7gx38r --seed 42 --out videos/ep_seed42.mp4
#
import argparse
import os
import yaml
import numpy as np
import imageio.v2 as imageio

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions

from models.actor_critic import ActorCriticConv, ActorCritic


def load_config(run_path: str) -> dict:
    cfg_path = os.path.join(run_path, "config.yaml")
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
    # unwrap wandb-style {"value": ...}
    cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
    return cfg


def build_env_and_net(cfg: dict):
    env_name = cfg["ENV_NAME"]
    is_classic = False

    if env_name == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
        from craftax.craftax.constants import Action

        env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        net = ActorCriticConv(len(Action), cfg["LAYER_SIZE"])

    elif env_name == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        from craftax.craftax.constants import Action

        env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        net = ActorCritic(len(Action), cfg["LAYER_SIZE"])

    elif env_name == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        net = ActorCriticConv(len(Action), cfg["LAYER_SIZE"])
        is_classic = True

    elif env_name == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicSymbolicEnv(CraftaxClassicSymbolicEnv.default_static_params())
        net = ActorCritic(len(Action), cfg["LAYER_SIZE"])
        is_classic = True

    else:
        raise ValueError(f"Unknown ENV_NAME: {env_name}")

    env_params = env.default_params
    return env, env_params, net, is_classic


def restore_state(run_path: str, net, env, env_params, ckpt_step: int) -> TrainState:
    # Initialize parameter tree with correct shapes, then restore checkpoint onto it.
    dummy_obs = jnp.zeros((1, *env.observation_space(env_params).shape))
    init_rng = jax.random.PRNGKey(0)
    params = net.init(init_rng, dummy_obs)

    # Optimizer is irrelevant for inference; TrainState is just a container for params.
    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    mgr = CheckpointManager(
        os.path.join(run_path, "policies"),
        PyTreeCheckpointer(),
        CheckpointManagerOptions(create=False),
    )
    state = mgr.restore(ckpt_step, items=state)
    return state


def make_renderer(env, env_params, is_classic: bool):
    # NOTE: This usually requires a display; on headless use xvfb-run.
    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer

    return CraftaxRenderer(env, env_params, pixel_render_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Directory containing config.yaml and policies/")
    ap.add_argument("--seed", type=int, required=True, help="Seed for env reset + action sampling")
    ap.add_argument("--out", required=True, help="Output MP4 path (e.g., videos/ep_seed42.mp4)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--ckpt_step", type=int, default=None, help="Override checkpoint step (default: TOTAL_TIMESTEPS)")
    ap.add_argument("--deterministic", action="store_true", help="Use argmax action instead of sampling")
    args = ap.parse_args()

    cfg = load_config(args.path)
    ckpt_step = int(args.ckpt_step if args.ckpt_step is not None else cfg["TOTAL_TIMESTEPS"])

    env, env_params, net, is_classic = build_env_and_net(cfg)
    state = restore_state(args.path, net, env, env_params, ckpt_step)
    renderer = make_renderer(env, env_params, is_classic)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = imageio.get_writer(args.out, fps=args.fps)

    # Rollout: one episode
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(key=reset_rng)

    done = False
    steps = 0

    # initial frame
    writer.append_data(np.asarray(renderer.render(env_state)))

    while (not done) and (steps < args.max_steps):
        obs_b = jnp.expand_dims(obs, axis=0)
        pi, _ = net.apply(state.params, obs_b)

        rng, act_rng = jax.random.split(rng)
        if args.deterministic:
            probs = np.asarray(pi.probs)
            action = int(np.argmax(probs.reshape(-1)))
        else:
            action = int(pi.sample(seed=act_rng)[0])

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)

        writer.append_data(np.asarray(renderer.render(env_state)))
        steps += 1

    writer.close()
    print(f"Saved MP4: {args.out}")
    print(f"env={cfg['ENV_NAME']} ckpt_step={ckpt_step} seed={args.seed} steps={steps} done={done}")
    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")


if __name__ == "__main__":
    main()

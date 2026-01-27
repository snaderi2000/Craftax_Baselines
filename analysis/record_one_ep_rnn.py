import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import numpy as np
import imageio.v2 as imageio

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions

# IMPORTANT: import your RNN network from wherever ppo_rnn.py defines it.
# Change this import to match your repo layout.
# Example: from ppo_rnn import ActorCriticRNN
from ppo_rnn import ActorCriticRNN  # <-- adjust if your file lives elsewhere


def load_config(run_path: str) -> dict:
    with open(os.path.join(run_path, "config.yaml"), "r") as f:
        raw = yaml.safe_load(f)
    cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
    cfg["NUM_ENVS"] = 1
    return cfg


def build_env(cfg: dict):
    env_name = cfg["ENV_NAME"]
    is_classic = False

    if env_name == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
        from craftax.craftax.constants import Action
        env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        action_dim = len(Action)

    elif env_name == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
        from craftax.craftax_classic.constants import Action
        env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        action_dim = len(Action)
        is_classic = True

    else:
        raise ValueError(f"This recorder supports pixel envs only. Got ENV_NAME={env_name}")

    env_params = env.default_params
    return env, env_params, action_dim, is_classic


def make_renderer(env, env_params, is_classic: bool):
    # Requires Xvfb on headless servers
    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
    return CraftaxRenderer(env, env_params, pixel_render_size=1)


def restore_state(run_path: str, network, env, env_params, ckpt_step: int):
    # Build dummy inputs that match ActorCriticRNN signature:
    # __call__(hidden, (obs, dones)) where obs is (T,B,H,W,C) and dones is (T,B)
    # We'll use T=1, B=1 for initialization.
    obs_shape = env.observation_space(env_params).shape  # e.g. (63,63,3)
    dummy_obs = jnp.zeros((1, 1, *obs_shape), dtype=jnp.uint8)
    dummy_dones = jnp.zeros((1, 1), dtype=bool)

    # Hidden size: your code uses GRUCell(features=ins.shape[1]) where ins is (B, 256),
    # so the recurrent size is 256. We'll allocate (B, 256).
    dummy_hidden = jnp.zeros((1, 256), dtype=jnp.float32)

    rng = jax.random.PRNGKey(0)
    params = network.init(rng, dummy_hidden, (dummy_obs, dummy_dones))

    # Optimizer irrelevant for inference
    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    mgr = CheckpointManager(
        os.path.join(run_path, "policies"),
        PyTreeCheckpointer(),
        CheckpointManagerOptions(create=False),
    )
    state = mgr.restore(ckpt_step, items=state)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Directory containing config.yaml and policies/")
    ap.add_argument("--seed", type=int, required=True, help="Seed for reset + action sampling")
    ap.add_argument("--out", required=True, help="Output mp4 path")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--ckpt_step", type=int, default=None, help="Override checkpoint step")
    ap.add_argument("--deterministic", action="store_true", help="Use argmax action (no sampling)")
    args = ap.parse_args()

    cfg = load_config(args.path)
    ckpt_step = int(args.ckpt_step if args.ckpt_step is not None else cfg["TOTAL_TIMESTEPS"])

    env, env_params, action_dim, is_classic = build_env(cfg)

    # Instantiate the RNN network exactly like training.
    # Your ActorCriticRNN signature: ActorCriticRNN(action_dim, config)
    network = ActorCriticRNN(action_dim=action_dim, config=cfg)

    # Restore checkpoint into the RNN parameter tree
    state = restore_state(args.path, network, env, env_params, ckpt_step)

    # Renderer (headless requires xvfb-run)
    renderer = make_renderer(env, env_params, is_classic)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps)

    # Rollout
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(key=reset_rng)

    hidden = jnp.zeros((1, 256), dtype=jnp.float32)  # (B, hidden)
    done = False
    steps = 0

    # write initial frame
    writer.append_data(np.asarray(renderer.render(env_state)))

    while (not done) and (steps < args.max_steps):
        # ActorCriticRNN expects (T,B,...) so T=1,B=1
        obs_tb = jnp.asarray(obs)[None, None, ...]               # (1,1,H,W,C)
        dones_tb = jnp.asarray(done)[None, None]                 # (1,1)

        hidden, pi, value = network.apply(state.params, hidden, (obs_tb, dones_tb))

        rng, act_rng = jax.random.split(rng)
        if args.deterministic:
            probs = np.asarray(pi.probs).reshape(-1)
            action = int(np.argmax(probs))
        else:
            action = int(pi.sample(seed=act_rng).reshape(-1)[0])

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)

        writer.append_data(np.asarray(renderer.render(env_state)))
        steps += 1

    writer.close()
    print(f"Saved MP4: {args.out}")
    print(f"ckpt_step={ckpt_step} seed={args.seed} steps={steps} done={done}")
    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")


if __name__ == "__main__":
    main()

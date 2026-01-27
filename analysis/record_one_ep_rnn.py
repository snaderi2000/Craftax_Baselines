import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import numpy as np
import imageio.v2 as imageio

import jax
import jax.numpy as jnp
from orbax.checkpoint import PyTreeCheckpointer

from models.actor_critic import ActorCriticConv, ActorCritic


def load_config(run_path: str) -> dict:
    with open(os.path.join(run_path, "config.yaml"), "r") as f:
        raw = yaml.safe_load(f)
    cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
    cfg["NUM_ENVS"] = 1
    return cfg


def build_env_and_network(cfg: dict):
    is_classic = False

    if cfg["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        from craftax.craftax.constants import Action
        env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), cfg["LAYER_SIZE"])

    elif cfg["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
        from craftax.craftax.constants import Action
        env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), cfg["LAYER_SIZE"])

    elif cfg["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
        from craftax.craftax_classic.constants import Action
        env = CraftaxClassicSymbolicEnv(CraftaxClassicSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), cfg["LAYER_SIZE"])
        is_classic = True

    elif cfg["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
        from craftax.craftax_classic.constants import Action
        env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), cfg["LAYER_SIZE"])
        is_classic = True

    else:
        raise ValueError(f"Unknown ENV_NAME: {cfg['ENV_NAME']}")

    env_params = env.default_params
    return env, env_params, network, is_classic


def make_renderer(env, env_params, is_classic: bool):
    # Requires Xvfb in headless mode
    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
    return CraftaxRenderer(env, env_params, pixel_render_size=1)


def load_orbax_params(run_path: str, step: int):
    ckpt_dir = os.path.join(run_path, "policies", str(step), "default")
    ckpt = PyTreeCheckpointer().restore(ckpt_dir)

    # From your inspect: top-level dict keys = opt_state, params, step
    # And ckpt["params"] has a single key "params"
    params = ckpt["params"]["params"]
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Dir containing config.yaml and policies/")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--ckpt_step", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true", help="Argmax action instead of sampling")
    args = ap.parse_args()

    cfg = load_config(args.path)
    step = int(args.ckpt_step if args.ckpt_step is not None else cfg["TOTAL_TIMESTEPS"])

    env, env_params, network, is_classic = build_env_and_network(cfg)
    renderer = make_renderer(env, env_params, is_classic)

    params = load_orbax_params(args.path, step)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(key=reset_rng)

    done = False
    steps = 0

    # initial frame
    writer.append_data(np.asarray(renderer.render(env_state)))

    while (not done) and (steps < args.max_steps):
        obs_b = jnp.expand_dims(obs, axis=0)  # (1, ...)
        pi, _ = network.apply(params, obs_b)

        rng, act_rng = jax.random.split(rng)
        if args.deterministic:
            probs = np.asarray(pi.probs).reshape(-1)
            action = int(np.argmax(probs))
        else:
            action = int(pi.sample(seed=act_rng)[0])

        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)

        writer.append_data(np.asarray(renderer.render(env_state)))
        steps += 1

    writer.close()
    print(f"Saved MP4: {args.out}")
    print(f"env={cfg['ENV_NAME']} ckpt_step={step} seed={args.seed} steps={steps} done={done}")
    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")


if __name__ == "__main__":
    main()

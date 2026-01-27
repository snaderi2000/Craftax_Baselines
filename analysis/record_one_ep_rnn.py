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
# analysis/record_one_ep_rnn.py
# Headless (xvfb-run) PPO_RNN policy rollout -> MP4 for a single episode.
#
# Example:
#   xvfb-run -s "-screen 0 1280x720x24" \
#   python analysis/record_one_ep_rnn.py \
#     --path /home/lnazaa/Craftax_Baselines/wandb/run-20260126_162348-kp7gx38r/files \
#     --seed 42 \
#     --out videos/episode_seed42.mp4
#
# Optional:
#   --ckpt_step 1000000   (override)
#   --deterministic       (argmax policy)
#   --max_steps 5000
#   --fps 30
#   --pixel_render_size 1

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import numpy as np
import imageio.v2 as imageio

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import functools

from flax.linen.initializers import orthogonal
from orbax.checkpoint import PyTreeCheckpointer

# ---------------------------
# Model definition (matches your PPO_RNN script)
# ---------------------------

class ImpalaResBlock(nn.Module):
    channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.relu(x)
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        return x + residual


class ImpalaStack(nn.Module):
    channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ImpalaResBlock(self.channels)(x)
        x = ImpalaResBlock(self.channels)(x)
        return x


class DenseResBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.width, kernel_init=orthogonal(2))(x)
        return x + residual


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x  # ins: (B, F), resets: (B,)
        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs: (T,B,63,63,3), dones: (T,B)

        x_enc = obs.astype(jnp.float32)
        for ch in (64, 64, 128):
            x_enc = ImpalaStack(ch)(x_enc)
        x_enc = nn.relu(x_enc)

        # z_t: (T,B,8192) typically
        z_t = x_enc.reshape((*x_enc.shape[:2], -1))

        # bridge -> 256
        rnn_in = nn.LayerNorm()(z_t)
        rnn_in = nn.Dense(256, kernel_init=orthogonal(2))(rnn_in)
        rnn_in = nn.relu(rnn_in)

        # RNN update (scan over T)
        hidden, y_t = ScannedRNN()(hidden, (rnn_in, dones))
        y_t = nn.relu(y_t)

        # concat (T,B, 256+8192)
        shared = jnp.concatenate([y_t, z_t], axis=-1)

        # actor head
        h = nn.LayerNorm()(shared)
        h = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h)
        h = nn.relu(h)
        h = DenseResBlock(self.config["LAYER_SIZE"])(h)
        h = DenseResBlock(self.config["LAYER_SIZE"])(h)
        h = nn.relu(h)
        h = nn.LayerNorm()(h)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)
        pi = distrax.Categorical(logits=logits)

        # critic head
        v = nn.LayerNorm()(shared)
        v = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(v)
        v = nn.relu(v)
        v = DenseResBlock(self.config["LAYER_SIZE"])(v)
        v = DenseResBlock(self.config["LAYER_SIZE"])(v)
        v = nn.relu(v)
        v = nn.LayerNorm()(v)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(v)

        return hidden, pi, jnp.squeeze(value, axis=-1)


# ---------------------------
# Utilities
# ---------------------------

def load_config(run_path: str) -> dict:
    with open(os.path.join(run_path, "config.yaml"), "r") as f:
        raw = yaml.safe_load(f)
    cfg = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
    # Ensure LAYER_SIZE exists and is int
    if "LAYER_SIZE" in cfg:
        cfg["LAYER_SIZE"] = int(cfg["LAYER_SIZE"])
    return cfg


def build_pixels_env(cfg: dict):
    """
    Builds a single-env Craftax *pixels* environment matching ENV_NAME
    and returns (env, env_params, action_dim, is_classic).
    """
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
        raise ValueError(
            f"This recorder is for *pixel* envs only. Got ENV_NAME={env_name}. "
            f"Use a different viewer for symbolic."
        )

    return env, env.default_params, action_dim, is_classic


def make_renderer(env, env_params, is_classic: bool, pixel_render_size: int):
    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
    return CraftaxRenderer(env, env_params, pixel_render_size=pixel_render_size)


def load_rnn_params(run_path: str, step: int):
    """
    Loads TrainState checkpoint saved by your PPO_RNN script and returns the *Flax params*
    pytree at ckpt["params"]["params"].
    """
    ckpt_dir = os.path.join(run_path, "policies", str(step), "default")
    ckpt = PyTreeCheckpointer().restore(ckpt_dir)  # dict with keys: step, opt_state, params
    return ckpt["params"]["params"]


def render_rgb(renderer, env_state):
    """
    CraftaxRenderer.render(env_state) draws to a pygame surface and returns None.
    We read pixels back from the pygame display surface (works under xvfb-run).
    """
    import pygame
    import pygame.surfarray

    renderer.render(env_state)

    surf = None
    # Try to grab a surface from renderer
    for name in ("screen", "_screen", "surface", "_surface", "window", "_window"):
        if hasattr(renderer, name):
            surf = getattr(renderer, name)
            break

    # Fallback to global display surface
    if surf is None:
        surf = pygame.display.get_surface()

    if surf is None:
        raise RuntimeError(
            "Could not get a pygame surface. Renderer may not be initialized or "
            "CraftaxRenderer implementation differs."
        )

    arr = pygame.surfarray.array3d(surf)               # (W,H,3)
    arr = np.transpose(arr, (1, 0, 2)).astype(np.uint8)  # (H,W,3)
    return arr


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Directory containing config.yaml and policies/")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--ckpt_step", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true", help="Argmax policy instead of sampling")
    ap.add_argument("--pixel_render_size", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.path)

    # Figure out which checkpoint step to use
    if args.ckpt_step is not None:
        ckpt_step = int(args.ckpt_step)
    else:
        if "TOTAL_TIMESTEPS" not in cfg:
            raise ValueError("config.yaml missing TOTAL_TIMESTEPS; pass --ckpt_step explicitly.")
        ckpt_step = int(cfg["TOTAL_TIMESTEPS"])

    # Build env + renderer
    env, env_params, action_dim, is_classic = build_pixels_env(cfg)
    renderer = make_renderer(env, env_params, is_classic, pixel_render_size=args.pixel_render_size)

    # Load params (RNN)
    params = load_rnn_params(args.path, ckpt_step)

    # Build RNN network (must match training)
    net = ActorCriticRNN(action_dim=action_dim, config=cfg)

    # Hidden state (B, 256) – fixed by your architecture
    B = 1
    hidden = jnp.zeros((B, 256), dtype=jnp.float32)

    # Output
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps)

    # Rollout (one episode)
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)

    obs, env_state = env.reset(key=reset_rng)
    done = False
    steps = 0

    # Initial frame
    writer.append_data(render_rgb(renderer, env_state))

    while (not done) and (steps < args.max_steps):
        # Prepare inputs: (T=1,B=1,...)
        obs_tb = jnp.asarray(obs)[None, None, ...]              # (1,1,H,W,C)
        dones_tb = jnp.asarray(done)[None, None]                # (1,1)

        hidden, pi, _ = net.apply(params, hidden, (obs_tb, dones_tb))

        rng, act_rng = jax.random.split(rng)
        if args.deterministic:
            probs = np.asarray(pi.probs).reshape(-1)
            action = int(np.argmax(probs))
        else:
            action = int(pi.sample(seed=act_rng).reshape(-1)[0])

        rng, step_rng = jax.random.split(rng)
        obs, env_state, _, done, _ = env.step(step_rng, env_state, action, env_params)

        writer.append_data(render_rgb(renderer, env_state))
        steps += 1

    writer.close()
    print(f"Saved MP4: {args.out}")
    print(f"ENV_NAME={cfg.get('ENV_NAME')} ckpt_step={ckpt_step} seed={args.seed} steps={steps} done={done}")
    print(f"jax_backend={jax.default_backend()} devices={jax.devices()}")


if __name__ == "__main__":
    main()

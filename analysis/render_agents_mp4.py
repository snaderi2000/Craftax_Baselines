#!/usr/bin/env python3
import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
import pygame

from flax.training.train_state import TrainState
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions

from wrappers import AutoResetEnvWrapper
from models.actor_critic import ActorCritic, ActorCriticConv


def load_config(path: str) -> Dict:
    with open(path) as f:
        raw = yaml.load(f, Loader=yaml.Loader)
    cfg = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "value" in v:
            cfg[k] = v["value"]
    return cfg


def infer_config_path_from_ckpt(ckpt_path: str) -> Optional[str]:
    # Try to locate a sibling config.yaml under a parent directory named "files"
    path = os.path.abspath(ckpt_path)
    cur = path
    for _ in range(6):
        base = os.path.basename(cur)
        if base == "files":
            candidate = os.path.join(cur, "config.yaml")
            return candidate if os.path.exists(candidate) else None
        cur = os.path.dirname(cur)
    # Fallback: look two levels up (…/policies/<step>/[default] → …/files/config.yaml)
    try:
        files_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        candidate = os.path.join(files_dir, "config.yaml")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    return None


def make_env_and_model(config: Dict):
    name = config["ENV_NAME"]
    is_classic = "Classic" in name
    is_pixels = name.endswith("Pixels-v1")

    if is_pixels:
        if is_classic:
            from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv
            from craftax.craftax_classic.constants import Action
            env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        else:
            from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
            from craftax.craftax.constants import Action
            env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])  # pixels path
    else:
        if is_classic:
            from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
            from craftax.craftax_classic.constants import Action
            env = CraftaxClassicSymbolicEnv(CraftaxClassicSymbolicEnv.default_static_params())
        else:
            from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
            from craftax.craftax.constants import Action
            env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), config["LAYER_SIZE"])  # symbolic path

    env = AutoResetEnvWrapper(env)
    return env, network, is_classic, is_pixels


def build_renderer(is_classic: bool, env, env_params, pixel_scale: int):
    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
    renderer = CraftaxRenderer(env, env_params, pixel_render_size=pixel_scale)
    return renderer


def restore_ckpt(ckpt_path: str, template_items: Dict):
    cp = PyTreeCheckpointer()

    # If they pointed to the leaf shard dir (e.g., …/default), restore directly
    if os.path.basename(ckpt_path) == "default":
        return cp.restore(ckpt_path, template_items)

    # Else they likely pointed to the <step> directory; use a manager
    # Extract numeric step if present, else try best-effort
    base = os.path.basename(ckpt_path)
    step_match = re.match(r"^(\d+)$", base)
    step = int(step_match.group(1)) if step_match else None

    mgr = CheckpointManager(
        ckpt_path,
        cp,
        CheckpointManagerOptions(max_to_keep=1, create=False),
    )

    if step is None:
        # Probe latest if step not parseable
        latest = mgr.latest_step()
        if latest is None:
            raise ValueError(f"No checkpoints found under: {ckpt_path}")
        step = latest
    return mgr.restore(step, template_items)


def capture_surface_frame() -> np.ndarray:
    surf = pygame.display.get_surface()
    arr = pygame.surfarray.array3d(surf)  # (W, H, 3)
    frame = np.transpose(arr, (1, 0, 2))  # (H, W, 3)
    return frame


def label_from_ckpt_path(path: str) -> str:
    # Use the parent folder (e.g., policy_2M) or the step number
    base = os.path.basename(os.path.dirname(path))  # parent of 'default' or last component
    return base


def run_agent_videos(
    ckpt_path: str,
    config_path: str,
    out_dir: str,
    num_episodes: int,
    seed: int,
    fps: int,
    pixel_scale: int,
    max_steps: int,
    custom_label: Optional[str] = None,
):
    cfg = load_config(config_path)
    cfg["NUM_ENVS"] = 1

    env, net, is_classic, is_pixels = make_env_and_model(cfg)
    env_params = env.default_params

    # Dummy init to get shapes
    rng = jax.random.PRNGKey(0)
    rng, init_key = jax.random.split(rng)

    obs0, env_state0 = env.reset(init_key, env_params)
    init_x = jnp.expand_dims(obs0, 0)

    # Build initial params and a holder TrainState for convenience
    params_init = net.init(init_key, init_x)
    tx = optax.identity()
    state = TrainState.create(apply_fn=net.apply, params=params_init, tx=tx)

    # Restore params from checkpoint; handle nested collections (e.g., batch_stats) if present
    template = {"params": params_init["params"]}
    restored = restore_ckpt(ckpt_path, template)
    params = restored["params"]

    # Build renderer if symbolic (we need to map to pixels)
    renderer = None
    if not is_pixels:
        renderer = build_renderer(is_classic, env, env_params, pixel_scale)

    # Naming
    label = custom_label or label_from_ckpt_path(ckpt_path)
    agent_dir = os.path.join(out_dir, label)
    os.makedirs(agent_dir, exist_ok=True)

    # Deterministic episode seeds
    base_key = jax.random.PRNGKey(seed)

    for epi in range(1, num_episodes + 1):
        print(f"[{label}] Episode {epi}/{num_episodes}")
        frames: List[np.ndarray] = []

        ep_key = jax.random.fold_in(base_key, epi)
        reset_key = jax.random.fold_in(ep_key, 0)
        step_key = jax.random.fold_in(ep_key, 1)

        obs, env_state = env.reset(reset_key, env_params)
        done = False
        steps = 0

        while (not bool(done)) and (steps < max_steps):
            # Render/capture frame
            if is_pixels:
                frame = np.array(obs)
            else:
                # Draw env_state to window and capture
                renderer.render(env_state)
                frame = capture_surface_frame()
            frames.append(frame)

            # Select action deterministically (mode)
            obs_b = jnp.expand_dims(obs, 0)
            pi, _v = net.apply({"params": params}, obs_b)
            a = int(np.asarray(pi.mode()[0]))

            # Step environment with deterministic RNG stream per episode
            step_key, use_key = jax.random.split(step_key)
            obs, env_state, reward, done, info = env.step(use_key, env_state, a, env_params)
            steps += 1

        # Write mp4
        out_path = os.path.join(agent_dir, f"ep_{epi:02d}.mp4")
        writer = imageio.get_writer(out_path, fps=fps)
        for im in frames:
            writer.append_data(im)
        writer.close()
        print(f" → wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Render one or more PPO checkpoints to mp4 videos with shared episode seeds.")
    p.add_argument("--agent", dest="agents", action="append", required=True,
                   help="Path to a checkpoint directory (…/policies/<step>/[default]). Repeat for multiple agents.")
    p.add_argument("--config_path", default=None,
                   help="Optional config.yaml path applied to all agents. If omitted, inferred from each agent path.")
    p.add_argument("--label", dest="labels", action="append", default=None,
                   help="Optional label per --agent (repeat to match). Defaults to parent folder name.")
    p.add_argument("--out_dir", default="videos",
                   help="Output directory root for rendered videos.")
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--pixel_scale", type=int, default=3,
                   help="Pixel scaling for renderer when mapping symbolic→pixels.")
    p.add_argument("--max_steps", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    agents: List[str] = args.agents
    labels: Optional[List[str]] = args.labels

    if labels is not None and len(labels) != len(agents):
        raise ValueError("If providing --label, repeat it once per --agent or omit entirely.")

    for i, ckpt in enumerate(agents):
        cfg_path = args.config_path or infer_config_path_from_ckpt(ckpt)
        if cfg_path is None or not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not locate config.yaml for agent: {ckpt}. Provide --config_path explicitly.")

        label = labels[i] if labels is not None else None
        run_agent_videos(
            ckpt_path=ckpt,
            config_path=cfg_path,
            out_dir=args.out_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            fps=args.fps,
            pixel_scale=args.pixel_scale,
            max_steps=args.max_steps,
            custom_label=label,
        )


if __name__ == "__main__":
    main()



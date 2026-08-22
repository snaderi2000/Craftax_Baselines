import argparse
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
import numpy as np
from PIL import Image

from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.render_episode_step import _render_labeled_step
from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)


def _parse_steps(raw: str) -> list[int]:
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render selected PPO episode steps side by side.")
    parser.add_argument("--run_path", default="wandb/run-20260528_225318-5v1h672r")
    parser.add_argument("--out_dir", default="concept_mapping/runs/episode19_value_plot")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode", type=int, default=19)
    parser.add_argument("--steps", type=str, default="10,80,300,610")
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=6)
    parser.add_argument("--gap", type=int, default=18)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted_steps = _parse_steps(args.steps)
    wanted_set = set(wanted_steps)

    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )

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
    captured = {}
    for episode in range(args.episode + 1):
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, env_state = reset_jit(reset_rng, env_params)
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
            if episode == args.episode and step in wanted_set:
                captured[step] = {
                    "obs": np.asarray(obs),
                    "action": action_i,
                    "value": value_f,
                    "reward": reward_f,
                    "done": done_b,
                }
            obs = next_obs
            if episode == args.episode and wanted_set.issubset(captured.keys()):
                break
            if done_b:
                break
        if episode == args.episode:
            break

    missing = [step for step in wanted_steps if step not in captured]
    if missing:
        raise RuntimeError(f"Could not capture requested steps: {missing}")

    panels = []
    for step in wanted_steps:
        item = captured[step]
        panels.append(
            _render_labeled_step(
                item["obs"],
                args.episode,
                step,
                item["value"],
                item["action"],
                item["reward"],
                item["done"],
                args.block_size,
                args.display_scale,
            )
        )

    width = sum(panel.width for panel in panels) + args.gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    grid = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        grid.paste(panel, (x, 0))
        x += panel.width + args.gap

    steps_name = "_".join(str(step) for step in wanted_steps)
    out_path = os.path.join(args.out_dir, f"episode_{args.episode:02d}_steps_{steps_name}.png")
    grid.save(out_path)

    print(f"checkpoint_step={checkpoint_step}")
    for step in wanted_steps:
        item = captured[step]
        print(
            f"step={step} value={item['value']:.6f} "
            f"action={item['action']}:{_action_name(item['action'])} "
            f"reward={item['reward']:.6f} done={item['done']}"
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

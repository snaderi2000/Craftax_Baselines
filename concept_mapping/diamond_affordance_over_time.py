import argparse
import csv
import os
import sys
from collections import deque

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
from PIL import Image, ImageDraw

from craftax.craftax_classic.constants import BlockType, OBS_DIM, load_all_textures
from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import (
    INV_NAMES,
    MAP_CHANNELS,
    MAP_DIM,
    decode_obs,
    render_local_view,
)


DIAMOND_INV_INDEX = 4
IRON_PICKAXE_INV_INDEX = 8


def _set_block(map_view: np.ndarray, row: int, col: int, block: BlockType) -> None:
    map_view[row, col, : len(BlockType)] = 0.0
    map_view[row, col, block.value] = 1.0


def _cover_visible_diamond(obs: np.ndarray, replacement: BlockType = BlockType.STONE) -> np.ndarray:
    out = obs.copy()
    map_view = out[:MAP_DIM].reshape(*OBS_DIM, MAP_CHANNELS)
    blocks = np.argmax(map_view[..., : len(BlockType)], axis=-1)
    for row, col in np.argwhere(blocks == BlockType.DIAMOND.value):
        _set_block(map_view, int(row), int(col), replacement)
    return out


def _remove_iron_pickaxe(obs: np.ndarray) -> np.ndarray:
    out = obs.copy()
    out[MAP_DIM + IRON_PICKAXE_INV_INDEX] = 0.0
    return out


def _has_visible_diamond(obs: np.ndarray) -> bool:
    blocks = decode_obs(obs).blocks
    return bool(np.any(blocks == BlockType.DIAMOND.value))


def _has_iron_pickaxe(obs: np.ndarray) -> bool:
    return bool(obs[MAP_DIM + IRON_PICKAXE_INV_INDEX] > 0.05)


def _diamond_count(obs: np.ndarray) -> float:
    return float(obs[MAP_DIM + DIAMOND_INV_INDEX] * 10.0)


def _make_counterfactuals(base_obs: np.ndarray) -> dict[str, np.ndarray]:
    d = base_obs.astype(np.float32)
    c = _remove_iron_pickaxe(d)
    b = _cover_visible_diamond(d)
    a = _remove_iron_pickaxe(b)
    return {
        "A_no_diamond_no_pickaxe": a,
        "B_no_diamond_pickaxe": b,
        "C_diamond_no_pickaxe": c,
        "D_diamond_pickaxe": d,
    }


def _rollout_collect_states(args, network, train_state, env, env_params) -> list[dict]:
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    @jax.jit
    def policy_apply(params, obs, rng):
        pi, _value = network.apply(params, obs[None, ...])
        if args.greedy:
            action = jnp.argmax(pi.logits[0])
            return action, rng
        rng, action_rng = jax.random.split(rng)
        return pi.sample(seed=action_rng)[0], rng

    rng = jax.random.PRNGKey(args.seed)
    collected = []
    for episode in range(args.max_episodes):
        if len(collected) >= args.target_states:
            break
        rng, reset_rng, policy_rng = jax.random.split(rng, 3)
        obs, env_state = reset_jit(reset_rng, env_params)
        history = deque(maxlen=args.lookback + 1)
        prev_diamond = _diamond_count(np.asarray(obs))
        for step in range(args.max_steps):
            obs_np = np.asarray(obs, dtype=np.float32)
            history.append((step, obs_np))
            action, policy_rng = policy_apply(train_state.params, obs, policy_rng)
            action_i = int(np.asarray(action))
            rng, step_rng = jax.random.split(rng)
            next_obs, env_state, reward, done, _info = step_jit(
                step_rng, env_state, action, env_params
            )
            next_obs_np = np.asarray(next_obs, dtype=np.float32)
            next_diamond = _diamond_count(next_obs_np)
            if next_diamond > prev_diamond + 0.5 and len(history) > args.lookback:
                base_step, base_obs = history[0]
                if _has_visible_diamond(base_obs) and _has_iron_pickaxe(base_obs):
                    collected.append(
                        {
                            "episode": episode,
                            "collect_step": step,
                            "base_step": base_step,
                            "action": action_i,
                            "action_name": _action_name(action_i),
                            "reward": float(np.asarray(reward)),
                            "base_obs": base_obs,
                        }
                    )
                    print(
                        f"collected {len(collected)}/{args.target_states}: "
                        f"episode={episode} base_step={base_step} collect_step={step}",
                        flush=True,
                    )
                    break
            prev_diamond = next_diamond
            obs = next_obs
            if bool(np.asarray(done)):
                break
        if episode % 25 == 0:
            print(f"scanned episode={episode} collected={len(collected)}", flush=True)
    return collected


def _eval_values(network, params, obs_by_key: dict[str, np.ndarray], batch_size: int) -> dict[str, np.ndarray]:
    keys = list(obs_by_key.keys())
    all_obs = {k: np.asarray(v, dtype=np.float32) for k, v in obs_by_key.items()}

    @jax.jit
    def apply_batch(x):
        _pi, values = network.apply(params, x)
        return values

    out = {}
    for key in keys:
        arr = all_obs[key]
        vals = []
        for start in range(0, arr.shape[0], batch_size):
            end = min(start + batch_size, arr.shape[0])
            vals.append(np.asarray(apply_batch(jnp.asarray(arr[start:end]))))
        out[key] = np.concatenate(vals, axis=0)
    return out


def _write_state_npz(samples: list[dict], out_path: str) -> None:
    base_obs = np.stack([s["base_obs"] for s in samples], axis=0)
    episodes = np.asarray([s["episode"] for s in samples], dtype=np.int32)
    base_steps = np.asarray([s["base_step"] for s in samples], dtype=np.int32)
    collect_steps = np.asarray([s["collect_step"] for s in samples], dtype=np.int32)
    np.savez_compressed(
        out_path,
        base_obs=base_obs,
        episodes=episodes,
        base_steps=base_steps,
        collect_steps=collect_steps,
    )


def _draw_text_panel(width: int, lines: list[str]) -> Image.Image:
    pad = 8
    line_h = 18
    panel = Image.new("RGB", (width, pad * 2 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(panel)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(0, 0, 0))
    return panel


def _render_counterfactual_sheet(
    cf: dict[str, np.ndarray],
    values: dict[str, float],
    sample_meta: dict,
    out_path: str,
    block_size: int,
    display_scale: int,
    textures: dict,
    panel_textures: dict,
) -> None:
    titles = [
        ("A_no_diamond_no_pickaxe", "A: no diamond, no pickaxe"),
        ("B_no_diamond_pickaxe", "B: no diamond, pickaxe"),
        ("C_diamond_no_pickaxe", "C: diamond, no pickaxe"),
        ("D_diamond_pickaxe", "D: diamond, pickaxe"),
    ]
    frames = []
    for key, title in titles:
        img = render_local_view(
            cf[key],
            block_size=block_size,
            include_inventory=True,
            display_scale=display_scale,
            textures=textures,
            panel_textures=panel_textures,
        )
        header = _draw_text_panel(img.width, [title, f"V={values[key]:.4f}"])
        frame = Image.new("RGB", (img.width, header.height + img.height), "white")
        frame.paste(header, (0, 0))
        frame.paste(img, (0, header.height))
        frames.append(frame)

    gap = 16
    cell_w = max(f.width for f in frames)
    cell_h = max(f.height for f in frames)
    footer_lines = [
        f"episode={sample_meta['episode']} base_step={sample_meta['base_step']} collect_step={sample_meta['collect_step']}",
        "Platonic rule: V(D) > {V(A), V(C)} > V(B); A/C order free",
        f"Observed: D={values['D_diamond_pickaxe']:.3f}, C={values['C_diamond_no_pickaxe']:.3f}, "
        f"A={values['A_no_diamond_no_pickaxe']:.3f}, B={values['B_no_diamond_pickaxe']:.3f}",
    ]
    footer = _draw_text_panel(2 * cell_w + gap, footer_lines)
    sheet = Image.new("RGB", (2 * cell_w + gap, 2 * cell_h + gap + footer.height), "white")
    positions = [(0, 0), (cell_w + gap, 0), (0, cell_h + gap), (cell_w + gap, cell_h + gap)]
    for frame, pos in zip(frames, positions):
        sheet.paste(frame, pos)
    sheet.paste(footer, (0, 2 * cell_h + gap))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track diamond/pickaxe critic counterfactual ordering across PPO checkpoints."
    )
    parser.add_argument("--run_path", default="wandb/run-20260606_064930-g568agfo")
    parser.add_argument("--out_dir", default="concept_mapping/runs/diamond_affordance_over_time")
    parser.add_argument("--collection_timestep", type=int, default=9999941632)
    parser.add_argument(
        "--checkpoints",
        default="65536,262144,524288,720896,1048576,10485760,104857600,705036288,1000013824,9999941632",
    )
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_episodes", type=int, default=700)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--target_states", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--render_examples", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inspection_dir = os.path.join(args.out_dir, "inspection")
    os.makedirs(inspection_dir, exist_ok=True)

    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)

    print(f"Collecting states from checkpoint {args.collection_timestep}", flush=True)
    collection_state, collection_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.collection_timestep
    )
    samples = _rollout_collect_states(args, network, collection_state, env, env_params)
    if not samples:
        raise RuntimeError("No diamond t-lookback states collected.")
    print(f"Collected {len(samples)} states", flush=True)
    _write_state_npz(samples, os.path.join(args.out_dir, "diamond_tminus_states.npz"))

    cfs_per_sample = [_make_counterfactuals(s["base_obs"]) for s in samples]
    cf_arrays = {
        key: np.stack([cf[key] for cf in cfs_per_sample], axis=0)
        for key in cfs_per_sample[0]
    }

    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    summary_path = os.path.join(args.out_dir, "ordering_summary.csv")
    values_path = os.path.join(args.out_dir, "counterfactual_values.csv")
    with open(summary_path, "w", newline="") as sf, open(values_path, "w", newline="") as vf:
        sw = csv.writer(sf)
        vw = csv.writer(vf)
        sw.writerow(
            [
                "checkpoint",
                "n_states",
                "strict_pct",
                "affordance_pct",
                "platonic_pairwise_pct",
                "platonic_middle_free_pct",
                "d_gt_c_pct",
                "b_gt_a_pct",
                "mean_interaction",
                "mean_A",
                "mean_B",
                "mean_C",
                "mean_D",
            ]
        )
        vw.writerow(
            [
                "checkpoint",
                "sample_index",
                "episode",
                "base_step",
                "collect_step",
                "A_no_diamond_no_pickaxe",
                "B_no_diamond_pickaxe",
                "C_diamond_no_pickaxe",
                "D_diamond_pickaxe",
                "strict_correct",
                "affordance_correct",
                "platonic_pairwise_correct",
                "platonic_middle_free_correct",
                "interaction",
            ]
        )
        checkpoint_values_for_render = None
        for ckpt in checkpoints:
            print(f"Evaluating checkpoint {ckpt}", flush=True)
            train_state, restored_step = _restore_train_state(
                args.run_path, config, network, env, env_params, ckpt
            )
            values = _eval_values(network, train_state.params, cf_arrays, args.batch_size)
            a = values["A_no_diamond_no_pickaxe"]
            b = values["B_no_diamond_pickaxe"]
            c = values["C_diamond_no_pickaxe"]
            d = values["D_diamond_pickaxe"]
            strict = (d > c) & (c > a) & (a > b)
            affordance = (d > c) & (d > b)
            platonic_pairwise = (d > c) & (b > a)
            platonic_middle_free = (d > a) & (d > c) & (a > b) & (c > b)
            interaction = (d - b) - (c - a)
            sw.writerow(
                [
                    restored_step,
                    len(samples),
                    float(strict.mean()),
                    float(affordance.mean()),
                    float(platonic_pairwise.mean()),
                    float(platonic_middle_free.mean()),
                    float((d > c).mean()),
                    float((b > a).mean()),
                    float(interaction.mean()),
                    float(a.mean()),
                    float(b.mean()),
                    float(c.mean()),
                    float(d.mean()),
                ]
            )
            for i, s in enumerate(samples):
                vw.writerow(
                    [
                        restored_step,
                        i,
                        s["episode"],
                        s["base_step"],
                        s["collect_step"],
                        float(a[i]),
                        float(b[i]),
                        float(c[i]),
                        float(d[i]),
                        bool(strict[i]),
                        bool(affordance[i]),
                        bool(platonic_pairwise[i]),
                        bool(platonic_middle_free[i]),
                        float(interaction[i]),
                    ]
                )
            if restored_step == checkpoints[-1]:
                checkpoint_values_for_render = values
            print(
                f"checkpoint={restored_step} strict={strict.mean():.3f} "
                f"affordance={affordance.mean():.3f} "
                f"platonic_pairwise={platonic_pairwise.mean():.3f} "
                f"platonic_middle_free={platonic_middle_free.mean():.3f} "
                f"interaction={interaction.mean():.3f}",
                flush=True,
            )

    if checkpoint_values_for_render is not None:
        textures = load_all_textures(args.block_size)
        panel_textures = (
            textures
            if args.display_scale == 1
            else load_all_textures(args.block_size * args.display_scale)
        )
        rng = np.random.default_rng(args.seed)
        n_render = min(args.render_examples, len(samples))
        render_indices = rng.choice(len(samples), size=n_render, replace=False)
        for rank, idx in enumerate(render_indices):
            values = {key: float(checkpoint_values_for_render[key][idx]) for key in cf_arrays}
            _render_counterfactual_sheet(
                cfs_per_sample[idx],
                values,
                samples[idx],
                os.path.join(inspection_dir, f"sample_{rank:03d}_idx_{idx:03d}.png"),
                args.block_size,
                args.display_scale,
                textures,
                panel_textures,
            )

    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved values: {values_path}", flush=True)
    print(f"Saved inspection sheets: {inspection_dir}", flush=True)


if __name__ == "__main__":
    main()

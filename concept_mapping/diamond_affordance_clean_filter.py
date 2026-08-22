import argparse
import csv
import hashlib
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

from craftax.craftax_classic.constants import BlockType, OBS_DIM, load_all_textures
from craftax.craftax_env import make_craftax_env_from_name

from concept_mapping.diamond_affordance_over_time import (
    DIAMOND_INV_INDEX,
    IRON_PICKAXE_INV_INDEX,
    _diamond_count,
    _eval_values,
    _has_iron_pickaxe,
    _has_visible_diamond,
    _make_counterfactuals,
    _render_counterfactual_sheet,
    _write_state_npz,
)
from concept_mapping.render_ppo_episodes import (
    _action_name,
    _init_network,
    _load_wandb_config,
    _restore_train_state,
)
from concept_mapping.visualize_symbolic_trajectory import MAP_DIM, decode_obs


PLAYER_ROW = OBS_DIM[0] // 2
PLAYER_COL = OBS_DIM[1] // 2
CLOSE_OFFSETS = [
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def _obs_key(obs: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()


def _inventory_counts(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs[MAP_DIM : MAP_DIM + 12], dtype=np.float32) * 10.0


def _near_block(obs: np.ndarray, block_type: BlockType) -> bool:
    blocks = decode_obs(obs).blocks
    for dr, dc in CLOSE_OFFSETS:
        r = PLAYER_ROW + dr
        c = PLAYER_COL + dc
        if 0 <= r < blocks.shape[0] and 0 <= c < blocks.shape[1]:
            if int(blocks[r, c]) == block_type.value:
                return True
    return False


def _can_immediately_make_iron_pickaxe(obs: np.ndarray) -> bool:
    inv = _inventory_counts(obs)
    has_resources = inv[0] >= 1 and inv[1] >= 1 and inv[2] >= 1 and inv[3] >= 1
    has_stations = _near_block(obs, BlockType.CRAFTING_TABLE) and _near_block(obs, BlockType.FURNACE)
    return bool(has_resources and has_stations)


def _clean_valid(obs: np.ndarray) -> tuple[bool, str]:
    if not _has_visible_diamond(obs):
        return False, "no_visible_diamond"
    if not _has_iron_pickaxe(obs):
        return False, "no_iron_pickaxe"
    if _diamond_count(obs) > 0.5:
        return False, "already_has_diamond"
    no_pickaxe = np.array(obs, copy=True)
    no_pickaxe[MAP_DIM + IRON_PICKAXE_INV_INDEX] = 0.0
    if _can_immediately_make_iron_pickaxe(no_pickaxe):
        return False, "can_remake_iron_pickaxe"
    return True, "valid"


def _load_existing_samples(path: str) -> list[dict]:
    data = np.load(path)
    samples = []
    for i, obs in enumerate(data["base_obs"]):
        samples.append(
            {
                "episode": int(data["episodes"][i]),
                "base_step": int(data["base_steps"][i]),
                "collect_step": int(data["collect_steps"][i]),
                "action": -1,
                "action_name": "",
                "reward": float("nan"),
                "base_obs": np.asarray(obs, dtype=np.float32),
                "source": "existing",
            }
        )
    return samples


def _filter_existing(samples: list[dict]) -> tuple[list[dict], dict[str, int]]:
    counts: dict[str, int] = {}
    kept = []
    seen = set()
    for sample in samples:
        ok, reason = _clean_valid(sample["base_obs"])
        counts[reason] = counts.get(reason, 0) + 1
        key = _obs_key(sample["base_obs"])
        if ok and key not in seen:
            kept.append(sample)
            seen.add(key)
    return kept, counts


def _collect_more(args, network, train_state, env, env_params, existing: list[dict]) -> list[dict]:
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

    collected = list(existing)
    seen = {_obs_key(s["base_obs"]) for s in collected}
    rng = jax.random.PRNGKey(args.extra_seed)
    scanned_candidates = 0
    rejected: dict[str, int] = {}
    for episode in range(args.max_extra_episodes):
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
                scanned_candidates += 1
                base_step, base_obs = history[0]
                ok, reason = _clean_valid(base_obs)
                rejected[reason] = rejected.get(reason, 0) + 1
                key = _obs_key(base_obs)
                if ok and key not in seen:
                    collected.append(
                        {
                            "episode": episode,
                            "collect_step": step,
                            "base_step": base_step,
                            "action": action_i,
                            "action_name": _action_name(action_i),
                            "reward": float(np.asarray(reward)),
                            "base_obs": base_obs,
                            "source": "extra",
                        }
                    )
                    seen.add(key)
                    print(
                        f"clean collected {len(collected)}/{args.target_states}: "
                        f"extra_episode={episode} base_step={base_step} collect_step={step}",
                        flush=True,
                    )
                break
            prev_diamond = next_diamond
            obs = next_obs
            if bool(np.asarray(done)):
                break
        if episode % 50 == 0:
            print(
                f"extra scanned episode={episode} clean_total={len(collected)} "
                f"candidate_rejections={rejected}",
                flush=True,
            )
    print(f"extra scanned diamond candidates={scanned_candidates} rejection_counts={rejected}", flush=True)
    return collected[: args.target_states]


def _evaluate_and_write(args, network, train_state, samples: list[dict], checkpoint_step: int) -> None:
    cfs_per_sample = [_make_counterfactuals(s["base_obs"]) for s in samples]
    cf_arrays = {
        key: np.stack([cf[key] for cf in cfs_per_sample], axis=0)
        for key in cfs_per_sample[0]
    }
    values = _eval_values(network, train_state.params, cf_arrays, args.batch_size)
    a = values["A_no_diamond_no_pickaxe"]
    b = values["B_no_diamond_pickaxe"]
    c = values["C_diamond_no_pickaxe"]
    d = values["D_diamond_pickaxe"]
    middle_free = (d > a) & (d > c) & (a > b) & (c > b)
    interaction = (d - b) - (c - a)

    summary_path = os.path.join(args.out_dir, "clean_ordering_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["checkpoint", checkpoint_step])
        writer.writerow(["n_states", len(samples)])
        writer.writerow(["existing_valid_used", sum(s["source"] == "existing" for s in samples)])
        writer.writerow(["extra_valid_used", sum(s["source"] == "extra" for s in samples)])
        writer.writerow(["middle_free_success_rate", float(middle_free.mean())])
        writer.writerow(["D_gt_A_rate", float((d > a).mean())])
        writer.writerow(["D_gt_C_rate", float((d > c).mean())])
        writer.writerow(["A_gt_B_rate", float((a > b).mean())])
        writer.writerow(["C_gt_B_rate", float((c > b).mean())])
        writer.writerow(["mean_interaction", float(interaction.mean())])
        writer.writerow(["mean_A", float(a.mean())])
        writer.writerow(["mean_B", float(b.mean())])
        writer.writerow(["mean_C", float(c.mean())])
        writer.writerow(["mean_D", float(d.mean())])

    values_path = os.path.join(args.out_dir, "clean_counterfactual_values.csv")
    with open(values_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_index",
                "source",
                "episode",
                "base_step",
                "collect_step",
                "A_no_diamond_no_pickaxe",
                "B_no_diamond_pickaxe",
                "C_diamond_no_pickaxe",
                "D_diamond_pickaxe",
                "middle_free_correct",
                "interaction",
            ]
        )
        for i, sample in enumerate(samples):
            writer.writerow(
                [
                    i,
                    sample["source"],
                    sample["episode"],
                    sample["base_step"],
                    sample["collect_step"],
                    float(a[i]),
                    float(b[i]),
                    float(c[i]),
                    float(d[i]),
                    bool(middle_free[i]),
                    float(interaction[i]),
                ]
            )

    _write_state_npz(samples, os.path.join(args.out_dir, "clean_diamond_tminus_states.npz"))

    inspection_dir = os.path.join(args.out_dir, "inspection")
    os.makedirs(inspection_dir, exist_ok=True)
    textures = load_all_textures(args.block_size)
    panel_textures = (
        textures
        if args.display_scale == 1
        else load_all_textures(args.block_size * args.display_scale)
    )
    rng = np.random.default_rng(args.seed)
    render_indices = rng.choice(len(samples), size=min(args.render_examples, len(samples)), replace=False)
    for rank, idx in enumerate(render_indices):
        value_dict = {key: float(values[key][idx]) for key in cf_arrays}
        _render_counterfactual_sheet(
            cfs_per_sample[idx],
            value_dict,
            samples[idx],
            os.path.join(inspection_dir, f"sample_{rank:03d}_idx_{idx:03d}.png"),
            args.block_size,
            args.display_scale,
            textures,
            panel_textures,
        )

    print(f"Saved clean summary: {summary_path}", flush=True)
    print(f"Saved clean values: {values_path}", flush=True)
    print(f"Saved clean states: {os.path.join(args.out_dir, 'clean_diamond_tminus_states.npz')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean diamond affordance analysis with incremental collection.")
    parser.add_argument("--run_path", default="wandb/run-20260630_214658-m0mw4end")
    parser.add_argument("--existing_run_dir", default="concept_mapping/runs/diamond_affordance_platonic_500_final_m0mw4end")
    parser.add_argument("--out_dir", default="concept_mapping/runs/diamond_affordance_clean_500_final_m0mw4end")
    parser.add_argument("--timestep", type=int, default=9999941632)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--target_states", type=int, default=500)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Collect a completely new dataset instead of reusing valid states "
            "from --existing_run_dir."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--extra_seed", type=int, default=107)
    parser.add_argument("--max_extra_episodes", type=int, default=4000)
    parser.add_argument("--max_steps", type=int, default=4096)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--render_examples", type=int, default=24)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.fresh:
        valid_existing = []
        print("Fresh collection requested: not reusing existing states", flush=True)
    else:
        existing_path = os.path.join(args.existing_run_dir, "diamond_tminus_states.npz")
        existing_samples = _load_existing_samples(existing_path)
        valid_existing, existing_counts = _filter_existing(existing_samples)
        print(f"Existing states={len(existing_samples)} valid_existing={len(valid_existing)} counts={existing_counts}", flush=True)

    config = _load_wandb_config(args.run_path)
    env_name = args.env_name or config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, checkpoint_step = _restore_train_state(
        args.run_path, config, network, env, env_params, args.timestep
    )

    samples = valid_existing
    if len(samples) < args.target_states:
        print(f"Need {args.target_states - len(samples)} additional clean states", flush=True)
        samples = _collect_more(args, network, train_state, env, env_params, samples)
    if len(samples) < args.target_states:
        raise RuntimeError(f"Only collected {len(samples)} clean states; need {args.target_states}")

    _evaluate_and_write(args, network, train_state, samples, checkpoint_step)


if __name__ == "__main__":
    main()

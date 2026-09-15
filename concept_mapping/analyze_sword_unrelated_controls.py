"""Compare iron-sword representation alignment with related/unrelated concepts.

The script builds fixed first-completion banks for Collect Drink and Place Plant,
evaluates their critic transition directions at the same stride-three checkpoints,
and compares them with the saved Iron Sword and Stone Sword directions.  The
aggregate pass also evaluates identical banks under multiple randomly initialized
networks to provide an architecture-and-symbolic-encoding baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from craftax.craftax_classic.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.analyze_achievement_stabilized_effect_over_checkpoints import selected_steps
from concept_mapping.analyze_achievement_transition_coherence import (
    MASTER, DEFAULT_RUN, TransitionBank, apply_hidden_and_value,
    checkpoint_path, run_worker,
)
from concept_mapping.analyze_diamond_representation_coherence import _coherence_statistics
from concept_mapping.plot_achievement_value_effects_with_success import log_bin_mean
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config


MAIN_ROOT = MASTER / "achievement_transition_coherence_stride3"
OUT = MAIN_ROOT / "sword_unrelated_controls"
BANK_PATH = MASTER / "unrelated_achievement_transition_fixed_bank.npz"
MAIN_BANK_PATH = MASTER / "achievement_transition_fixed_bank.npz"
SUCCESS_CSV = (
    MASTER / "achievement_stabilized_effect_over_checkpoints_stride3" /
    "wandb_achievement_success_history.csv"
)
UNRELATED = ("collect_drink", "place_plant")


def build_or_load_bank(path: Path) -> TransitionBank:
    if not path.exists():
        enum_names = [item.name.lower() for item in Achievement]
        target_indices = [enum_names.index(name) for name in UNRELATED]
        before, after, concept_index, episode_ids, completion_steps = [], [], [], [], []
        for episode_path in sorted((MASTER / "episodes").glob("episode_*.npz")):
            with np.load(episode_path) as data:
                observations = np.asarray(data["observations"], dtype=np.float32)
                achievements = np.concatenate([
                    np.asarray(data["achievements"]),
                    np.asarray(data["terminal_achievements"])[None],
                ])
                terminal_observation = np.asarray(data["terminal_observation"], dtype=np.float32)
                episode_id = int(data["episode_id"])
            for concept, achievement_index in enumerate(target_indices):
                hits = np.flatnonzero(achievements[:, achievement_index] > 0)
                if not len(hits) or hits[0] == 0:
                    continue
                after_index = int(hits[0])
                before.append(observations[after_index - 1])
                after.append(observations[after_index] if after_index < len(observations)
                             else terminal_observation)
                concept_index.append(concept)
                episode_ids.append(episode_id)
                completion_steps.append(after_index)
        payload = {
            "before": np.asarray(before, dtype=np.float32),
            "after": np.asarray(after, dtype=np.float32),
            "concept_index": np.asarray(concept_index, dtype=np.int16),
            "episode_id": np.asarray(episode_ids, dtype=np.int32),
            "completion_step": np.asarray(completion_steps, dtype=np.int32),
            "concept_names": np.asarray(UNRELATED),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)

    with np.load(path) as data:
        before = np.asarray(data["before"], dtype=np.float32)
        after = np.asarray(data["after"], dtype=np.float32)
        concept_index = np.asarray(data["concept_index"], dtype=np.int16)
        names = tuple(np.asarray(data["concept_names"]).astype(str).tolist())
    counts = np.bincount(concept_index, minlength=len(names))
    return TransitionBank(names, before, after, concept_index, counts)


def vector_at(root: Path, concept: str, checkpoint: int) -> np.ndarray:
    path = checkpoint_path(root, concept, checkpoint)
    with np.load(path) as data:
        vector = np.asarray(data["top_eigenvector"], dtype=np.float64)
    return vector / np.linalg.norm(vector)


def load_main_sword_banks() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(MAIN_BANK_PATH) as data:
        names = np.asarray(data["concept_names"]).astype(str).tolist()
        indices = np.asarray(data["concept_index"])
        before = np.asarray(data["before"], dtype=np.float32)
        after = np.asarray(data["after"], dtype=np.float32)
    result = {}
    for name in ("make_stone_sword", "make_iron_sword"):
        mask = indices == names.index(name)
        result[name] = (before[mask], after[mask])
    return result


def random_initialization_baseline(args, unrelated_bank: TransitionBank) -> list[dict]:
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    activation = str(config.get("ACTIVATION", "tanh"))
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    init_obs = jnp.zeros((1, *env.observation_space(env_params).shape), dtype=jnp.float32)
    banks = load_main_sword_banks()
    for concept, name in enumerate(unrelated_bank.names):
        mask = unrelated_bank.concept_index == concept
        banks[name] = (unrelated_bank.before[mask], unrelated_bank.after[mask])

    packed_parts, slices, cursor = [], {}, 0
    for name, (before, after) in banks.items():
        packed_parts.extend((before, after))
        slices[name] = (slice(cursor, cursor + len(before)),
                        slice(cursor + len(before), cursor + 2 * len(before)))
        cursor += 2 * len(before)
    packed = np.concatenate(packed_parts)

    records = []
    for seed in range(args.random_seeds):
        params = network.init(jax.random.PRNGKey(seed), init_obs)
        hidden, _ = apply_hidden_and_value(params, packed, activation, args.batch_size)
        vectors = {}
        for name, (before_slice, after_slice) in slices.items():
            delta_h = hidden[after_slice] - hidden[before_slice]
            _stats, _eigenvalues, vector, _mean, _norms = _coherence_statistics(delta_h)
            vectors[name] = vector / np.linalg.norm(vector)
        iron = vectors["make_iron_sword"]
        stone = abs(float(iron @ vectors["make_stone_sword"]))
        drink = abs(float(iron @ vectors["collect_drink"]))
        plant = abs(float(iron @ vectors["place_plant"]))
        records.append({
            "seed": seed,
            "iron_stone": stone,
            "iron_collect_drink": drink,
            "iron_place_plant": plant,
            "selectivity": stone - 0.5 * (drink + plant),
        })
        print(f"random baseline seed {seed + 1}/{args.random_seeds}", flush=True)
    return records


def aggregate(args, steps: list[int], bank: TransitionBank) -> None:
    missing = {
        name: sum(not checkpoint_path(Path(args.out_dir), name, step).exists() for step in steps)
        for name in bank.names
    }
    if any(missing.values()):
        raise RuntimeError(f"Supplemental sweep incomplete: {missing}")

    rows = []
    for step in steps:
        iron = vector_at(MAIN_ROOT, "make_iron_sword", step)
        stone = vector_at(MAIN_ROOT, "make_stone_sword", step)
        drink = vector_at(Path(args.out_dir), "collect_drink", step)
        plant = vector_at(Path(args.out_dir), "place_plant", step)
        iron_stone = abs(float(iron @ stone))
        iron_drink = abs(float(iron @ drink))
        iron_plant = abs(float(iron @ plant))
        rows.append({
            "checkpoint": step,
            "iron_stone": iron_stone,
            "iron_collect_drink": iron_drink,
            "iron_place_plant": iron_plant,
            "unrelated_mean": 0.5 * (iron_drink + iron_plant),
            "sword_selectivity": iron_stone - 0.5 * (iron_drink + iron_plant),
        })
    trained_csv = Path(args.out_dir) / "iron_sword_related_vs_unrelated.csv"
    with trained_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    baseline = random_initialization_baseline(args, bank)
    baseline_csv = Path(args.out_dir) / "random_initialization_alignment_baseline.csv"
    with baseline_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline[0]))
        writer.writeheader(); writer.writerows(baseline)

    random_summary = {}
    for field in ("iron_stone", "iron_collect_drink", "iron_place_plant", "selectivity"):
        values = np.asarray([row[field] for row in baseline])
        random_summary[field] = {
            "mean": float(values.mean()), "q025": float(np.quantile(values, .025)),
            "q975": float(np.quantile(values, .975)),
        }
    (Path(args.out_dir) / "random_initialization_alignment_summary.json").write_text(
        json.dumps(random_summary, indent=2) + "\n"
    )

    with SUCCESS_CSV.open(newline="") as handle:
        success_rows = list(csv.DictReader(handle))
    sx, sy = [], []
    for row in success_rows:
        value = row.get("make_iron_sword", "")
        if value not in (None, ""):
            sx.append(float(row["environment_steps"])); sy.append(float(value))
    success_x, success_y = log_bin_mean(np.asarray(sx), np.asarray(sy), bins=500)

    x = np.asarray([row["checkpoint"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.6), sharex=True,
                             constrained_layout=True)
    colors = {"iron_stone": "#7c3aed", "iron_collect_drink": "#0891b2",
              "iron_place_plant": "#ea580c"}
    labels = {"iron_stone": "Iron sword vs. stone sword",
              "iron_collect_drink": "Iron sword vs. collect drink",
              "iron_place_plant": "Iron sword vs. place plant"}
    top, bottom = axes
    for field in colors:
        y = np.asarray([row[field] for row in rows])
        top.plot(x, y, color=colors[field], linewidth=2.0, label=labels[field])
        summary = random_summary[field]
        top.axhspan(summary["q025"], summary["q975"], color=colors[field], alpha=.07)
        top.axhline(summary["mean"], color=colors[field], linestyle=":", linewidth=1.1)
    top.set_ylim(0, 1)
    top.set_ylabel("Absolute eigenvector alignment")
    top.set_title("Related and unrelated alignment controls")
    top.grid(True, which="both", alpha=.20)
    top.legend(frameon=False, ncol=3)

    selectivity = np.asarray([row["sword_selectivity"] for row in rows])
    bottom.plot(x, selectivity, color="#1d4ed8", linewidth=2.1,
                label="Iron–stone minus mean unrelated alignment")
    rs = random_summary["selectivity"]
    bottom.axhspan(rs["q025"], rs["q975"], color="#64748b", alpha=.16,
                   label="Random-init 95% interval")
    bottom.axhline(0, color="#475569", linestyle="--", linewidth=1.0)
    right = bottom.twinx()
    right.plot(success_x, success_y, color="#dc2626", linewidth=1.65,
               alpha=.9, label="Iron-sword success")
    right.set_ylim(0, 100)
    right.set_ylabel("Make Iron Sword success rate (%)", color="#b91c1c")
    right.tick_params(axis="y", colors="#b91c1c")
    bottom.set_ylabel("Sword-specific alignment advantage")
    bottom.set_title("Iron-sword directional selectivity over unrelated achievements")
    bottom.grid(True, which="both", alpha=.20)
    h1, l1 = bottom.get_legend_handles_labels(); h2, l2 = right.get_legend_handles_labels()
    bottom.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    bottom.set_xscale("log")
    bottom.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9],
                      [r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$"])
    bottom.set_xlabel("Training environment steps")
    fig.suptitle("Is the iron-sword direction specifically related to swords?", fontsize=17)
    png = Path(args.out_dir) / "iron_sword_related_vs_unrelated_and_random_baseline.png"
    pdf = Path(args.out_dir) / "iron_sword_related_vs_unrelated_and_random_baseline.pdf"
    fig.savefig(png, dpi=300); fig.savefig(pdf); plt.close(fig)
    print(trained_csv); print(baseline_csv); print(png); print(pdf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--bank", type=Path, default=BANK_PATH)
    parser.add_argument("--out_dir", type=Path, default=OUT)
    parser.add_argument("--checkpoint_stride", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--worker_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--random_seeds", type=int, default=32)
    parser.add_argument("--build_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    bank = build_or_load_bank(args.bank)
    print(f"bank counts: {dict(zip(bank.names, bank.counts.tolist()))}", flush=True)
    if args.build_only:
        return
    steps = selected_steps(args.run_path, args.checkpoint_stride)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args, steps, bank)
    else:
        run_worker(args, steps, bank)


if __name__ == "__main__":
    main()

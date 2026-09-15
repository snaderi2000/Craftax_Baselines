"""Compare sword-transition directions with mechanically matched crafting controls.

The primary control for Make Iron Sword is Make Iron Pickaxe.  In Craftax
Classic these actions have identical prerequisites and consume one wood, stone,
coal, and iron; only the output inventory coordinate differs.  This script
reuses the frozen natural first-completion bank and the saved stride-three
eigenvectors.  It does not restore trained checkpoints or perform PPO training.

For an architecture/encoding baseline, the same fixed transition banks are
evaluated under independently initialized networks.  Each seed is cached
atomically, making that pass resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.analyze_achievement_transition_coherence import (
    DEFAULT_RUN,
    MASTER,
    apply_hidden_and_value,
)
from concept_mapping.analyze_diamond_representation_coherence import _coherence_statistics
from concept_mapping.plot_achievement_value_effects_with_success import log_bin_mean
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config


COHERENCE_ROOT = MASTER / "achievement_transition_coherence_stride3"
DEFAULT_OUT = COHERENCE_ROOT / "sword_matched_crafting_controls"
DEFAULT_BANK = MASTER / "achievement_transition_fixed_bank.npz"
SUCCESS_CSV = (
    MASTER / "achievement_stabilized_effect_over_checkpoints_stride3" /
    "wandb_achievement_success_history.csv"
)
CONCEPTS = (
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
)


def _atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Encountered a zero or non-finite transition direction")
    return vector / norm


def checkpoint_vectors(root: Path) -> tuple[list[int], dict[str, dict[int, np.ndarray]]]:
    vectors: dict[str, dict[int, np.ndarray]] = {}
    for concept in CONCEPTS:
        concept_vectors = {}
        for path in (root / concept / "per_checkpoint").glob("checkpoint_*.npz"):
            with np.load(path) as data:
                step = int(data["checkpoint"])
                concept_vectors[step] = _unit(data["top_eigenvector"])
        if not concept_vectors:
            raise RuntimeError(f"No saved checkpoint directions for {concept}")
        vectors[concept] = concept_vectors
    shared = sorted(set.intersection(*(set(item) for item in vectors.values())))
    if not shared:
        raise RuntimeError("No checkpoints shared by all sword and pickaxe concepts")
    return shared, vectors


def alignment_metrics(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    iron_sword = vectors["make_iron_sword"]
    result = {
        "iron_sword__wood_sword": abs(float(iron_sword @ vectors["make_wood_sword"])),
        "iron_sword__stone_sword": abs(float(iron_sword @ vectors["make_stone_sword"])),
        "iron_sword__wood_pickaxe": abs(float(iron_sword @ vectors["make_wood_pickaxe"])),
        "iron_sword__stone_pickaxe": abs(float(iron_sword @ vectors["make_stone_pickaxe"])),
        "iron_sword__iron_pickaxe": abs(float(iron_sword @ vectors["make_iron_pickaxe"])),
        "wood_sword__wood_pickaxe": abs(float(
            vectors["make_wood_sword"] @ vectors["make_wood_pickaxe"]
        )),
        "stone_sword__stone_pickaxe": abs(float(
            vectors["make_stone_sword"] @ vectors["make_stone_pickaxe"]
        )),
    }
    # Primary test: earlier sword-family reuse versus an exactly recipe-matched
    # non-sword craft. Positive values favor sword specificity.
    result["matched_sword_selectivity"] = (
        result["iron_sword__stone_sword"] - result["iron_sword__iron_pickaxe"]
    )
    result["matched_sword_selectivity_wood"] = (
        result["iron_sword__wood_sword"] - result["iron_sword__iron_pickaxe"]
    )
    return result


def trained_rows(root: Path) -> list[dict]:
    steps, by_concept = checkpoint_vectors(root)
    rows = []
    for step in steps:
        row = {"checkpoint": step}
        row.update(alignment_metrics({name: by_concept[name][step] for name in CONCEPTS}))
        rows.append(row)
    return rows


def load_fixed_banks(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    with np.load(path) as data:
        names = np.asarray(data["concept_names"]).astype(str).tolist()
        indices = np.asarray(data["concept_index"])
        before = np.asarray(data["before"], dtype=np.float32)
        after = np.asarray(data["after"], dtype=np.float32)
        episode_ids = np.asarray(data["episode_id"])
    banks = {}
    counts = {}
    episode_hashes = {}
    for concept in CONCEPTS:
        if concept not in names:
            raise RuntimeError(f"{concept} absent from fixed bank {path}")
        mask = indices == names.index(concept)
        banks[concept] = (before[mask], after[mask])
        counts[concept] = int(mask.sum())
        episode_hashes[concept] = hashlib.sha256(
            np.ascontiguousarray(episode_ids[mask]).tobytes()
        ).hexdigest()
    metadata = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "counts": counts,
        "episode_id_sha256": episode_hashes,
    }
    return banks, metadata


def pack_banks(banks: dict[str, tuple[np.ndarray, np.ndarray]]):
    parts, slices, cursor = [], {}, 0
    for concept in CONCEPTS:
        before, after = banks[concept]
        parts.extend((before, after))
        slices[concept] = (
            slice(cursor, cursor + len(before)),
            slice(cursor + len(before), cursor + 2 * len(before)),
        )
        cursor += 2 * len(before)
    return np.concatenate(parts), slices


def compute_random_seed(seed: int, network, init_obs, packed, slices,
                        activation: str, batch_size: int) -> dict[str, float]:
    params = network.init(jax.random.PRNGKey(seed), init_obs)
    hidden, _ = apply_hidden_and_value(params, packed, activation, batch_size)
    directions = {}
    for concept in CONCEPTS:
        before_slice, after_slice = slices[concept]
        delta_h = hidden[after_slice] - hidden[before_slice]
        _stats, _eigenvalues, vector, _mean, _norms = _coherence_statistics(delta_h)
        directions[concept] = _unit(vector)
    return alignment_metrics(directions)


def random_rows(args, banks, bank_sha256: str) -> list[dict]:
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    activation = str(config.get("ACTIVATION", "tanh"))
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    init_obs = jnp.zeros((1, *env.observation_space(env_params).shape), dtype=jnp.float32)
    packed, slices = pack_banks(banks)
    config_sha256 = hashlib.sha256(repr(sorted(
        (str(key), repr(value)) for key, value in config.items()
    )).encode()).hexdigest()
    cache = Path(args.out_dir) / "random_seed_cache"
    rows = []
    for seed in range(args.random_seeds):
        path = cache / f"seed_{seed:03d}.npz"
        if path.exists():
            with np.load(path) as data:
                valid = (
                    "bank_sha256" in data.files and "config_sha256" in data.files and
                    str(data["bank_sha256"]) == bank_sha256 and
                    str(data["config_sha256"]) == config_sha256 and
                    int(data["seed"]) == seed
                )
                if valid:
                    provenance = {"seed", "bank_sha256", "config_sha256"}
                    metrics = {
                        field: float(data[field]) for field in data.files
                        if field not in provenance
                    }
                else:
                    metrics = None
        else:
            metrics = None
        if metrics is None:
            metrics = compute_random_seed(
                seed, network, init_obs, packed, slices, activation, args.batch_size
            )
            _atomic_savez(
                path, seed=np.int32(seed), bank_sha256=np.asarray(bank_sha256),
                config_sha256=np.asarray(config_sha256),
                **{field: np.float64(value) for field, value in metrics.items()},
            )
        rows.append({"seed": seed, **metrics})
        print(f"random baseline seed {seed + 1}/{args.random_seeds}", flush=True)
    return rows


def summarize_random(rows: list[dict]) -> dict:
    summary = {}
    for field in rows[0]:
        if field == "seed":
            continue
        values = np.asarray([row[field] for row in rows], dtype=float)
        summary[field] = {
            "mean": float(values.mean()),
            "q025": float(np.quantile(values, .025)),
            "q975": float(np.quantile(values, .975)),
        }
    return summary


def success_curve() -> tuple[np.ndarray, np.ndarray, float]:
    with SUCCESS_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    x, y = [], []
    for row in rows:
        value = row.get("make_iron_sword", "")
        if value not in (None, ""):
            x.append(float(row["environment_steps"]))
            y.append(float(value))
    x_array, y_array = np.asarray(x), np.asarray(y)
    positive = x_array[y_array > 0]
    first_positive = float(positive.min()) if len(positive) else float("nan")
    bx, by = log_bin_mean(x_array, y_array, bins=500)
    return bx, by, first_positive


def nearest_row(rows: list[dict], target: float) -> dict:
    return min(rows, key=lambda row: abs(row["checkpoint"] - target))


def result_summary(rows: list[dict], random_summary: dict, first_success: float) -> dict:
    before = [row for row in rows if row["checkpoint"] < first_success]
    selectivity = "matched_sword_selectivity"
    peak = max(before, key=lambda row: row[selectivity]) if before else None
    rs = random_summary[selectivity]
    above = [row[selectivity] > rs["q975"] for row in before]
    runs, start = [], None
    for index, is_above in enumerate(above + [False]):
        if is_above and start is None:
            start = index
        elif not is_above and start is not None:
            runs.append((start, index - 1))
            start = None
    longest = max(runs, key=lambda run: run[1] - run[0] + 1) if runs else None
    longest_payload = None if longest is None else {
        "n_sampled_checkpoints": longest[1] - longest[0] + 1,
        "first_checkpoint": before[longest[0]]["checkpoint"],
        "last_checkpoint": before[longest[1]]["checkpoint"],
    }
    return {
        "primary_contrast": (
            "abs(v_make_iron_sword dot v_make_stone_sword) - "
            "abs(v_make_iron_sword dot v_make_iron_pickaxe)"
        ),
        "first_nonzero_logged_iron_sword_success_step": first_success,
        "checkpoint_near_4p19e6": nearest_row(rows, 4.19e6),
        "maximum_prebehavioral_selectivity": peak,
        "final_checkpoint": rows[-1],
        "random_initialization_primary_selectivity": rs,
        "n_prebehavioral_checkpoints": len(before),
        "n_prebehavioral_above_random_q975": int(sum(above)),
        "fraction_prebehavioral_above_random_q975": float(np.mean(above)) if before else float("nan"),
        "longest_prebehavioral_run_above_random_q975": longest_payload,
        "fraction_all_checkpoints_above_random_q975": float(np.mean([
            row[selectivity] > rs["q975"] for row in rows
        ])),
    }


def plot_results(args, rows, random_summary, success_x, success_y) -> None:
    x = np.asarray([row["checkpoint"] for row in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.0), sharex=True,
                             constrained_layout=True)
    top, middle, bottom = axes

    alignment_fields = (
        ("iron_sword__stone_sword", "Iron sword vs. stone sword", "#7c3aed"),
        ("iron_sword__iron_pickaxe", "Iron sword vs. iron pickaxe (exact recipe match)", "#dc2626"),
        ("iron_sword__wood_sword", "Iron sword vs. wood sword", "#2563eb"),
        ("iron_sword__stone_pickaxe", "Iron sword vs. stone pickaxe", "#d97706"),
    )
    for field, label, color in alignment_fields:
        top.plot(x, [row[field] for row in rows], color=color, linewidth=1.9, label=label)
        rs = random_summary[field]
        top.axhline(rs["mean"], color=color, linestyle=":", linewidth=.9, alpha=.75)
    top.set_ylim(0, 1)
    top.set_ylabel("Absolute eigenvector alignment")
    top.set_title("Iron-sword alignment: sword relatives vs. matched crafting controls")
    top.grid(True, which="both", alpha=.20)
    top.legend(frameon=False, ncol=2)

    same_tier = (
        ("wood_sword__wood_pickaxe", "Wood sword vs. wood pickaxe", "#059669"),
        ("stone_sword__stone_pickaxe", "Stone sword vs. stone pickaxe", "#d97706"),
        ("iron_sword__iron_pickaxe", "Iron sword vs. iron pickaxe", "#dc2626"),
    )
    for field, label, color in same_tier:
        middle.plot(x, [row[field] for row in rows], color=color, linewidth=1.9, label=label)
        rs = random_summary[field]
        middle.axhspan(rs["q025"], rs["q975"], color=color, alpha=.045)
    middle.set_ylim(0, 1)
    middle.set_ylabel("Absolute eigenvector alignment")
    middle.set_title("Within-tier sword–pickaxe alignment")
    middle.grid(True, which="both", alpha=.20)
    middle.legend(frameon=False, ncol=3)

    field = "matched_sword_selectivity"
    bottom.plot(x, [row[field] for row in rows], color="#1d4ed8", linewidth=2.1,
                label="Iron–stone-sword minus iron–iron-pickaxe")
    rs = random_summary[field]
    bottom.axhspan(rs["q025"], rs["q975"], color="#64748b", alpha=.17,
                   label="Random-init 95% interval")
    bottom.axhline(0, color="#475569", linestyle="--", linewidth=1.0)
    right = bottom.twinx()
    right.plot(success_x, success_y, color="#dc2626", linewidth=1.6, alpha=.85,
               label="Iron-sword success")
    right.set_ylim(0, 100)
    right.set_ylabel("Make Iron Sword success rate (%)", color="#b91c1c")
    right.tick_params(axis="y", colors="#b91c1c")
    bottom.set_ylabel("Matched sword selectivity")
    bottom.set_title("Does sword-family alignment exceed exact recipe-matched crafting?")
    bottom.grid(True, which="both", alpha=.20)
    h1, l1 = bottom.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    bottom.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    bottom.set_xscale("log")
    bottom.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9],
                      [r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$"])
    bottom.set_xlabel("Training environment steps")
    fig.suptitle("Matched crafting control for sword-direction reuse", fontsize=17)
    out = Path(args.out_dir)
    fig.savefig(out / "iron_sword_matched_crafting_controls.png", dpi=300)
    fig.savefig(out / "iron_sword_matched_crafting_controls.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN)
    parser.add_argument("--coherence_root", type=Path, default=COHERENCE_ROOT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random_seeds", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--skip_random", action="store_true")
    args = parser.parse_args()
    if args.random_seeds <= 0 or args.batch_size <= 0:
        raise ValueError("random_seeds and batch_size must be positive")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    rows = trained_rows(args.coherence_root)
    trained_path = Path(args.out_dir) / "iron_sword_matched_crafting_alignment.csv"
    _write_csv(trained_path, rows)
    banks, bank_metadata = load_fixed_banks(args.bank)
    if args.skip_random:
        print(trained_path)
        return

    baseline = random_rows(args, banks, bank_metadata["sha256"])
    baseline_path = Path(args.out_dir) / "random_initialization_matched_baseline.csv"
    _write_csv(baseline_path, baseline)
    random_summary = summarize_random(baseline)
    random_summary_path = Path(args.out_dir) / "random_initialization_matched_summary.json"
    random_summary_path.write_text(json.dumps(random_summary, indent=2) + "\n")
    success_x, success_y, first_success = success_curve()
    summary = result_summary(rows, random_summary, first_success)
    summary_path = Path(args.out_dir) / "matched_crafting_result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    metadata = {
        "run_path": args.run_path,
        "coherence_root": str(args.coherence_root),
        "n_checkpoints": len(rows),
        "random_seeds": args.random_seeds,
        "random_seed_range": [0, args.random_seeds - 1],
        "concepts": list(CONCEPTS),
        "fixed_transition_bank": bank_metadata,
        "representation": "post-activation Dense_6 critic output immediately before scalar Dense_7",
        "direction_statistic": "top eigenvector of uncentered E[Delta h Delta h^T]",
        "primary_matched_control": "make_iron_pickaxe",
        "matching_rationale": (
            "Make Iron Sword and Make Iron Pickaxe both require a table and furnace, "
            "consume one wood, stone, coal, and iron, and differ only in output slot."
        ),
        "warning": (
            "Natural transition banks may differ in surrounding state distribution and "
            "completion time; the control matches the environment transition mechanics, "
            "not the full observational distribution."
        ),
    }
    (Path(args.out_dir) / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    plot_results(args, rows, random_summary, success_x, success_y)
    for path in (trained_path, baseline_path, random_summary_path, summary_path):
        print(path)


if __name__ == "__main__":
    main()

"""Checkpoint sweep for the conditional wood-pickaxe harvesting affordance.

Computes value, policy-DO-logit/probability, and critic-representation 2x2
interactions on the same frozen stone/coal states at every PPO checkpoint.
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
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)
    jax.tree = _JaxTreeCompat()
import jax.numpy as jnp
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from craftax.craftax_classic.constants import Action
from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.analyze_diamond_representation_coherence import _coherence_statistics
from concept_mapping.critic_representation import critic_pre_value_representation, validate_critic_representation
from concept_mapping.plot_diamond_ordering_over_checkpoints import _checkpoint_steps
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


DEFAULT_RUN = "wandb/run-20260830_231509-616wuss8"
DEFAULT_BANK = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/wood_pickaxe_functional/wood_pickaxe_functional_bank.npz"
DEFAULT_OUT = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/wood_pickaxe_functional/checkpoint_sweep"


def _load_bank(path: Path):
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _apply(params, observations: np.ndarray, network, activation: str, batch_size: int):
    @jax.jit
    def apply_batch(parameters, batch):
        h, value = critic_pre_value_representation(parameters, batch, activation)
        policy, _ = network.apply(parameters, batch)
        logits = policy.logits[:, Action.DO.value]
        probability = jax.nn.softmax(policy.logits, axis=-1)[:, Action.DO.value]
        return h, value, logits, probability
    chunks = [[], [], [], []]
    for start in range(0, len(observations), batch_size):
        output = apply_batch(params, jnp.asarray(observations[start:start + batch_size]))
        for target, values in zip(chunks, output):
            target.append(np.asarray(values))
    return tuple(np.concatenate(values) for values in chunks)


def _output_path(out_dir: Path, step: int) -> Path:
    return out_dir / "per_checkpoint" / f"checkpoint_{step}.npz"


def _run_worker(args, steps: list[int], bank: dict):
    config = _load_wandb_config(args.run_path)
    config["ENV_NAME"] = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    activation = str(config.get("ACTIVATION", "tanh"))
    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    packed = np.concatenate([bank[key].astype(np.float32) for key in "ABCD"])
    n = len(bank["A"])
    validated = False
    for step in steps:
        path = _output_path(Path(args.out_dir), step)
        if path.exists():
            continue
        print(f"worker {args.worker_index}: checkpoint {step:,}", flush=True)
        state, restored = _restore_train_state(args.run_path, config, network, env, env_params, step)
        if not validated:
            check = validate_critic_representation(network, state.params, jnp.asarray(bank["D"][:32]), activation)
            if check["value_reconstruction_max_abs_error"] > 1e-5:
                raise RuntimeError(check)
            print(f"worker {args.worker_index}: representation validated {check}", flush=True)
            validated = True
        h, value, logit, probability = _apply(state.params, packed, network, activation, args.batch_size)
        hs = dict(zip("ABCD", np.split(h, 4)))
        vs = dict(zip("ABCD", np.split(value, 4)))
        ls = dict(zip("ABCD", np.split(logit, 4)))
        ps = dict(zip("ABCD", np.split(probability, 4)))
        interaction_h = (hs["D"] - hs["C"]) - (hs["B"] - hs["A"])
        interaction_v = (vs["D"] - vs["C"]) - (vs["B"] - vs["A"])
        interaction_l = (ls["D"] - ls["C"]) - (ls["B"] - ls["A"])
        interaction_p = (ps["D"] - ps["C"]) - (ps["B"] - ps["A"])
        stats, eigenvalues, top_vector, mean_delta, norms = _coherence_statistics(interaction_h)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": np.int64(restored), "n_states": np.int32(n),
            "eigenvalues": eigenvalues, "lambda_1": np.float64(stats["lambda_1"]),
            "total_eigenvalue_variance": np.float64(stats["total_eigenvalue_variance"]),
            "top_eigenvalue_fraction": np.float64(stats["top_eigenvalue_fraction"]),
            "top_eigenvector": top_vector, "mean_interaction_h": mean_delta,
            "interaction_h_norms": norms,
            "interaction_value": interaction_v, "interaction_do_logit": interaction_l,
            "interaction_do_probability": interaction_p,
            "tool_effect_value_resource_present": vs["D"] - vs["C"],
            "tool_effect_value_resource_absent": vs["B"] - vs["A"],
            "tool_effect_logit_resource_present": ls["D"] - ls["C"],
            "tool_effect_logit_resource_absent": ls["B"] - ls["A"],
        }
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(path)


def _group_summary(values: np.ndarray, mask: np.ndarray):
    selected = values[mask]
    return float(selected.mean()), float(np.median(selected)), float((selected > 0).mean())


def _cluster_bootstrap(values: np.ndarray, episodes: np.ndarray, mask: np.ndarray, rng, draws=5000):
    values, episodes = values[mask], episodes[mask]
    unique = np.unique(episodes)
    means, fractions = np.empty(draws), np.empty(draws)
    groups = {episode: values[episodes == episode] for episode in unique}
    for draw in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        draw_values = np.concatenate([groups[episode] for episode in sampled])
        means[draw] = draw_values.mean(); fractions[draw] = (draw_values > 0).mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])], [float(x) for x in np.quantile(fractions, [0.025, 0.975])]


def _plot_lines(rows: list[dict], field: str, ylabel: str, path: Path, percent=False):
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    colors = {"pooled": "#111827", "stone": "#2563eb", "coal": "#d97706"}
    for group in ("pooled", "stone", "coal"):
        selected = [row for row in rows if row["group"] == group]
        x = np.asarray([row["checkpoint"] for row in selected], float)
        y = np.asarray([row[field] for row in selected], float) * (100 if percent else 1)
        ax.plot(x, y, color=colors[group], linewidth=1.8, label=group)
    ax.set_xscale("log"); ax.set_xlabel("Training environment steps"); ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=.25); ax.legend(frameon=False)
    fig.savefig(path.with_suffix(".png"), dpi=300); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def _aggregate(args, steps: list[int], bank: dict):
    out_dir = Path(args.out_dir)
    missing = [step for step in steps if not _output_path(out_dir, step).exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} checkpoints; first missing={missing[:5]}")
    loaded = []
    for step in steps:
        with np.load(_output_path(out_dir, step)) as data:
            loaded.append({key: np.asarray(data[key]) for key in data.files})
    final_vector = np.asarray(loaded[-1]["top_eigenvector"], float)
    final_vector /= np.linalg.norm(final_vector)
    representation_rows, value_rows = [], []
    resources = bank["resource"].astype(str)
    group_masks = {"pooled": np.ones(len(resources), bool), "stone": resources == "stone", "coal": resources == "coal"}
    for data in loaded:
        vector = np.asarray(data["top_eigenvector"], float); vector /= np.linalg.norm(vector)
        representation_rows.append({
            "checkpoint": int(data["checkpoint"]), "lambda_1": float(data["lambda_1"]),
            "top_eigenvalue_fraction": float(data["top_eigenvalue_fraction"]),
            "alignment_to_final": float(abs(np.dot(vector, final_vector))),
        })
        for group, mask in group_masks.items():
            value_mean, value_median, value_positive = _group_summary(data["interaction_value"], mask)
            logit_mean, logit_median, logit_positive = _group_summary(data["interaction_do_logit"], mask)
            probability_mean, probability_median, probability_positive = _group_summary(data["interaction_do_probability"], mask)
            value_rows.append({"checkpoint": int(data["checkpoint"]), "group": group,
                               "mean_value_interaction": value_mean, "median_value_interaction": value_median, "fraction_value_interaction_positive": value_positive,
                               "mean_do_logit_interaction": logit_mean, "median_do_logit_interaction": logit_median, "fraction_do_logit_interaction_positive": logit_positive,
                               "mean_do_probability_interaction": probability_mean, "median_do_probability_interaction": probability_median, "fraction_do_probability_interaction_positive": probability_positive})
    with (out_dir / "representation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(representation_rows[0])); writer.writeheader(); writer.writerows(representation_rows)
    with (out_dir / "value_policy_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(value_rows[0])); writer.writeheader(); writer.writerows(value_rows)
    x = np.asarray([row["checkpoint"] for row in representation_rows], float)
    for field, ylabel, filename in (("lambda_1", r"$\lambda_1$ of interaction residual", "representation_lambda1"),
                                    ("top_eigenvalue_fraction", r"$\lambda_1/\sum_j\lambda_j$", "representation_coherence"),
                                    ("alignment_to_final", "Alignment to final direction", "representation_alignment")):
        fig, ax = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
        ax.plot(x, [r[field] for r in representation_rows], color="#7c3aed", linewidth=1.8)
        ax.set_xscale("log"); ax.set_xlabel("Training environment steps"); ax.set_ylabel(ylabel); ax.grid(True, which="both", alpha=.25)
        fig.savefig(out_dir / f"{filename}.png", dpi=300); fig.savefig(out_dir / f"{filename}.pdf"); plt.close(fig)
    _plot_lines(value_rows, "mean_value_interaction", r"Mean value interaction $I_V$", out_dir / "mean_value_interaction")
    _plot_lines(value_rows, "fraction_value_interaction_positive", r"States with $I_V>0$ (%)", out_dir / "value_interaction_success", percent=True)
    _plot_lines(value_rows, "mean_do_logit_interaction", r"Mean DO-logit interaction $I_\pi$", out_dir / "mean_policy_interaction")
    _plot_lines(value_rows, "fraction_do_logit_interaction_positive", r"States with $I_\pi>0$ (%)", out_dir / "policy_interaction_success", percent=True)
    final = loaded[-1]
    rng = np.random.default_rng(50)
    final_summary = {"checkpoint": int(final["checkpoint"]), "n_states": len(resources), "groups": {}}
    for group, mask in group_masks.items():
        group_result = {}
        for label, field in (("value", "interaction_value"), ("do_logit", "interaction_do_logit"), ("do_probability", "interaction_do_probability")):
            mean, median, fraction = _group_summary(final[field], mask)
            mean_ci, fraction_ci = _cluster_bootstrap(final[field], bank["episode_id"], mask, rng)
            group_result[label] = {"mean": mean, "median": median, "fraction_positive": fraction, "cluster_bootstrap_95ci_mean": mean_ci, "cluster_bootstrap_95ci_fraction_positive": fraction_ci}
        final_summary["groups"][group] = group_result
    (out_dir / "final_checkpoint_summary.json").write_text(json.dumps(final_summary, indent=2))
    (out_dir / "sweep_metadata.json").write_text(json.dumps({"run_path": args.run_path, "bank": args.bank, "n_checkpoints": len(steps), "representation_interaction": "(h_D-h_C)-(h_B-h_A)", "value_interaction": "(V_D-V_C)-(V_B-V_A)", "policy_interaction": "same difference-in-differences on DO logit/probability"}, indent=2))
    print(json.dumps(final_summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN); parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--out_dir", default=DEFAULT_OUT); parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--worker_index", type=int, default=0); parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--aggregate_only", action="store_true"); parser.add_argument("--checkpoints", default="")
    args = parser.parse_args()
    steps = [int(x) for x in args.checkpoints.split(",") if x.strip()] if args.checkpoints else _checkpoint_steps(args.run_path)
    bank = _load_bank(Path(args.bank)); Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(args, steps, bank); return
    worker_steps = [step for index, step in enumerate(steps) if index % args.num_workers == args.worker_index]
    print(f"worker {args.worker_index}/{args.num_workers}: {len(worker_steps)} checkpoints", flush=True)
    _run_worker(args, worker_steps, bank)


if __name__ == "__main__":
    main()

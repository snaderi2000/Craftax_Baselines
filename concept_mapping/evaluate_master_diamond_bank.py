"""Evaluate the frozen master-dataset diamond/pickaxe bank at one PPO checkpoint.

This intentionally uses only the two clean counterfactuals: C is a visible
diamond without an iron pickaxe, and D is the identical state with one.  Thus
``D > C`` is the direct diamond-collectability test requested for the initial
master-bank replication.
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
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from concept_mapping.diamond_affordance_over_time import _eval_values, _make_counterfactuals
from concept_mapping.render_ppo_episodes import _init_network, _load_wandb_config, _restore_train_state


DEFAULT_RUN_PATH = "wandb/run-20260830_231509-616wuss8"
DEFAULT_BANK = "concept_mapping/runs/master_symbolic_1p5b_1000episodes/concept_index/diamond_collectible_state_bank.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_path", default=DEFAULT_RUN_PATH)
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--timestep", type=int, default=None, help="Defaults to the run's final checkpoint.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out_dir", default=None, help="Defaults beside --bank in master_diamond_replication.")
    args = parser.parse_args()

    bank_path = Path(args.bank)
    out_dir = Path(args.out_dir) if args.out_dir else bank_path.parent / "master_diamond_replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(bank_path) as bank:
        base = np.asarray(bank["base_obs"], dtype=np.float32)
        episodes = np.asarray(bank["episodes"], dtype=np.int32)
        steps = np.asarray(bank["base_steps"], dtype=np.int32)
        source_npz = np.asarray(bank["source_npz"])
    counterfactuals = [_make_counterfactuals(obs) for obs in base]
    arrays = {name: np.stack([cf[name] for cf in counterfactuals]) for name in ("C_diamond_no_pickaxe", "D_diamond_pickaxe")}

    config = _load_wandb_config(args.run_path)
    env_name = config.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    config["ENV_NAME"] = env_name
    env = make_craftax_env_from_name(env_name, auto_reset=False)
    env_params = env.default_params
    network = _init_network(config, env, env_params)
    train_state, restored_step = _restore_train_state(args.run_path, config, network, env, env_params, args.timestep)
    values = _eval_values(network, train_state.params, arrays, args.batch_size)
    c, d = values["C_diamond_no_pickaxe"], values["D_diamond_pickaxe"]
    delta = d - c
    passed = d > c
    with (out_dir / "diamond_pickaxe_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "episode", "t", "source_npz", "C_diamond_no_pickaxe", "D_diamond_pickaxe", "delta_D_minus_C", "d_gt_c"])
        writer.writeheader()
        for i in range(len(base)):
            writer.writerow({"sample_index": i, "episode": int(episodes[i]), "t": int(steps[i]), "source_npz": str(source_npz[i]), "C_diamond_no_pickaxe": float(c[i]), "D_diamond_pickaxe": float(d[i]), "delta_D_minus_C": float(delta[i]), "d_gt_c": bool(passed[i])})
    summary = {
        "checkpoint": int(restored_step), "n_states": int(len(base)), "d_gt_c_rate": float(passed.mean()),
        "mean_delta_D_minus_C": float(delta.mean()), "median_delta_D_minus_C": float(np.median(delta)),
        "positive_delta_count": int(passed.sum()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_dir / 'diamond_pickaxe_values.csv'}")


if __name__ == "__main__":
    main()

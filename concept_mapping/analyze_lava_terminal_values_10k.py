"""Evaluate the frozen critic immediately before lava-ending terminal actions."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from concept_mapping.analyze_wood_sword_combat_followup import (
    _init_network, _load_wandb_config, _restore_train_state, evaluate_values,
    make_craftax_env_from_name,
)


ROOT = Path(__file__).resolve().parent / "runs"
EPISODE_SETS = ((50, "master_symbolic_10b_5000episodes"),
                (51, "master_symbolic_10b_5000episodes_seed51"))
RUN = "wandb/run-20260630_214658-m0mw4end"
CHECKPOINT = 9_999_941_632
OUT = ROOT / "lava_terminal_values_10b_10000episodes"


def summarize(values):
    n = len(values)
    result = {
        "n": n,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q05": float(np.quantile(values, .05)),
        "q25": float(np.quantile(values, .25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, .75)),
        "q95": float(np.quantile(values, .95)),
        "max": float(values.max()),
    }
    for name, mask in (("negative", values < 0), ("exact_zero", values == 0),
                       ("positive", values > 0),
                       ("absolute_le_0.01", np.abs(values) <= .01),
                       ("absolute_le_0.05", np.abs(values) <= .05),
                       ("absolute_le_0.10", np.abs(values) <= .10)):
        result[name + "_count"] = int(mask.sum())
        result[name + "_rate"] = float(mask.mean())
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, observations = [], []
    for seed, folder in EPISODE_SETS:
        for path in sorted((ROOT / folder / "episodes").glob("episode_*.npz")):
            with np.load(path) as data:
                terminal = data["terminal_observation"]
                center_block = int(terminal[:1323].reshape(7, 9, 21)[3, 4, :17].argmax())
                if not bool(data["dones"][-1]) or center_block != 14:
                    continue
                observations.append(data["observations"][-1].astype(np.float32, copy=True))
                achievements = data["terminal_achievements"]
                intrinsic = data["intrinsics"][-1]
                rows.append(dict(
                    seed=seed, episode_id=int(data["episode_id"]),
                    timestep=int(data["step"][-1]), episode_length=len(data["actions"]),
                    action=int(data["actions"][-1]), collected_diamond=bool(achievements[19]),
                    health=int(intrinsic[0]), food=int(intrinsic[1]),
                    water=int(intrinsic[2]), energy=int(intrinsic[3]),
                ))
    observations = np.stack(observations)
    assert len(rows) == 6904
    config = _load_wandb_config(RUN)
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    network = _init_network(config, env, env.default_params)
    state, checkpoint = _restore_train_state(RUN, config, network, env, env.default_params, CHECKPOINT)
    values = evaluate_values(state.params, network, observations, 512).astype(float)
    for row, value in zip(rows, values):
        row["critic_value"] = float(value)
    with (OUT / "per_episode_values.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    diamond = np.asarray([row["collected_diamond"] for row in rows])
    summary = {
        "checkpoint": checkpoint,
        "state_definition": "Final saved pre-action observation immediately before the action whose terminal observation places the player on lava.",
        "near_zero_definitions": "Reported at |V| <= 0.01, 0.05, and 0.10; exact zero is separately counted.",
        "all_lava_endings": summarize(values),
        "diamond_collected": summarize(values[diamond]),
        "diamond_not_collected": summarize(values[~diamond]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

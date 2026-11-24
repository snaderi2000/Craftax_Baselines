"""Replay a saved crop-plus-world evaluation using one observed map and the same weights."""
import argparse
import csv
from functools import partial
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cap_reward_shaped_runs import try_evaluation_lock
from run_ppo_study import require_free
from run_reward_shaped_architectures import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be unique")
    locks = [try_evaluation_lock(i) for i in args.gpus]
    if None in locks:
        raise RuntimeError("Requested evaluation GPU is reserved")
    require_free(args.gpus)
    os.environ.update(environment(args.gpus))
    from evaluate_reward_shaped_ppo import Achievement, build_evaluator, jax, jnp, ocp, make_craftax_env_from_name
    from fixed_world_seeds import read_world_seeds
    from models.actor_critic_map_mlp import map_mlp_network
    from symbolic_map_memory import GLOBAL_MAP_FEATURES, LOCAL_SIZE, with_map_memory

    original = args.teacher / "evaluation_wallclock"
    reference = json.loads((original / "summary.json").read_text())
    with (original / "episodes.csv").open() as f:
        rows = list(csv.DictReader(f))
    cfg = json.loads((args.teacher / "inference_config.json").read_text())
    if (cfg.get("SYMBOLIC_ENCODER") != "map_mlp" or cfg.get("RECURRENT") or
            not cfg.get("MAP_GLOBAL_READOUT") or cfg.get("MAP_FULL_TERRAIN") or
            cfg.get("MAP_SINGLE_READOUT") or cfg.get("MAP_LANDMARKS")):
        raise ValueError("Expected a feedforward crop-plus-observed-world checkpoint")
    bank = np.asarray(read_world_seeds(reference["world_seeds_file"]), dtype=np.uint32)
    if (reference["trials_per_world"] != 1 or [int(row["world_seed"]) for row in rows] != bank.tolist()
            or [int(row["episode"]) for row in rows] != list(range(len(bank)))):
        raise ValueError("Expected one ordered reference episode per world")
    batch = json.loads((args.teacher / "experiment_metadata.json").read_text())["evaluation"]["batch"]
    devices = jax.local_devices(backend="gpu")
    if len(devices) != len(args.gpus):
        raise RuntimeError("Visible devices differ from requested GPUs")
    env = with_map_memory(make_craftax_env_from_name(cfg["ENV_NAME"], False), dict(cfg, MAP_SINGLE_READOUT=True))
    assert env.observation_space(env.default_params).shape == (LOCAL_SIZE + GLOBAL_MAP_FEATURES,)
    net = map_mlp_network(cfg)
    reader = SimpleNamespace(apply=partial(net.apply, method=net.from_observed_world))
    checkpoint = Path(reference["checkpoint"])
    weights = ocp.PyTreeCheckpointer().restore(str(checkpoint / "default" if (checkpoint / "default").is_dir() else checkpoint))["params"]
    evaluate = build_evaluator(env, reader, weights, cfg, seed=reference["evaluation_seed"], batch=batch,
                               world_seeds=jnp.asarray(bank), max_steps=reference["max_episode_steps"],
                               temperature=reference["temperature"], resource_timing=True)
    evaluate_shards = jax.pmap(lambda ids: evaluate(ids), devices=devices)
    stride = batch * len(devices)
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for start in range(0, len(bank), stride):
        ids = np.arange(start, start+stride).reshape(len(devices), batch)
        outputs = jax.device_get(evaluate_shards(ids))
        _, finished, returns, shaped, lengths, achievements, lava, death, timeout, first = outputs
        for d, lane in np.ndindex(ids.shape):
            i = ids[d, lane]
            if i >= len(rows):
                continue
            ach = achievements[d, lane]
            r = rows[i]
            new = dict(episode=int(i), world_seed=int(bank[i]), all_22=bool(ach.all()),
                       length=int(lengths[d, lane]), terminated=bool(finished[d, lane]),
                       lava=bool(lava[d, lane]), nonlava_death=bool(death[d, lane]), timeout=bool(timeout[d, lane]),
                       original_return=float(returns[d, lane]), shaped_return=float(shaped[d, lane]))
            for k, field in enumerate(("first_drink_step", "first_diamond_step")):
                tick = int(first[d, lane, k])
                new[field] = tick if tick >= 0 else ""
            new.update({a.name.lower(): int(ach[a.value]) for a in Achievement})
            same = all(str(v) == r[k] for k, v in new.items() if k not in ("original_return", "shaped_return"))
            records.append(dict(**new, reference_all_22=r["all_22"] == "True", matched_outcomes_and_timings=same))
        print(f"Evaluated {len(records)}/{len(rows)}", flush=True)
    with (args.out / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    report = dict(checkpoint=str(checkpoint), reference=str(original.resolve()),
                  world_bank=reference["world_seeds_file"], seed=reference["evaluation_seed"],
                  episode_cap=reference["max_episode_steps"], temperature=reference["temperature"], batch_per_gpu=batch,
                  episodes=len(records), one_map_all_22=sum(r["all_22"] for r in records),
                  reference_all_22=sum(r["reference_all_22"] for r in records),
                  gained=sum(r["all_22"] and not r["reference_all_22"] for r in records),
                  lost=sum(r["reference_all_22"] and not r["all_22"] for r in records),
                  matched_outcomes_and_timings=sum(r["matched_outcomes_and_timings"] for r in records))
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

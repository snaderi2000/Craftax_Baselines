"""Ripe versus unripe target plant at first EAT_PLANT states, frozen critic."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_wood_sword_combat_followup import (
    save_csv, evaluate_values, _load_wandb_config, _init_network,
    _restore_train_state, make_craftax_env_from_name,
)

ROOT = Path(__file__).resolve().parent / "runs"
OUT = ROOT / "plant_collectability_10b_10000episodes_ripe_vs_unripe"


def bootstrap(effect, draws=10_000, seed=50):
    rng = np.random.default_rng(seed)
    n = len(effect)
    success = (effect > 0).astype(float)
    estimates = np.empty((draws, 2))
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        indices = rng.integers(0, n, (size, n))
        estimates[start:start + size, 0] = success[indices].mean(axis=1)
        estimates[start:start + size, 1] = effect[indices].mean(axis=1)
    return np.percentile(estimates, (2.5, 97.5), axis=0)


def main():
    OUT.mkdir(exist_ok=True)
    rows, factual, no_event = [], [], 0
    for seed, folder in [(50, "master_symbolic_10b_5000episodes"),
                         (51, "master_symbolic_10b_5000episodes_seed51")]:
        for path in sorted((ROOT / folder / "episodes").glob("episode_*.npz")):
            with np.load(path) as data:
                flags = data["achievements"][:, 11]
                terminal_flag = data["terminal_achievements"][11]
                times = np.flatnonzero(np.diff(np.r_[flags, terminal_flag].astype(int)) > 0)
                if len(times) == 0:
                    no_event += 1
                    continue
                t = int(times[0])
                direction = int(data["direction"][t])
                row = 3 + [0, 0, -1, 1][direction]
                col = 4 + [-1, 1, 0, 0][direction]
                assert data["actions"][t] == 5 and not data["sleeping"][t]
                assert data["visible_blocks"][t, row, col] == 16 and not flags[t]
                obs = data["observations"][t].copy()
                assert obs[(row * 9 + col) * 21 + 16] == 1
                factual.append(obs)
                rows.append(dict(
                    seed=seed, episode_id=int(data["episode_id"]), timestep=t,
                    target_row=row, target_col=col,
                    health=int(data["intrinsics"][t, 0]),
                    food=int(data["intrinsics"][t, 1]),
                    water=int(data["intrinsics"][t, 2]),
                    energy=int(data["intrinsics"][t, 3]),
                    inventory=json.dumps(data["inventory"][t].astype(int).tolist()),
                    achievement_flags=json.dumps(data["achievements"][t].astype(int).tolist()),
                ))
    n = len(rows)
    assert (n, no_event) == (1883, 8117)
    assert len({(r["seed"], r["episode_id"]) for r in rows}) == n
    ripe = np.stack(factual)
    unripe = ripe.copy()
    offsets = np.array([(r["target_row"] * 9 + r["target_col"]) * 21 for r in rows])
    unripe[np.arange(n), offsets + 16] = 0
    unripe[np.arange(n), offsets + 15] = 1
    changes = np.sum(ripe != unripe, axis=1)
    assert np.all(changes == 2)
    assert np.array_equal(ripe[:, 1323:], unripe[:, 1323:])
    assert np.all(unripe[np.arange(n), offsets + 15] == 1)
    np.savez_compressed(OUT / "base_states.npz", observations=ripe,
                        seed=[r["seed"] for r in rows],
                        episode_id=[r["episode_id"] for r in rows])
    print(f"Validated {n} first-ripe-plant states", flush=True)

    run = "wandb/run-20260630_214658-m0mw4end"
    config = _load_wandb_config(run)
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    net = _init_network(config, env, env.default_params)
    state, checkpoint = _restore_train_state(
        run, config, net, env, env.default_params, 9999941632)
    value_ripe = evaluate_values(state.params, net, ripe, 512).astype(float)
    value_unripe = evaluate_values(state.params, net, unripe, 512).astype(float)
    effect = value_ripe - value_unripe

    pilot = list(csv.DictReader((ROOT / "collectability_three_tasks_10b_pilot" /
                                  "plant/per_state_results.csv").open()))
    lookup = {(r["seed"], r["episode_id"], r["timestep"]): i for i, r in enumerate(rows)}
    max_error = 0.0
    for sample in pilot:
        index = lookup[tuple(int(sample[k]) for k in ("seed", "episode_id", "timestep"))]
        for actual, key in ((value_ripe[index], "V_ripe"),
                            (value_unripe[index], "V_unripe")):
            max_error = max(max_error, abs(actual - float(sample[key])))
            assert np.isclose(actual, float(sample[key]), atol=1e-5, rtol=1e-5)

    for i, record in enumerate(rows):
        record.update(V_ripe=float(value_ripe[i]), V_unripe=float(value_unripe[i]),
                      ripe_effect=float(effect[i]), success=bool(effect[i] > 0))
    save_csv(OUT / "per_state_results.csv", rows)
    summaries = []
    for group, mask in (("all", np.ones(n, dtype=bool)),
                        ("food_below_9", np.array([r["food"] < 9 for r in rows])),
                        ("food_9", np.array([r["food"] == 9 for r in rows]))):
        values = effect[mask]
        ci = bootstrap(values)
        summaries.append(dict(
            group=group, n=len(values), success_count=int((values > 0).sum()),
            success_rate=float((values > 0).mean()),
            success_ci_low=float(ci[0, 0]), success_ci_high=float(ci[1, 0]),
            mean_effect=float(values.mean()), median_effect=float(np.median(values)),
            mean_effect_ci_low=float(ci[0, 1]), mean_effect_ci_high=float(ci[1, 1]),
        ))
    save_csv(OUT / "summary.csv", summaries)
    metadata = dict(checkpoint=checkpoint, n=n, no_first_plant_achievement=no_event,
                    criterion="V(ripe target plant) > V(unripe target plant)",
                    selected="One awake pre-action state before first EAT_PLANT achievement per episode; facing ripe plant; DO action.",
                    intervention="Only the target RIPE_PLANT block channel changes to PLANT; inventory and other features preserved.",
                    bootstrap="10,000 paired episode resamples per group, seed 50, percentile 95% intervals.",
                    pilot_reference_max_error=max_error,
                    caveat="Unripe plants may later ripen; this is a current ripe-versus-unripe opportunity probe.")
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(summary=summaries, metadata=metadata), indent=2), flush=True)


if __name__ == "__main__":
    main()

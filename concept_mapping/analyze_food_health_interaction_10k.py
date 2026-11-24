"""Evaluate health x food level x visible-cow counterfactuals on 10k episodes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from craftax.craftax_classic.constants import BlockType

from concept_mapping.analyze_wood_sword_combat_followup import (
    _init_network, _load_wandb_config, _restore_train_state, evaluate_values,
    make_craftax_env_from_name, save_csv,
)
from concept_mapping.visualize_symbolic_trajectory import MAP_CHANNELS, MAP_DIM


ROOT = Path(__file__).resolve().parent / "runs"
EPISODE_SETS = ((50, "master_symbolic_10b_5000episodes"),
                (51, "master_symbolic_10b_5000episodes_seed51"))
OUT = ROOT / "food_health_interaction_10b_10000episodes"
RUN = "wandb/run-20260630_214658-m0mw4end"
CHECKPOINT = 9_999_941_632
HEALTH_OFFSET = MAP_DIM + 12
FOOD_OFFSET = MAP_DIM + 13
COW_CHANNEL = len(BlockType) + 1
LOOKBACK = 5


def collect_states():
    rows, observations = [], []
    episodes_with_restoration = 0
    no_eligible = 0
    for seed, folder in EPISODE_SETS:
        for path in sorted((ROOT / folder / "episodes").glob("episode_*.npz")):
            with np.load(path) as data:
                intrinsics = data["intrinsics"]
                after = np.empty(len(intrinsics), dtype=np.int16)
                after[:-1] = intrinsics[1:, 1]
                after[-1] = int(round(float(data["terminal_observation"][FOOD_OFFSET]) * 10))
                events = np.flatnonzero(after > intrinsics[:, 1])
                episodes_with_restoration += bool(len(events))
                selected = None
                for event_t in events:
                    direction = int(data["direction"][event_t])
                    target_row = 3 + (0, 0, -1, 1)[direction]
                    target_col = 4 + (-1, 1, 0, 0)[direction]
                    # Distinguish cow consumption from any other possible
                    # source of increased food.
                    if (int(data["actions"][event_t]) != 5 or
                            not data["visible_mobs"][event_t, target_row, target_col, 1]):
                        continue
                    base_t = int(event_t) - LOOKBACK
                    if base_t >= 0 and np.any(data["visible_mobs"][base_t, :, :, 1]):
                        selected = (base_t, int(event_t))
                        break
                if selected is None:
                    no_eligible += 1
                    continue
                base_t, event_t = selected
                assert int(data["actions"][event_t]) == 5
                observation = data["observations"][base_t].astype(np.float32, copy=True)
                cow_mask = observation[:MAP_DIM].reshape(7, 9, MAP_CHANNELS)[..., COW_CHANNEL] > .5
                assert cow_mask.any()
                observations.append(observation)
                rows.append(dict(
                    seed=seed, episode_id=int(data["episode_id"]), base_timestep=base_t,
                    restoration_timestep=event_t, lookback=LOOKBACK,
                    original_health=int(intrinsics[base_t, 0]), original_food=int(intrinsics[base_t, 1]),
                    original_drink=int(intrinsics[base_t, 2]), original_energy=int(intrinsics[base_t, 3]),
                    food_before=int(intrinsics[event_t, 1]), food_after=int(after[event_t]),
                    action=int(data["actions"][event_t]), visible_cows=int(cow_mask.sum()),
                ))
    factual = np.stack(observations)
    assert len(rows) == 9692
    assert len({(row["seed"], row["episode_id"]) for row in rows}) == len(rows)
    return rows, factual, episodes_with_restoration, no_eligible


def make_counterfactuals(factual):
    n = len(factual)
    cow_mask = factual[:, :MAP_DIM].reshape(n, 7, 9, MAP_CHANNELS)[..., COW_CHANNEL] > .5
    assert np.all(cow_mask.reshape(n, -1).sum(axis=1) >= 1)
    without_cow = factual.copy()
    no_cow_view = without_cow[:, :MAP_DIM].reshape(n, 7, 9, MAP_CHANNELS)
    no_cow_view[..., COW_CHANNEL][cow_mask] = 0
    assert not np.any(no_cow_view[..., COW_CHANNEL])
    branches = {}
    for health in (2, 9):
        for food in (0, 9):
            for cow in (False, True):
                key = f"h{health}_f{food}_{'cow' if cow else 'no_cow'}"
                obs = (factual if cow else without_cow).copy()
                obs[:, HEALTH_OFFSET] = health / 10
                obs[:, FOOD_OFFSET] = food / 10
                assert np.allclose(obs[:, HEALTH_OFFSET], health / 10)
                assert np.allclose(obs[:, FOOD_OFFSET], food / 10)
                branches[key] = obs
    for health in (2, 9):
        for food in (0, 9):
            present = branches[f"h{health}_f{food}_cow"]
            absent = branches[f"h{health}_f{food}_no_cow"]
            assert np.array_equal(present[:, MAP_DIM:], absent[:, MAP_DIM:])
            assert np.all(np.count_nonzero(present[:, :MAP_DIM] != absent[:, :MAP_DIM], axis=1)
                          == cow_mask.reshape(n, -1).sum(axis=1))
    return branches


def bootstrap(metrics, draws=10_000, seed=50):
    rng = np.random.default_rng(seed)
    n, k = metrics.shape
    estimates = np.empty((draws, k), dtype=np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, n, (count, n))
        estimates[start:start + count] = metrics[indices].mean(axis=1)
    return np.percentile(estimates, (2.5, 97.5), axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, factual, episodes_with_restoration, no_eligible = collect_states()
    np.savez_compressed(OUT / "base_states.npz", observations=factual,
                        seed=[row["seed"] for row in rows], episode_id=[row["episode_id"] for row in rows],
                        base_timestep=[row["base_timestep"] for row in rows])
    branches = make_counterfactuals(factual)
    config = _load_wandb_config(RUN)
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
    network = _init_network(config, env, env.default_params)
    state, checkpoint = _restore_train_state(RUN, config, network, env, env.default_params, CHECKPOINT)
    values = {key: evaluate_values(state.params, network, obs, 512).astype(float)
              for key, obs in branches.items()}
    delta = {(h, food): values[f"h{h}_f{food}_cow"] - values[f"h{h}_f{food}_no_cow"]
             for h in (2, 9) for food in (0, 9)}
    focal = delta[2, 0]
    primary = focal > np.maximum.reduce([np.zeros(len(focal)), delta[2, 9], delta[9, 0], delta[9, 9]])
    positive = focal > 0
    hunger_h2 = focal > delta[2, 9]
    health_f0 = focal > delta[9, 0]
    interaction3 = (delta[2, 0] - delta[2, 9]) - (delta[9, 0] - delta[9, 9])
    interaction_positive = interaction3 > 0
    names = ("primary_success_rate", "focal_positive_rate", "hunger_modulation_h2_rate",
             "health_modulation_f0_rate", "three_way_positive_rate",
             "mean_focal_cow_effect", "mean_three_way_interaction")
    metrics = np.column_stack([primary, positive, hunger_h2, health_f0,
                               interaction_positive, focal, interaction3]).astype(float)
    ci = bootstrap(metrics)
    for i, row in enumerate(rows):
        row.update({"V_" + key: float(value[i]) for key, value in values.items()})
        for (health, food), effect in delta.items():
            row[f"delta_cow_h{health}_f{food}"] = float(effect[i])
        row.update(primary_success=bool(primary[i]), focal_positive=bool(positive[i]),
                   hunger_modulation_h2=bool(hunger_h2[i]), health_modulation_f0=bool(health_f0[i]),
                   three_way_interaction=float(interaction3[i]),
                   three_way_positive=bool(interaction_positive[i]))
    save_csv(OUT / "per_state_results.csv", rows)
    summary = dict(
        checkpoint=checkpoint, n=len(rows), total_episodes=10_000,
        episodes_with_any_food_restoration=episodes_with_restoration,
        episodes_without_eligible_tminus5_visible_cow_state=no_eligible,
        selection="First eligible state per episode exactly five steps before an observed food-level increase caused by a DO action targeting a cow; cow visible in the base observation.",
        conditions="health {2,9} x food {0,9} x all visible cows {present, removed}; eight branches per state.",
        primary_criterion="delta_cow(health=2,food=0) > max(0, delta_cow(2,9), delta_cow(9,0), delta_cow(9,9)).",
        three_way_interaction="[delta_cow(2,0)-delta_cow(2,9)]-[delta_cow(9,0)-delta_cow(9,9)].",
        bootstrap="10,000 paired episode bootstrap resamples, seed 50; percentile 95% intervals. One state per episode.",
        validation="All non-target observation coordinates preserved; all visible cow channels cleared; no time, reward, action, achievement, or rollout changes.",
    )
    for index, name in enumerate(names):
        summary[name] = float(metrics[:, index].mean())
        summary[name + "_95ci"] = [float(ci[0, index]), float(ci[1, index])]
    summary["median_focal_cow_effect"] = float(np.median(focal))
    summary["median_three_way_interaction"] = float(np.median(interaction3))
    summary["cow_effects"] = {
        f"h{health}_f{food}": {"mean": float(effect.mean()), "median": float(np.median(effect)),
                                "positive_rate": float((effect > 0).mean())}
        for (health, food), effect in delta.items()
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Replay the 108 V2 losses and retain crop and survival events through termination."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "runs/global_map_ppo_200m"
TRAIN = ROOT / "global/map_mlp"
OUT = ROOT / "harvest_audit"
BATCH = 64  # Original paired evaluation.


def archive_fields(obs, raw):
    import jax.numpy as jnp
    from craftax.craftax_classic.constants import Achievement, BlockType, DIRECTIONS

    lanes = jnp.arange(raw.map.shape[0])
    crops = raw.growing_plants_positions
    tiles = raw.map[lanes[:, None], crops[..., 0], crops[..., 1]]
    ripe = raw.growing_plants_mask & (tiles == BlockType.RIPE_PLANT.value)
    delta = jnp.abs(crops - raw.player_position[:, None])
    seen = (delta[..., 0] <= 3) & (delta[..., 1] <= 4)
    front = raw.player_position + DIRECTIONS[raw.player_direction]
    front_tile = raw.map[lanes, front[:, 0], front[:, 1]]
    in_bounds = ((front >= 0) & (front < jnp.asarray(raw.map.shape[1:]))).all(axis=1)
    clear = in_bounds & ~raw.is_sleeping
    for name in ("cows", "zombies", "skeletons"):
        mob = getattr(raw, name)
        clear &= ~((mob.position == front[:, None]).all(axis=-1) & mob.mask).any(axis=1)

    live = obs[:, :1323].reshape(-1, 7, 9, 21)
    return dict(
        planted=raw.achievements[:, Achievement.PLACE_PLANT.value],
        ate=raw.achievements[:, Achievement.EAT_PLANT.value],
        diamond=raw.achievements[:, Achievement.COLLECT_DIAMOND.value],
        crop_active=raw.growing_plants_mask, crop_ripe=ripe, crop_seen=seen & raw.growing_plants_mask,
        crop_adjacent=(delta.sum(axis=-1) == 1) & ripe,
        ready=clear & (front_tile == BlockType.RIPE_PLANT.value),
        drink_ready=clear & (front_tile == BlockType.WATER.value),
        water_visible=live[..., BlockType.WATER.value].any(axis=(1, 2)),
        position=raw.player_position.astype(jnp.int8), crop_positions=crops.astype(jnp.int8),
        health=raw.player_health.astype(jnp.int8), drink=raw.player_drink.astype(jnp.int8),
        energy=raw.player_energy.astype(jnp.int8), food=raw.player_food.astype(jnp.int8),
        sleeping=raw.is_sleeping, recover=raw.player_recover,
    )


def first_step(condition):
    indices = np.flatnonzero(condition)
    return int(indices[0]) if len(indices) else None


def episode_audit(row):
    episode = int(row["episode"])
    with np.load(OUT / f"episode_{episode}.npz") as trace:
        n = len(trace["actions"])
        plant = first_step(trace["planted"])
        ate = first_step(trace["ate"])
        ripe = first_step(trace["crop_ripe"].any(axis=1))
        # A terminal observation has no subsequent player decision.
        visible = first_step((trace["crop_ripe"][:-1] & trace["crop_seen"][:-1]).any(axis=1))
        adjacent = first_step(trace["crop_adjacent"][:-1].any(axis=1))
        ready = first_step(trace["ready"][:-1])
        harvest = first_step(trace["ready"][:-1] & (trace["actions"] == 5))
        assert ate == (harvest + 1 if harvest is not None else None), episode
        if ate is not None:
            category = "harvested"
        elif ripe is None:
            assert row["timeout"] == "False", episode
            category = "died_before_ripe"
        elif ripe == n:
            category = "ripe_on_terminal_step"
        elif ready is not None:
            category = "harvest_ready"
        elif adjacent is not None:
            category = "ripe_adjacent"
        elif visible is not None:
            category = "ripe_visible_only"
        else:
            category = "ripe_out_of_view"

        deficit = ((trace["drink"] == 0) | (trace["food"] == 0)
                   | ((trace["energy"] == 0) & ~trace["sleeping"]))
        # update_player_intrinsics runs after combat. Any missing necessity
        # reduces recovery by 1 awake or 0.5 asleep. Crossing -15 costs one HP.
        loss_rate = 0.5 if trace["sleeping"][-1] else 1
        vital_damage = deficit[-1] and trace["recover"][-2] - loss_rate < -15
        if row["lava"] == "True":
            ending = "lava"
        elif row["timeout"] == "True":
            ending = "time_limit"
        elif vital_damage and trace["health"][-1] == trace["health"][-2] - 1:
            ending = "depleted_vitals"
        elif vital_damage:
            ending = "combat_and_depleted_vitals"
        else:
            ending = "combat"
        last_supplied = np.flatnonzero(~deficit)
        final_deficit = int(last_supplied[-1]) + 1 if len(last_supplied) else 0
        skipped = (trace["planted"][:-1] & trace["drink_ready"][:-1]
                   & (trace["drink"][:-1] <= 3) & (trace["actions"] != 5))
        return dict(episode=episode, world_seed=row["world_seed"], length=n, category=category,
                    first_plant=plant, first_ripe=ripe, first_ripe_in_view=visible,
                    first_ripe_adjacent=adjacent, first_harvest_ready=ready, first_ate=ate,
                    first_skipped_direct_drink_while_low=first_step(skipped),
                    final_deficit_started=final_deficit if deficit[-1] else None,
                    first_diamond=first_step(trace["diamond"]),
                    terminal_health=int(trace["health"][-1]), terminal_water=int(trace["drink"][-1]),
                    terminal_food=int(trace["food"][-1]), terminal_energy=int(trace["energy"][-1]),
                    terminal_sleeping=int(trace["sleeping"][-1]), final_damage_or_ending=ending)


def replay_first_drink(env, net, params, cfg, meta, seeds, rows):
    """Change one skipped DO on a low-water state, preserving subsequent RNG keys."""
    from typing import NamedTuple
    from evaluate_reward_shaped_ppo import Achievement, build_evaluator, jax, jnp
    from craftax.craftax_classic.constants import Action

    audits = [episode_audit(row) for row in rows]
    targets = {a["episode"]: a["first_skipped_direct_drink_while_low"] for a in audits
               if a["category"] == "died_before_ripe" and a["first_skipped_direct_drink_while_low"] is not None}
    ids = sorted(targets)
    lookup_seeds = jnp.array([seeds[i] for i in ids], jnp.uint32)
    lookup_steps = jnp.array([targets[i] for i in ids])

    class BranchState(NamedTuple):
        env_state: object
        change_at: object

    class ChangeOneAction:
        default_params = env.default_params

        def reset(self, key, params):
            obs, state = env.reset(key, params)
            # Standard fixed-world resets use PRNGKey(world_seed).
            matches = lookup_seeds == jax.random.key_data(key)[-1]
            time = jnp.where(matches.any(), lookup_steps[jnp.argmax(matches)], -1)
            return obs, BranchState(state, time)

        def step(self, key, state, action, params):
            perform = state.env_state.elapsed == state.change_at
            action = jnp.where(perform, Action.DO.value, action)
            obs, new, reward, done, info = env.step(key, state.env_state, action, params)
            return obs, BranchState(new, state.change_at), reward, done, info

    def fields(obs, raw):
        return dict(position=raw.player_position, drink=raw.player_drink,
                    health=raw.player_health, food=raw.player_food, energy=raw.player_energy,
                    ripe=(raw.growing_plants_mask & (raw.growing_plants_age >= 600)).any(axis=1))

    run = build_evaluator(ChangeOneAction(), net, params, cfg,
                          seed=meta["evaluation_seed"], batch=BATCH,
                          world_seeds=jnp.array(seeds, jnp.uint32), temperature=meta["temperature"],
                          max_steps=meta["max_episode_steps"], archive_fields=fields)
    out = []
    for offset in range(0, len(ids), BATCH):
        selected = ids[offset:offset + BATCH]
        padded = selected + [selected[-1]] * (BATCH - len(selected))
        print(f"Branching {len(selected)} missed drinks", flush=True)
        _, _, _, _, lengths, flags, lava, other, timeout, (frames, transitions) = jax.device_get(run(jnp.array(padded)))
        for lane, ep in enumerate(selected):
            t, n = targets[ep], int(lengths[lane])
            with np.load(OUT / f"episode_{ep}.npz") as baseline:
                for key in ("position", "drink", "health", "food", "energy"):
                    assert np.array_equal(frames[key][:t+1, lane], baseline[key][:t+1]), (ep, key)
                # The archived actions are the original policy proposals, before the override.
                assert np.array_equal(transitions["actions"][:t+1, lane], baseline["actions"][:t+1]), ep
                assert frames["drink"][t+1, lane] == baseline["drink"][t] + 1, ep
                out.append(dict(episode=ep, world_seed=seeds[ep], branch_state_step=t,
                                drink_before=int(baseline["drink"][t]), original_length=len(baseline["actions"]),
                                original_action=Action(int(baseline["actions"][t])).name, branch_length=n,
                                crop_ripened=int(frames["ripe"][:n+1, lane].any()),
                                ate_plant=int(flags[lane, Achievement.EAT_PLANT.value]),
                                all_22=int(flags[lane].all()), achievements=int(flags[lane].sum()),
                                ending="lava" if lava[lane] else "nonlava_death" if other[lane] else "timeout"))
    with (OUT / "first_drink_branches.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=out[0])
        writer.writeheader()
        writer.writerows(out)
    print(f"Validated all {len(out)} branches through the changed action", flush=True)


def main():
    from cap_reward_shaped_runs import try_evaluation_lock
    from run_ppo_study import require_free
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-first-drink", action="store_true")
    args = parser.parse_args()
    gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])
    lock = try_evaluation_lock(gpu)
    if lock is None:
        raise RuntimeError(f"GPU {gpu} is reserved")
    require_free(gpu)
    from evaluate_reward_shaped_ppo import Achievement, build_evaluator, jax, jnp, make_craftax_env_from_name, ocp
    from fixed_world_seeds import read_world_seeds
    from models.actor_critic_map_mlp import map_mlp_network
    from symbolic_map_memory import with_map_memory
    if jax.default_backend() != "gpu":
        raise RuntimeError("Expected GPU execution")
    cfg = json.loads((TRAIN / "inference_config.json").read_text())
    original = TRAIN / "evaluation_wallclock"
    meta = json.loads((original / "summary.json").read_text())
    with (original / "episodes.csv").open() as src:
        rows = sorted((r for r in csv.DictReader(src) if r["all_22"] == "False"),
                      key=lambda r: int(r["length"]))
    seeds = read_world_seeds(meta["world_seeds_file"])
    env = with_map_memory(make_craftax_env_from_name(cfg["ENV_NAME"], False), cfg)
    net = map_mlp_network(cfg, 17)
    params = ocp.PyTreeCheckpointer().restore(str(Path(meta["checkpoint"]) / "default"))["params"]
    OUT.mkdir(exist_ok=True)
    if args.branch_first_drink:
        replay_first_drink(env, net, params, cfg, meta, seeds, rows)
        return
    for start in range(0, len(rows), BATCH):
        selected = rows[start:start+BATCH]
        ids = [int(r["episode"]) for r in selected]
        padded = ids + [ids[-1]] * (BATCH - len(ids))
        limit = max(int(r["length"]) for r in selected)
        evaluate = build_evaluator(env, net, params, cfg, seed=meta["evaluation_seed"],
                                   batch=BATCH, world_seeds=jnp.array(seeds, jnp.uint32),
                                   temperature=meta["temperature"], max_steps=limit,
                                   archive_fields=archive_fields)
        print(f"Replaying {len(ids)} episodes, up to {limit} steps", flush=True)
        result = jax.device_get(evaluate(jnp.array(padded)))
        _, done, returns, shaped, lengths, achievements, lava, nonlava, timeout, (frames, transitions) = result
        for i, (episode, row) in enumerate(zip(ids, selected)):
            n = int(lengths[i])
            expected_flags = [bool(int(row[a.name.lower()])) for a in Achievement]
            assert done[i] and n == int(row["length"]) and int(row["world_seed"]) == seeds[episode], episode
            assert np.array_equal(achievements[i], expected_flags), episode
            assert all(bool(actual) == (row[key] == "True") for actual, key in
                       [(lava[i], "lava"), (nonlava[i], "nonlava_death"), (timeout[i], "timeout")]), episode
            assert np.allclose([returns[i], shaped[i]],
                               [float(row["original_return"]), float(row["shaped_return"])], atol=1e-5), episode
            trace = {key: value[:n+1, i] for key, value in frames.items()}
            trace["actions"] = transitions["actions"][:n, i]
            np.savez_compressed(OUT / f"episode_{episode}.npz", **trace)
        print(f"Validated {start + len(ids)}/{len(rows)} against the original evaluation", flush=True)

    audits = sorted((episode_audit(row) for row in rows), key=lambda a: a["episode"])
    with (OUT / "episode_audit.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=audits[0])
        writer.writeheader()
        writer.writerows(audits)


if __name__ == "__main__":
    main()

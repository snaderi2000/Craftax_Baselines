"""Collect and audit observed-map snapshots from a frozen Map-MLP on fresh worlds."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from cap_reward_shaped_runs import try_evaluation_lock
from run_ppo_study import require_free


RUNS = Path(__file__).resolve().parent / "runs"
OUT = RUNS / "regional_diamond_map"
STEPS = (0, 64, 256, 512)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=512)
    args = parser.parse_args()
    if os.getenv("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the requested GPU")
    lock = try_evaluation_lock(args.gpu)
    if lock is None:
        raise RuntimeError("Evaluation GPU is reserved")
    require_free(args.gpu)
    from evaluate_reward_shaped_ppo import jax, jnp, make_craftax_env_from_name, ocp
    from models.actor_critic_map_mlp import map_mlp_network
    from symbolic_map_memory import with_map_memory
    if jax.default_backend() != "gpu":
        raise RuntimeError("Expected GPU execution")
    teacher = RUNS / "global_map_ppo_200m/global/map_mlp"
    cfg = json.loads((teacher / "inference_config.json").read_text())
    checkpoint = Path(json.loads((teacher / "progress.json").read_text())["final_checkpoint"])
    net = map_mlp_network(cfg)
    params = ocp.PyTreeCheckpointer().restore(str(checkpoint / "default"))["params"]
    env = with_map_memory(make_craftax_env_from_name(cfg["ENV_NAME"], False), cfg)
    rows, cols = jnp.indices((64, 64))

    def audit(state, previous_map, previous_time):
        # Independent reference built from the simulator only inside the measured footprint.
        raw = state.env_state
        visible = (abs(rows - raw.player_position[0]) <= 3) & (abs(cols - raw.player_position[1]) <= 4)
        reference = jnp.where(visible, raw.map.astype(jnp.uint8) + 1, previous_map)
        times = jnp.where(visible, state.elapsed, previous_time)
        errors = ((reference != state.terrain) | (times != state.last_seen)).sum()
        return reference, times, errors

    def collect(world):
        reset, key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(609127), world))
        obs, state = env.reset(reset, env.default_params)
        reference, times, errors = audit(state, jnp.zeros_like(state.terrain),
                                        jnp.full_like(state.last_seen, -1))
        carry = (obs, state, key, jnp.array(False), reference, times, errors)

        def advance(_, carry):
            obs, state, key, ended, reference, times, errors = carry
            key, action_key, dynamics = jax.random.split(key, 3)
            policy, _ = net.apply(params, obs)
            action = jax.random.categorical(action_key, policy.logits)
            obs_next, next_state, _, done, _ = env.step(dynamics, state, action, env.default_params)
            stop = ended | done
            # An early death retains its last pre-action state at all later readouts.
            obs = jnp.where(stop, obs, obs_next)
            state = jax.tree_util.tree_map(lambda old, new: jnp.where(stop, old, new), state, next_state)
            reference, times, mistakes = audit(state, reference, times)
            return obs, state, key, stop, reference, times, errors + jnp.where(stop, 0, mistakes)

        def record(carry):
            state = carry[1]
            return dict(terrain=state.terrain, last_seen=state.last_seen, elapsed=state.elapsed,
                        position=state.position, truth=state.env_state.map.astype(jnp.uint8), errors=carry[-1])

        snapshots = [record(carry)]
        for start, end in zip(STEPS, STEPS[1:]):
            carry = jax.lax.fori_loop(0, end - start, advance, carry)
            snapshots.append(record(carry))
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *snapshots)

    batch = 32
    if args.worlds <= 0 or args.worlds % batch:
        raise ValueError("World count must be a positive multiple of 32")
    run = jax.jit(jax.vmap(collect))
    chunks = []
    for start in range(0, args.worlds, batch):
        chunks.append(jax.device_get(run(jnp.arange(start, start + batch, dtype=jnp.uint32))))
        if start % 128 == 0:
            print(f"Collected {start + batch}/{args.worlds}", flush=True)
    data = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    assert data["errors"].max() == 0, "An observation or timestamp disagreed with the independent reference"
    with (RUNS / "lava_only_gru_concat_dp2_10b_seed50/world_transitions_1000/episodes.csv").open() as f:
        archived = {r["initial_map_sha256"] for r in csv.DictReader(f)}
    hashes = {hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest() for m in data["truth"][:, 0]}
    assert len(hashes) == args.worlds and hashes.isdisjoint(archived)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "fresh_worlds.npz", **data, steps=STEPS, checkpoint=str(checkpoint))
    print(f"Verified {len(hashes)} distinct new worlds; saved {len(STEPS)} snapshots per world", flush=True)


if __name__ == "__main__":
    main()

"""Create return-to-plant drill starts from the visible pilot's saved states."""
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from craftax.craftax_classic.constants import BlockType, DIRECTIONS, SOLID_BLOCKS
from collect_recovery_bank import select
from evaluate_reward_shaped_ppo import select_policy_actions

RIPE = BlockType.RIPE_PLANT.value
CENTER = jnp.array([3, 4])
DETOUR_LIMIT = 48


def terrain(obs):
    return obs[:, :1323].reshape(-1, 7, 9, 21)[..., :17]


def build_offscreen_bank(part, source, destination, starts, actor, step, flag):
    records = json.loads((source / f"{part}_bank.json").read_text())["records"]
    if (destination / f"{part}_bank.msgpack").exists():
        raise FileExistsError(f"{part} out-of-view bank already exists")
    seeds = np.array([r['world'] for r in records], np.uint32)
    count = len(records)
    lanes = jnp.arange(count)
    obs, state, _ = starts
    local = jnp.indices((7, 9)).transpose(1, 2, 0)
    dist = abs(local - CENTER).sum(-1)
    ripe = terrain(obs)[..., RIPE] > .5
    assert np.asarray(ripe.any((1, 2))).all()
    nearest = jnp.where(ripe, dist, 99).reshape(count, -1).argmin(-1)
    target = state.env_state.player_position + jnp.stack((nearest // 9, nearest % 9), -1) - CENTER
    assert np.all(np.asarray(state.env_state.map[lanes, target[:, 0], target[:, 1]]) == RIPE)
    moves = DIRECTIONS[1:5]

    @jax.jit
    def attempt(initial, keys, natural, enabled):
        def tick(carry, _):
            snap, keys, ended, age, elapsed = carry
            o, s, h = snap
            live = enabled & ~ended & (age < 4)
            ks = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
            nh, logits = actor(o, h)
            def sample_walk(_):
                tiles = terrain(o)[:, CENTER[0] + moves[:, 0], CENTER[1] + moves[:, 1]].argmax(-1)
                safe = ~jnp.isin(tiles, SOLID_BLOCKS) & (tiles != BlockType.LAVA.value) & (tiles != BlockType.OUT_OF_BOUNDS.value)
                delta = s.env_state.player_position - target
                gain = abs(delta[:, None, :] + moves).sum(-1) - abs(delta).sum(-1)[:, None]
                weights = jnp.where(safe, jnp.where(age[:, None] == 0, 1.25 * gain, 0.), -30.)
                return 1 + select_policy_actions(weights, ks[:, 1])
            action = jax.lax.cond(natural, lambda _: select_policy_actions(logits, ks[:, 1]), sample_walk, None)
            no, ns, _, done, _ = step(ks[:, 2], s, action)
            unseen = ~(terrain(no)[..., RIPE] > .5).any((1, 2))
            age = jnp.where(live, jnp.where(unseen, age + 1, 0), age)
            ended |= live & (done | ns.env_state.achievements[:, flag])
            return (select(live, (no, ns, nh), snap), ks[:, 0], ended, age, elapsed + live), None
        zeros = jnp.zeros(count, jnp.int32)
        (snap, _, ended, age, elapsed), _ = jax.lax.scan(tick, (initial, keys, zeros.astype(bool), zeros, zeros), None, length=DETOUR_LIMIT)
        return snap, enabled & ~ended & (age >= 4), elapsed

    selected = starts
    sources = np.full(count, -1)
    ticks = np.zeros(count, int)
    total_slots = active_steps = 0
    for trial in range(4):
        keys = jax.vmap(lambda s: jax.random.fold_in(jax.random.PRNGKey(s), 78329 + trial))(seeds)
        snap, ok, elapsed = attempt(starts, keys, trial == 0, sources == -1)
        ok = np.asarray(ok)
        ticks[ok] = np.asarray(elapsed)[ok]
        sources[ok] = trial
        selected = select(ok, snap, selected)
        total_slots += count * DETOUR_LIMIT
        active_steps += int(np.sum(elapsed))
        if (sources >= 0).all():
            break
    take = np.flatnonzero(sources >= 0)
    if not len(take):
        raise RuntimeError("No valid offscreen starts")
    output = jax.tree_util.tree_map(lambda x: np.asarray(x)[take], selected)
    o, s, h = output
    coords = np.asarray(target)[take]
    delta = abs(coords - s.env_state.player_position)
    assert np.all((delta[:, 0] > 3) | (delta[:, 1] > 4))
    assert not np.any(np.asarray(terrain(o)[..., RIPE]) > .5)
    assert np.all(s.env_state.map[np.arange(len(take)), coords[:, 0], coords[:, 1]] == RIPE)
    assert np.isfinite(h).all() and not np.any(s.env_state.achievements[:, flag])
    metadata = [dict(world=int(seeds[i]), target=np.asarray(target[i]).tolist(),
                     preparation="natural" if sources[i] == 0 else "detour",
                     attempt=int(sources[i]), elapsed=int(ticks[i]), distance=int(delta[j].sum()))
                for j, i in enumerate(take)]
    (destination / f"{part}_bank.msgpack").write_bytes(serialization.to_bytes(output))
    (destination / f"{part}_bank.json").write_text(json.dumps(dict(records=metadata, candidates=count,
        simulator_slots=total_slots, active_steps=active_steps, excluded=seeds[sources < 0].tolist()), indent=2))
    print(part, len(take), "natural", sum(sources == 0), "slots", total_slots, flush=True)

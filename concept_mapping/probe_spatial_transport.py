"""Matched offline test of reversible vs freely learned cardinal motion in a terrain memory."""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from types import SimpleNamespace

if not hasattr(jax, "tree"):
    jax.tree = SimpleNamespace(map=jax.tree_util.tree_map, leaves=jax.tree_util.tree_leaves,
                               reduce=jax.tree_util.tree_reduce, structure=jax.tree_util.tree_structure)

from concept_mapping.spatial_transport_queries import BANDS, LOCAL, RESOURCES, STRIDE

FREQ = 32
CHUNK = 128
BATCH = 16
EPOCHS = 3
SEEDS = (32150, 32151)
ARMS = ("free", "reversible")


def init(seed):
    # Equal initial functions. The free arm can learn inconsistent inverse moves.
    frequency = jax.random.uniform(jax.random.PRNGKey(seed), (2, FREQ), minval=-jnp.pi, maxval=jnp.pi)
    return dict(direction=jnp.stack([frequency, -frequency]), gate=jnp.zeros(17), gain=jnp.array(1.))


def address(offset, frequency):
    angle = offset @ frequency
    return jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], -1) / jnp.sqrt(FREQ)


def move(memory, delta, direction):
    angle = -(jnp.maximum(delta, 0) @ direction[0] + jnp.maximum(-delta, 0) @ direction[1])
    c, s = jnp.cos(angle)[..., None], jnp.sin(angle)[..., None]
    x, y = jnp.split(memory, 2, axis=-2)
    return jnp.concatenate([c*x-s*y, s*x+c*y], -2)


def forward(params, memory, tiles, motion, query, reversible):
    frequency = (params["direction"][0] - params["direction"][1]) / 2.
    direction = jnp.stack([frequency, -frequency]) if reversible else params["direction"]
    local_key = address(jnp.asarray(LOCAL, jnp.float32), frequency)
    gram = local_key.T @ local_key
    normalizer = jnp.linalg.norm(gram) + 1e-6
    # Each terrain label gets its own value channel. Only exposed tiles enter writes.
    def block(state, xs):
        tile_block, move_block, offsets = xs
        def step(old, frame):
            view, disp = frame
            moved = move(old, disp.astype(jnp.float32), direction)
            gate = jax.nn.sigmoid(params["gate"][view].mean(-1))[:, None, None]
            target = jax.nn.one_hot(view, 17)
            correction = local_key.T @ target - gram @ moved
            return moved + gate * correction / normalizer, None
        state, _ = jax.lax.scan(step, state, (tile_block, move_block))
        keys = address(offsets.astype(jnp.float32), frequency)
        relevant = state[..., jnp.array(RESOURCES)]
        logits = jnp.einsum("barpk,bkr->barp", keys, relevant)
        return state, (logits[..., 0] - logits[..., 1]) * jax.nn.softplus(params["gain"])
    def grouped(a):
        return a.swapaxes(0, 1).reshape((-1, STRIDE, *a.shape[0:1], *a.shape[2:]))
    return jax.lax.scan(block, memory, (grouped(tiles), grouped(motion), query.swapaxes(0, 1)))


def check():
    p = init(1)
    f = p["direction"][0]
    offset = jnp.array([[2., 5.]])
    d = jnp.array([[1., 0.]])
    key = address(offset, f)[..., None]
    shifted = move(key, d, p["direction"])
    np.testing.assert_allclose(shifted[..., 0], address(offset-d, f), atol=5e-6)
    np.testing.assert_allclose(move(shifted, -d, p["direction"]), key, atol=5e-6)
    np.testing.assert_allclose(move(key, d*0, p["direction"]), key, atol=1e-7)
    blank = jnp.zeros((2, 2*FREQ, 17))
    tiles = jnp.ones((2, 64, 63), dtype=jnp.uint8)
    motion = jnp.zeros((2, 64, 2), dtype=jnp.int8).at[:, 2, 0].set(1)
    queries = jnp.zeros((2, 2, 4, 2, 2, 2))
    def run(p, h, t, m, q, tied):
        return forward(p, h, t, m, q, tied)
    h, scores = run(p, blank, tiles, motion, queries, True)
    h1, s1 = run(p, blank, tiles[:, :32], motion[:, :32], queries[:, :1], False)
    h2, s2 = run(p, h1, tiles[:, 32:], motion[:, 32:], queries[:, 1:], False)
    np.testing.assert_allclose(h, h2, atol=3e-6)
    np.testing.assert_allclose(scores, jnp.concatenate([s1,s2]), atol=1e-7)


def run(out, arm, seed, disable_motion=False, *, initializer=init, memory_forward=forward):
    reversible = arm == "reversible"
    params = initializer(seed)
    if disable_motion:
        with np.load(out / f"{arm}_{seed}.npz") as fitted:
            params = {k: jnp.asarray(fitted[k]) for k in params}
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(.001))
    opt_state = optimizer.init(params)
    @jax.jit
    def train(p, opt_state, h, tiles, motion, query, mask):
        def loss(p):
            carry, margin = memory_forward(p, h, tiles, motion, query, reversible)
            mask_t = mask.swapaxes(0, 1)
            n = mask_t.sum((0, 1))
            sums = (jax.nn.softplus(-margin) * mask_t).sum((0, 1))
            return (sums / n.clip(1)).sum() / (n > 0).sum().clip(1), carry
        (_, carry), grad = jax.value_and_grad(loss, has_aux=True)(p)
        updates, opt_state = optimizer.update(grad, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, carry
    predict = jax.jit(lambda p,h,t,m,q: memory_forward(p,h,t,m,q,reversible))
    for mode in (("test",) if disable_motion else ("train", "test")):
        with np.load(out / f"{mode}.npz") as data:
            arrays = dict(data)
        N, horizon = arrays["tiles"].shape[:2]
        assert N % BATCH == horizon % CHUNK == 0
        rng = np.random.default_rng(seed)
        counts = np.zeros((N, len(BANDS), 2, 2))
        for epoch in range(EPOCHS if mode == "train" else 1):
            order = rng.permutation(N) if mode == "train" else np.arange(N)
            for ids in order.reshape(-1, BATCH):
                carry = jnp.zeros((BATCH, 2*FREQ, 17))
                for t in range(0, horizon, CHUNK):
                    span = slice(t, t+CHUNK)
                    qspan = slice(t//STRIDE, (t+CHUNK)//STRIDE)
                    tiles = arrays["tiles"][ids, span]
                    motion = arrays["motion"][ids, span]
                    if disable_motion:
                        motion = np.zeros_like(motion)
                    query = arrays["query"][ids, qspan]
                    mask = arrays["valid"][ids, qspan]
                    if mode == "train":
                        if mask.any():
                            params, opt_state, carry = train(params, opt_state, carry, tiles, motion, query, mask)
                        else:
                            carry, _ = predict(params, carry, tiles, motion, query)
                    else:
                        carry, margin = predict(params, carry, tiles, motion, query)
                        margin = np.asarray(margin).swapaxes(0, 1)
                        if not np.isfinite(margin).all():
                            raise ArithmeticError("Nonfinite test predictions")
                        counts[ids, ..., 0] += (((margin > 0) + .5*(margin == 0))*mask).sum(1)
                        counts[ids, ..., 1] += mask.sum(1)
            jax.block_until_ready(carry)
            print(arm, seed, mode, epoch+1, flush=True)
    suffix = "_motion_off" if disable_motion else ""
    np.savez(out / f"{arm}_{seed}{suffix}.npz", counts=counts, **jax.device_get(params))


def report(out):
    counts = np.stack([[np.load(out / f"{arm}_{s}.npz")["counts"]
                       for s in SEEDS] for arm in ARMS])
    assert np.array_equal(counts[..., 1], np.broadcast_to(counts[0,0,...,1], counts[...,1].shape))
    # Each world counts equally within an age/resource stratum. Exclude the short
    # training-only band. Average paired differences over both optimization seeds.
    hits, n = counts[:, :, :, 1:, :, 0], counts[0, 0, :, 1:, :, 1]
    scores = np.where(n > 0, hits / np.maximum(n, 1), np.nan)
    paired = (scores[1] - scores[0]).mean(0)
    rng = np.random.default_rng(941853)
    sample = rng.integers(len(paired), size=(10000, len(paired)))
    boot = np.nanmean(paired[sample], axis=1)
    effect = np.nanmean(paired, axis=0)
    mean = np.nanmean(scores, axis=2).mean(1)
    ci = np.quantile(boot.mean((1,2)), [.025,.975])
    enough = np.all((n > 0).sum(0) >= 30)
    passed = bool(enough and ci[0] > 0 and (effect.mean(0) > 0).all())
    with (out / "results.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["age_min","age_max","resource","worlds","pairs","free_accuracy","reversible_accuracy","delta_ci025","delta_ci975"])
        for b,(lo,hi) in enumerate(BANDS[1:]):
            for r,name in enumerate(("water","diamond")):
                w.writerow([lo,hi,name,int((n[:,b,r]>0).sum()),int(n[:,b,r].sum()),*mean[:,b,r],*np.quantile(boot[:,b,r],[.025,.975])])
    result = dict(gate_passed=passed, adequate_support=bool(enough), macro_gain=float(effect.mean()),
                  macro_ci=ci.tolist(), seed_macro_gains=np.nanmean(scores[1]-scores[0],axis=1).mean((1,2)).tolist())
    print(result, flush=True)
    (out / "decision.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("action",choices=("check","report",*ARMS))
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--disable-motion", action="store_true", help="Evaluate a fitted memory with zero displacement.")
    args=parser.parse_args()
    if args.action == "report":
        report(args.out)
    else:
        check()
        if args.action != "check":
            if args.seed is None: parser.error("training needs --seed")
            run(args.out, args.action, args.seed, args.disable_motion)

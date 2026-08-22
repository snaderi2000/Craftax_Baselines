import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from concept_mapping.models import TransitionConceptEncoder
from concept_mapping.train_concepts import load_npz_transitions


def minibatch_kmeans(
    x: np.ndarray,
    k: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(x.shape[0], size=k, replace=False)
    centroids = x[init_idx].astype(np.float32).copy()
    counts = np.zeros(k, dtype=np.int64)

    for step in range(1, steps + 1):
        idx = rng.integers(0, x.shape[0], size=batch_size)
        batch = x[idx]
        dists = (
            np.sum(batch * batch, axis=1, keepdims=True)
            - 2.0 * batch @ centroids.T
            + np.sum(centroids * centroids, axis=1)[None, :]
        )
        labels = np.argmin(dists, axis=1)
        for point, label in zip(batch, labels):
            counts[label] += 1
            eta = 1.0 / float(counts[label])
            centroids[label] = (1.0 - eta) * centroids[label] + eta * point
        if step == 1 or step % 100 == 0:
            used = int(np.sum(counts > 0))
            print(f"step={step} used_clusters={used}/{k}")

    return centroids, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--params", required=True, help="concept_encoder_params.msgpack")
    parser.add_argument("--out_dir", default="concept_mapping/runs/debug")
    parser.add_argument("--num_actions", type=int, default=17)
    parser.add_argument("--num_clusters", type=int, default=512)
    parser.add_argument("--embed_batch_size", type=int, default=8192)
    parser.add_argument("--kmeans_steps", type=int, default=2000)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    transitions = load_npz_transitions(args.data, args.max_transitions, args.seed)
    n = transitions["obs"].shape[0]
    obs_dim = transitions["obs"].shape[-1]

    model = TransitionConceptEncoder(num_actions=args.num_actions)
    init_params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
    )
    with open(args.params, "rb") as f:
        params = serialization.from_bytes(init_params, f.read())

    @jax.jit
    def embed_batch(obs, action, next_obs):
        out = model.apply(params, obs, action, next_obs)
        return out["nu"]

    embeddings = []
    for start in range(0, n, args.embed_batch_size):
        end = min(start + args.embed_batch_size, n)
        emb = embed_batch(
            jnp.asarray(transitions["obs"][start:end]),
            jnp.asarray(transitions["action"][start:end]),
            jnp.asarray(transitions["next_obs"][start:end]),
        )
        embeddings.append(np.asarray(emb, dtype=np.float32))
        if start == 0 or end == n or (start // args.embed_batch_size) % 10 == 0:
            print(f"embedded {end:,}/{n:,}")
    x = np.concatenate(embeddings, axis=0)

    centroids, counts = minibatch_kmeans(
        x,
        k=args.num_clusters,
        steps=args.kmeans_steps,
        batch_size=args.kmeans_batch_size,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "concept_kmeans.npz")
    np.savez_compressed(out_path, centroids=centroids, counts=counts)
    print(f"Saved centroids to {out_path}")


if __name__ == "__main__":
    main()

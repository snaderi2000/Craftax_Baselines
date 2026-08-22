import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax

if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)

    jax.tree = _JaxTreeCompat()

import jax.numpy as jnp
import numpy as np
from flax import serialization

from concept_mapping.analyze_action_outcomes import (
    assign_to_centroids,
    compute_basic_flags,
    parse_actions,
    render_action_outcomes,
    write_action_summary,
    write_outcome_summary,
)
from concept_mapping.fit_kmeans import minibatch_kmeans
from concept_mapping.inspect_concepts import load_transition_subset
from concept_mapping.models import ICMTransitionConceptEncoder, TransitionConceptEncoder
from concept_mapping.visualize_symbolic_trajectory import _action_name


def load_encoder(params_path: str, obs_dim: int, num_actions: int, model_type: str):
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Missing encoder params at {params_path}. "
            "Train one with concept_mapping/train_concepts.py or "
            "concept_mapping/train_icm_concepts.py first."
        )
    if model_type == "icm":
        model = ICMTransitionConceptEncoder(num_actions=num_actions)
    elif model_type == "infonce":
        model = TransitionConceptEncoder(num_actions=num_actions)
    else:
        raise ValueError("--model_type must be icm or infonce")
    init_params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
    )
    with open(params_path, "rb") as f:
        params = serialization.from_bytes(init_params, f.read())
    return model, params


def embed_transitions(
    model,
    params,
    data: dict[str, np.ndarray],
    batch_size: int,
    embedding: str,
) -> np.ndarray:
    if embedding not in {"nu", "psi", "state_delta", "concept"}:
        raise ValueError("--embedding must be one of: nu, psi, state_delta, concept")

    @jax.jit
    def embed_batch(obs, action, next_obs):
        out = model.apply(params, obs, action, next_obs)
        if embedding == "state_delta":
            return out["next_state"] - out["state"]
        return out[embedding]

    xs = []
    n = data["obs"].shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = embed_batch(
            jnp.asarray(data["obs"][start:end]),
            jnp.asarray(data["action"][start:end]),
            jnp.asarray(data["next_obs"][start:end]),
        )
        xs.append(np.asarray(x, dtype=np.float32))
        if start == 0 or end == n or (start // batch_size) % 10 == 0:
            print(f"embedded {end:,}/{n:,}", flush=True)
    return np.concatenate(xs, axis=0)


def append_reward_done(
    x: np.ndarray,
    data: dict[str, np.ndarray],
    reward_scale: float,
    done_scale: float,
) -> np.ndarray:
    reward_col = data["reward"][:, None].astype(np.float32) * reward_scale
    done_col = data["done"][:, None].astype(np.float32) * done_scale
    return np.concatenate([x.astype(np.float32), reward_col, done_col], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action-conditioned outcome clustering using SSL transition embeddings."
    )
    parser.add_argument("--data", default="concept_mapping/data/ppo_10b_symbolic")
    parser.add_argument("--params", required=True, help="concept_encoder_params.msgpack")
    parser.add_argument("--model_type", choices=["infonce", "icm"], default="infonce")
    parser.add_argument("--out_dir", default="concept_mapping/runs/ppo_10b_action_outcomes_embedded")
    parser.add_argument("--max_transitions", type=int, default=1_000_000)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--num_actions", type=int, default=17)
    parser.add_argument("--clusters_per_action", type=int, default=8)
    parser.add_argument(
        "--cluster_actions",
        type=str,
        default=None,
        help="Comma-separated action ids to cluster. Defaults to all actions.",
    )
    parser.add_argument("--min_action_count", type=int, default=64)
    parser.add_argument("--embedding", choices=["nu", "psi", "state_delta", "concept"], default="nu")
    parser.add_argument("--embed_batch_size", type=int, default=8192)
    parser.add_argument("--kmeans_steps", type=int, default=500)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--include_reward_done", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--done_scale", type=float, default=1.0)
    parser.add_argument("--no_effect_eps", type=float, default=1e-7)
    parser.add_argument("--reward_eps", type=float, default=1e-6)
    parser.add_argument("--render_actions", type=str, default="5,10,11,12,13,16")
    parser.add_argument(
        "--render_outcomes",
        type=str,
        default=None,
        help="Comma-separated outcome ids to render. Defaults to all clustered outcomes.",
    )
    parser.add_argument("--examples_per_outcome", type=int, default=6)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.model_type == "icm" and args.embedding in {"nu", "psi"}:
        raise ValueError("ICM model supports --embedding concept or state_delta.")
    if args.model_type == "infonce" and args.embedding == "concept":
        raise ValueError("InfoNCE model supports --embedding nu, psi, or state_delta.")

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_transition_subset(args.data, args.max_transitions, args.seed, args.max_shards)
    print(f"Loaded {data['obs'].shape[0]:,} transitions")

    delta, delta_l1, no_effect, positive, negative = compute_basic_flags(
        data, args.no_effect_eps, args.reward_eps
    )
    write_action_summary(
        os.path.join(args.out_dir, "action_waste_summary.csv"),
        data,
        delta_l1,
        no_effect,
        positive,
        negative,
        args.num_actions,
    )

    model, params = load_encoder(
        args.params,
        data["obs"].shape[-1],
        args.num_actions,
        args.model_type,
    )
    features = embed_transitions(
        model,
        params,
        data,
        batch_size=args.embed_batch_size,
        embedding=args.embedding,
    )
    if args.include_reward_done:
        features = append_reward_done(
            features,
            data,
            reward_scale=args.reward_scale,
            done_scale=args.done_scale,
        )
    print(f"feature_dim={features.shape[-1]} model_type={args.model_type} embedding={args.embedding}")

    labels_by_action = {}
    dists_by_action = {}
    centroids_by_action = {}
    cluster_actions = parse_actions(args.cluster_actions, args.num_actions)
    for action in cluster_actions:
        idx = np.where(data["action"] == action)[0]
        if idx.size < args.min_action_count:
            print(f"action={action}:{_action_name(action)} skipped count={idx.size}")
            continue
        k = min(args.clusters_per_action, idx.size)
        print(
            f"Clustering action={action}:{_action_name(action)} count={idx.size} k={k}",
            flush=True,
        )
        centroids, _ = minibatch_kmeans(
            features[idx],
            k=k,
            steps=args.kmeans_steps,
            batch_size=min(args.kmeans_batch_size, idx.size),
            seed=args.seed + action,
        )
        labels, dists = assign_to_centroids(features[idx], centroids)
        labels_by_action[action] = labels
        dists_by_action[action] = dists
        centroids_by_action[action] = centroids

    write_outcome_summary(
        os.path.join(args.out_dir, "action_outcome_summary.csv"),
        data,
        labels_by_action,
        dists_by_action,
        no_effect,
        positive,
        negative,
        args.num_actions,
    )

    np.savez_compressed(
        os.path.join(args.out_dir, "action_outcome_embedding_clusters.npz"),
        feature_dim=np.array([features.shape[-1]], dtype=np.int32),
        embedding=np.array([args.embedding]),
        model_type=np.array([args.model_type]),
        **{f"centroids_action_{a}": c for a, c in centroids_by_action.items()},
    )

    render_action_outcomes(
        args.out_dir,
        data,
        labels_by_action,
        dists_by_action,
        parse_actions(args.render_actions, args.num_actions),
        parse_actions(args.render_outcomes, args.clusters_per_action) if args.render_outcomes else None,
        args.examples_per_outcome,
        args.block_size,
        args.display_scale,
    )
    print(f"Saved embedded action-outcome summaries to {args.out_dir}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from concept_mapping.analyze_action_outcomes import (
    assign_to_centroids,
    make_outcome_features,
    save_transition_grid,
)
from concept_mapping.inspect_concepts import load_transition_subset, render_transition_pair
from concept_mapping.visualize_symbolic_trajectory import _action_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render more examples from an existing raw action-outcome k-means run."
    )
    parser.add_argument("--data", default="concept_mapping/data/ppo_10b_symbolic")
    parser.add_argument("--run_dir", required=True, help="Directory with action_outcome_clusters.npz")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--action", type=int, required=True)
    parser.add_argument("--outcome", type=int, required=True)
    parser.add_argument("--max_transitions", type=int, default=500_000)
    parser.add_argument("--max_shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--done_scale", type=float, default=5.0)
    parser.add_argument("--examples", type=int, default=40)
    parser.add_argument("--block_size", type=int, default=7)
    parser.add_argument("--display_scale", type=int, default=4)
    args = parser.parse_args()

    clusters_path = os.path.join(args.run_dir, "action_outcome_clusters.npz")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Missing {clusters_path}")
    clusters = np.load(clusters_path)
    key = f"centroids_action_{args.action}"
    if key not in clusters.files:
        raise ValueError(f"{key} not found in {clusters_path}. Available: {clusters.files}")
    centroids = np.asarray(clusters[key], dtype=np.float32)

    data = load_transition_subset(args.data, args.max_transitions, args.seed, args.max_shards)
    delta = data["next_obs"] - data["obs"]
    features = make_outcome_features(
        data,
        delta,
        delta_scale=args.delta_scale,
        reward_scale=args.reward_scale,
        done_scale=args.done_scale,
    )

    action_idx = np.where(data["action"] == args.action)[0]
    if action_idx.size == 0:
        raise ValueError(f"No transitions found for action {args.action}:{_action_name(args.action)}")
    labels, dists = assign_to_centroids(features[action_idx], centroids)
    local = np.where(labels == args.outcome)[0]
    if local.size == 0:
        raise ValueError(f"No examples found for outcome {args.outcome}")

    local = local[np.argsort(dists[local])[: args.examples]]
    frames = []
    for rank, local_i in enumerate(local):
        i = action_idx[local_i]
        frames.append(
            render_transition_pair(
                data["obs"][i],
                data["next_obs"][i],
                int(data["action"][i]),
                float(data["reward"][i]),
                bool(data["done"][i]),
                args.outcome,
                rank,
                float(dists[local_i]),
                args.block_size,
                args.display_scale,
            )
        )

    out_dir = args.out_dir or os.path.join(args.run_dir, "more_examples")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir,
        f"action_{args.action:02d}_{_action_name(args.action)}_outcome_{args.outcome:02d}_{len(frames)}examples.png",
    )
    save_transition_grid(frames, path, cols=2)
    print(f"Rendered {len(frames)} examples from {args.run_dir}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

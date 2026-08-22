import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from PIL import Image

from concept_mapping.fit_kmeans import minibatch_kmeans
from concept_mapping.inspect_concepts import load_transition_subset, render_transition_pair
from concept_mapping.visualize_symbolic_trajectory import (
    _action_name,
)


def compute_basic_flags(data: dict[str, np.ndarray], no_effect_eps: float, reward_eps: float):
    delta = data["next_obs"] - data["obs"]
    delta_l1 = np.mean(np.abs(delta), axis=-1)
    reward = data["reward"]
    done = data["done"]
    no_effect = (delta_l1 <= no_effect_eps) & (np.abs(reward) <= reward_eps) & (~done)
    positive = reward > reward_eps
    negative = reward < -reward_eps
    return delta, delta_l1, no_effect, positive, negative


def make_outcome_features(
    data: dict[str, np.ndarray],
    delta: np.ndarray,
    delta_scale: float,
    reward_scale: float,
    done_scale: float,
) -> np.ndarray:
    reward_col = data["reward"][:, None].astype(np.float32) * reward_scale
    done_col = data["done"][:, None].astype(np.float32) * done_scale
    return np.concatenate([delta.astype(np.float32) * delta_scale, reward_col, done_col], axis=-1)


def assign_to_centroids(x: np.ndarray, centroids: np.ndarray, batch_size: int = 16384):
    labels = np.empty((x.shape[0],), dtype=np.int32)
    dists_out = np.empty((x.shape[0],), dtype=np.float32)
    centroid_norms = np.sum(centroids * centroids, axis=1)[None, :]
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        batch = x[start:end]
        dists = (
            np.sum(batch * batch, axis=1, keepdims=True)
            - 2.0 * batch @ centroids.T
            + centroid_norms
        )
        labels[start:end] = np.argmin(dists, axis=1)
        dists_out[start:end] = np.min(dists, axis=1)
    return labels, dists_out


def write_action_summary(
    path: str,
    data: dict[str, np.ndarray],
    delta_l1: np.ndarray,
    no_effect: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    num_actions: int,
) -> None:
    with open(path, "w") as f:
        f.write(
            "action,action_name,count,wasted_rate,ineffective_rate,positive_rate,negative_rate,"
            "done_rate,mean_reward,mean_delta_l1\n"
        )
        for action in range(num_actions):
            idx = data["action"] == action
            count = int(idx.sum())
            if count == 0:
                continue
            f.write(
                f"{action},{_action_name(action)},{count},"
                f"{float(no_effect[idx].mean()):.6f},"
                f"{float(((~positive[idx]) & (~data['done'][idx])).mean()):.6f},"
                f"{float(positive[idx].mean()):.6f},"
                f"{float(negative[idx].mean()):.6f},"
                f"{float(data['done'][idx].mean()):.6f},"
                f"{float(data['reward'][idx].mean()):.6f},"
                f"{float(delta_l1[idx].mean()):.8f}\n"
            )


def write_outcome_summary(
    path: str,
    data: dict[str, np.ndarray],
    labels_by_action: dict[int, np.ndarray],
    dists_by_action: dict[int, np.ndarray],
    no_effect: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    num_actions: int,
) -> None:
    with open(path, "w") as f:
        f.write(
            "action,action_name,outcome,count,wasted_rate,ineffective_rate,positive_rate,negative_rate,"
            "done_rate,mean_reward,mean_dist\n"
        )
        for action in range(num_actions):
            action_idx = np.where(data["action"] == action)[0]
            if action_idx.size == 0 or action not in labels_by_action:
                continue
            labels = labels_by_action[action]
            dists = dists_by_action[action]
            for outcome in np.argsort(-np.bincount(labels)):
                local = labels == outcome
                if local.sum() == 0:
                    continue
                global_idx = action_idx[local]
                f.write(
                    f"{action},{_action_name(action)},{int(outcome)},{int(local.sum())},"
                    f"{float(no_effect[global_idx].mean()):.6f},"
                    f"{float(((~positive[global_idx]) & (~data['done'][global_idx])).mean()):.6f},"
                    f"{float(positive[global_idx].mean()):.6f},"
                    f"{float(negative[global_idx].mean()):.6f},"
                    f"{float(data['done'][global_idx].mean()):.6f},"
                    f"{float(data['reward'][global_idx].mean()):.6f},"
                    f"{float(dists[local].mean()):.6f}\n"
                )


def save_transition_grid(frames: list[Image.Image], path: str, cols: int = 2) -> None:
    if not frames:
        return
    cols = min(cols, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * frames[0].width, rows * frames[0].height), "white")
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * frame.width, (i // cols) * frame.height))
    sheet.save(path)


def render_action_outcomes(
    out_dir: str,
    data: dict[str, np.ndarray],
    labels_by_action: dict[int, np.ndarray],
    dists_by_action: dict[int, np.ndarray],
    action_ids: list[int],
    outcome_ids: list[int] | None,
    examples_per_outcome: int,
    block_size: int,
    display_scale: int,
) -> None:
    render_dir = os.path.join(out_dir, "outcome_sheets")
    os.makedirs(render_dir, exist_ok=True)
    for action in action_ids:
        action_idx = np.where(data["action"] == action)[0]
        if action_idx.size == 0 or action not in labels_by_action:
            continue
        labels = labels_by_action[action]
        dists = dists_by_action[action]
        for outcome in np.argsort(-np.bincount(labels)):
            if outcome_ids is not None and int(outcome) not in outcome_ids:
                continue
            local = np.where(labels == outcome)[0]
            if local.size == 0:
                continue
            local = local[np.argsort(dists[local])[:examples_per_outcome]]
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
                        action,
                        rank,
                        float(dists[local_i]),
                        block_size,
                        display_scale,
                    )
                )
            path = os.path.join(render_dir, f"action_{action:02d}_{_action_name(action)}_outcome_{int(outcome):02d}.png")
            save_transition_grid(frames, path)


def parse_actions(raw: str | None, num_actions: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(range(num_actions))
    return [int(x) for x in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="concept_mapping/data/ppo_10b_symbolic")
    parser.add_argument("--out_dir", default="concept_mapping/runs/ppo_10b_action_outcomes")
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
    parser.add_argument("--kmeans_steps", type=int, default=500)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--done_scale", type=float, default=5.0)
    parser.add_argument("--no_effect_eps", type=float, default=1e-7)
    parser.add_argument("--reward_eps", type=float, default=1e-6)
    parser.add_argument("--render_actions", type=str, default="5,10,12,13,16")
    parser.add_argument(
        "--render_outcomes",
        type=str,
        default=None,
        help="Comma-separated outcome ids to render. Defaults to all clustered outcomes.",
    )
    parser.add_argument("--examples_per_outcome", type=int, default=6)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--display_scale", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_transition_subset(args.data, args.max_transitions, args.seed, args.max_shards)
    print(f"Loaded {data['obs'].shape[0]:,} transitions")

    delta, delta_l1, no_effect, positive, negative = compute_basic_flags(
        data, args.no_effect_eps, args.reward_eps
    )
    features = make_outcome_features(
        data,
        delta,
        delta_scale=args.delta_scale,
        reward_scale=args.reward_scale,
        done_scale=args.done_scale,
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
        print(f"Clustering action={action}:{_action_name(action)} count={idx.size} k={k}", flush=True)
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
        os.path.join(args.out_dir, "action_outcome_clusters.npz"),
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

    print(f"Saved action summaries and outcome sheets to {args.out_dir}")


if __name__ == "__main__":
    main()

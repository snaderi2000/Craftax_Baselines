import argparse
import glob
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from PIL import Image, ImageDraw

from concept_mapping.models import TransitionConceptEncoder
from concept_mapping.visualize_symbolic_trajectory import (
    _action_name,
    _summary_text,
    render_local_view,
)


def _npz_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "transitions_*.npz")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"No NPZ files found in {path}")
        return paths
    return [path]


def load_transition_subset(
    path: str,
    max_transitions: int | None,
    seed: int,
    max_shards: int | None = None,
) -> dict[str, np.ndarray]:
    paths = _npz_paths(path)
    if max_shards is not None:
        paths = paths[:max_shards]
    rng = np.random.default_rng(seed)
    per_shard = None
    if max_transitions is not None:
        per_shard = int(np.ceil(max_transitions / max(1, len(paths))))

    chunks = []
    for p in paths:
        d = np.load(p)
        action_key = "action" if "action" in d.files else "actions"
        obs = np.asarray(d["obs"], dtype=np.float32).reshape(-1, d["obs"].shape[-1])
        n = obs.shape[0]
        if per_shard is not None and n > per_shard:
            idx = rng.choice(n, size=per_shard, replace=False)
        else:
            idx = np.arange(n)
        action_arr = np.asarray(d[action_key], dtype=np.int32).reshape(-1)
        chunk = {
            "obs": obs[idx],
            "next_obs": np.asarray(d["next_obs"], dtype=np.float32).reshape(-1, d["next_obs"].shape[-1])[idx],
            "action": action_arr[idx],
            "reward": np.asarray(d["reward"], dtype=np.float32).reshape(-1)[idx]
            if "reward" in d.files
            else np.zeros(idx.shape[0], dtype=np.float32),
            "done": np.asarray(d["done"], dtype=bool).reshape(-1)[idx]
            if "done" in d.files
            else np.zeros(idx.shape[0], dtype=bool),
        }
        chunks.append(chunk)
        print(f"loaded {idx.shape[0]:,} transitions from {p}", flush=True)

    out = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0].keys()}
    if max_transitions is not None and out["obs"].shape[0] > max_transitions:
        idx = rng.choice(out["obs"].shape[0], size=max_transitions, replace=False)
        out = {k: v[idx] for k, v in out.items()}
    return out


def load_params(params_path: str, obs_dim: int, num_actions: int):
    model = TransitionConceptEncoder(num_actions=num_actions)
    init_params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
    )
    with open(params_path, "rb") as f:
        params = serialization.from_bytes(init_params, f.read())
    return model, params


def embed_transitions(model, params, data: dict[str, np.ndarray], batch_size: int) -> np.ndarray:
    @jax.jit
    def embed_batch(obs, action, next_obs):
        out = model.apply(params, obs, action, next_obs)
        return out["nu"]

    embeddings = []
    n = data["obs"].shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = embed_batch(
            jnp.asarray(data["obs"][start:end]),
            jnp.asarray(data["action"][start:end]),
            jnp.asarray(data["next_obs"][start:end]),
        )
        embeddings.append(np.asarray(emb, dtype=np.float32))
        if start == 0 or end == n or (start // batch_size) % 10 == 0:
            print(f"embedded {end:,}/{n:,}", flush=True)
    return np.concatenate(embeddings, axis=0)


def assign_clusters(embeddings: np.ndarray, centroids: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty((embeddings.shape[0],), dtype=np.int32)
    nearest_dists = np.empty((embeddings.shape[0],), dtype=np.float32)
    centroid_norms = np.sum(centroids * centroids, axis=1)[None, :]
    for start in range(0, embeddings.shape[0], batch_size):
        end = min(start + batch_size, embeddings.shape[0])
        x = embeddings[start:end]
        dists = np.sum(x * x, axis=1, keepdims=True) - 2.0 * x @ centroids.T + centroid_norms
        labels[start:end] = np.argmin(dists, axis=1)
        nearest_dists[start:end] = np.min(dists, axis=1)
    return labels, nearest_dists


def draw_text_panel(width: int, lines: list[str], title: str | None = None) -> Image.Image:
    pad = 8
    line_h = 15
    all_lines = ([title] if title else []) + lines
    img = Image.new("RGB", (width, pad * 2 + line_h * len(all_lines)), "white")
    draw = ImageDraw.Draw(img)
    y = pad
    for i, line in enumerate(all_lines):
        fill = (0, 0, 0) if i > 0 or not title else (30, 60, 120)
        draw.text((pad, y), line, fill=fill)
        y += line_h
    return img


def render_transition_pair(
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: int,
    reward: float,
    done: bool,
    cluster_id: int,
    rank: int,
    dist: float,
    block_size: int,
    display_scale: int = 1,
) -> Image.Image:
    left = render_local_view(obs, block_size, display_scale=display_scale)
    right = render_local_view(next_obs, block_size, display_scale=display_scale)
    gap = 10
    width = left.width + right.width + gap
    top = Image.new("RGB", (width, left.height), "white")
    top.paste(left, (0, 0))
    top.paste(right, (left.width + gap, 0))

    lines = _summary_text(obs, action, reward, done, rank, cluster_id)
    lines[0] = f"cluster={cluster_id} example={rank} action={int(action)}:{_action_name(int(action))}"
    lines.append(f"centroid_dist={dist:.5f}")
    panel = draw_text_panel(width, lines, title="s_t  ->  s_{t+1}")

    out = Image.new("RGB", (width, top.height + panel.height), "white")
    out.paste(top, (0, 0))
    out.paste(panel, (0, top.height))
    return out


def save_cluster_sheet(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    nearest_dists: np.ndarray,
    cluster_id: int,
    out_dir: str,
    examples_per_cluster: int,
    block_size: int,
    display_scale: int,
) -> None:
    idx = np.where(labels == cluster_id)[0]
    if idx.size == 0:
        return
    idx = idx[np.argsort(nearest_dists[idx])[:examples_per_cluster]]
    frames = []
    for rank, i in enumerate(idx):
        frames.append(
            render_transition_pair(
                data["obs"][i],
                data["next_obs"][i],
                int(data["action"][i]),
                float(data["reward"][i]),
                bool(data["done"][i]),
                cluster_id,
                rank,
                float(nearest_dists[i]),
                block_size,
                display_scale,
            )
        )
    cols = min(2, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    sheet = Image.new("RGB", (cols * frames[0].width, rows * frames[0].height), "white")
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % cols) * frame.width, (i // cols) * frame.height))
    sheet.save(os.path.join(out_dir, f"cluster_{cluster_id:04d}.png"))


def write_summary(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    nearest_dists: np.ndarray,
    out_path: str,
) -> np.ndarray:
    num_clusters = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=num_clusters)
    order = np.argsort(-counts)
    with open(out_path, "w") as f:
        f.write("cluster,count,mean_dist,mean_reward,done_rate,top_actions\n")
        for cid in order:
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            actions = data["action"][idx]
            action_counts = np.bincount(actions, minlength=17)
            top = np.argsort(-action_counts)[:5]
            top_actions = " ".join(
                f"{int(a)}:{_action_name(int(a))}:{int(action_counts[a])}" for a in top if action_counts[a] > 0
            )
            f.write(
                f"{cid},{idx.size},{float(nearest_dists[idx].mean()):.6f},"
                f"{float(data['reward'][idx].mean()):.6f},"
                f"{float(data['done'][idx].mean()):.6f},{top_actions}\n"
            )
    return order


def parse_cluster_ids(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(x) for x in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="concept_mapping/data/ppo_10b_symbolic")
    parser.add_argument("--params", default="concept_mapping/runs/ppo_10b_concepts/concept_encoder_params.msgpack")
    parser.add_argument("--kmeans", default="concept_mapping/runs/ppo_10b_concepts/concept_kmeans.npz")
    parser.add_argument("--out_dir", default="concept_mapping/runs/ppo_10b_concepts/inspection")
    parser.add_argument("--num_actions", type=int, default=17)
    parser.add_argument("--max_transitions", type=int, default=500_000)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--embed_batch_size", type=int, default=8192)
    parser.add_argument("--assign_batch_size", type=int, default=16384)
    parser.add_argument("--top_clusters", type=int, default=24)
    parser.add_argument("--cluster_ids", type=str, default=None, help="Comma-separated cluster ids to render.")
    parser.add_argument("--examples_per_cluster", type=int, default=6)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--display_scale", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_transition_subset(args.data, args.max_transitions, args.seed, args.max_shards)
    print(f"Loaded {data['obs'].shape[0]:,} transitions")

    kmeans = np.load(args.kmeans)
    centroids = np.asarray(kmeans["centroids"], dtype=np.float32)
    model, params = load_params(args.params, data["obs"].shape[-1], args.num_actions)
    embeddings = embed_transitions(model, params, data, args.embed_batch_size)
    labels, nearest_dists = assign_clusters(embeddings, centroids, args.assign_batch_size)

    np.savez_compressed(
        os.path.join(args.out_dir, "assignments.npz"),
        labels=labels,
        nearest_dists=nearest_dists,
    )
    order = write_summary(
        data,
        labels,
        nearest_dists,
        os.path.join(args.out_dir, "cluster_summary.csv"),
    )
    print(f"Saved summary to {os.path.join(args.out_dir, 'cluster_summary.csv')}")

    cluster_ids = parse_cluster_ids(args.cluster_ids)
    if cluster_ids is None:
        cluster_ids = [int(x) for x in order[: args.top_clusters]]
    print(f"Rendering clusters: {cluster_ids}")
    for cid in cluster_ids:
        save_cluster_sheet(
            data,
            labels,
            nearest_dists,
            cid,
            args.out_dir,
            args.examples_per_cluster,
            args.block_size,
            args.display_scale,
        )
    print(f"Saved cluster sheets to {args.out_dir}")


if __name__ == "__main__":
    main()

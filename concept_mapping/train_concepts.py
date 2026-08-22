import argparse
import glob
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax

# Compatibility for newer Optax/Flax code running on JAX versions before
# `jax.tree` was introduced (some environments only expose `jax.tree_util`).
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)

    jax.tree = _JaxTreeCompat()
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from concept_mapping.models import TransitionConceptEncoder, symmetric_info_nce


def _npz_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "transitions_*.npz")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(path, "*.npz")))
        if not paths:
            raise FileNotFoundError(f"No NPZ files found in {path}")
        return paths
    return [path]


def _load_one_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    keys = set(data.files)
    action_key = "action" if "action" in keys else "actions"
    required = {"obs", "next_obs", action_key}
    missing = required - keys
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    obs = np.asarray(data["obs"], dtype=np.float32)
    next_obs = np.asarray(data["next_obs"], dtype=np.float32)
    actions = np.asarray(data[action_key], dtype=np.int32)
    obs = obs.reshape(-1, obs.shape[-1])
    next_obs = next_obs.reshape(-1, next_obs.shape[-1])
    actions = actions.reshape(-1)
    out = {"obs": obs, "next_obs": next_obs, "action": actions}
    if "reward" in keys:
        out["reward"] = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
    if "done" in keys:
        out["done"] = np.asarray(data["done"], dtype=bool).reshape(-1)
    return out


def load_npz_transitions(
    path: str,
    max_transitions: int | None = None,
    seed: int = 0,
    max_shards: int | None = None,
) -> dict[str, np.ndarray]:
    paths = _npz_paths(path)
    if max_shards is not None:
        paths = paths[:max_shards]
    rng = np.random.default_rng(seed)
    per_shard = None
    if max_transitions is not None:
        per_shard = int(np.ceil(max_transitions / max(1, len(paths))))

    shards = []
    for p in paths:
        shard = _load_one_npz(p)
        n = shard["obs"].shape[0]
        if per_shard is not None and n > per_shard:
            idx = rng.choice(n, size=per_shard, replace=False)
            shard = {k: v[idx] for k, v in shard.items()}
        shards.append(shard)
        print(f"loaded {shard['obs'].shape[0]:,} transitions from {p}", flush=True)

    keys = set.intersection(*(set(s.keys()) for s in shards))
    out = {
        key: np.concatenate([s[key] for s in shards], axis=0)
        for key in sorted(keys)
    }
    if max_transitions is not None and out["obs"].shape[0] > max_transitions:
        rng = np.random.default_rng(seed)
        idx = rng.choice(out["obs"].shape[0], size=max_transitions, replace=False)
        out = {k: v[idx] for k, v in out.items()}
    return out


def make_state(params, tx):
    return TrainState.create(apply_fn=None, params=params, tx=tx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="NPZ with obs, next_obs, action/actions")
    parser.add_argument("--out_dir", default="concept_mapping/runs/debug")
    parser.add_argument("--num_actions", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--max_transitions", type=int, default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--forward_coef", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    transitions = load_npz_transitions(args.data, args.max_transitions, args.seed, args.max_shards)
    n = transitions["obs"].shape[0]
    obs_dim = transitions["obs"].shape[-1]
    print(f"Loaded {n:,} transitions, obs_dim={obs_dim}")

    model = TransitionConceptEncoder(num_actions=args.num_actions)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    params = model.init(
        init_rng,
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(args.lr),
    )

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            out = model.apply(params, batch["obs"], batch["action"], batch["next_obs"])
            pred_loss = symmetric_info_nce(out["psi"], out["nu"], args.temperature)
            forward_loss = jnp.mean(jnp.square(out["pred_next_state"] - jax.lax.stop_gradient(out["next_state"])))
            loss = pred_loss + args.forward_coef * forward_loss
            return loss, {"loss": loss, "pred_loss": pred_loss, "forward_loss": forward_loss}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    os.makedirs(args.out_dir, exist_ok=True)
    np_rng = np.random.default_rng(args.seed)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("step,elapsed_sec,steps_per_sec,transitions_per_sec,eta_sec,loss,pred_loss,forward_loss\n")

    t0 = time.time()
    last_t = t0
    last_step = 0
    for step in range(1, args.steps + 1):
        idx = np_rng.integers(0, n, size=args.batch_size)
        batch = {
            "obs": jnp.asarray(transitions["obs"][idx]),
            "next_obs": jnp.asarray(transitions["next_obs"][idx]),
            "action": jnp.asarray(transitions["action"][idx]),
        }
        state, metrics = train_step(state, batch)
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            now = time.time()
            elapsed = now - t0
            interval = max(now - last_t, 1e-9)
            interval_steps = step - last_step
            steps_per_sec = interval_steps / interval
            transitions_per_sec = steps_per_sec * args.batch_size
            eta_sec = (args.steps - step) / max(steps_per_sec, 1e-9)
            metrics_f = {k: float(v) for k, v in metrics.items()}
            msg = " ".join(f"{k}={v:.4f}" for k, v in metrics_f.items())
            print(
                f"step={step}/{args.steps} elapsed={elapsed/60:.1f}m "
                f"speed={steps_per_sec:.2f} steps/s "
                f"samples={transitions_per_sec:,.0f}/s eta={eta_sec/60:.1f}m {msg}",
                flush=True,
            )
            with open(metrics_path, "a") as f:
                f.write(
                    f"{step},{elapsed:.6f},{steps_per_sec:.6f},{transitions_per_sec:.6f},"
                    f"{eta_sec:.6f},{metrics_f['loss']:.8f},{metrics_f['pred_loss']:.8f},"
                    f"{metrics_f['forward_loss']:.8f}\n"
                )
            last_t = now
            last_step = step

    ckpt_path = os.path.join(args.out_dir, "concept_encoder_params.msgpack")
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(state.params))
    print(f"Training complete. Params saved to {ckpt_path}")


if __name__ == "__main__":
    main()

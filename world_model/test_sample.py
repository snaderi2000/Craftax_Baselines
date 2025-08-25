# test_sample.py
import argparse
import jax
import jax.numpy as jnp

from sample import make_replay_samplers   # <-- IMPORTANT: not from sample.py

def frac_early_dones(dones_window: jnp.ndarray) -> float:
    """Fraction of windows where a 'done' occurs before the last step."""
    if dones_window.shape[1] <= 1:
        return 0.0
    early = dones_window[:, :-1].any(axis=1)
    return float(early.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault_uid", required=True)
    ap.add_argument("--rel_dir", default="/home/synaderi/Craftax_Baselines")
    ap.add_argument("--T_wm", type=int, default=20)
    ap.add_argument("--batch_envs", type=int, default=48)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    sample_train, sample_val, info = make_replay_samplers(
        vault_uid=args.vault_uid,
        rel_dir=args.rel_dir,
        T_wm=args.T_wm,
        batch_envs=args.batch_envs,
    )
    print(f"[D] Loaded replay: B={info['B']}  T_train={info['T_train']}  T_val={info['T_val']}")

    rng = jax.random.PRNGKey(0)

    # Train sampler checks
    for i in range(args.iters):
        rng, sub = jax.random.split(rng)
        batch = sample_train(sub)  # returns a plain dict now
        obs, actions, rewards, dones = (
            batch["obs"], batch["actions"], batch["rewards"], batch["dones"]
        )

        print(f"\n[train] iter {i}")
        print("  obs     :", tuple(obs.shape),  obs.dtype)
        print("  actions :", tuple(actions.shape), actions.dtype)
        print("  rewards :", tuple(rewards.shape), rewards.dtype)
        print("  dones   :", tuple(dones.shape),   dones.dtype)

        print("  obs[min,max]     :", int(obs.min()), int(obs.max()))
        print("  rewards mean/std :", float(rewards.mean()), float(rewards.std()))
        print("  unique actions    :", int(jnp.unique(actions).size))
        print(f"  frac windows w/ early done: {frac_early_dones(dones):.3f}")

        # sanity
        assert obs.ndim == 5 and obs.shape[2:] == (63, 63, 3)
        assert actions.shape[:2] == rewards.shape[:2] == dones.shape[:2] == obs.shape[:2]
        assert obs.dtype == jnp.float32 and actions.dtype in (jnp.int32, jnp.int64)
        assert rewards.dtype == jnp.float32 and dones.dtype == jnp.bool_

    # Val sampler once
    rng, sub = jax.random.split(rng)
    vb = sample_val(sub)
    print("\n[val] one batch:")
    print("  obs     :", tuple(vb['obs'].shape), vb['obs'].dtype)
    print("  actions :", tuple(vb['actions'].shape), vb['actions'].dtype)
    print("  rewards :", tuple(vb['rewards'].shape), vb['rewards'].dtype)
    print("  dones   :", tuple(vb['dones'].shape),   vb['dones'].dtype)
    print("  frac windows w/ early done:", f"{frac_early_dones(vb['dones']):.3f}")

    print("\n✅ Sampler looks good if shapes & stats are sane.")

if __name__ == "__main__":
    main()

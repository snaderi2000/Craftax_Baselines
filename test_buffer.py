# test_buffer.py
import argparse
import numpy as np
import jax.numpy as jnp
from flashbax.vault import Vault

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel_dir", default="/home/synaderi/Craftax_Baselines")
    ap.add_argument("--vault_name", default="craftax_replay_buffer")
    ap.add_argument("--vault_uid", default="my_first_buffer_run")
    args = ap.parse_args()

    # Open the exact run
    v = Vault(vault_name=args.vault_name, rel_dir=args.rel_dir, vault_uid=args.vault_uid)
    print("vault_index (timesteps per env):", v.vault_index)

    # Read everything that was written
    buf = v.read()
    exp = buf.experience

    obs     = exp["obs"]      # (B, T, 63, 63, 3) uint8
    actions = exp["actions"]  # (B, T) int32
    rewards = exp["rewards"]  # (B, T) float32
    dones   = exp["dones"]    # (B, T) bool

    B, T = actions.shape
    print(f"\nshapes:")
    print(f"  obs     {tuple(obs.shape)}  {obs.dtype}")
    print(f"  actions {tuple(actions.shape)}  {actions.dtype}")
    print(f"  rewards {tuple(rewards.shape)}  {rewards.dtype}")
    print(f"  dones   {tuple(dones.shape)}  {dones.dtype}")

    total = B * T
    print(f"\nsummary:")
    print(f"  envs (B)         = {B}")
    print(f"  timesteps/env (T) = {T}")
    print(f"  total transitions = {total}")

    # quick sanity stats
    print("\nquick stats:")
    print("  obs min/max:", int(obs.min()), int(obs.max()))
    print("  rewards mean/std:", float(rewards.mean()), float(rewards.std()))
    print("  done count:", int(dones.sum()))
    ua_env0 = int(jnp.unique(actions[0]).size)
    ua_all  = int(jnp.unique(actions).size)
    print(f"  unique actions (env0/global): {ua_env0}/{ua_all}")

    # per-env episode counts & avg lengths
    done_per_env = dones.sum(axis=1)
    avg_len_per_env = T / jnp.maximum(1, done_per_env)
    print("  per-env done count (min/mean/max):",
          int(done_per_env.min()), float(done_per_env.mean()), int(done_per_env.max()))
    print("  per-env avg length (min/mean/max):",
          float(avg_len_per_env.min()), float(avg_len_per_env.mean()), float(avg_len_per_env.max()))
    print("  dones at final timestep:", int(dones[:, -1].sum()))

    # quick episodic returns over completed episodes (flattened over envs)
    flat_r = rewards.reshape(-1)
    flat_d = dones.reshape(-1)
    end_idx = np.nonzero(np.array(flat_d))[0]
    ep_returns, prev = [], -1
    for e in end_idx:
        ep_returns.append(float(flat_r[prev+1:e+1].sum()))
        prev = e
    if ep_returns:
        q = np.quantile(ep_returns, [0.0, 0.5, 0.9, 0.99])
        print(f"  episode returns: n={len(ep_returns)} "
              f"min/median/p90/p99={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}")
    else:
        print("  (no completed episodes found in this slice)")

if __name__ == "__main__":
    main()

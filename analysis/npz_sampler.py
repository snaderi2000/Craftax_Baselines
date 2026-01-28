"""
Sampler for NPZ replay data files.

Provides the same interface as world_model/sample.py but reads from NPZ files
instead of flashbax Vaults. This avoids GPU memory issues during data loading.

Usage:
    from analysis.npz_sampler import make_replay_samplers_from_npz
    
    sample_train, sample_val, info = make_replay_samplers_from_npz(
        npz_path="./replay_data/my_buffer/replay_data.npz",
        T_wm=20,
        batch_size=48,
    )
    
    # Sample a batch
    batch = sample_train(rng_key)
    # batch has keys: obs, actions, rewards, dones
"""

import jax
import jax.numpy as jnp
import numpy as np


def make_replay_samplers_from_npz(
    npz_path: str,
    T_wm: int = 20,
    batch_size: int = 48,
    train_pct: float = 0.9,
):
    """
    Create train/val samplers from an NPZ file.
    
    Args:
        npz_path: Path to the NPZ file containing replay data
        T_wm: Sequence length to sample (world model context length)
        batch_size: Number of sequences per batch
        train_pct: Fraction of episodes to use for training (rest for validation)
    
    Returns:
        sample_train: Function(rng_key) -> batch dict
        sample_val: Function(rng_key) -> batch dict  
        info: Dict with dataset statistics
    """
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    obs = data["obs"]        # (B, T, 63, 63, 3) float32
    actions = data["actions"]  # (B, T) int32
    rewards = data["rewards"]  # (B, T) float32
    dones = data["dones"]      # (B, T) bool
    
    B, T = actions.shape
    print(f"Loaded {B} episodes with {T} max timesteps each")
    
    # Split into train/val by episodes
    n_train = int(B * train_pct)
    
    train_data = {
        "obs": jnp.array(obs[:n_train]),
        "actions": jnp.array(actions[:n_train]),
        "rewards": jnp.array(rewards[:n_train]),
        "dones": jnp.array(dones[:n_train]),
    }
    
    val_data = {
        "obs": jnp.array(obs[n_train:]),
        "actions": jnp.array(actions[n_train:]),
        "rewards": jnp.array(rewards[n_train:]),
        "dones": jnp.array(dones[n_train:]),
    }
    
    B_train = train_data["actions"].shape[0]
    B_val = val_data["actions"].shape[0]
    
    print(f"Train: {B_train} episodes, Val: {B_val} episodes")
    
    def _sample_trajectories(data_dict, rng_key, batch_size, seq_len):
        """Sample random trajectory windows from the data."""
        B_data, T_data = data_dict["actions"].shape
        
        # Maximum valid start index for a window of length seq_len
        max_start = T_data - seq_len
        
        # Sample random episodes and start positions
        rng_ep, rng_start = jax.random.split(rng_key)
        episode_idxs = jax.random.randint(rng_ep, (batch_size,), 0, B_data)
        start_idxs = jax.random.randint(rng_start, (batch_size,), 0, max_start + 1)
        
        # Extract windows using vmap
        def extract_window(ep_idx, start_idx):
            return {
                "obs": jax.lax.dynamic_slice(
                    data_dict["obs"][ep_idx], 
                    (start_idx, 0, 0, 0), 
                    (seq_len, 63, 63, 3)
                ),
                "actions": jax.lax.dynamic_slice(
                    data_dict["actions"][ep_idx],
                    (start_idx,),
                    (seq_len,)
                ),
                "rewards": jax.lax.dynamic_slice(
                    data_dict["rewards"][ep_idx],
                    (start_idx,),
                    (seq_len,)
                ),
                "dones": jax.lax.dynamic_slice(
                    data_dict["dones"][ep_idx],
                    (start_idx,),
                    (seq_len,)
                ),
            }
        
        batch = jax.vmap(extract_window)(episode_idxs, start_idxs)
        return batch
    
    @jax.jit
    def sample_train(rng_key):
        batch = _sample_trajectories(train_data, rng_key, batch_size, T_wm)
        # Normalize obs to [0, 1] if stored as float32 in [0, 255] range
        # (Our generate_data.py stores as float32 already normalized)
        return batch
    
    @jax.jit
    def sample_val(rng_key):
        batch = _sample_trajectories(val_data, rng_key, batch_size, T_wm)
        return batch
    
    info = {
        "B_train": B_train,
        "B_val": B_val,
        "T": T,
        "T_wm": T_wm,
        "batch_size": batch_size,
    }
    
    return sample_train, sample_val, info


# Convenience function matching the vault-based API
def make_replay_samplers(
    npz_path: str,
    T_wm: int = 20,
    batch_envs: int = 48,
    train_pct=(0, 90),
    val_pct=(90, 100),
):
    """
    Drop-in replacement for world_model/sample.py's make_replay_samplers.
    
    Note: train_pct/val_pct are tuples for API compatibility but we only 
    use the split point (train_pct[1] / 100).
    """
    train_frac = train_pct[1] / 100.0
    sample_train, sample_val, info = make_replay_samplers_from_npz(
        npz_path=npz_path,
        T_wm=T_wm,
        batch_size=batch_envs,
        train_pct=train_frac,
    )
    
    # Wrap to match the expected output format from world_model/sample.py
    def _to_plain_dict(batch):
        """Match the output format of world_model/sample.py"""
        return {
            "obs": batch["obs"].astype(jnp.float32),  # Already float32
            "actions": batch["actions"],
            "rewards": batch["rewards"],
            "dones": batch["dones"],
        }
    
    def wrapped_train(rng_key):
        return _to_plain_dict(sample_train(rng_key))
    
    def wrapped_val(rng_key):
        return _to_plain_dict(sample_val(rng_key))
    
    return wrapped_train, wrapped_val, info


if __name__ == "__main__":
    # Quick test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str, required=True)
    parser.add_argument("--T_wm", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    sample_train, sample_val, info = make_replay_samplers_from_npz(
        args.npz_path,
        T_wm=args.T_wm,
        batch_size=args.batch_size,
    )
    
    print(f"\nDataset info: {info}")
    
    # Test sampling
    rng = jax.random.PRNGKey(0)
    batch = sample_train(rng)
    
    print(f"\nSampled batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} ({v.dtype})")
    
    print("\n✅ Sampler working correctly!")

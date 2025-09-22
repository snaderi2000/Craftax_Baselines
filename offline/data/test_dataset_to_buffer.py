# offline/data/test_dataset_to_buffer.py
"""
Test pipeline for streaming data from .npz files into a Tianshou ReplayBuffer.
"""

import os
import numpy as np
from offline.data.npz_dataset import NPZStreamDataset
from offline.data.buffer_wrapper import NPZReplayBuffer
from torch.utils.data import DataLoader



def test_dataset_to_buffer():
    # Paths
    data_dir = "craftax_classic_200M_dataset"  # <-- Adjust if your data folder is elsewhere
    assert os.path.exists(data_dir), f"Data directory not found: {data_dir}"

    # Settings
    obs_dim = 1345
    action_dim = 17
    buffer_size = 5000   # Small for testing

    print(f"Loading from directory: {data_dir}")

    # 1. Create streaming dataset
    dataset = NPZStreamDataset(data_dir)
    print(f"Found {len(dataset.files)} .npz files for streaming.")

    # 2. Create replay buffer
    buffer = NPZReplayBuffer(buffer_size=buffer_size, obs_dim=obs_dim, action_dim=action_dim)

    # 3. Load the first batch from the dataset
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    batch = next(iter(loader))

    print("Batch keys:", batch.keys())
    print("Obs shape:", batch["obs"].shape)
    print("Actions shape:", batch["action"].shape)

    # 4. Push the batch into buffer
    buffer.push_batch(
        obs=batch["obs"].numpy(),
        actions=batch["action"].numpy(),
        rewards=batch["reward"].numpy(),
        dones=batch["done"].numpy(),
        next_obs=batch["next_obs"].numpy()
    )

    # 5. Validate buffer state
    assert len(buffer) == 64, f"Buffer size mismatch: expected 64, got {len(buffer)}"

    # 6. Sample from buffer
    sampled_batch, indices = buffer.sample(batch_size=8)
    print("Sampled obs shape:", sampled_batch.obs.shape)
    print("Sampled act shape:", sampled_batch.act.shape)

    assert sampled_batch.obs.shape == (8, obs_dim)
    assert sampled_batch.act.shape == (8,)

    print("test_dataset_to_buffer.py PASSED!")


if __name__ == "__main__":
    test_dataset_to_buffer()


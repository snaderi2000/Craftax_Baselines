"""
Test pipeline for streaming data from .npz files into a Tianshou ReplayBuffer.
"""

import os
from torch.utils.data import DataLoader
from offline.data.npz_dataset import NPZStreamDataset
from offline.data.buffer_wrapper import NPZReplayBuffer


def test_dataset_to_buffer():
    # Paths
    data_dir = "craftax_classic_200M_dataset"  # <-- Adjust this if needed
    assert os.path.exists(data_dir), f"Data directory not found: {data_dir}"

    # Settings
    obs_dim = 1345
    action_dim = 17
    buffer_size = 5000  # Small for testing

    print(f"Loading from directory: {data_dir}")

    # 1. Create streaming dataset
    dataset = NPZStreamDataset(data_dir)
    print(f"Found {len(dataset.files)} .npz files for streaming.")

    # 2. Create replay buffer
    buffer = NPZReplayBuffer(buffer_size=buffer_size, obs_dim=obs_dim, action_dim=action_dim)

    # 3. Create DataLoader for streaming
    loader = DataLoader(dataset, batch_size=64, num_workers=4)  # shuffle=False for IterableDataset

    # 4. Grab a single batch
    batch = next(iter(loader))

    print("Batch keys:", batch.keys())
    print("Obs shape:", batch["obs"].shape)
    print("Actions shape:", batch["action"].shape)

    # 5. Push the batch into the replay buffer
    buffer.push_batch(
        obs=batch["obs"].numpy(),
        actions=batch["action"].numpy(),
        rewards=batch["reward"].numpy(),
        dones=batch["done"].numpy(),
        next_obs=batch["next_obs"].numpy(),
    )

    # 6. Validate buffer state
    assert len(buffer) == 64, f"Buffer size mismatch: expected 64, got {len(buffer)}"

    # 7. Sample from buffer
    sampled_batch, indices = buffer.sample(batch_size=8)
    print("Sampled obs shape:", sampled_batch.obs.shape)
    print("Sampled act shape:", sampled_batch.act.shape)

    assert sampled_batch.obs.shape == (8, obs_dim)
    assert sampled_batch.act.shape == (8,)

    print("test_dataset_to_buffer.py PASSED!")


if __name__ == "__main__":
    test_dataset_to_buffer()

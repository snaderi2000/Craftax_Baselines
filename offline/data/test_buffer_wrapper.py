import numpy as np
from offline.data.buffer_wrapper import NPZReplayBuffer

def test_buffer_wrapper():
    # Create fake data to simulate streaming from dataset
    batch_size = 8
    obs_dim = 1345
    action_dim = 1

    obs = np.random.rand(batch_size, obs_dim).astype(np.float32)
    next_obs = np.random.rand(batch_size, obs_dim).astype(np.float32)
    actions = np.random.randint(0, 17, size=(batch_size,), dtype=np.int32)
    rewards = np.random.rand(batch_size).astype(np.float32)
    dones = np.random.randint(0, 2, size=(batch_size,), dtype=np.bool_)

    # Create buffer
    buffer = NPZReplayBuffer(buffer_size=100, obs_dim=obs_dim, action_dim=action_dim)



    # Push batch
    buffer.push_batch(obs, actions, rewards, dones, next_obs)

    # Check the buffer now has correct data
    assert len(buffer) == batch_size, "Replay buffer should contain all pushed transitions"

    # Sample a batch
    sampled = buffer.sample(batch_size=4)
    print("Sampled obs shape:", sampled.obs.shape)
    print("Sampled actions shape:", sampled.act.shape)

    assert sampled.obs.shape == (4, obs_dim)
    assert sampled.act.ndim == 1

    print("buffer_wrapper.py test PASSED!")

if __name__ == "__main__":
    test_buffer_wrapper()

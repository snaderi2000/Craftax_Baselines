# offline/data/test_buffer_wrapper.py
import numpy as np
from offline.data.buffer_wrapper import NPZReplayBuffer


def test_buffer_wrapper():
    obs_dim = 1345
    action_dim = 17
    buffer_size = 100

    # Create buffer
    buffer = NPZReplayBuffer(buffer_size=buffer_size, obs_dim=obs_dim, action_dim=action_dim)

    # Fake data
    batch_size = 10
    obs = np.random.rand(batch_size, obs_dim).astype(np.float32)
    next_obs = np.random.rand(batch_size, obs_dim).astype(np.float32)
    actions = np.random.randint(0, action_dim, size=(batch_size,))
    rewards = np.random.rand(batch_size).astype(np.float32)
    dones = np.zeros(batch_size, dtype=bool)

    # Push to buffer
    buffer.push_batch(obs, actions, rewards, dones, next_obs)

    # Assertions
    assert len(buffer) == batch_size, f"Buffer length mismatch: {len(buffer)} vs {batch_size}"
    assert buffer.obs.shape == (buffer_size, obs_dim), "Observation shape mismatch"
    assert buffer.act.shape == (buffer_size,), "Action shape mismatch"

    # Sample a batch
    sampled_batch, indices = buffer.sample(batch_size=4)
    print("Sampled obs shape:", sampled_batch.obs.shape)
    print("Sampled actions shape:", sampled_batch.act.shape)

    assert sampled_batch.obs.shape[1] == obs_dim, "Sampled obs_dim mismatch"
    assert sampled_batch.act.shape[0] == 4, "Sample size mismatch"

    print("buffer_wrapper.py test PASSED!")


if __name__ == "__main__":
    test_buffer_wrapper()

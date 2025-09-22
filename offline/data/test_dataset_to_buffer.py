from torch.utils.data import DataLoader
from offline.data.npz_dataset import NPZStreamDataset
from offline.data.buffer_wrapper import NPZReplayBuffer

def test_dataset_to_buffer():
    data_dir = "craftax_classic_200M_dataset"
    dataset = NPZStreamDataset(data_dir, fields=("obs", "action", "reward", "done", "next_obs"))
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)

    buffer = NPZReplayBuffer(buffer_size=100, obs_dim=1345, action_dim=1)

    # Stream one batch and push into buffer
    obs, actions, rewards, dones, next_obs = next(iter(dataloader))
    buffer.push_batch(obs.numpy(), actions.numpy(), rewards.numpy(), dones.numpy(), next_obs.numpy())

    # Verify
    print("Buffer size:", len(buffer))
    sample = buffer.sample(4)
    print("Sampled obs shape:", sample.obs.shape)
    print("Sampled actions shape:", sample.act.shape)

if __name__ == "__main__":
    test_dataset_to_buffer()

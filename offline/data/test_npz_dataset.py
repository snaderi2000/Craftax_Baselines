import os
from torch.utils.data import DataLoader
from offline.data.npz_dataset import NPZStreamDataset

def test_npz_dataset():
    # Path to your directory containing .npz files
    data_dir = "craftax_classic_200M_dataset"

    # Ensure directory exists
    assert os.path.exists(data_dir), f"{data_dir} does not exist!"

    # Create dataset
    dataset = NPZStreamDataset(data_dir, fields=("obs", "action"), shuffle=False)

    # Wrap with DataLoader
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

    # Fetch first batch
    batch = next(iter(dataloader))

    obs, actions = batch
    print("Obs shape:", obs.shape)
    print("Actions shape:", actions.shape)

    # Sanity checks
    assert obs.shape[1] == 1345, "Observation vector size must be 1345!"
    assert actions.ndim == 1, "Actions should be a 1D vector for discrete space"

    print("npz_dataset.py test PASSED!")

if __name__ == "__main__":
    test_npz_dataset()

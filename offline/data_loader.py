# data_loader.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

class CraftaxNPZDataset(IterableDataset):
    def __init__(self, data_dir, subset_fraction=1.0, shuffle_files=True, seed=42):
        self.files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npz")))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        self.shuffle_files = shuffle_files
        self.rng = np.random.default_rng(seed)
        self.subset_fraction = subset_fraction

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle_files:
            self.rng.shuffle(files)

        for file_path in files:
            data = np.load(file_path, allow_pickle=False)
            N = data["obs"].shape[0]

            # Subset sampling
            if self.subset_fraction < 1.0:
                keep_count = int(N * self.subset_fraction)
                indices = self.rng.choice(N, size=keep_count, replace=False)
            else:
                indices = np.arange(N)

            for i in indices:
                yield {
                    "obs": torch.from_numpy(data["obs"][i]).float(),
                    "action": torch.tensor(data["action"][i], dtype=torch.long),
                    "reward": torch.tensor(data["reward"][i], dtype=torch.float32),
                    "done": torch.tensor(data["done"][i], dtype=torch.bool),
                    "next_obs": torch.from_numpy(data["next_obs"][i]).float(),
                    "log_prob": torch.tensor(data["log_prob"][i], dtype=torch.float32),
                    "value": torch.tensor(data["value"][i], dtype=torch.float32),
                }

def make_dataloader(data_dir, batch_size=1024, subset_fraction=1.0, num_workers=4):
    dataset = CraftaxNPZDataset(
        data_dir=data_dir,
        subset_fraction=subset_fraction,
        shuffle_files=True,
        seed=42,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

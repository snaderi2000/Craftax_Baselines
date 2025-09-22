from torch.utils.data import IterableDataset
import numpy as np
import os
import random

class NPZStreamDataset(IterableDataset):
    def __init__(self, data_dir, shuffle=True):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        self.shuffle = shuffle

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)

        for f in files:
            data = np.load(f)
            length = len(data["obs"])
            for i in range(length):
                # Return a dict for each transition
                yield {
                    "obs": data["obs"][i],
                    "action": data["action"][i],
                    "reward": data["reward"][i],
                    "done": data["done"][i],
                    "next_obs": data["next_obs"][i],
                }

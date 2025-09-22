from torch.utils.data import IterableDataset
import numpy as np
import os
import random

class NPZStreamDataset(IterableDataset):
    def __init__(self, data_dir, fields=("obs", "action"), shuffle=True):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        self.shuffle = shuffle
        self.fields = fields  # allows us to only load obs/action for BC

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)
        for f in files:
            data = np.load(f)
            arrays = [data[k] for k in self.fields]
            length = len(arrays[0])
            for i in range(length):
                yield tuple(arr[i] for arr in arrays)

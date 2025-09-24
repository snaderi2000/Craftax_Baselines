import numpy as np
import h5py
from tqdm import tqdm
import os

OUTPUT_H5 = "craftax_last50_cleaned.h5"

EARLY_FRAC = 0.3
MID_FRAC = 0.4
LATE_FRAC = 0.3  # auto remainder
SOURCE_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"  # folder with .npz files

def trim_start(data):
    """Trim partial episode at start."""
    first_done_idx = np.where(data["done"])[0][0]
    if first_done_idx > 0:
        return {k: v[first_done_idx+1:] for k, v in data.items()}
    return data

def trim_end(data):
    """Trim partial episode at end."""
    last_done_idx = np.where(data["done"])[0][-1]
    if last_done_idx < len(data["done"]) - 1:
        return {k: v[:last_done_idx+1] for k, v in data.items()}
    return data

def compute_game_phase(done):
    """0 = early, 1 = mid, 2 = late"""
    done_indices = np.where(done)[0]
    starts = np.concatenate(([0], done_indices + 1))
    labels = np.zeros_like(done, dtype=np.int8)

    for s, e in zip(starts, done_indices + 1):
        length = e - s
        if length <= 0: continue
        e_cut = s + int(EARLY_FRAC * length)
        m_cut = e_cut + int(MID_FRAC * length)
        labels[s:e_cut] = 0
        labels[e_cut:m_cut] = 1
        labels[m_cut:e] = 2
    return labels

# Get subset of files
files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith(".npz")],
               key=lambda x: int(x.split('_')[-1].split('.')[0]))[-50:]

# Pre-compute total steps
total_steps = sum(np.load(os.path.join(SOURCE_DIR, f))["obs"].shape[0] for f in files)

# Setup HDF5
first_file = np.load(os.path.join(SOURCE_DIR, files[0]))
keys = [k for k in first_file.keys() if k != "info"]

with h5py.File(OUTPUT_H5, "w") as h5f:
    datasets = {}
    for k in keys:
        shape = (total_steps,) + (first_file[k].shape[1:] if first_file[k].ndim > 1 else ())
        datasets[k] = h5f.create_dataset(k, shape=shape, dtype=first_file[k].dtype)
    datasets["game_phase"] = h5f.create_dataset("game_phase", shape=(total_steps,), dtype=np.int8)

    idx = 0
    for i, fname in enumerate(tqdm(files, desc="Processing last 50 files")):
        path = os.path.join(SOURCE_DIR, fname)
        data = np.load(path)

        # Trim starts
        data = trim_start(data)

        # Trim end if final file
        if i == len(files) - 1:
            data = trim_end(data)

        # Compute game phase
        phase_labels = compute_game_phase(data["done"])

        steps = data["obs"].shape[0]
        for k in keys:
            datasets[k][idx:idx+steps] = data[k]
        datasets["game_phase"][idx:idx+steps] = phase_labels
        idx += steps

    # Resize to exact size after trimming
    for k in datasets.keys():
        datasets[k].resize((idx,) + datasets[k].shape[1:])

print(f"✅ Finished dataset: {OUTPUT_H5}, total transitions: {idx}")

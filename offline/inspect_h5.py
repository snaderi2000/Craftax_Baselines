import h5py
import numpy as np

H5_PATH = "episodes_compressed.h5"

print(f"Inspecting file: {H5_PATH}\n")

with h5py.File(H5_PATH, 'r') as hf:
    # Get a list of all episodes (groups) in the file
    all_episode_keys = list(hf.keys())
    print(f"Found {len(all_episode_keys)} total episodes.")
    
    # --- Inspect the first episode ---
    if len(all_episode_keys) > 0:
        first_episode_key = all_episode_keys[0]
        print(f"\n--- Structure of '{first_episode_key}' ---")
        
        # List all datasets (obs, action, etc.) within the episode group
        episode_group = hf[first_episode_key]
        for key in episode_group.keys():
            dataset = episode_group[key]
            # Check if compression was applied
            compression = dataset.compression if dataset.compression else "None"
            print(f"  - Dataset: {key}, Shape: {dataset.shape}, Dtype: {dataset.dtype}, Compression: {compression}")
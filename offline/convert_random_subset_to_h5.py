# convert_random_subset_to_h5.py
import os
import random
import numpy as np
from d3rlpy.dataset import MDPDataset
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"
OUTPUT_PATH = "craftax_dataset_10percent.h5"
NUM_FILES_TO_SAMPLE = 305     # 10% of 3051 total files
SEED = 42                     # For reproducibility
# ------------------------

def convert_random_subset_to_h5(data_dir, output_path, num_files=305, seed=42):
    # Collect all .npz files
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    total_files = len(all_files)
    if total_files == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    
    print(f"Found {total_files} total files in {data_dir}")

    # Fix random seed and randomly sample
    random.seed(seed)
    sampled_files = random.sample(all_files, min(num_files, total_files))
    print(f"Randomly selected {len(sampled_files)} files using seed {seed}")

    # Initialize lists to collect arrays
    obs_list, next_obs_list = [], []
    actions, rewards, dones = [], [], []

    # Process each sampled file
    for file in tqdm(sampled_files, desc="Converting"):
        path = os.path.join(data_dir, file)
        data = np.load(path, allow_pickle=False)

        obs_list.append(data["obs"])
        next_obs_list.append(data["next_obs"])
        actions.append(data["action"])
        rewards.append(data["reward"])
        dones.append(data["done"].astype(np.float32))  # Ensure float32

    # Concatenate into final arrays
    observations = np.concatenate(obs_list, axis=0)
    next_observations = np.concatenate(next_obs_list, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    dones = np.concatenate(dones, axis=0)

    print(f"Final subset size: {observations.shape[0]} transitions")

    # Create MDPDataset for d3rlpy
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=dones,
    )

    # Save to disk
    dataset.dump(output_path)
    print(f"Saved random subset MDPDataset to {output_path}")


if __name__ == "__main__":
    convert_random_subset_to_h5(DATA_DIR, OUTPUT_PATH, NUM_FILES_TO_SAMPLE, SEED)

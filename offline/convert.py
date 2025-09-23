import os
import random
import numpy as np
from tqdm import tqdm
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"  # folder with .npz files
OUTPUT_PATH = "craftax_dataset_64files_minimal_v2.h5"  # output h5 file
NUM_FILES_TO_SAMPLE = 64  # number of .npz files to include
SEED = 42  # random seed for reproducibility
# -----------------------

def convert_random_npz_to_h5(data_dir, output_path, num_files=64, seed=42):
    # Find all .npz files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    if len(all_files) == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")

    print(f"Found {len(all_files)} total .npz files in {data_dir}")

    # Randomly sample files
    random.seed(seed)
    sampled_files = random.sample(all_files, min(num_files, len(all_files)))
    print(f"Randomly selected {len(sampled_files)} files using seed {seed}")

    # Collect all episodes in a list
    all_episodes = []

    # Process each file
    for path in tqdm(sampled_files, desc="Processing files"):
        # Load with allow_pickle=True to handle 'info' object array safely
        data = np.load(path, allow_pickle=True)

        # Extract only the needed arrays
        obs = data["obs"]            # shape = (N, obs_dim)
        actions = data["action"]     # shape = (N,)
        rewards = data["reward"]     # shape = (N,)
        dones = data["done"].astype(np.float32)  # shape = (N,)

        # Split into episodes using done=True
        start = 0
        for i, done_flag in enumerate(dones):
            if done_flag:  # episode ends at this index
                episode = Episode(
                    observations=obs[start:i+1],
                    actions=actions[start:i+1],
                    rewards=rewards[start:i+1],
                    terminals=dones[start:i+1],
                )
                all_episodes.append(episode)
                start = i + 1  # next episode starts here

    print(f"Finished building {len(all_episodes)} total episodes.")

    # Create ReplayBuffer *after* episodes are collected
    buffer = FIFOBuffer(limit=None)
    replay_buffer = ReplayBuffer(buffer, episodes=all_episodes)

    # Save ReplayBuffer to disk
    with open(output_path, "wb") as f:
        replay_buffer.dump(f)

    print(f"Saved ReplayBuffer to {output_path}")


if __name__ == "__main__":
    convert_random_npz_to_h5(DATA_DIR, OUTPUT_PATH, NUM_FILES_TO_SAMPLE, SEED)

import os
import random
import numpy as np
from tqdm import tqdm
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"  # folder with .npz files
OUTPUT_PATH = "craftax_dataset_64files_minimal_v2.h5"  # output file
NUM_FILES_TO_SAMPLE = 64  # how many files to include
SEED = 42  # for reproducibility
# -----------------------

def convert_random_npz_to_h5(data_dir, output_path, num_files=64, seed=42):
    # Collect all .npz files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    if len(all_files) == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    
    print(f"Found {len(all_files)} total .npz files in {data_dir}")

    # Randomly sample files
    random.seed(seed)
    sampled_files = random.sample(all_files, min(num_files, len(all_files)))
    print(f"Randomly selected {len(sampled_files)} files with seed {seed}")

    # Create ReplayBuffer
    buffer = FIFOBuffer(limit=None)
    replay_buffer = ReplayBuffer(buffer)

    # Process each sampled file
    for path in tqdm(sampled_files, desc="Processing files"):
        # Load with pickle because 'info' is an object array (we will ignore it)
        data = np.load(path, allow_pickle=True)

        # Keep only essential arrays
        obs = data["obs"]           # shape = (N, obs_dim)
        actions = data["action"]    # shape = (N,)
        rewards = data["reward"]    # shape = (N,)
        dones = data["done"].astype(np.float32)  # shape = (N,)

        # Split file into episodes using done=True as boundary
        start = 0
        for i, done_flag in enumerate(dones):
            if done_flag:  # episode ends at this index
                episode = Episode(
                    observations=obs[start:i+1],
                    actions=actions[start:i+1],
                    rewards=rewards[start:i+1],
                    terminals=dones[start:i+1],
                )
                replay_buffer.append_episode(episode)
                start = i + 1  # next episode starts here

    print(f"Finished building ReplayBuffer with {len(replay_buffer.episodes)} total episodes.")

    # Save ReplayBuffer to a single compact .h5 file
    with open(output_path, "wb") as f:
        replay_buffer.dump(f)

    print(f"Saved minimal ReplayBuffer to {output_path}")


if __name__ == "__main__":
    convert_random_npz_to_h5(DATA_DIR, OUTPUT_PATH, NUM_FILES_TO_SAMPLE, SEED)

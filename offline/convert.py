import os
import random
import numpy as np
from tqdm import tqdm
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"  # folder with .npz files
OUTPUT_PATH = "craftax_dataset_cleaned.h5"  # output single .h5 file
NUM_FILES_TO_SAMPLE = None  # set to None to use all files, or e.g. 64 to sample a subset
SEED = 42  # random seed for reproducibility
# -----------------------


def split_into_episodes(obs, actions, rewards, dones):
    """
    Split flat transition arrays into individual episodes using done=True flags.
    """
    episodes = []
    start = 0
    for i, done_flag in enumerate(dones):
        if done_flag:
            episode = Episode(
                observations=obs[start:i+1],
                actions=actions[start:i+1],
                rewards=rewards[start:i+1],
                terminals=dones[start:i+1],  # <-- FIXED HERE
            )
            episodes.append(episode)
            start = i + 1
    return episodes


def convert_npz_to_replay_buffer(data_dir, output_path, num_files=None, seed=42):
    """
    Load multiple .npz files, split into episodes, and save as a single ReplayBuffer .h5 file.
    """
    # Collect all .npz files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    if len(all_files) == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    print(f"Found {len(all_files)} total .npz files in {data_dir}")

    # Randomly sample files if specified
    if num_files is not None:
        random.seed(seed)
        sampled_files = random.sample(all_files, min(num_files, len(all_files)))
        print(f"Randomly selected {len(sampled_files)} files using seed {seed}")
    else:
        sampled_files = all_files
        print("Using all files.")

    # Collect all episodes
    all_episodes = []

    for path in tqdm(sampled_files, desc="Processing files"):
        # allow_pickle=True is required because 'info' is stored as an object array
        data = np.load(path, allow_pickle=True)

        obs = data["obs"]
        actions = data["action"]
        rewards = data["reward"]
        dones = data["done"].astype(np.float32)

        # Split into episodes using done=True
        episodes = split_into_episodes(obs, actions, rewards, dones)
        all_episodes.extend(episodes)

    print(f"Total episodes collected: {len(all_episodes)}")

    # Create ReplayBuffer from episodes
    buffer = FIFOBuffer(limit=None)
    replay_buffer = ReplayBuffer(buffer, episodes=all_episodes)

    # Save to disk
    with open(output_path, "wb") as f:
        replay_buffer.dump(f)

    print(f"Saved ReplayBuffer to {output_path}")


if __name__ == "__main__":
    convert_npz_to_replay_buffer(DATA_DIR, OUTPUT_PATH, NUM_FILES_TO_SAMPLE, SEED)

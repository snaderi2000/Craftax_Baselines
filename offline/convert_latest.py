import os
import numpy as np
from tqdm import tqdm
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"  # folder with .npz files
OUTPUT_PATH = "craftax_dataset_cleaned.h5"  # output file
NUM_FILES_TO_SAMPLE = 64  # or set to 64 to sample a subset
SEED = 42
# -----------------------

def split_into_episodes(obs, actions, rewards, dones):
    """
    Convert raw step-wise data into a list of d3rlpy Episode objects.
    """
    episodes = []
    start = 0
    for i, done_flag in enumerate(dones):
        if done_flag:
            ep_obs = obs[start:i+1]
            ep_actions = actions[start:i+1]
            ep_rewards = rewards[start:i+1]

            # terminated=True since this episode ends here
            episode = Episode(ep_obs, ep_actions, ep_rewards, True)
            episodes.append(episode)

            start = i + 1  # start next episode after this

    return episodes


def convert_npz_to_replay_buffer(data_dir, output_path, num_files=None, seed=42, end_batch=None):
    # Collect all files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    if len(all_files) == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    print(f"Found {len(all_files)} total .npz files")

    # Optionally filter by batch index if `end_batch` is provided.
    # Filenames are expected like 'batch_<num>.npz'. If parsing fails for a file,
    # it will be ignored when `end_batch` is used.
    if end_batch is not None:
        def parse_batch_index(path):
            name = os.path.basename(path)
            if name.startswith("batch_") and name.endswith(".npz"):
                try:
                    num = int(name[len("batch_"):-len(".npz")])
                    return num
                except ValueError:
                    return None
            return None

        filtered = []
        for p in all_files:
            idx = parse_batch_index(p)
            if idx is None:
                continue
            if idx <= end_batch:
                filtered.append(p)

        if len(filtered) == 0:
            raise RuntimeError(f"No batch files found with index <= {end_batch}")
        all_files = filtered
        print(f"Filtered to {len(all_files)} files with batch index <= {end_batch}")

    # Select the last `num_files` by modification time (most recent files)
    if num_files:
        # sort by modification time ascending and take the last `num_files`
        all_files.sort(key=lambda p: os.path.getmtime(p))
        files = all_files[-min(num_files, len(all_files)):]
        print(f"Selected the last {len(files)} files by modification time")
    else:
        files = all_files
        print("Using all files.")

    all_episodes = []

    # Process files
    for path in tqdm(files, desc="Processing files"):
        data = np.load(path, allow_pickle=True)
        obs = data["obs"]
        actions = data["action"]
        rewards = data["reward"]
        dones = data["done"]

        # Split into episodes
        episodes = split_into_episodes(obs, actions, rewards, dones)
        all_episodes.extend(episodes)

    print(f"Total episodes collected: {len(all_episodes)}")

    # Create ReplayBuffer
    buffer = FIFOBuffer(limit=None)
    replay_buffer = ReplayBuffer(buffer, episodes=all_episodes)

    # Save to file
    with open(output_path, "w+b") as f:
        replay_buffer.dump(f)

    print(f"Saved ReplayBuffer to {output_path}")


if __name__ == "__main__":
    convert_npz_to_replay_buffer(DATA_DIR, OUTPUT_PATH, NUM_FILES_TO_SAMPLE, SEED)

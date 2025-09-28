import os
import glob
import numpy as np
import h5py
from tqdm import tqdm

# --- Configuration ---
# The directory containing all your raw .npz files
DATA_DIR = "../craftax_classic_200M_dataset" 

# Set to process only the 50 most recent files
NUM_FILES_TO_PROCESS = 80


# The name of your final, subset dataset file
H5_OUTPUT_PATH = f"episodes_mixed_{NUM_FILES_TO_PROCESS}_files.h5"

def reconstruct_episodes_from_subset(file_paths):
    """Loads a subset of .npz files and reconstructs episodes."""
    print(f"Processing a subset of {len(file_paths)} raw trajectory files.")
    
    with np.load(file_paths[0], allow_pickle=True) as data:
        all_keys = [key for key in data.keys() if key != 'info']
    
    all_data = {key: [] for key in all_keys}

    for path in tqdm(file_paths, desc="Loading raw batches into RAM"):
        with np.load(path, allow_pickle=True) as data:
            for key in all_keys:
                all_data[key].append(data[key])

    print("Concatenating data from subset...")
    for key in tqdm(all_keys, desc="Concatenating arrays"):
        all_data[key] = np.concatenate(all_data[key], axis=0)

    print("Reconstructing episodes from subset...")
    episodes_dict = {}
    unique_ids = np.unique(all_data['episode_id'])

    for episode_id in tqdm(unique_ids, desc="Reconstructing episodes"):
        mask = (all_data['episode_id'] == episode_id)
        steps_in_episode = all_data['step_in_episode'][mask]
        sort_indices = np.argsort(steps_in_episode)
        
        episode_trajectory = {}
        for key in all_keys:
            episode_trajectory[key] = all_data[key][mask][sort_indices]
        
        episodes_dict[episode_id] = episode_trajectory

    return episodes_dict

def filter_complete_episodes(episodes_dict):
    """
    Filters a dictionary of episodes, keeping only those that both
    start at step 0 AND contain a 'done=True' flag.
    """
    complete_episodes = {}
    print("\nFiltering for complete episodes (start at step 0 AND contain a done=True)...")
    for episode_id, episode_data in episodes_dict.items():
        steps = episode_data.get('step_in_episode')
        dones = episode_data.get('done')
        
        if steps is not None and dones is not None and len(steps) > 0 and steps[0] == 0 and np.any(dones):
            complete_episodes[episode_id] = episode_data
            
    return complete_episodes

def save_episodes_to_hdf5(episodes_dict, output_path):
    """Saves the final dictionary of episodes to a compressed HDF5 file."""
    print(f"Saving {len(episodes_dict)} complete episodes to '{output_path}'...")
    keys_to_compress = {'obs', 'next_obs'}
    with h5py.File(output_path, 'w') as hf:
        for episode_id, episode_data in tqdm(episodes_dict.items(), desc="Saving to HDF5"):
            episode_group = hf.create_group(f"episode_{episode_id}")
            for key, data in episode_data.items():
                if key in keys_to_compress:
                    episode_group.create_dataset(
                        key, data=data, compression="gzip", compression_opts=4
                    )
                else:
                    episode_group.create_dataset(key, data=data)

if __name__ == "__main__":
    # 1. Select the last N files
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    #subset_files = all_files[-NUM_FILES_TO_PROCESS:]
    total_files = len(all_files)
    print(f"Selecting {CHUNK_SIZE} files from the beginning, middle, and end...")
     
    # 2. Reconstruct episodes from this subset
    reconstructed_subset = reconstruct_episodes_from_subset(subset_files)
    print(f"\nFound {len(reconstructed_subset)} unique episode fragments in the subset.")

    # 3. Filter out incomplete episodes
    complete_episodes_subset = filter_complete_episodes(reconstructed_subset)
    
    # 4. Save the final, high-quality subset
    save_episodes_to_hdf5(complete_episodes_subset, H5_OUTPUT_PATH)

    print(f"\n✅ Success! Your subset dataset with {len(complete_episodes_subset)} complete episodes is ready at '{H5_OUTPUT_PATH}'.")

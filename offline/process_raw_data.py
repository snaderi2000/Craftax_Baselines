import os
import glob
import numpy as np
import h5py
from tqdm import tqdm

def reconstruct_episodes_from_npz(data_directory):
    """
    Loads all raw .npz files from a directory, concatenates them, and
    structures the data into a dictionary of complete, ordered episodes.
    """
    # Find all batch files and sort them to process in order
    file_paths = sorted(glob.glob(os.path.join(data_directory, "*.npz")))
    if not file_paths:
        raise FileNotFoundError(f"Error: No .npz files found in '{data_directory}'")

    print(f"Found {len(file_paths)} raw trajectory files to process.")

    # Load the first file to get the data keys
    with np.load(file_paths[0], allow_pickle=True) as data:
        all_keys = [key for key in data.keys() if key != 'info'] # Exclude non-array 'info'
    
    # Initialize a dictionary of lists to hold all concatenated data
    all_data = {key: [] for key in all_keys}

    # Load data from each file and append to the lists
    for path in tqdm(file_paths, desc="Phase 1: Loading raw batches"):
        try:
            with np.load(path, allow_pickle=True) as data:
                for key in all_keys:
                    all_data[key].append(data[key])
        except Exception as e:
            print(f"Warning: Could not load file {path}. Error: {e}")
            continue

    # Concatenate all lists of arrays into single large arrays
    print("Concatenating all data...")
    for key in tqdm(all_keys, desc="Concatenating arrays"):
        all_data[key] = np.concatenate(all_data[key], axis=0)

    # Group transitions into a dictionary of complete episodes
    print("Reconstructing episodes...")
    episodes_dict = {}
    unique_ids = np.unique(all_data['episode_id'])

    for episode_id in tqdm(unique_ids, desc="Processing episodes"):
        mask = (all_data['episode_id'] == episode_id)
        steps_in_episode = all_data['step_in_episode'][mask]
        sort_indices = np.argsort(steps_in_episode)
        
        episode_trajectory = {}
        for key in all_keys:
            episode_trajectory[key] = all_data[key][mask][sort_indices]
        
        episodes_dict[episode_id] = episode_trajectory

    return episodes_dict

def save_episodes_to_compressed_hdf5(episodes_dict, output_path):
    """
    Saves a dictionary of episodes to a single HDF5 file, with selective compression.
    """
    print(f"Creating compressed HDF5 file at '{output_path}'...")
    keys_to_compress = {'obs', 'next_obs'}

    with h5py.File(output_path, 'w') as hf:
        for episode_id, episode_data in tqdm(episodes_dict.items(), desc="Phase 2: Saving to HDF5"):
            episode_group = hf.create_group(f"episode_{episode_id}")
            
            for key, data in episode_data.items():
                if key in keys_to_compress:
                    episode_group.create_dataset(
                        key, data=data, compression="gzip", compression_opts=4
                    )
                else:
                    episode_group.create_dataset(key, data=data)

if __name__ == "__main__":
    # --- Configuration ---
    # The directory containing your raw .npz files
    DATA_DIR = "../craftax_classic_1M_dataset" 
    # The name of your final, processed dataset file
    H5_OUTPUT_PATH = "episodes_compressed.h5"

    # --- Run the Pipeline ---
    print("Starting the full data processing pipeline...")
    
    # PHASE 1: Reconstruct episodes from all .npz files into memory
    reconstructed_episodes = reconstruct_episodes_from_npz(DATA_DIR)
    
    # PHASE 2: Save the reconstructed episodes to a compressed HDF5 file
    save_episodes_to_compressed_hdf5(reconstructed_episodes, H5_OUTPUT_PATH)

    print(f"\n✅ Success! Your final dataset is ready at '{H5_OUTPUT_PATH}'.")
    print("You can now use this file to load data for training your agent.")
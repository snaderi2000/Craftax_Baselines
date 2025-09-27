import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# --- Configuration ---
DATA_DIR = "./craftax_classic_1M_dataset"
H5_OUTPUT_PATH = "episodes_compressed_parr.h5"
TEMP_DIR = "./temp_episode_chunks" # Directory to store intermediate files
NUM_WORKERS = 100 # Match this to the number of CPU cores you have (e.g., --cpus-per-task=50)

def process_chunk(file_chunk, temp_output_path):
    """Worker function: processes a chunk of .npz files and saves a temporary dict."""
    all_data = {}
    try:
        # Load the first file to get keys
        with np.load(file_chunk[0], allow_pickle=True) as data:
            all_keys = [key for key in data.keys() if key != 'info']
        
        all_data = {key: [] for key in all_keys}

        # Load all files in the chunk
        for path in file_chunk:
            with np.load(path, allow_pickle=True) as data:
                for key in all_keys:
                    all_data[key].append(data[key])
        
        # Concatenate and reconstruct episodes for this chunk
        for key in all_keys:
            all_data[key] = np.concatenate(all_data[key], axis=0)
            
        episodes_dict = {}
        unique_ids = np.unique(all_data['episode_id'])
        for episode_id in unique_ids:
            mask = (all_data['episode_id'] == episode_id)
            steps_in_episode = all_data['step_in_episode'][mask]
            sort_indices = np.argsort(steps_in_episode)
            
            episode_trajectory = {key: all_data[key][mask][sort_indices] for key in all_keys}
            episodes_dict[episode_id] = episode_trajectory
            
        # Save the result of this chunk to a temporary file
        with open(temp_output_path, 'wb') as f:
            pickle.dump(episodes_dict, f)
            
        return temp_output_path
    except Exception as e:
        return f"Error processing chunk for {temp_output_path}: {e}"

def merge_chunks(temp_files):
    """Merges all temporary dictionaries into a single master dictionary."""
    master_dict = {}
    for temp_file in tqdm(temp_files, desc="Merging chunks"):
        with open(temp_file, 'rb') as f:
            chunk_dict = pickle.load(f)
            for episode_id, episode_data in chunk_dict.items():
                if episode_id not in master_dict:
                    master_dict[episode_id] = {key: [] for key in episode_data.keys()}
                for key, data in episode_data.items():
                    master_dict[episode_id][key].append(data)
    
    # Final concatenation and sorting within each merged episode
    print("Finalizing merged episodes...")
    final_episodes = {}
    for episode_id, data_lists in tqdm(master_dict.items(), desc="Finalizing episodes"):
        final_trajectory = {}
        for key, list_of_arrays in data_lists.items():
            final_trajectory[key] = np.concatenate(list_of_arrays, axis=0)
            
        # Final sort
        steps = final_trajectory['step_in_episode']
        sort_indices = np.argsort(steps)
        for key in final_trajectory.keys():
            final_trajectory[key] = final_trajectory[key][sort_indices]
            
        final_episodes[episode_id] = final_trajectory
        
    return final_episodes

def filter_complete_episodes(episodes_dict):
    """Filters for episodes that start at step 0 and contain a done flag."""
    complete_episodes = {}
    print("\nFiltering for complete episodes...")
    for episode_id, episode_data in episodes_dict.items():
        if 'step_in_episode' in episode_data and 'done' in episode_data:
            if episode_data['step_in_episode'][0] == 0 and np.any(episode_data['done']):
                complete_episodes[episode_id] = episode_data
    return complete_episodes

def save_episodes_to_hdf5(episodes_dict, output_path):
    """Saves the final dictionary to a compressed HDF5 file."""
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
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 1. SPLIT files into chunks
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    chunk_size = math.ceil(len(all_files) / NUM_WORKERS)
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    
    temp_files = []
    
    # 2. PROCESS chunks in parallel
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_chunk, chunk, f"{TEMP_DIR}/chunk_{i}.pkl"): i for i, chunk in enumerate(file_chunks)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Chunks"):
            result = future.result()
            if "Error" in str(result):
                print(result)
            else:
                temp_files.append(result)

    # 3. MERGE the results from all workers
    final_reconstructed_episodes = merge_chunks(temp_files)
    
    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
    os.rmdir(TEMP_DIR)
    
    # 4. FINALIZE: Filter and save the complete dataset
    complete_episodes = filter_complete_episodes(final_reconstructed_episodes)
    save_episodes_to_hdf5(complete_episodes, H5_OUTPUT_PATH)

    print(f"\n✅ Success! Parallel processing complete. Final dataset with {len(complete_episodes)} episodes is ready at '{H5_OUTPUT_PATH}'.")
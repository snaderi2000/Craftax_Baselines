import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
import math

# --- Configuration ---
DATA_DIR = "../craftax_classic_1M_dataset"
H5_OUTPUT_PATH = "episodes_compressed_chunk.h5"
# The number of .npz files to load and process in each batch
CHUNK_SIZE = 10 
# ---

def process_data_in_chunks(data_directory, output_path, chunk_size):
    """
    Processes raw .npz files in chunks to build a structured, compressed
    HDF5 file, balancing speed and memory safety.
    """
    all_files = sorted(glob.glob(os.path.join(data_directory, "*.npz")))
    if not all_files:
        raise FileNotFoundError(f"Error: No .npz files found in '{data_directory}'")

    # Split the list of all files into smaller chunks
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    print(f"Found {len(all_files)} files. Split into {len(file_chunks)} chunks of size {chunk_size}.")

    keys_to_compress = {'obs', 'next_obs'}
    
    # Open the HDF5 file once for writing
    with h5py.File(output_path, 'w') as hf:
        # Process one chunk of files at a time
        for file_chunk in tqdm(file_chunks, desc="Processing Chunks"):
            # Temporarily hold all data from the current chunk in memory
            chunk_data = {}
            with np.load(file_chunk[0], allow_pickle=True) as data:
                 all_keys = [key for key in data.keys() if key != 'info']
            
            chunk_data = {key: [] for key in all_keys}

            for path in file_chunk:
                with np.load(path, allow_pickle=True) as data:
                    for key in all_keys:
                        chunk_data[key].append(data[key])
            
            for key in all_keys:
                chunk_data[key] = np.concatenate(chunk_data[key], axis=0)

            # Reconstruct and append episodes from this chunk's data
            unique_ids_in_chunk = np.unique(chunk_data['episode_id'])
            for episode_id in unique_ids_in_chunk:
                mask = (chunk_data['episode_id'] == episode_id)
                episode_group = hf.require_group(f"episode_{episode_id}")
                
                for key in all_keys:
                    new_data = chunk_data[key][mask]
                    if key not in episode_group:
                        max_shape = (None,) + new_data.shape[1:]
                        chunks = (128,) + new_data.shape[1:] if len(new_data.shape) > 1 else (128,)
                        if key in keys_to_compress:
                            episode_group.create_dataset(
                                key, data=new_data, maxshape=max_shape, chunks=chunks,
                                compression="gzip", compression_opts=4
                            )
                        else:
                            episode_group.create_dataset(
                                key, data=new_data, maxshape=max_shape, chunks=chunks
                            )
                    else:
                        dataset = episode_group[key]
                        old_size = dataset.shape[0]
                        dataset.resize(old_size + len(new_data), axis=0)
                        dataset[old_size:] = new_data

    # Final sorting pass remains the same
    print("\nSorting data within each episode in the HDF5 file...")
    with h5py.File(output_path, 'a') as hf:
        for episode_key in tqdm(hf.keys(), desc="Sorting episodes"):
            episode_group = hf[episode_key]
            if 'step_in_episode' not in episode_group: continue
            
            steps = episode_group['step_in_episode'][:]
            sort_indices = np.argsort(steps)
            
            if not np.array_equal(steps, np.arange(len(steps))):
                for key in episode_group.keys():
                    sorted_data = episode_group[key][:][sort_indices]
                    episode_group[key][...] = sorted_data

if __name__ == "__main__":
    process_data_in_chunks(DATA_DIR, H5_OUTPUT_PATH, CHUNK_SIZE)
    print(f"\n✅ Success! Your pipeline is complete. Final dataset is at '{H5_OUTPUT_PATH}'.")
import os
import glob
import numpy as np
import h5py
from tqdm import tqdm
import pickle

# --- Configuration ---
DATA_DIR = "../craftax_classic_1M_dataset"
H5_OUTPUT_PATH = "1M_combined_dataset_compressed.h5"

COMPRESS_KEYS = {'obs', 'next_obs'}

def save_dataset_safely(group, key, array):
    """
    Save array to HDF5, handling object dtype, scalar arrays,
    and compression for certain keys.
    """
    # Ensure scalars become 1D arrays so we can iterate
    array = np.atleast_1d(array)

    if array.dtype == object:
        # Try string detection safely
        try:
            if np.all([isinstance(x, str) for x in array]):
                str_dtype = h5py.string_dtype(encoding='utf-8')
                group.create_dataset(key, data=array.astype(str), dtype=str_dtype)
            else:
                # Fallback: store as pickled bytes for generic objects
                pickled = np.array([np.void(pickle.dumps(x)) for x in array])
                group.create_dataset(key, data=pickled)
        except Exception as e:
            print(f"Warning: Could not process key '{key}' as object: {e}")
            pickled = np.array([np.void(pickle.dumps(x)) for x in array])
            group.create_dataset(key, data=pickled)
    else:
        # Numeric array path
        if key in COMPRESS_KEYS:
            group.create_dataset(
                key, data=array, compression="gzip", compression_opts=4
            )
        else:
            group.create_dataset(key, data=array)

def combine_npz_to_h5(npz_files, output_path):
    print(f"Found {len(npz_files)} .npz files.")
    with h5py.File(output_path, 'w') as hf:
        for i, file_path in enumerate(tqdm(npz_files, desc="Processing .npz files")):
            group_name = f"file_{i:04d}"
            group = hf.create_group(group_name)

            with np.load(file_path, allow_pickle=True) as data:
                for key in data.files:
                    save_dataset_safely(group, key, data[key])

    print(f"\n✅ Done! Combined HDF5 saved at: {output_path}")

if __name__ == "__main__":
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {DATA_DIR}")

    combine_npz_to_h5(npz_files, H5_OUTPUT_PATH)


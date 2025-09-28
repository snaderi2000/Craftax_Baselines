import os
import glob
import numpy as np
import duckdb
import zlib
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
DATA_DIR = "../craftax_classic_1M_dataset"  # Directory with raw .npz files
DB_PATH = "transitions.duckdb"                # Output DuckDB database
COMPRESSION_LEVEL = 3                         # zlib compression level (1-9)

# ==============================
# DUCKDB SETUP
# ==============================
def init_duckdb(db_path):
    con = duckdb.connect(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS transitions (
        episode_id BIGINT,
        step_in_episode BIGINT,
        action INT,
        reward FLOAT,
        done BOOLEAN,
        obs BLOB,
        next_obs BLOB
    );
    """)
    return con

# ==============================
# UTILS
# ==============================
def compress_array(arr):
    """Compress a NumPy array into bytes using zlib."""
    return zlib.compress(arr.astype(np.float32).tobytes(), level=COMPRESSION_LEVEL)

# ==============================
# PROCESS SINGLE SHARD
# ==============================
def process_npz_file(con, file_path):
    """Load a single .npz shard and insert its rows into DuckDB."""
    data = np.load(file_path, allow_pickle=True)

    episode_ids = data["episode_id"]
    step_ids = data["step_in_episode"]
    actions = data["action"]
    rewards = data["reward"]
    dones = data["done"]
    obs = data["obs"]
    next_obs = data["next_obs"]

    # Prepare rows for batch insert
    rows = [
        (
            int(episode_ids[i]),
            int(step_ids[i]),
            int(actions[i]),
            float(rewards[i]),
            bool(dones[i]),
            compress_array(obs[i]),
            compress_array(next_obs[i]),
        )
        for i in range(len(episode_ids))
    ]

    con.executemany("""
        INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)

# ==============================
# MAIN LOOP
# ==============================
def main():
    # Initialize DuckDB connection
    con = init_duckdb(DB_PATH)

    # Get all .npz files
    npz_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"Found {len(npz_files)} .npz files to process.")

    # Process each shard one at a time
    for file_path in tqdm(npz_files, desc="Inserting shards"):
        process_npz_file(con, file_path)
        con.commit()  # commit after each file to avoid losing progress
        # Optional: delete numpy arrays to free memory
        del file_path

    print(f"\n✅ Done! All transitions inserted into DuckDB at '{DB_PATH}'")

if __name__ == "__main__":
    main()


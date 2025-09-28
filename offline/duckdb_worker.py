import os
import argparse
import glob
import numpy as np
import duckdb
from tqdm import tqdm

import zlib
import numpy as np

def compress_array(arr: np.ndarray) -> bytes:
    """Compress a numpy array to bytes."""
    return zlib.compress(arr.tobytes())

def decompress_array(blob: bytes, dtype=np.float32, shape=None) -> np.ndarray:
    """Decompress a numpy array back from bytes."""
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def process_worker(input_dir, output_db, shard_index, total_shards, chunk_size=2000):
    """
    Each worker:
    - Selects files based on shard_index
    - Loads each .npz file
    - Inserts its data into a DuckDB database
    """
    print(f"[Worker {shard_index}] 🚀 Starting...")

    # Step 1. Get list of files and shard them
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    files_for_this_worker = [f for i, f in enumerate(all_files) if i % total_shards == shard_index]

    print(f"[Worker {shard_index}] Found {len(files_for_this_worker)} files to process.")
    if len(files_for_this_worker) == 0:
        print(f"[Worker {shard_index}] ❌ No files assigned. Exiting.")
        return

    # Step 2. Connect to DuckDB
    con = duckdb.connect(output_db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS transitions (
            episode_id BIGINT,
            step_in_episode INT,
            obs BLOB,
            next_obs BLOB,
            action INT,
            reward FLOAT
        )
    """)
    print(f"[Worker {shard_index}] Connected to database: {output_db}")

    # Step 3. Process files one at a time
    for file_path in files_for_this_worker:
        print(f"[Worker {shard_index}] Loading file: {file_path}")
        try:
            with np.load(file_path, allow_pickle=True) as data:
                # Validate required keys
                required_keys = ["obs", "next_obs", "action", "reward", "episode_id", "step_in_episode"]
                for key in required_keys:
                    if key not in data:
                        raise ValueError(f"Missing key '{key}' in {file_path}")

                # Load arrays
                obs = data["obs"].astype(np.float32)
                next_obs = data["next_obs"].astype(np.float32)
                actions = data["action"].astype(np.int32)
                rewards = data["reward"].astype(np.float32)
                episode_ids = data["episode_id"].astype(np.int64)
                steps = data["step_in_episode"].astype(np.int32)

                total_rows = obs.shape[0]
                print(f"[Worker {shard_index}] Shapes: obs={obs.shape}, next_obs={next_obs.shape}, total_rows={total_rows}")

                # Step 4. Insert into DuckDB using chunked batches
                print(f"[Worker {shard_index}] Starting batch insert of {total_rows} rows...")

                con.execute("BEGIN TRANSACTION;")  # wrap all chunks in one transaction

                for start in range(0, total_rows, chunk_size):
                    end = min(start + chunk_size, total_rows)

                    batch_data = [
                        (
                            int(episode_ids[i]),
                            int(steps[i]),
                            compress_array(obs[i]),          # compress to BLOB
                            compress_array(next_obs[i]),
                            int(actions[i]),
                            float(rewards[i]),
                        )
                        for i in range(start, end)
                    ]

                    con.executemany("""
                        INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?)
                    """, batch_data)

                    if start % (chunk_size * 10) == 0:
                        print(f"[Worker {shard_index}] Inserted {end}/{total_rows} rows from {os.path.basename(file_path)}")

                con.execute("COMMIT;")
                print(f"[Worker {shard_index}] ✅ Finished inserting {total_rows} rows from {file_path}")

        except Exception as e:
            print(f"[Worker {shard_index}] ❗ ERROR processing {file_path}: {e}")
            continue

    print(f"[Worker {shard_index}] 🎉 All files processed successfully.")
    con.close()

# ------------------------------
# CLI Entrypoint
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckDB Worker for NPZ ingestion")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .npz files")
    parser.add_argument("--output_db", type=str, required=True, help="Output DuckDB database file for this worker")
    parser.add_argument("--shard_index", type=int, required=True, help="Index of this worker [0..total_shards-1]")
    parser.add_argument("--total_shards", type=int, required=True, help="Total number of workers/shards")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Number of rows per insert batch (default=2000)")

    args = parser.parse_args()

    process_worker(
        input_dir=args.input_dir,
        output_db=args.output_db,
        shard_index=args.shard_index,
        total_shards=args.total_shards,
        chunk_size=args.chunk_size,
    )


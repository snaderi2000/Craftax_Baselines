import os
import argparse
import glob
import numpy as np
import duckdb
from tqdm import tqdm

def process_worker(input_dir, output_db, shard_index, total_shards):
    """
    Each worker:
    - Selects files based on shard_index
    - Loads each .npz file
    - Inserts its data into a DuckDB database
    """
    print(f"[Worker {shard_index}] Starting...")

    # Step 1. Get list of files and shard them
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    files_for_this_worker = [
        f for i, f in enumerate(all_files) if i % total_shards == shard_index
    ]

    print(f"[Worker {shard_index}] Found {len(files_for_this_worker)} files to process.")
    if len(files_for_this_worker) == 0:
        print(f"[Worker {shard_index}] No files assigned. Exiting.")
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

    # Step 3. Process files
    for file_path in tqdm(files_for_this_worker, desc=f"[Worker {shard_index}] Ingesting"):
        print(f"[Worker {shard_index}] Loading file: {file_path}")
        try:
            with np.load(file_path, allow_pickle=True) as data:
                keys = list(data.keys())
                print(f"[Worker {shard_index}] Loaded keys: {keys}")

                # Validate required keys
                required_keys = ["obs", "next_obs", "action", "reward", "episode_id", "step_in_episode"]
                for key in required_keys:
                    if key not in keys:
                        raise ValueError(f"Missing key '{key}' in {file_path}")

                obs = data["obs"].astype(np.float32)
                next_obs = data["next_obs"].astype(np.float32)
                actions = data["action"].astype(np.int32)
                rewards = data["reward"].astype(np.float32)
                episode_ids = data["episode_id"].astype(np.int64)
                steps = data["step_in_episode"].astype(np.int32)

                print(f"[Worker {shard_index}] Shapes: obs={obs.shape}, next_obs={next_obs.shape}, actions={actions.shape}, rewards={rewards.shape}")
                total_rows = obs.shape[0]
                print(f"[Worker {shard_index}] Total transitions in file: {total_rows}")

                # Step 4. Insert a single test row
                print(f"[Worker {shard_index}] Inserting first row as test...")
                con.execute("""
                    INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    int(episode_ids[0]),
                    int(steps[0]),
                    obs[0].tobytes(),
                    next_obs[0].tobytes(),
                    int(actions[0]),
                    float(rewards[0]),
                ))
                print(f"[Worker {shard_index}] First row insert successful!")

                # Step 5. Batch insert the rest
                print(f"[Worker {shard_index}] Preparing batch insert of remaining {total_rows-1} rows...")
                batch_data = [
                    (
                        int(episode_ids[i]),
                        int(steps[i]),
                        obs[i].tobytes(),
                        next_obs[i].tobytes(),
                        int(actions[i]),
                        float(rewards[i]),
                    )
                    for i in range(total_rows)
                ]
                con.executemany("""
                    INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?)
                """, batch_data)
                print(f"[Worker {shard_index}] Finished inserting {total_rows} rows from {file_path}")

        except Exception as e:
            print(f"[Worker {shard_index}] ERROR processing {file_path}: {e}")
            continue

    print(f"[Worker {shard_index}] ✅ All files processed successfully.")
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

    args = parser.parse_args()

    process_worker(
        input_dir=args.input_dir,
        output_db=args.output_db,
        shard_index=args.shard_index,
        total_shards=args.total_shards,
    )


import duckdb
import glob
import os

WORKER_DIR = "./worker_dbs"      # Folder where worker_*.duckdb files are stored
OUTPUT_DB = "merged.duckdb"      # Final merged database

def merge_worker_dbs():
    # Step 1. Collect all worker database paths
    worker_files = sorted(glob.glob(os.path.join(WORKER_DIR, "worker_*.duckdb")))
    print(f"Found {len(worker_files)} worker databases to merge.")

    if len(worker_files) == 0:
        raise RuntimeError("No worker databases found!")

    # Step 2. Connect to a new database for merging
    con = duckdb.connect(OUTPUT_DB)
    print(f"Created new merged database: {OUTPUT_DB}")

    # Step 3. Create table schema if not already present
    con.execute("""
        CREATE TABLE IF NOT EXISTS transitions (
            episode_id BIGINT,
            step_in_episode INT,
            obs BLOB,
            next_obs BLOB,
            action INT,
            reward FLOAT
        );
    """)
    print("Main table `transitions` created in merged database.")

    # Step 4. Loop over worker DBs and merge their data
    for db_path in worker_files:
        print(f"Merging {db_path} → {OUTPUT_DB}")
        con.execute(f"""
            ATTACH '{db_path}' AS worker_db;
            INSERT INTO main.transitions
            SELECT * FROM worker_db.transitions;
            DETACH worker_db;
        """)

    print(f"✅ Successfully merged {len(worker_files)} databases into {OUTPUT_DB}")
    con.close()

if __name__ == "__main__":
    merge_worker_dbs()


import duckdb
import numpy as np
import zlib
import d3rlpy
import random
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DB_PATH = "merged.duckdb"
OUTPUT_H5 = "latest_1024_episodes.h5"
OBS_SHAPE = (1345,)  # Shape of a single observation

# -----------------------------
# Helper: Decompress function
# -----------------------------
def decompress_array(blob, dtype=np.float32, shape=OBS_SHAPE):
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)

# -----------------------------
# Step 1. Connect to DB
# -----------------------------
con = duckdb.connect(DB_PATH)

# -----------------------------
# Step 2. Get the 2 most recent episodes per environment
# -----------------------------
# This query ranks episodes within each environment and selects the top 10.
# An "environment" is defined by integer dividing episode_id by 1,000,000.
# "Recency" is defined by the remainder, ordered descending.
latest_episodes_df = con.execute("""
    WITH ranked_episodes AS (
        SELECT
            episode_id,
            ROW_NUMBER() OVER (
                PARTITION BY (episode_id // 1000000)
                ORDER BY (episode_id % 1000000) DESC
            ) AS rank
        FROM (
            SELECT DISTINCT episode_id FROM transitions
        )
    )
    SELECT
        episode_id
    FROM ranked_episodes
    WHERE rank <= 1
    ORDER BY episode_id;
""").fetchdf()

# -----------------------------
# Step 3. Prepare episode IDs for the next query
# -----------------------------
final_episode_ids = latest_episodes_df['episode_id'].tolist()

print(f"Selected {len(final_episode_ids)} episodes to fetch.")

# -----------------------------
# Step 4. Fetch all transitions for final episodes
# -----------------------------
# This part of your code is correct and can be used as is.
placeholders = ",".join(map(str, final_episode_ids))
query = f"""
    SELECT episode_id, step_in_episode, obs, action, reward
    FROM transitions
    WHERE episode_id IN ({placeholders})
    ORDER BY episode_id ASC, step_in_episode ASC
"""
df = con.execute(query).fetchdf()
print(f"Fetched {len(df)} total transitions for the selected episodes.")

# -----------------------------
# Step 5. Build arrays for MDPDataset
# -----------------------------
observations, actions, rewards, terminals = [], [], [], []

print("Processing transitions into MDPDataset format...")
current_episode = None

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Decompress observation
    obs = decompress_array(row['obs'])
    observations.append(obs)
    actions.append(np.array([row['action']], dtype=np.float32))
    rewards.append(np.array([row['reward']], dtype=np.float32))

    # Detect if this is the last step of the current episode
    is_last_step = (
        idx == len(df)-1 or df.iloc[idx + 1]['episode_id'] != row['episode_id']
    )
    terminals.append(np.array([1.0 if is_last_step else 0.0], dtype=np.float32))

# Convert to NumPy arrays
observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)
rewards = np.array(rewards, dtype=np.float32)
terminals = np.array(terminals, dtype=np.float32)

print(f"Final shapes -> Observations: {observations.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}")

# -----------------------------
# Step 6. Save as MDPDataset
# -----------------------------
dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)

with open(OUTPUT_H5, "w+b") as f:
    dataset.dump(f)

print(f"✅ Saved latest episodes to {OUTPUT_H5}")


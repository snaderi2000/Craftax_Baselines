import duckdb
import numpy as np
import zlib
import d3rlpy
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DB_PATH = "merged.duckdb"
OUTPUT_H5 = "top_12000_episodes.h5"
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
# Step 2. Find top 1000 episodes by cumulative reward
# -----------------------------
top_episodes_df = con.execute("""
    SELECT episode_id, SUM(reward) as total_reward
    FROM transitions
    GROUP BY episode_id
    ORDER BY total_reward DESC
    LIMIT 12000
""").fetchdf()

top_episode_ids = top_episodes_df['episode_id'].tolist()
print(f"Top 1000 episodes selected. Highest reward: {top_episodes_df.iloc[0]['total_reward']}")

# -----------------------------
# Step 3. Extract all transitions for those episodes
# -----------------------------
placeholders = ",".join(map(str, top_episode_ids))
query = f"""
    SELECT episode_id, step_in_episode, obs, action, reward
    FROM transitions
    WHERE episode_id IN ({placeholders})
    ORDER BY episode_id ASC, step_in_episode ASC
"""
df = con.execute(query).fetchdf()
print(f"Fetched {len(df)} total transitions.")

# -----------------------------
# Step 4. Build arrays for MDPDataset
# -----------------------------
observations = []
actions = []
rewards = []
terminals = []

print("Processing transitions into MDPDataset format...")
current_episode = None
episode_indices = 0

for idx, row in tqdm(df.iterrows(), total=len(df)):
    obs = decompress_array(row['obs'])                # Decompress obs
    observations.append(obs)
    actions.append(np.array([row['action']], dtype=np.float32))
    rewards.append(np.array([row['reward']], dtype=np.float32))

    # Detect if this is the last step of an episode
    if current_episode != row['episode_id']:
        # Starting a new episode
        current_episode = row['episode_id']

    # Look ahead to see if the next row is a different episode
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
# Step 5. Save as MDPDataset
# -----------------------------
dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)

with open(OUTPUT_H5, "w+b") as f:
    dataset.dump(f)

print(f"✅ Saved top 1000 episodes to {OUTPUT_H5}")


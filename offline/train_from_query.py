import duckdb
import numpy as np
import zlib
import d3rlpy
from tqdm import tqdm
import pandas as pd

# ===================================================================
# 1. EXPERIMENT CONFIGURATION
# ===================================================================
# --- Data Config ---
DB_PATH = "merged.duckdb"
N_EPISODES = 200000000
OBS_SHAPE = (1345,)

# --- Training Config ---
WANDB_PROJECT = "Craftax-OfflineRL"
WANDB_RUN_NAME = f"cql-run-{N_EPISODES}-episodes-direct"
MODEL_SAVE_PATH = f"cql_model_{N_EPISODES}_episodes.d3"
TOTAL_STEPS = 750_000
EPOCH_STEPS = 150_000
BATCH_SIZE = 256

# ===================================================================
# 2. HELPER FUNCTION
# ===================================================================
def decompress_array(blob, dtype=np.float32, shape=OBS_SHAPE):
    """Decompresses a zlib-compressed binary blob into a NumPy array."""
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)

# ===================================================================
# 3. LOAD DATA FROM DATABASE
# ===================================================================
print("Connecting to database...")
con = duckdb.connect(DB_PATH, read_only=True)

print(f"Finding top {N_EPISODES} episodes by cumulative reward...")
# top_episodes_df = con.execute(f"""
#     SELECT episode_id, SUM(reward) as total_reward
#     FROM transitions
#     GROUP BY episode_id
#     ORDER BY total_reward DESC
#     LIMIT {N_EPISODES}
# """).fetchdf()


top_episodes_df = con.execute(f"""
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
    WHERE rank <= {N_EPISODES}
    ORDER BY episode_id;
""").fetchdf()


top_episode_ids = top_episodes_df['episode_id'].tolist()
#print(f"Top episodes selected. Highest reward: {top_episodes_df.iloc[0]['total_reward']:.2f}")

print("Fetching all transitions for selected episodes...")
placeholders = ",".join(map(str, top_episode_ids))
query = f"""
    SELECT episode_id, obs, action, reward
    FROM transitions
    WHERE episode_id IN ({placeholders})
    ORDER BY episode_id ASC, step_in_episode ASC
"""
df = con.execute(query).fetchdf()
print(f"Fetched {len(df)} total transitions from {len(top_episode_ids)} episodes.")
con.close()

# ===================================================================
# 4. BUILD REPLAY BUFFER DIRECTLY IN MEMORY
# ===================================================================
print("Processing transitions into d3rlpy Episodes...")
episodes = []

# Group the DataFrame by episode_id to process one episode at a time
for episode_id, episode_df in tqdm(df.groupby('episode_id'), desc="Building Episodes"):
    # Decompress observations for the entire episode at once
    observations = np.array(episode_df['obs'].apply(decompress_array).tolist(), dtype=np.float32)
    
    # Extract actions and rewards
    actions = episode_df['action'].to_numpy(dtype=np.float32).reshape(-1, 1)
    rewards = episode_df['reward'].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Create a d3rlpy Episode object. Since these are complete episodes from
    # a dataset, the `terminated` flag should be True.
    episode = d3rlpy.dataset.Episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=True
    )
    episodes.append(episode)

print(f"Created {len(episodes)} Episode objects.")

# Now, create the ReplayBuffer directly from the list of episodes
replay_buffer = d3rlpy.dataset.ReplayBuffer(
    buffer=d3rlpy.dataset.InfiniteBuffer(), # Use an infinite buffer
    episodes=episodes
)
print(f"Replay buffer created with {replay_buffer.transition_count} transitions.")

# ===================================================================
# 5. CONFIGURE AND TRAIN THE CQL ALGORITHM
# ===================================================================
print("Configuring Discrete CQL agent...")
cql = d3rlpy.algos.DiscreteCQLConfig(batch_size=BATCH_SIZE).create(device="cuda:0")

# Set up logging and evaluation
wandb_logger = d3rlpy.logging.WanDBAdapterFactory(project=WANDB_PROJECT)
td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator()
initial_state_value_evaluator = d3rlpy.metrics.InitialStateValueEstimationEvaluator()

print("🚀 Starting training...")
cql.fit(
    replay_buffer,
    n_steps=TOTAL_STEPS,
    n_steps_per_epoch=EPOCH_STEPS,
    evaluators={
        "td_error": td_error_evaluator,
        "initial_state_value": initial_state_value_evaluator,
    },
    experiment_name=WANDB_RUN_NAME,
    with_timestamp=True,
    show_progress=True,
    logger_adapter=wandb_logger
)

# ===================================================================
# 6. SAVE FINAL MODEL
# ===================================================================
cql.save(MODEL_SAVE_PATH)
print(f"\n🎉 Training complete! Final model saved to {MODEL_SAVE_PATH}")

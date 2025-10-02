import duckdb
import numpy as np
import zlib
import d3rlpy
from tqdm import tqdm
import pandas as pd
import argparse

# ===================================================================
# 1. EXPERIMENT CONFIGURATION
# ===================================================================
DB_PATH = "merged.duckdb"
OBS_SHAPE = (1345,)
WANDB_PROJECT = "Craftax-OfflineRL-Final-Comparison"
MODEL_SAVE_PATH_PREFIX = "cql_model"
BATCH_SIZE = 256
TOTAL_EPISODES_TO_SAMPLE = 10000

# ===================================================================
# 2. HELPER FUNCTION (Unchanged)
# ===================================================================
def decompress_array(blob, dtype=np.float32, shape=OBS_SHAPE):
    """Decompresses a zlib-compressed binary blob into a NumPy array."""
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)

# ===================================================================
# 3. MAIN SCRIPT LOGIC
# ===================================================================
def main(args):
    print(f"--- Running Experiment (Strategy: {args.strategy}) ---")
    
    con = duckdb.connect(DB_PATH, read_only=True)

    print("Fetching and ranking all episode rewards to create a fixed pool...")
    # This query selects a fixed, pseudo-random sample of 10,000 episodes
    # and also ranks them by reward percentile for the curriculum strategy.
    episode_pool_query = f"""
        WITH all_episode_rewards AS (
            SELECT episode_id, SUM(reward) as total_reward
            FROM transitions
            GROUP BY episode_id
        ),
        shuffled_episodes AS (
            SELECT episode_id, total_reward
            FROM all_episode_rewards
            ORDER BY HASH(episode_id)
            LIMIT {TOTAL_EPISODES_TO_SAMPLE}
        )
        SELECT
            episode_id,
            NTILE(100) OVER (ORDER BY total_reward) as percentile
        FROM shuffled_episodes
    """ 
    
    # Create a temporary table with our fixed pool of 10,000 episodes
    con.execute(f"CREATE OR REPLACE TEMP TABLE episode_pool AS ({episode_pool_query});")
    print(f"Created a fixed pool of {TOTAL_EPISODES_TO_SAMPLE} episodes for this run.")

    cql = d3rlpy.algos.DiscreteCQLConfig(batch_size=BATCH_SIZE).create(device="cuda:0")
    wandb_logger = d3rlpy.logging.WanDBAdapterFactory(project=WANDB_PROJECT)
    
    # Define the curriculum stages
    curriculum_stages = [
        {"name": "Stage 1: Easy", "min_percentile": 0, "max_percentile": 33, "steps": 250_000},
        {"name": "Stage 2: Medium", "min_percentile": 33, "max_percentile": 66, "steps": 250_000},
        {"name": "Stage 3: Hard", "min_percentile": 66, "max_percentile": 100, "steps": 250_000},
    ]
    total_steps = sum(s['steps'] for s in curriculum_stages)
    
    # Set a unique run name for W&B
    wandb_run_name = f"cql_{TOTAL_EPISODES_TO_SAMPLE}_episodes_{args.strategy}"

    if args.strategy == "control":
        # --- CONTROL STRATEGY ---
        print("Fetching all 10,000 episodes for the control group...")
        all_ids_df = con.execute("SELECT episode_id FROM episode_pool;").fetchdf()
        episode_ids = all_ids_df['episode_id'].tolist()
        
        placeholders = ",".join(map(str, episode_ids))
        query = f"""
            SELECT episode_id, obs, action, reward
            FROM transitions WHERE episode_id IN ({placeholders})
            ORDER BY episode_id ASC, step_in_episode ASC
        """
        df = con.execute(query).fetchdf()
        
        print("Building a single replay buffer...")
        episodes = []
        for ep_id, ep_df in tqdm(df.groupby('episode_id'), desc="Building Episodes"):
            observations = np.array(ep_df['obs'].apply(decompress_array).tolist(), dtype=np.float32)
            actions = ep_df['action'].to_numpy(dtype=np.float32).reshape(-1, 1)
            rewards = ep_df['reward'].to_numpy(dtype=np.float32).reshape(-1, 1)
            episodes.append(d3rlpy.dataset.Episode(
                observations=observations, actions=actions, rewards=rewards, terminated=True
            ))
        replay_buffer = d3rlpy.dataset.ReplayBuffer(buffer=d3rlpy.dataset.InfiniteBuffer(), episodes=episodes)

        print(f"🚀 Starting training for {total_steps} steps...")


        cql.fit(
            replay_buffer,
            n_steps=2_250_000,      # Use the appropriate number of steps
            n_steps_per_epoch=250000,         # Or your desired epoch size
            experiment_name=wandb_run_name,
            with_timestamp=True,
            show_progress=True,
            logger_adapter=wandb_logger
            # No 'evaluators' dictionary for speed and consistency
        )

    elif args.strategy == "curriculum":
        # --- CURRICULUM STRATEGY ---
        for i, stage in enumerate(curriculum_stages):
            print("\n" + "="*50)
            print(f"🚀 STARTING CURRICULUM: {stage['name']}")
            print("="*50)

            query_ids_sql = f"""
                SELECT episode_id FROM episode_pool
                WHERE percentile > {stage['min_percentile']} AND percentile <= {stage['max_percentile']};
            """
            stage_ids_df = con.execute(query_ids_sql).fetchdf()
            episode_ids = stage_ids_df['episode_id'].tolist()
            
            print(f"Fetching {len(episode_ids)} episodes for this stage...")
            placeholders = ",".join(map(str, episode_ids))
            query_transitions_sql = f"""
                SELECT episode_id, obs, action, reward
                FROM transitions WHERE episode_id IN ({placeholders})
                ORDER BY episode_id ASC, step_in_episode ASC
            """
            df = con.execute(query_transitions_sql).fetchdf()

            print("Building replay buffer for stage...")
            episodes = []
            for ep_id, ep_df in tqdm(df.groupby('episode_id'), desc="Building Episodes"):
                observations = np.array(ep_df['obs'].apply(decompress_array).tolist(), dtype=np.float32)
                actions = ep_df['action'].to_numpy(dtype=np.float32).reshape(-1, 1)
                rewards = ep_df['reward'].to_numpy(dtype=np.float32).reshape(-1, 1)
                episodes.append(d3rlpy.dataset.Episode(
                    observations=observations, actions=actions, rewards=rewards, terminated=True
                ))
            replay_buffer = d3rlpy.dataset.ReplayBuffer(buffer=d3rlpy.dataset.InfiniteBuffer(), episodes=episodes)
            
            logger_adapter_to_use = wandb_logger if i == 0 else "wandb"
            print(f"🚀 Starting training for {total_steps} steps...")

            cql.fit(
                replay_buffer,
                n_steps=total_steps,      # Use the appropriate number of steps
                n_steps_per_epoch=50000,         # Or your desired epoch size
                experiment_name=wandb_run_name,
                with_timestamp=True,
                show_progress=True,
                logger_adapter=wandb_logger
                # No 'evaluators' dictionary for speed and consistency
            )

    con.close()
    
    model_path = f"{MODEL_SAVE_PATH_PREFIX}_{args.strategy}.d3"
    cql.save(model_path)
    print(f"\n🎉 Training complete! Final model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled curriculum vs. standard training experiments.")
    parser.add_argument("--strategy", choices=["curriculum", "control"], required=True, help="Training strategy to use.")
    args = parser.parse_args()
    main(args)

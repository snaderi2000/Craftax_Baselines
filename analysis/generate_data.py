"""
Generate offline RL data from a trained PPO-RNN agent.

Usage:
    python analysis/generate_data.py \
        --run_path wandb/run-20260126_162348-kp7gx38r \
        --timestep 1000000 \
        --num_episodes 100 \
        --max_steps 700 \
        --output_dir ./replay_data/my_buffer
"""

import argparse
import json
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from tqdm import tqdm

from ppo_rnn import ActorCriticRNN, ScannedRNN
from craftax.craftax_env import make_craftax_env_from_name

# For flashbax vault
import flashbax as fbx
from flashbax.vault import Vault
from flashbax.utils import get_tree_shape_prefix


def load_trained_model(run_path: str, timestep: int):
    """Load config and trained parameters from a wandb run."""
    # Load config
    config_path = os.path.join(run_path, "files/config.yaml")
    with open(config_path) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)
        cfg = {k: v["value"] if isinstance(v, dict) and "value" in v else v 
               for k, v in raw_config.items()}
    
    # Setup environment to get action dim
    env_name = cfg.get("ENV_NAME", "Craftax-Classic-Pixels-v1")
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    action_dim = env.action_space(env_params).n
    
    # Initialize network
    network = ActorCriticRNN(action_dim=action_dim, config=cfg)
    
    # Load checkpoint
    ckpt_path = os.path.abspath(os.path.join(run_path, "files/policies"))
    mngr = ocp.CheckpointManager(ckpt_path, ocp.PyTreeCheckpointer())
    raw_data = mngr.restore(timestep)
    trained_params = raw_data['params']
    
    return network, trained_params, cfg, env, env_params


def generate_episodes_parallel(
    network,
    trained_params,
    env,
    env_params,
    num_episodes: int,
    max_steps: int,
    seed: int = 42,
):
    """
    Generate episodes in parallel using vectorized environment steps.
    
    Returns:
        dict with keys: obs, actions, rewards, dones
        Each has shape (num_episodes, max_steps, ...)
    """
    print(f"Generating {num_episodes} episodes with max {max_steps} steps each...")
    
    # Initialize storage (on CPU to save GPU memory)
    obs_shape = (63, 63, 3)
    all_obs = np.zeros((num_episodes, max_steps, *obs_shape), dtype=np.float32)
    all_actions = np.zeros((num_episodes, max_steps), dtype=np.int32)
    all_rewards = np.zeros((num_episodes, max_steps), dtype=np.float32)
    all_dones = np.zeros((num_episodes, max_steps), dtype=np.bool_)
    
    # Track which episodes have finished
    episode_lengths = np.zeros(num_episodes, dtype=np.int32)
    
    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    
    # Reset all environments in parallel
    rng, *env_rngs = jax.random.split(rng, num_episodes + 1)
    env_rngs = jnp.stack(env_rngs)
    
    # Vectorized reset
    reset_vmap = jax.vmap(env.reset, in_axes=(0, None))
    obs_batch, env_states = reset_vmap(env_rngs, env_params)
    
    # Initialize RNN hidden states for all episodes
    hstates = ScannedRNN.initialize_carry(num_episodes, 256)
    
    # Track done status
    dones = jnp.zeros(num_episodes, dtype=jnp.bool_)
    episode_done_permanent = np.zeros(num_episodes, dtype=np.bool_)
    
    # JIT compile the step functions
    @jax.jit
    def get_actions(params, hstates, obs_batch, dones_batch, rng):
        """Get actions for all environments."""
        # Add time dimension: (B,) -> (1, B)
        obs_in = obs_batch[None, :]  # (1, B, 63, 63, 3)
        dones_in = dones_batch[None, :]  # (1, B)
        
        new_hstates, pi, values = network.apply(params, hstates, (obs_in, dones_in))
        
        actions = pi.sample(seed=rng)
        actions = actions.squeeze(0)  # Remove time dim
        
        return new_hstates, actions
    
    step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    
    # Collect data
    for step_idx in tqdm(range(max_steps), desc="Collecting steps"):
        # Store current observations
        all_obs[:, step_idx] = np.array(obs_batch)
        
        # Get actions from policy
        rng, action_rng = jax.random.split(rng)
        hstates, actions = get_actions(trained_params, hstates, obs_batch, dones, action_rng)
        
        # Store actions
        all_actions[:, step_idx] = np.array(actions)
        
        # Step all environments
        rng, *step_rngs = jax.random.split(rng, num_episodes + 1)
        step_rngs = jnp.stack(step_rngs)
        
        obs_batch, env_states, rewards, step_dones, infos = step_vmap(
            step_rngs, env_states, actions, env_params
        )
        
        # Store rewards and dones
        all_rewards[:, step_idx] = np.array(rewards)
        all_dones[:, step_idx] = np.array(step_dones)
        
        # Update done tracking
        dones = step_dones
        
        # Track episode lengths (first time each episode ends)
        newly_done = np.array(step_dones) & ~episode_done_permanent
        episode_lengths = np.where(newly_done, step_idx + 1, episode_lengths)
        episode_done_permanent = episode_done_permanent | np.array(step_dones)
        
        # Note: We continue stepping even after done because env auto-resets
        # The 'dones' array marks where episodes ended
    
    # For episodes that never terminated, set length to max_steps
    episode_lengths = np.where(episode_lengths == 0, max_steps, episode_lengths)
    
    print(f"\nEpisode length statistics:")
    print(f"  Min: {episode_lengths.min()}")
    print(f"  Max: {episode_lengths.max()}")
    print(f"  Mean: {episode_lengths.mean():.1f}")
    print(f"  Episodes that reached max_steps: {(episode_lengths == max_steps).sum()}")
    
    return {
        "obs": all_obs,
        "actions": all_actions,
        "rewards": all_rewards,
        "dones": all_dones,
    }, episode_lengths


def save_to_vault(data: dict, output_dir: str, vault_name: str = "craftax_replay_buffer"):
    """Save collected data to a flashbax Vault."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to jax arrays with proper structure for flashbax
    experience = {
        "obs": jnp.array(data["obs"]),          # (B, T, 63, 63, 3) float32
        "actions": jnp.array(data["actions"]),   # (B, T) int32
        "rewards": jnp.array(data["rewards"]),   # (B, T) float32
        "dones": jnp.array(data["dones"]),       # (B, T) bool
    }
    
    B, T = experience["actions"].shape
    
    print(f"\nData shapes:")
    for k, v in experience.items():
        print(f"  {k}: {v.shape} ({v.dtype})")
    
    # Calculate memory usage
    total_bytes = sum(v.nbytes for v in experience.values())
    print(f"\nTotal data size: {total_bytes / 1e9:.2f} GB")
    
    # Create a trajectory buffer with matching structure
    # We create a buffer that can hold all our data
    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=T,
        min_length_time_axis=1,
        add_batch_size=B,
        sample_batch_size=1,
        sample_sequence_length=1,
        period=1,
    )
    
    # Create example item for initialization (single timestep per env)
    example_item = {
        "obs": jnp.zeros((B, 63, 63, 3), dtype=jnp.float32),
        "actions": jnp.zeros((B,), dtype=jnp.int32),
        "rewards": jnp.zeros((B,), dtype=jnp.float32),
        "dones": jnp.zeros((B,), dtype=jnp.bool_),
    }
    
    # Initialize buffer state
    buffer_state = buffer.init(example_item)
    
    # Add all timesteps to buffer
    print("\nAdding data to buffer...")
    for t in tqdm(range(T), desc="Adding timesteps"):
        timestep_data = {
            "obs": experience["obs"][:, t],
            "actions": experience["actions"][:, t],
            "rewards": experience["rewards"][:, t],
            "dones": experience["dones"][:, t],
        }
        buffer_state = buffer.add(buffer_state, timestep_data)
    
    # Setup vault paths
    vault_uid = os.path.basename(output_dir)
    rel_dir = os.path.dirname(os.path.abspath(output_dir))
    if not rel_dir:
        rel_dir = "."
    
    print(f"\nSaving to vault:")
    print(f"  Name: {vault_name}")
    print(f"  UID: {vault_uid}")
    print(f"  Directory: {rel_dir}")
    
    # Create vault with experience structure
    vault = Vault(
        vault_name=vault_name,
        experience_structure=buffer_state.experience,
        vault_uid=vault_uid,
        rel_dir=rel_dir,
    )
    
    # Write buffer to vault
    T_buf = get_tree_shape_prefix(buffer_state.experience, n_axes=2)[1]
    k = int(buffer_state.current_index)
    
    if bool(buffer_state.is_full):
        n1 = vault.write(buffer_state, source_interval=(k, T_buf), dest_start=vault.vault_index)
        n2 = vault.write(buffer_state, source_interval=(0, k), dest_start=vault.vault_index)
        n_written = n1 + n2
    else:
        n_written = vault.write(buffer_state, source_interval=(0, k), dest_start=vault.vault_index)
    
    print(f"✅ Vault created successfully! Wrote {n_written} timesteps.")
    return vault


def save_to_npz(data: dict, output_path: str):
    """Alternative: save to compressed NPZ file."""
    print(f"\nSaving to NPZ: {output_path}")
    np.savez_compressed(
        output_path,
        obs=data["obs"],
        actions=data["actions"],
        rewards=data["rewards"],
        dones=data["dones"],
    )
    file_size = os.path.getsize(output_path) / 1e9
    print(f"✅ Saved! File size: {file_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Generate offline RL data from trained PPO-RNN")
    parser.add_argument("--run_path", type=str, required=True,
                        help="Path to wandb run directory")
    parser.add_argument("--timestep", type=int, default=1000000,
                        help="Checkpoint timestep to load")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to generate")
    parser.add_argument("--max_steps", type=int, default=700,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./replay_data/default_buffer",
                        help="Output directory for the vault")
    parser.add_argument("--save_npz", action="store_true",
                        help="Also save as NPZ file (for debugging)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Offline Data Generation from Trained PPO-RNN Agent")
    print("=" * 60)
    print(f"Run path: {args.run_path}")
    print(f"Timestep: {args.timestep}")
    print(f"Num episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading trained model...")
    network, trained_params, cfg, env, env_params = load_trained_model(
        args.run_path, args.timestep
    )
    print(f"✅ Loaded model from {args.run_path}")
    print(f"   Environment: {cfg.get('ENV_NAME', 'unknown')}")
    
    # Generate data
    print("\n[2/3] Generating episodes...")
    data, episode_lengths = generate_episodes_parallel(
        network=network,
        trained_params=trained_params,
        env=env,
        env_params=env_params,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    # Save data
    print("\n[3/3] Saving data...")
    vault = save_to_vault(data, args.output_dir)
    
    if args.save_npz:
        npz_path = os.path.join(args.output_dir, "data.npz")
        save_to_npz(data, npz_path)
    
    # Save metadata
    metadata = {
        "run_path": args.run_path,
        "timestep": args.timestep,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "env_name": cfg.get("ENV_NAME", "unknown"),
        "episode_lengths": episode_lengths.tolist(),
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print(f"   Vault location: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

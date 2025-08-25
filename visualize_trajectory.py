# visualize_trajectory.py (Corrected)

import argparse
import imageio
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault

# This disables GPU for this simple script
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def visualize_trajectory(vault_name: str, vault_uid: str, output_path: str):
    """Loads a saved vault and visualizes the first trajectory as a GIF."""
    
    print(f"--- Loading Vault '{vault_name}' with UID '{vault_uid}' ---")
    
    try:
        # Recreate the buffer's structure to load the vault correctly
        example_item = {
            "obs": jnp.zeros((63, 63, 3), dtype=jnp.uint8),
            "actions": jnp.zeros((), dtype=jnp.int32),
            "rewards": jnp.zeros((), dtype=jnp.float32),
            "dones": jnp.zeros((), dtype=bool),
        }
        buffer_for_init = fbx.make_trajectory_buffer(
            max_length_time_axis=128000 // 48, # Must match save script
            min_length_time_axis=20, add_batch_size=48,
            sample_batch_size=32, sample_sequence_length=20, period=1,
        )
        dummy_state = buffer_for_init.init(example_item)
        
        # Initialize and read from the Vault
        vault = Vault(
            vault_name=vault_name,
            experience_structure=dummy_state.experience,
            vault_uid=vault_uid,
            rel_dir="."
        )
        buffer_state = vault.read()
        print("✅ Vault loaded successfully.")

        # --- FIX IS HERE ---
        # Get the number of valid timesteps directly from the vault object itself
        num_timesteps_added = vault.vault_index
        
        if num_timesteps_added == 0:
            print("Buffer is empty, nothing to visualize.")
            return
            
        # Extract the observation history for the first parallel environment
        first_trajectory_obs = buffer_state.experience['obs'][0]
        
        # Slice the trajectory to get only the frames that were actually collected
        valid_frames = first_trajectory_obs[:num_timesteps_added]
        
        # Save the frames as a GIF
        print(f"\nSaving trajectory of {len(valid_frames)} frames to '{output_path}'...")
        imageio.mimsave(output_path, valid_frames, fps=10)
        print(f"✅ GIF saved successfully.")

    except Exception as e:
        import traceback
        print(f"\n❌ An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vault_name", type=str, help="Name of the vault parent directory.")
    parser.add_argument("vault_uid", type=str, help="The specific UID of the vault to load.")
    parser.add_argument("--output", type=str, default="saved_trajectory.gif", help="Output filename for the GIF.")
    args = parser.parse_args()
    visualize_trajectory(args.vault_name, args.vault_uid, args.output)
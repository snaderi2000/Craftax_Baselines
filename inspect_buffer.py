import argparse
from flashbax.vault import Vault
import jax.numpy as jnp
import flashbax as fbx
import os

# This disables GPU for this simple script
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def inspect_buffer(vault_name: str, vault_uid: str):
    """Loads a specific Flashbax Vault and correctly inspects its contents."""
    
    print(f"--- Inspecting Vault '{vault_name}' with UID '{vault_uid}' ---")
    
    try:
        # Recreate the buffer's structure for validation
        example_item = {
            "obs": jnp.zeros((63, 63, 3), dtype=jnp.uint8),
            "actions": jnp.zeros((), dtype=jnp.int32),
            "rewards": jnp.zeros((), dtype=jnp.float32),
            "dones": jnp.zeros((), dtype=bool),
        }
        buffer_for_init = fbx.make_trajectory_buffer(
            max_length_time_axis=128000 // 48,
            min_length_time_axis=20, add_batch_size=48,
            sample_batch_size=32, sample_sequence_length=20, period=1,
        )
        dummy_state = buffer_for_init.init(example_item)
        
        # Initialize the Vault object. This loads the metadata and the vault_index.
        vault = Vault(
            vault_name=vault_name,
            experience_structure=dummy_state.experience,
            vault_uid=vault_uid,
            rel_dir="."
        )
        print("✅ Vault object initialized successfully.")

        # --- Definitive Check ---
        # The vault_index attribute holds the true number of saved timesteps
        total_timesteps_saved = vault.vault_index
        print(f"\nTotal timesteps found in vault: {total_timesteps_saved}")

        if total_timesteps_saved > 0:
            print("✅ Success! The buffer contains data and is ready for training.")
        else:
            print("❌ The buffer is empty.")

    except Exception as e:
        import traceback
        print(f"\n❌ An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vault_name", type=str, help="Name of the vault parent directory.")
    parser.add_argument("vault_uid", type=str, help="The specific, timestamped UID.")
    args = parser.parse_args()
    inspect_buffer(args.vault_name, args.vault_uid)


#python inspect_buffer.py craftax_replay_buffer my_first_buffer_run
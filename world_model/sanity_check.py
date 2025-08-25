import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

def run_sanity_check():
    """
    Verifies that the slicer-free logic correctly handles
    embedding and head selection.
    """
    print("🔬 Running sanity check for slicer-free logic...")

    # --- 1. Create a simple, predictable setup ---
    B, T, L, C = 1, 2, 3, 4 # Batch=1, 2 timesteps, 3 obs tokens, embed_dim=4
    tokens_per_block = L + 1
    T_flat = T * tokens_per_block
    
    # --- 2. Create "toy" data we can easily recognize ---
    # Obs tokens are small positive numbers, action tokens are large ones.
    # Timestep 1: obs=[10, 11, 12], act=[100]
    # Timestep 2: obs=[13, 14, 15], act=[101]
    tokens = jnp.array([[10, 11, 12, 100, 13, 14, 15, 101]])
    print(f"Input token sequence (flat):\n{tokens}\n")

    # --- 3. Create "special" embedding tables ---
    # obs_embed maps token `i` to a vector of `i`'s.
    # act_embed maps token `i` to a vector of `-i`'s.
    obs_vocab_size = 20
    act_vocab_size = 105
    
    # Create the embedding weights manually
    obs_embed_weights = jnp.arange(obs_vocab_size)[:, None] * jnp.ones((obs_vocab_size, C))
    act_embed_weights = -jnp.arange(act_vocab_size)[:, None] * jnp.ones((act_vocab_size, C))

    obs_embed = nn.Embed(num_embeddings=obs_vocab_size, features=C)
    act_embed = nn.Embed(num_embeddings=act_vocab_size, features=C)
    
    # Manually assign the weights to our simple embedding modules
    embed_params = {
        'params': {
            'embedding': obs_embed_weights
        }
    }
    act_embed_params = {
        'params': {
            'embedding': act_embed_weights
        }
    }

    # --- 4. Test the Embedding Logic (Embedder replacement) ---
    print("--- Testing Part 1: Embedding Logic ---")
    
    # This is the exact logic from WorldModel.__call__
    tokens_structured = tokens.reshape(B, T, tokens_per_block)
    obs_tokens = tokens_structured[..., :L]
    act_tokens = tokens_structured[..., -1]

    obs_embs = obs_embed.apply(embed_params, obs_tokens)
    act_embs = act_embed.apply(act_embed_params, act_tokens)
    
    sequences_structured = jnp.concatenate([obs_embs, act_embs[..., None, :]], axis=-2)
    embedded_sequence = sequences_structured.reshape(B, T_flat, C)

    print("Resulting embedded sequence:")
    print(embedded_sequence)
    
    # Verification
    # Check if the 4th element (first action) is embedded as -100
    is_act1_correct = jnp.all(embedded_sequence[0, 3] == -100)
    # Check if the 8th element (second action) is embedded as -101
    is_act2_correct = jnp.all(embedded_sequence[0, 7] == -101)

    if is_act1_correct and is_act2_correct:
        print("\n✅ SUCCESS: Observations and actions were embedded by different tables and correctly interleaved.")
    else:
        print("\n❌ FAILED: Embedding logic is incorrect.")

    # --- 5. Test the Head Selection Logic (Head replacement) ---
    print("\n--- Testing Part 2: Head Selection Logic ---")

    # Create a dummy output from the transformer with recognizable values
    # Each timestep `t` will have embeddings with the value `t+1`
    dummy_transformer_output = jnp.concatenate([
        jnp.ones((B, tokens_per_block, C)) * 1.0,  # Timestep 1 embeddings
        jnp.ones((B, tokens_per_block, C)) * 2.0,  # Timestep 2 embeddings
    ], axis=1).reshape(B, T_flat, C)
    
    print("\nDummy Transformer Output (values represent timestep):")
    print(dummy_transformer_output)
    
    # This is the exact logic from WorldModel.__call__
    x_structured = dummy_transformer_output.reshape(B, T, tokens_per_block, C)
    act_embs_for_pred = x_structured[..., -1, :] # Select only the action embedding
    
    print("\nEmbeddings selected for the Reward/End Heads:")
    print(act_embs_for_pred)
    
    # Verification
    # Check if the embedding for timestep 1 is selected
    is_timestep1_selected = jnp.all(act_embs_for_pred[0, 0] == 1.0)
    # Check if the embedding for timestep 2 is selected
    is_timestep2_selected = jnp.all(act_embs_for_pred[0, 1] == 2.0)
    
    if act_embs_for_pred.shape[1] == T and is_timestep1_selected and is_timestep2_selected:
        print("\n✅ SUCCESS: The logic correctly selected ONLY the action embeddings from each timestep.")
    else:
        print("\n❌ FAILED: Head selection logic is incorrect.")


if __name__ == "__main__":
    run_sanity_check()
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import Optional, Any
from einops import rearrange
import optax

# Import your modules
from transformer import Transformer, TransformerConfig
from kv_caching import KeysValues
from slicer import Embedder, Head, Slicer
# Assuming Tokenizer and Batch are available/mocked or typed as Any
Tokenizer = Any
Batch = Any

@struct.dataclass
class WorldModelOutput:
    output_sequence: jnp.ndarray
    logits_observations: jnp.ndarray
    logits_rewards: jnp.ndarray
    logits_ends: jnp.ndarray

@struct.dataclass
class LossWithIntermediateLosses:
    loss_obs: jnp.ndarray
    loss_rewards: jnp.ndarray
    loss_ends: jnp.ndarray
    total_loss: jnp.ndarray

class WorldModel(nn.Module):
    obs_vocab_size: int
    act_vocab_size: int
    config: TransformerConfig

    def setup(self):
        # 1. Define Patterns matching PyTorch exactly
        # [0, 0, ..., 1] -> Action is the last token
        self.act_pattern = jnp.zeros(self.config.tokens_per_block).at[-1].set(1.0)
        self.obs_pattern = 1.0 - self.act_pattern
        
        # Pattern for Observation Head: PyTorch uses 'all_but_last_obs_tokens_pattern'
        # It seems to mask out the token BEFORE the action? 
        # Let's trust the masks we generate here for the loss.
        # For JAX, we simply predict everywhere and mask the loss.
        
        # 2. Components
        self.transformer = Transformer(self.config)
        self.pos_emb = nn.Embed(self.config.max_tokens, self.config.embed_dim)
        
        # Embedder with Correct Lists
        self.embedder = Embedder(
            max_blocks=self.config.max_blocks,
            embed_dim=self.config.embed_dim,
            block_masks=[self.act_pattern, self.obs_pattern],
            vocab_sizes=[self.act_vocab_size, self.obs_vocab_size]
        )
        
        # 3. Heads
        tpb = self.config.tokens_per_block  # 65

        act_mask = jnp.zeros((tpb,), dtype=jnp.float32).at[-1].set(1.0)
        all_but_last_obs = jnp.ones((tpb,), dtype=jnp.float32).at[-2].set(0.0)

        self.head_observations = Head(
            embed_dim=self.config.embed_dim,
            output_dim=self.obs_vocab_size,
            block_mask=all_but_last_obs,
        )

        self.head_rewards = Head(
            embed_dim=self.config.embed_dim,
            output_dim=3,
            block_mask=act_mask,
        )

        self.head_ends = Head(
            embed_dim=self.config.embed_dim,
            output_dim=2,
            block_mask=act_mask,
        ) 

    def __call__(self, tokens, past_keys_values=None, deterministic=True):
        """
        Args:
            tokens: [Batch, Time] integer indices.
            past_keys_values: KVCache object (optional).
        """
        B, T = tokens.shape
        
        # Calculate start step for positional embeddings and masking
        if past_keys_values is not None:
            # Assuming all layers are synced, take index from first layer
            prev_steps = past_keys_values[0].index
        else:
            prev_steps = 0
            
        # 1. Embeddings
        # Mix Obs/Act embeddings
        x = self.embedder(tokens, start_step=prev_steps)
        
        # Add Learned Positional Embeddings
        pos_indices = jnp.arange(T) + prev_steps
        # Debug prints (comment out for cleaner output)
        # print("pos_indices.max():", pos_indices.max())
        # print("pos_emb.num_embeddings:", self.pos_emb.num_embeddings)

        x = x + self.pos_emb(pos_indices)
        
        # 2. Transformer Backbone
        # Returns (x, new_cache) if cache provided, else x
        trans_out = self.transformer(x, past_keys_values, deterministic=deterministic)
        
        if past_keys_values is not None:
            x_out, new_cache = trans_out
        else:
            x_out = trans_out
            new_cache = None
            
        # 3. Heads
        # In JAX, we simply run the heads on the full sequence.
        # We will handle masking/selecting specific tokens during Loss computation.
        logits_obs = self.head_observations(x_out, num_steps=T, prev_steps=prev_steps)
        logits_rew = self.head_rewards(x_out, num_steps=T, prev_steps=prev_steps)
        logits_ends = self.head_ends(x_out, num_steps=T, prev_steps=prev_steps)
        
        output = WorldModelOutput(
            output_sequence=x_out,
            logits_observations=logits_obs,
            logits_rewards=logits_rew,
            logits_ends=logits_ends
        )
        
        if past_keys_values is not None:
            return output, new_cache
        return output

    # --- Loss Computation Logic ---
    
    def compute_loss(self, batch, dropout_rng=None):
        """
        Calculates loss. Must be called via model.apply(..., method=model.compute_loss)
        """
        # 1. Prepare Inputs (Interleave Obs and Actions)
        # obs_tokens: [B, T, 64]
        obs_tokens = batch['obs_tokens'] 
        act_tokens = batch['actions'][..., None] # [B, T, 1]
        
        # Concatenate to [B, T, 65] then flatten
        B, T, _ = obs_tokens.shape
        tokens_block = jnp.concatenate([obs_tokens, act_tokens], axis=2)
        tokens_flat = tokens_block.reshape(B, -1) # [B, T*65]

        # Debug prints (comment out for cleaner output)
        # print("tokens_flat.shape:", tokens_flat.shape)
        # print("config.max_tokens:", self.config.max_tokens)

        
        # 2. Forward Pass (using bound self)
        # We pass the flattened tokens.
        # self.__call__ will run embedder -> transformer -> heads
        # It returns WorldModelOutput with logits for ALL steps.
        output = self.__call__(tokens_flat, deterministic=False)
        
        # 3. Compute Raw Labels
        labels_obs, labels_rew, labels_ends = self.compute_labels(
            obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding']
        )
        
        # 4. EXPAND LABELS & ALIGN SHAPES
        # We need to map the sparse labels onto the dense (B, T, 65) grid.
        
        # --- A. Prepare Rewards & Ends (Located at Index 64 / Action Token) ---
        labels_rew_reshaped = labels_rew.reshape(B, T, 1)
        labels_ends_reshaped = labels_ends.reshape(B, T, 1)
        
        target_rew_grid = jnp.full((B, T, 65), -100, dtype=jnp.int32)
        target_rew_grid = target_rew_grid.at[:, :, -1].set(labels_rew_reshaped.squeeze(-1))
        
        target_ends_grid = jnp.full((B, T, 65), -100, dtype=jnp.int32)
        target_ends_grid = target_ends_grid.at[:, :, -1].set(labels_ends_reshaped.squeeze(-1))
        
        # --- B. Prepare Observations (Located at Indices 0..63) ---
        target_obs_grid = jnp.full((B, T, 65), -100, dtype=jnp.int32)
        
        # Robustly place obs labels
        target_len = B * T * 64
        if labels_obs.size < target_len:
             labels_obs = jnp.pad(labels_obs, (0, target_len - labels_obs.size), constant_values=-100)
        elif labels_obs.size > target_len:
             labels_obs = labels_obs[:target_len]
        
        target_obs_grid = target_obs_grid.at[:, :, :-1].set(labels_obs.reshape(B, T, 64))

        # --- C. CRITICAL FIX: Flatten First, Then Slice ---
        # 1. Flatten to (Batch, Total_Tokens) e.g. (32, 1300)
        flat_rew = target_rew_grid.reshape(B, -1)
        flat_obs = target_obs_grid.reshape(B, -1)
        flat_ends = target_ends_grid.reshape(B, -1)

        # 2. Slice off the FIRST token (Target for the prediction made at step 0)
        # Shape becomes (Batch, 1299) -> Flatten to (41568,)
        labels_rew_flat = flat_rew[:, 1:].reshape(-1)
        labels_obs_flat = flat_obs[:, 1:].reshape(-1)
        labels_ends_flat = flat_ends[:, 1:].reshape(-1)

        # ------------------------------------------------------------------
        # 5. Calculate Cross Entropy (masked, stable)
        # ------------------------------------------------------------------

        def compute_masked_loss(logits_flat, labels_flat, vocab_size, name=""):
            """
            logits_flat: (N, C)
            labels_flat: (N,)
            """
            mask = (labels_flat != -100)
            
            # Clamp labels to valid range to prevent indexing errors
            safe_labels = jnp.where(mask, labels_flat, 0)
            safe_labels = jnp.clip(safe_labels, 0, vocab_size - 1)
            
            # Check for NaN in logits
            has_nan_logits = jnp.any(jnp.isnan(logits_flat))
            has_inf_logits = jnp.any(jnp.isinf(logits_flat))
            
            # Numerically stable cross-entropy
            # Clip logits to prevent overflow
            logits_clipped = jnp.clip(logits_flat, -50.0, 50.0)
            
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits_clipped, safe_labels
            )

            loss = jnp.where(mask, loss, 0.0)
            
            num_valid = mask.sum()
            avg_loss = jnp.where(num_valid > 0, loss.sum() / (num_valid + 1e-9), 0.0)
            
            # Debug: print if NaN detected (only during tracing, not during JIT)
            # jax.debug.print("{name} - nan_logits: {nan}, inf_logits: {inf}, valid: {v}, loss: {l}", 
            #                 name=name, nan=has_nan_logits, inf=has_inf_logits, v=num_valid, l=avg_loss)
            
            return avg_loss


        # ------------------------------------------------------------------
        # 6. Get logits from the output (already computed in __call__)
        # ------------------------------------------------------------------

        logits_obs  = output.logits_observations
        logits_rew  = output.logits_rewards
        logits_ends = output.logits_ends

        # ------------------------------------------------------------------
        # 7. Shift + flatten logits (autoregressive alignment)
        #    Predict token t+1 from logits at t
        # ------------------------------------------------------------------

        logits_obs_flat = logits_obs[:, :-1, :].reshape(-1, self.obs_vocab_size)
        logits_rew_flat = logits_rew[:, :-1, :].reshape(-1, 3)
        logits_ends_flat = logits_ends[:, :-1, :].reshape(-1, 2)

        # ------------------------------------------------------------------
        # 8. Compute losses
        # ------------------------------------------------------------------

        loss_obs = compute_masked_loss(logits_obs_flat, labels_obs_flat, self.obs_vocab_size, "obs")
        loss_rew = compute_masked_loss(logits_rew_flat, labels_rew_flat, 3, "rew")
        loss_ends = compute_masked_loss(logits_ends_flat, labels_ends_flat, 2, "ends")

        total_loss = loss_obs + loss_rew + loss_ends

        return LossWithIntermediateLosses(
            loss_obs=loss_obs,
            loss_rewards=loss_rew,
            loss_ends=loss_ends,
            total_loss=total_loss,
        )
        

    def compute_labels(self, obs_tokens, rewards, ends, mask_padding):
        """
        Compute labels for the world model training.
        
        Args:
            obs_tokens: (B, L, K) observation tokens where K=64
            rewards: (B, L) reward values
            ends: (B, L) episode end flags (0 or 1)
            mask_padding: (B, L) boolean mask, True = padding (ignore)
        
        Returns:
            labels_obs: (B*L*K,) flattened observation labels, -100 for ignored
            labels_rew: (B*L,) flattened reward labels (0,1,2), -100 for ignored
            labels_ends: (B*L,) flattened end labels (0,1), -100 for ignored
        """
        # mask_padding: True = Padding (Ignore)
        mask_valid = ~mask_padding  # True = valid data
        B, L, K = obs_tokens.shape
        
        # 1. Obs Labels - expand mask to match obs shape
        flat_obs = obs_tokens.reshape(B, L * K)
        # Expand mask [B, L] -> [B, L*K] by repeating each position K times
        mask_flat = jnp.repeat(mask_valid, K, axis=1)
        labels_obs = jnp.where(mask_flat, flat_obs, -100).reshape(-1)
        
        # 2. Reward Labels: {-1, 0, 1} -> {0, 1, 2}
        rewards_cls = jnp.sign(rewards).astype(jnp.int32) + 1
        labels_rew = jnp.where(mask_valid, rewards_cls, -100).reshape(-1)
        
        # 3. Ends Labels: ensure integer type
        ends_int = ends.astype(jnp.int32)
        labels_ends = jnp.where(mask_valid, ends_int, -100).reshape(-1)
        
        return labels_obs, labels_rew, labels_ends
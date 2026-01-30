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
from slicer import Embedder, Head
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
        self.head_observations = Head(self.config.embed_dim, self.obs_vocab_size)
        self.head_rewards = Head(self.config.embed_dim, 3)
        self.head_ends = Head(self.config.embed_dim, 2)

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
        logits_obs = self.head_observations(x_out)
        logits_rew = self.head_rewards(x_out)
        logits_ends = self.head_ends(x_out)
        
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
        
        # 2. Forward Pass (using bound self)
        # We pass the flattened tokens.
        # self.__call__ will run embedder -> transformer -> heads
        # It returns WorldModelOutput with logits for ALL steps.
        output = self.__call__(tokens_flat, deterministic=False, rngs={'dropout': dropout_rng} if dropout_rng is not None else None)
        
        # 3. Compute Raw Labels
        labels_obs, labels_rew, labels_ends = self.compute_labels(
            obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding']
        )
        
        # 4. APPLY MASKS (Critical for Slicer/Head alignment)
        # We must ignore predictions from the "wrong" heads at specific steps.
        
        # Generate mask for Action positions (where Rewards/Ends are valid)
        # Shape: [T*65]
        total_steps = tokens_flat.shape[1]
        is_action_step = Slicer.compute_mask(total_steps, 0, self.act_pattern)
        # Broadcast to batch: [B, TotalSteps]
        is_action_mask = jnp.tile(is_action_step, (B, 1))
        
        # REWARD & ENDS LOSS
        # Only valid where is_action_mask is True
        labels_rew = jnp.where(is_action_mask.reshape(-1), labels_rew, -100)
        labels_ends = jnp.where(is_action_mask.reshape(-1), labels_ends, -100)
        
        # OBS LOSS
        # Only valid where is_action_mask is FALSE (i.e. it is an observation token)
        # Note: PyTorch had specific slicing, but generally we predict Obs at Obs steps.
        labels_obs = jnp.where(~is_action_mask.reshape(-1), labels_obs, -100)

        # 5. Calculate Cross Entropy
        
        # Obs
        logits_obs = output.logits_observations[:, :-1].reshape(-1, self.obs_vocab_size)
        loss_obs = optax.softmax_cross_entropy_with_integer_labels(logits_obs, labels_obs)
        loss_obs = (loss_obs * (labels_obs != -100)).sum() / ((labels_obs != -100).sum() + 1e-9)

        # Rewards
        logits_rew = output.logits_rewards.reshape(-1, 3)
        loss_rew = optax.softmax_cross_entropy_with_integer_labels(logits_rew, labels_rew)
        loss_rew = (loss_rew * (labels_rew != -100)).sum() / ((labels_rew != -100).sum() + 1e-9)

        # Ends
        logits_ends = output.logits_ends.reshape(-1, 2)
        loss_ends = optax.softmax_cross_entropy_with_integer_labels(logits_ends_flat, labels_ends)
        loss_ends = (loss_ends * (labels_ends != -100)).sum() / ((labels_ends != -100).sum() + 1e-9)

        total_loss = loss_obs + loss_rew + loss_ends
        
        return LossWithIntermediateLosses(loss_obs, loss_rew, loss_ends, total_loss)

    def compute_labels(self, obs_tokens, rewards, ends, mask_padding):
        # mask_padding: True = Padding (Ignore)
        mask_valid = ~mask_padding
        B, L, K = obs_tokens.shape
        
        # 1. Obs Labels
        # Flatten obs tokens: [B, L*K]
        # We shift by 1 (Predict Next)
        # We need to insert -100 (ignore) for Action positions in the stream?
        # The PyTorch code:
        # labels_observations = rearrange(obs_tokens, 'b t k -> b (t k)')[:, 1:]
        # It essentially trains on Obs->Obs and Obs->Act? 
        # Actually PyTorch code uses obs_tokens only.
        
        flat_obs = obs_tokens.reshape(B, L * K)
        # Apply padding mask
        # Expand mask [B, L] -> [B, L, K] -> [B, L*K]
        mask_flat = jnp.repeat(mask_valid, K, axis=1)
        
        labels_obs = jnp.where(mask_flat, flat_obs, -100)
        
        # The stream we fed in was [Obs...Act]. The output size is L*(K+1).
        # We need to map labels to that stream. 
        # This part requires distinct alignment with the PyTorch logic 
        # which separates logits_observations logic.
        
        # For simplicity in translation:
        # We construct the full target sequence [Obs_2..Act_1..Obs_Next]
        # But let's stick to the PyTorch return values:
        # It returns flattened labels.
        
        # Only shift obs:
        labels_obs = labels_obs[:, 1:].reshape(-1)
        
        # 2. Reward Labels
        # Rewards are {-1, 0, 1} -> shift to {0, 1, 2}
        rewards_cls = jnp.sign(rewards).astype(jnp.int32) + 1
        labels_rew = jnp.where(mask_valid, rewards_cls, -100).reshape(-1)
        
        # 3. Ends Labels
        labels_ends = jnp.where(mask_valid, ends, -100).astype(jnp.int32).reshape(-1)
        
        return labels_obs, labels_rew, labels_ends
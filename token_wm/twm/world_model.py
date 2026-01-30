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
        # 1. Define Patterns for interleaving
        # Act is the LAST token in the block
        # [0, 0, ..., 0, 1]
        act_pattern = jnp.zeros(self.config.tokens_per_block)
        act_pattern = act_pattern.at[-1].set(1.0)
        self.act_tokens_pattern = act_pattern
        
        # 2. Components
        self.transformer = Transformer(self.config)
        
        # Positional Embedding (Learned)
        # Note: We also have RoPE in the transformer, but we keep this 
        # to match the PyTorch reference implementation exactly.
        self.pos_emb = nn.Embed(self.config.max_tokens, self.config.embed_dim)
        
        self.embedder = Embedder(
            max_blocks=self.config.max_blocks,
            embed_dim=self.config.embed_dim,
            # List of patterns (Action mask, Obs mask)
            block_masks=[act_pattern, obs_pattern],
            # List of vocab sizes (Action vocab, Obs vocab)
            vocab_sizes=[self.act_vocab_size, self.obs_vocab_size]
        ) 
        
        # 3. Output Heads
        # Observation Head (Next token prediction)
        self.head_observations = Head(
            embed_dim=self.config.embed_dim, 
            output_dim=self.obs_vocab_size
        )
        
        # Reward Head (3 classes: -1, 0, 1)
        self.head_rewards = Head(
            embed_dim=self.config.embed_dim,
            output_dim=3
        )
        
        # Ends Head (2 classes: False, True)
        self.head_ends = Head(
            embed_dim=self.config.embed_dim,
            output_dim=2
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
    
    def compute_loss(self, batch, tokenizer_encode_fn, params, dropout_rng):
        """
        Standalone loss function compatible with JAX transformations.
        batch: Dictionary containing 'observations', 'actions', 'rewards', 'ends', 'mask_padding'
        """
        # 1. Tokenize Observations (Assuming we have VQVAE indices already or encode on fly)
        # For efficiency in JAX, pre-tokenizing data is preferred. 
        # Assuming batch['obs_tokens'] exists or calculating it here:
        # obs_tokens shape: [Batch, Length, K_patches]
        obs_tokens = batch['obs_tokens'] 
        
        # 2. Prepare Inputs
        # act_tokens: [Batch, Length] -> [Batch, Length, 1]
        act_tokens = batch['actions'][..., None]
        
        # Concatenate: [Obs_1...Obs_K, Act]
        # rearrange (B, L, K) and (B, L, 1) -> (B, L * (K+1))
        # We do this manually or via reshape
        B, L, K = obs_tokens.shape
        # Interleave: We construct the flat sequence
        # This requires careful reshaping to match the [Obs, Obs, ..., Act] pattern
        
        # Concatenate along last dim
        tokens_block = jnp.concatenate([obs_tokens, act_tokens], axis=2) # [B, L, K+1]
        tokens_flat = tokens_block.reshape(B, L * (K + 1))
        
        # 3. Forward Pass
        # We call the model with 'params' (passed from train state)
        # self.apply is used inside train step
        # output = self.apply(...)
        # For this method structure, we assume it's running inside a bound module or passing apply_fn
        # We'll assume standard Flax calling convention here (call returns output)
        output = self.__call__(tokens_flat, deterministic=False)

        # 4. Compute Labels
        labels_obs, labels_rew, labels_ends = self.compute_labels(
            obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding']
        )
        
        # 5. Compute Losses
        
        # Observation Loss: Predict NEXT token
        # logits: [B, T, Vocab] -> Shift right for prediction
        # labels: [B, T] (already shifted in compute_labels)
        
        # We only care about predicting Obs tokens, not Action tokens.
        # However, the standard AR objective usually trains on everything.
        # The PyTorch code sliced specific logits.
        
        # PyTorch Reference:
        # logits_observations = outputs.logits_observations[:, :-1]
        # labels_observations = ... [:, 1:]
        
        logits_obs_flat = output.logits_observations[:, :-1].reshape(-1, self.obs_vocab_size)
        loss_obs = optax.softmax_cross_entropy_with_integer_labels(logits_obs_flat, labels_obs)
        
        # Masking padding/invalid targets (-100 used in PyTorch)
        # JAX doesn't support -100 masking natively in loss, we must multiply by mask
        mask_obs = (labels_obs != -100)
        loss_obs = (loss_obs * mask_obs).sum() / (mask_obs.sum() + 1e-9)

        # Reward Loss
        # Only valid at Action positions
        logits_rew_flat = output.logits_rewards.reshape(-1, 3)
        loss_rew = optax.softmax_cross_entropy_with_integer_labels(logits_rew_flat, labels_rew)
        mask_rew = (labels_rew != -100)
        loss_rew = (loss_rew * mask_rew).sum() / (mask_rew.sum() + 1e-9)
        
        # Ends Loss
        logits_ends_flat = output.logits_ends.reshape(-1, 2)
        loss_ends = optax.softmax_cross_entropy_with_integer_labels(logits_ends_flat, labels_ends)
        mask_ends = (labels_ends != -100)
        loss_ends = (loss_ends * mask_ends).sum() / (mask_ends.sum() + 1e-9)

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
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.struct import dataclass
from einops import rearrange
from typing import Any, List, Optional, Tuple

# Import modules directly
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .kv_caching import KVCacheState

@dataclass
class WorldModelOutput:
    output_sequence: jnp.ndarray
    logits_observations: jnp.ndarray
    logits_rewards: jnp.ndarray
    logits_ends: jnp.ndarray

@dataclass
class LossWithIntermediateLosses:
    total_loss: jnp.ndarray
    loss_obs: jnp.ndarray
    loss_rewards: jnp.ndarray
    loss_ends: jnp.ndarray

class WorldModel(nn.Module):
    obs_vocab_size: int
    act_vocab_size: int
    config: TransformerConfig

    def setup(self):
        self.transformer = Transformer(self.config)
        self.pos_emb = nn.Embed(self.config.max_tokens, self.config.embed_dim)

        # Define embedding tables directly
        self.obs_embed = nn.Embed(self.obs_vocab_size, self.config.embed_dim)
        self.act_embed = nn.Embed(self.act_vocab_size, self.config.embed_dim)

        # Define prediction heads directly
        self.head_observations = nn.Sequential([
            nn.Dense(self.config.embed_dim), nn.relu, nn.Dense(self.obs_vocab_size)
        ])
        self.head_rewards = nn.Sequential([
            nn.Dense(self.config.embed_dim), nn.relu, nn.Dense(3)
        ])
        self.head_ends = nn.Sequential([
            nn.Dense(self.config.embed_dim), nn.relu, nn.Dense(2)
        ])

    def __call__(self, tokens: jnp.ndarray, past_keys_values: Optional[List[KVCacheState]] = None, *, train: bool) -> Tuple[WorldModelOutput, Optional[List[KVCacheState]]]:
        B, T_flat = tokens.shape
        tokens_per_block = self.config.tokens_per_block
        L = tokens_per_block - 1  # Number of obs tokens

        # Ensure the flat token sequence length is compatible with the block structure
        assert T_flat % tokens_per_block == 0, f"Total tokens {T_flat} not divisible by block size {tokens_per_block}"
        T = T_flat // tokens_per_block # This is the number of timesteps

        # --- Embedding (replaces Embedder) ---
        tokens_structured = tokens.reshape(B, T, tokens_per_block)
        obs_tokens = tokens_structured[..., :L]
        act_tokens = tokens_structured[..., -1]

        obs_embs = self.obs_embed(obs_tokens)
        act_embs = self.act_embed(act_tokens)

        sequences_structured = jnp.concatenate([obs_embs, act_embs[..., None, :]], axis=-2)
        sequences_flat = sequences_structured.reshape(B, T_flat, self.config.embed_dim)

        # Add positional embeddings
        num_steps = tokens.shape[1]
        prev_steps = 0 if past_keys_values is None else past_keys_values[0].k.size
        pos_indices = prev_steps + jnp.arange(num_steps)
        sequences = sequences_flat + self.pos_emb(pos_indices)

        # --- Transformer ---
        x_flat, new_keys_values = self.transformer(sequences, past_keys_values, train=train)
        
        # --- Prediction Heads (replaces Head) ---
        x_structured = x_flat.reshape(B, T, tokens_per_block, self.config.embed_dim)
        
        # Select embeddings for observation prediction (all but the last obs token)
        # This aligns with predicting the next token in the sequence
        obs_embs_for_pred = x_structured[..., :-1, :] 
        
        # Select action embeddings for reward and end prediction
        act_embs_for_pred = x_structured[..., -1, :]

        # Flatten the time and token dimensions for the prediction heads
        B, T, L_pred, C = obs_embs_for_pred.shape
        logits_observations = self.head_observations(obs_embs_for_pred.reshape(B, T * L_pred, C))
        logits_rewards = self.head_rewards(act_embs_for_pred)
        logits_ends = self.head_ends(act_embs_for_pred)

        output = WorldModelOutput(
            output_sequence=x_flat,
            logits_observations=logits_observations,
            logits_rewards=logits_rewards,
            logits_ends=logits_ends
        )
        return output, new_keys_values

# --- Standalone Functions for Loss Calculation (JAX Idiom) ---

def _masked_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    labels_one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, labels_one_hot)
    mask = (labels != -100)
    return (loss * mask).sum() / jnp.maximum(mask.sum(), 1)

def compute_labels_world_model(
    obs_tokens: jnp.ndarray,
    rewards: jnp.ndarray,
    ends: jnp.ndarray,
    mask_padding: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes the ground-truth labels for the world model's predictions."""
    mask_fill = jnp.logical_not(mask_padding)
    
    # Observation labels are the next token in the sequence
    labels_obs_unraveled = rearrange(obs_tokens, 'b t k -> b (t k)')
    labels_obs_shifted = labels_obs_unraveled[:, 1:]
    
    # Mask out padding tokens
    mask_fill_obs = jnp.repeat(mask_fill, obs_tokens.shape[-1], axis=1)[:, :-1]
    labels_observations = jnp.where(mask_fill_obs, -100, labels_obs_shifted).flatten()
    
    # Reward and end labels
    labels_rewards = (jnp.sign(rewards) + 1).astype(jnp.int32)
    labels_rewards = jnp.where(mask_fill, -100, labels_rewards).flatten()
    labels_ends = jnp.where(mask_fill, -100, ends.astype(jnp.int32)).flatten()

    return labels_observations, labels_rewards, labels_ends

def compute_wm_loss(
    world_model_params: Any,
    tokenizer_params: Any,
    world_model: WorldModel,
    tokenizer: Tokenizer,
    batch: dict,
    rngs: dict
) -> LossWithIntermediateLosses:
    # 1. Tokenize observations
    obs_5d = batch['observations']
    B, T, H, W, C = obs_5d.shape
    obs_4d = obs_5d.reshape(B * T, H, W, C)

    encode_output = tokenizer.apply(
        {'params': tokenizer_params},
        obs_4d,
        train=False,
        method=tokenizer.encode
    )
    obs_tokens = encode_output.tokens.reshape(B, T, -1)

    # 2. Prepare the input sequence for the World Model
    act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
    tokens = rearrange(jnp.concatenate((obs_tokens, act_tokens), axis=2), 'b l k1 -> b (l k1)')

    # 3. Forward pass through the World Model
    outputs, _ = world_model.apply(
        {'params': world_model_params},
        tokens,
        train=True,
        rngs=rngs
    )

    # 4. Compute ground-truth labels
    labels_observations, labels_rewards, labels_ends = compute_labels_world_model(
        obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding']
    )
    
    # 5. Compute individual losses

    # The logits tensor has predictions for a sequence of length N,
    # but labels are for a sequence of N-1. We slice off the last prediction.
    logits_obs_for_loss = outputs.logits_observations[:, :-1, :]

    loss_obs = _masked_cross_entropy(
        rearrange(logits_obs_for_loss, 'b s v -> (b s) v'), # Note the new rearrange pattern
        labels_observations
    )

    loss_rewards = _masked_cross_entropy(
        rearrange(outputs.logits_rewards, 'b t e -> (b t) e'),
        labels_rewards
    )
    loss_ends = _masked_cross_entropy(
        rearrange(outputs.logits_ends, 'b t e -> (b t) e'),
        labels_ends
    )

    total_loss = loss_obs + loss_rewards + loss_ends

    return LossWithIntermediateLosses(
        total_loss=total_loss,
        loss_obs=loss_obs,
        loss_rewards=loss_rewards,
        loss_ends=loss_ends
    )
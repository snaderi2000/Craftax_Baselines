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
        jax.debug.print(
            "[POS EMB] prev_steps={prev}, num_steps={steps}, pos_indices_head={pi}",
            prev=prev_steps,
            steps=num_steps,
            pi=pos_indices[:5],
        )
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
    obs_tokens: jnp.ndarray,   # (B, T, K)
    rewards: jnp.ndarray,      # (B, T)
    ends: jnp.ndarray,         # (B, T)  (0/1)
    mask_padding: jnp.ndarray  # (B, T)  bool
):
    """
      - Flatten obs tokens to (B, T*K), shift by 1 for next-token targets
      - Use -100 to ignore loss where mask is True
      - Rewards mapped to {-1,0,1} -> {0,1,2}
    NOTE: This assumes mask_padding==True means 'keep' (valid). If in your pipeline
          mask_padding==True means 'padding', flip the NOT below.
    """
    # At most one 'done' per sequence (optional guard)
    # assert bool(jnp.all(jnp.sum(ends, axis=1) <= 1)), "Each sequence should have ≤1 done."
    jax.debug.print(
        "[CHECK] max done per sequence = {x}",
        x=jnp.max(jnp.sum(ends, axis=1))
    )

    # PyTorch used: mask_fill = ~mask_padding, and masked_fill(..., -100)
    # Here: mask_fill == True where we want to fill (ignore) with -100
    mask_fill = jnp.logical_not(mask_padding)   # flip this if your convention differs

    # ---- Observation token labels ----
    # Broadcast mask_fill to obs_token shape (B, T, K)
    mask_fill_obs = jnp.broadcast_to(mask_fill[..., None], obs_tokens.shape)
    # Set ignored positions to -100 *before* flattening & shifting
    obs_tokens_masked = jnp.where(mask_fill_obs, -100, obs_tokens)
    # Flatten to (B, T*K) then shift by 1 (global next-token)
    labels_observations = rearrange(obs_tokens_masked, 'b t k -> b (t k)')[:, 1:]
    labels_observations = labels_observations.reshape(-1)  # (B*(T*K-1),)

    # ---- Reward labels ----
    labels_rewards = (jnp.sign(rewards) + 1).astype(jnp.int32)  # {-1,0,1}->{0,1,2}
    labels_rewards = jnp.where(mask_fill, -100, labels_rewards).reshape(-1)

    # ---- End (done) labels ----
    labels_ends = ends.astype(jnp.int32)
    labels_ends = jnp.where(mask_fill, -100, labels_ends).reshape(-1)

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
    obs_4d = obs_5d.reshape(B * T, H, W, C).astype(jnp.float32) / 255.0 #normalized 

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
    # Debug shapes of inputs & labels before computing losses
    jax.debug.print(
        "[WM LOSS] Obs tokens shape=({ob},{ot},{ok}), Actions shape=({ab},{at}), Rewards shape=({rb},{rt}), Ends shape=({eb},{et})",
        ob=jnp.asarray(obs_tokens.shape[0]),
        ot=jnp.asarray(obs_tokens.shape[1]),
        ok=jnp.asarray(obs_tokens.shape[2]),
        ab=jnp.asarray(batch['actions'].shape[0]),
        at=jnp.asarray(batch['actions'].shape[1]),
        rb=jnp.asarray(batch['rewards'].shape[0]),
        rt=jnp.asarray(batch['rewards'].shape[1]),
        eb=jnp.asarray(batch['ends'].shape[0]),
        et=jnp.asarray(batch['ends'].shape[1]),
    )
    jax.debug.print(
        "[WM LOSS] Labels: Obs={lo_shape}, Rewards={lr_shape}, Ends={le_shape}",
        lo_shape=jnp.asarray(labels_observations.shape),
        lr_shape=jnp.asarray(labels_rewards.shape),
        le_shape=jnp.asarray(labels_ends.shape),
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
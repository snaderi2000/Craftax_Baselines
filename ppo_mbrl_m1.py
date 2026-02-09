"""
M1 Model-Based RL Implementation for Craftax
Based on "Improving Transformer World Models for Data-Efficient RL"

Uses the RNN-based IMPALA policy from ppo_m1_best.py (ActorCriticRNN)

PURE M1 (NOT Dyna):
- Before T_BP: Collect data, train VQ-VAE, train TWM
- After T_BP: Train policy ONLY on imagined data from TWM
- Real environment data is ONLY used to train the world model, NOT the policy

MEMORY OPTIMIZED VERSION:
- Uses separate JIT functions instead of one giant JIT
- Python outer loop to avoid tracing entire training
- Each step is JIT compiled independently

Algorithm 1 from the paper (M1):
1. Collect data from environment → replay buffer (for WM training)
2. Update world model (VQ-VAE + TWM) on real data
3. Update policy ONLY on imagined data (after T_BP steps)
"""

import argparse
import os
import sys
import time
import pickle
import functools
from typing import NamedTuple, Dict, Any, Tuple
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import distrax
from flax import struct
from flax.training.train_state import TrainState
from flax.training import orbax_utils
from flax.linen.initializers import constant, orthogonal
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

# Craftax
from craftax.craftax_env import make_craftax_env_from_name

# Local modules
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from logz.batch_logging import create_log_dict, batch_log

# World model modules
from token_wm.tokenizer.vqvae import VQVAE
from token_wm.twm.world_model import WorldModel
from token_wm.twm.transformer import TransformerConfig
from token_wm.twm.kv_caching import KeysValues

import wandb


# =============================================================================
# Network Architecture (from ppo_m1_best.py)
# =============================================================================

class ImpalaResBlock(nn.Module):
    channels: int
    groups: int = 32
    
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.relu(x)
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        return x + residual


class ImpalaStack(nn.Module):
    channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ImpalaResBlock(self.channels)(x)
        x = ImpalaResBlock(self.channels)(x)
        return x


class DenseResBlock(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Dense(self.width, kernel_init=orthogonal(2))(x)
        return x + residual


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    """RNN-based Actor-Critic network (IMPALA + GRU)."""
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        # 1. IMPALA CNN Encoder
        x_enc = obs.astype(jnp.float32)
        for ch in (64, 64, 128):
            x_enc = ImpalaStack(ch)(x_enc)
        x_enc = nn.relu(x_enc)
        z_t = x_enc.reshape((*x_enc.shape[:2], -1))

        # 2. RNN Bridge
        rnn_input_features = nn.LayerNorm()(z_t)
        rnn_input_features = nn.Dense(256, kernel_init=orthogonal(2))(rnn_input_features)
        rnn_input_features = nn.relu(rnn_input_features)

        # 3. RNN (GRU)
        rnn_in = (rnn_input_features, dones)
        hidden, y_t = ScannedRNN()(hidden, rnn_in)
        y_t = nn.relu(y_t)

        # 4. Concat z_t and y_t
        shared_input = jnp.concatenate([y_t, z_t], axis=-1)

        # 5. Actor Head
        h_actor = nn.LayerNorm()(shared_input)
        h_actor = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h_actor)
        h_actor = nn.relu(h_actor)
        h_actor = DenseResBlock(self.config["LAYER_SIZE"])(h_actor)
        h_actor = DenseResBlock(self.config["LAYER_SIZE"])(h_actor)
        h_actor = nn.relu(h_actor)
        h_actor = nn.LayerNorm()(h_actor)
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h_actor)
        pi = distrax.Categorical(logits=actor_logits)

        # 6. Critic Head
        h_critic = nn.LayerNorm()(shared_input)
        h_critic = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h_critic)
        h_critic = nn.relu(h_critic)
        h_critic = DenseResBlock(self.config["LAYER_SIZE"])(h_critic)
        h_critic = DenseResBlock(self.config["LAYER_SIZE"])(h_critic)
        h_critic = nn.relu(h_critic)
        h_critic = nn.LayerNorm()(h_critic)
        critic_value = nn.Dense(1, kernel_init=orthogonal(1.0))(h_critic)

        return hidden, pi, jnp.squeeze(critic_value, axis=-1)


# =============================================================================
# Data Structures
# =============================================================================

class Transition(NamedTuple):
    """Transition for PPO training."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


# =============================================================================
# World Model Components
# =============================================================================

def create_vqvae(config):
    return VQVAE(
        num_embeddings=config["VQVAE_CODEBOOK_SIZE"],
        embedding_dim=config["VQVAE_EMBED_DIM"],
    )


def create_twm(config):
    twm_config = TransformerConfig(
        tokens_per_block=config["TOKENS_PER_BLOCK"],
        max_blocks=config["TWM_SEQ_LEN"],
        attention='causal',
        num_layers=config["TWM_NUM_LAYERS"],
        num_heads=config["TWM_NUM_HEADS"],
        embed_dim=config["TWM_EMBED_DIM"],
        embed_pdrop=config["TWM_DROPOUT"],
        resid_pdrop=config["TWM_DROPOUT"],
        attn_pdrop=config["TWM_DROPOUT"],
    )
    return WorldModel(
        obs_vocab_size=config["VQVAE_CODEBOOK_SIZE"],
        act_vocab_size=config["NUM_ACTIONS"],
        config=twm_config,
    )


def create_vqvae_train_state(config, rng, sample_obs):
    vqvae = create_vqvae(config)
    params = vqvae.init(rng, sample_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["VQVAE_LR"]),
    )
    return TrainState.create(apply_fn=vqvae.apply, params=params, tx=tx)


def create_twm_train_state(config, rng, sample_tokens):
    twm = create_twm(config)
    params = twm.init(rng, sample_tokens)
    tx = optax.chain(
        optax.clip_by_global_norm(config["TWM_MAX_GRAD_NORM"]),
        optax.adam(config["TWM_LR"]),
    )
    return TrainState.create(apply_fn=twm.apply, params=params, tx=tx)


# =============================================================================
# JIT-Compiled Step Functions (Separate JITs for memory efficiency)
# =============================================================================

def make_env_rollout_fn(env, env_params, network, config):
    """Create JIT-compiled environment rollout function."""
    
    @jax.jit
    def env_rollout(policy_state, env_state, last_obs, last_done, hstate, rng):
        """Collect NUM_STEPS of environment data (for world model training)."""
        
        def _env_step(carry, _):
            policy_state, env_state, last_obs, last_done, hstate, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = network.apply(policy_state.params, hstate, ac_in)
            action = pi.sample(seed=action_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)
            
            obsv, env_state, reward, done, info = env.step(step_rng, env_state, action, env_params)
            
            transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
            return (policy_state, env_state, obsv, done, hstate, rng), transition
        
        initial_hstate = hstate
        carry = (policy_state, env_state, last_obs, last_done, hstate, rng)
        carry, traj = jax.lax.scan(_env_step, carry, None, config["NUM_STEPS"])
        _, env_state, last_obs, last_done, hstate, rng = carry
        
        return traj, env_state, last_obs, last_done, hstate, initial_hstate, rng
    
    return env_rollout


def make_vqvae_update_fn(vqvae, config):
    """Create JIT-compiled VQ-VAE update function."""
    
    def _vqvae_loss_fn(params, obs_batch):
        recon, tokens, total_loss, metrics = vqvae.apply(params, obs_batch, method=vqvae.get_vq_loss)
        return total_loss, metrics
    
    @jax.jit
    def vqvae_update(vqvae_state, obs_batch, num_updates):
        """Update VQ-VAE on observation batch."""
        def _update_step(state, _):
            grad_fn = jax.value_and_grad(_vqvae_loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(state.params, obs_batch)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        vqvae_state, losses = jax.lax.scan(_update_step, vqvae_state, None, num_updates)
        return vqvae_state, losses[-1]
    
    return vqvae_update


def make_twm_update_fn(twm, vqvae, config):
    """Create JIT-compiled TWM update function."""
    
    def _twm_loss_fn(params, obs_tokens, actions, mask):
        """Compute TWM loss on tokenized sequences."""
        # Interleave obs tokens and action tokens: [o0, a0, o1, a1, ...]
        B, T, L = obs_tokens.shape  # (batch, time, tokens_per_frame)
        
        # Build input sequence
        # For each timestep: 64 obs tokens + 1 action token = 65 tokens
        seq_len = T * config["TOKENS_PER_BLOCK"]
        
        # Create interleaved sequence
        input_tokens = jnp.zeros((B, seq_len), dtype=jnp.int32)
        
        # Fill in observation tokens and action tokens
        for t in range(T):
            start_idx = t * config["TOKENS_PER_BLOCK"]
            # Obs tokens for this frame
            input_tokens = input_tokens.at[:, start_idx:start_idx+L].set(obs_tokens[:, t, :])
            # Action token (if not last frame)
            if t < T - 1:
                input_tokens = input_tokens.at[:, start_idx+L].set(actions[:, t])
        
        # Forward pass
        output = twm.apply(params, input_tokens, deterministic=False)
        
        # Compute loss on next-token prediction
        # Target is shifted input
        target_tokens = jnp.roll(input_tokens, -1, axis=1)
        
        # Get observation logits and compute cross-entropy
        obs_logits = output.logits_observations  # (B, seq_len, vocab_size)
        
        # Flatten for loss computation
        logits_flat = obs_logits[:, :-1, :].reshape(-1, obs_logits.shape[-1])
        targets_flat = target_tokens[:, :-1].reshape(-1)
        
        # Cross-entropy loss with mask
        mask_flat = jnp.ones_like(targets_flat, dtype=jnp.float32)  # Simple mask for now
        
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        targets_one_hot = jax.nn.one_hot(targets_flat, obs_logits.shape[-1])
        loss = -jnp.sum(log_probs * targets_one_hot, axis=-1)
        loss = jnp.sum(loss * mask_flat) / jnp.maximum(jnp.sum(mask_flat), 1.0)
        
        return loss
    
    @jax.jit
    def twm_update(twm_state, vqvae_state, obs_batch, actions_batch, rng):
        """Update TWM on observation/action sequences."""
        # Tokenize observations
        B, T = obs_batch.shape[:2]
        obs_flat = obs_batch.reshape(B * T, 63, 63, 3)
        tokens_flat = vqvae.apply(vqvae_state.params, obs_flat, method=vqvae.encode)
        obs_tokens = tokens_flat.reshape(B, T, -1)
        
        # Compute gradient and update
        mask = jnp.ones((B, T))  # Simple mask
        grad_fn = jax.value_and_grad(_twm_loss_fn)
        loss, grads = grad_fn(twm_state.params, obs_tokens, actions_batch, mask)
        twm_state = twm_state.apply_gradients(grads=grads)
        
        return twm_state, loss
    
    return twm_update


def make_imagination_fn(network, vqvae, twm, config):
    """Create JIT-compiled imagination rollout + PPO function (M1 style)."""
    
    def _calculate_gae(traj_batch, last_val, last_done):
        def _get_advantages(carry, transition):
            gae, next_value, next_done = carry
            done, value, reward = transition.done, transition.value, transition.reward
            delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
            return (gae, value, done), gae
        
        _, advantages = jax.lax.scan(
            _get_advantages, (jnp.zeros_like(last_val), last_val, last_done),
            traj_batch, reverse=True, unroll=16,
        )
        return advantages, advantages + traj_batch.value
    
    def _ppo_loss_fn(params, init_hstate, traj_batch, gae, targets):
        _, pi, value = network.apply(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
        log_prob = pi.log_prob(traj_batch.action)
        
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        
        entropy = pi.entropy().mean()
        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        return total_loss, (value_loss, loss_actor, entropy)
    
    def _ppo_update_minbatch(train_state, batch_info):
        init_hstate, traj_batch, advantages, targets = batch_info
        grad_fn = jax.value_and_grad(_ppo_loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(train_state.params, init_hstate, traj_batch, advantages, targets)
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, (loss, aux)
    
    def _ppo_update_epoch(update_state, _):
        policy_state, init_hstate, traj_batch, advantages, targets, rng = update_state
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        rng, perm_rng = jax.random.split(rng)
        N = traj_batch.obs.shape[1]  # num envs dimension
        permutation = jax.random.permutation(perm_rng, N)
        batch = (init_hstate, traj_batch, advantages, targets)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
        
        # Reshape for minibatches
        num_minibatches = config["NUM_MINIBATCHES"]
        minibatches = jax.tree.map(
            lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], num_minibatches, -1] + list(x.shape[2:])), 1, 0),
            shuffled_batch,
        )
        policy_state, losses = jax.lax.scan(_ppo_update_minbatch, policy_state, minibatches)
        return (policy_state, init_hstate, traj_batch, advantages, targets, rng), losses
    
    @jax.jit
    def imagination_step(policy_state, vqvae_state, twm_state, buffer_obs, buffer_size, rng):
        """
        M1: Single imagination rollout + PPO update.
        Policy is trained ONLY on imagined data.
        """
        N = config["IMAGINATION_BATCH_SIZE"]
        rng, sample_rng, imagine_rng = jax.random.split(rng, 3)
        
        # Sample starting states from buffer
        sample_idx = jax.random.randint(sample_rng, (N,), 0, jnp.maximum(buffer_size, 1))
        start_obs = buffer_obs[sample_idx]
        start_done = jnp.zeros(N)
        start_hstate = ScannedRNN.initialize_carry(N, 256)
        
        # Tokenize starting observation
        start_tokens = vqvae.apply(vqvae_state.params, start_obs, method=vqvae.encode)
        
        # Initialize KV cache
        cache = KeysValues.init(
            n=N,
            num_heads=config["TWM_NUM_HEADS"],
            max_tokens=config["TWM_SEQ_LEN"] * config["TOKENS_PER_BLOCK"],
            embed_dim=config["TWM_EMBED_DIM"],
            num_layers=config["TWM_NUM_LAYERS"],
        )
        
        # Feed initial observation to TWM
        _, cache = twm.apply(twm_state.params, start_tokens, past_keys_values=cache, deterministic=True)
        
        # Imagination loop
        def _imagine_step(carry, _):
            current_obs, hstate, current_done, cache, rng = carry
            rng, action_rng, gen_rng = jax.random.split(rng, 3)
            
            # Policy takes action
            ac_in = (current_obs[np.newaxis, :], current_done[np.newaxis, :])
            hstate, pi, value = network.apply(policy_state.params, hstate, ac_in)
            action = pi.sample(seed=action_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)
            
            # Feed action to TWM
            action_tokens = action.reshape(N, 1)
            output, cache = twm.apply(twm_state.params, action_tokens, past_keys_values=cache, deterministic=True)
            
            # Sample reward and done from TWM predictions
            rew_logits = output.logits_rewards[:, -1, :]
            done_logits = output.logits_ends[:, -1, :]
            
            rng, rew_rng, done_rng = jax.random.split(rng, 3)
            # Reward: probability of positive reward
            reward_prob = jax.nn.softmax(rew_logits)[:, 1]
            reward = reward_prob  # Use soft reward for smoother gradients
            
            # Done: probability of episode end
            done_prob = jax.nn.softmax(done_logits)[:, 1]
            new_done = jax.random.bernoulli(done_rng, done_prob).astype(jnp.float32)
            
            # Generate next observation tokens autoregressively
            obs_logits = output.logits_observations[:, -1, :]
            next_token = jax.random.categorical(gen_rng, obs_logits / config["TWM_TEMPERATURE"], axis=-1)
            
            def _gen_token(carry, _):
                token, cache, rng = carry
                token_input = token.reshape(N, 1)
                output, cache = twm.apply(twm_state.params, token_input, past_keys_values=cache, deterministic=True)
                rng, gen_rng = jax.random.split(rng)
                obs_logits = output.logits_observations[:, -1, :]
                next_tok = jax.random.categorical(gen_rng, obs_logits / config["TWM_TEMPERATURE"], axis=-1)
                return (next_tok, cache, rng), next_tok
            
            (last_token, cache, rng), generated_tokens = jax.lax.scan(
                _gen_token, (next_token, cache, rng), None, 63
            )
            
            # Feed last token to complete frame
            last_input = last_token.reshape(N, 1)
            _, cache = twm.apply(twm_state.params, last_input, past_keys_values=cache, deterministic=True)
            
            # Stack tokens and decode
            all_next_tokens = jnp.concatenate([next_token.reshape(N, 1), generated_tokens.T], axis=1)
            next_obs = vqvae.apply(vqvae_state.params, all_next_tokens, method=vqvae.decode_tokens)
            next_obs = next_obs[:, :63, :63, :]
            
            transition = Transition(
                done=current_done, action=action, value=value,
                reward=reward, log_prob=log_prob, obs=current_obs, info=None,
            )
            return (next_obs, hstate, new_done, cache, rng), transition
        
        carry = (start_obs, start_hstate, start_done, cache, imagine_rng)
        carry, imag_traj = jax.lax.scan(_imagine_step, carry, None, config["TWM_ROLLOUT_LEN"])
        final_obs, final_hstate, final_done, _, _ = carry
        
        # Get final value
        ac_in = (final_obs[np.newaxis, :], final_done[np.newaxis, :])
        _, _, last_val = network.apply(policy_state.params, final_hstate, ac_in)
        last_val = last_val.squeeze(0)
        
        # PPO on imagined data (M1: this is the ONLY policy training)
        imag_advantages, imag_targets = _calculate_gae(imag_traj, last_val, final_done)
        init_hstate_batch = start_hstate[None, :]
        
        rng, ppo_rng = jax.random.split(rng)
        ppo_state = (policy_state, init_hstate_batch, imag_traj, imag_advantages, imag_targets, ppo_rng)
        ppo_state, _ = jax.lax.scan(_ppo_update_epoch, ppo_state, None, config["UPDATE_EPOCHS_WM"])
        policy_state = ppo_state[0]
        
        return policy_state, rng
    
    return imagination_step


def make_buffer_update_fn(config):
    """Create JIT-compiled buffer update function."""
    
    @jax.jit
    def update_buffer(buffer_obs, buffer_actions, buffer_rewards, buffer_dones, 
                      buffer_ptr, buffer_size, traj_obs, traj_actions, traj_rewards, traj_dones):
        """Update circular buffer with new trajectory data."""
        num_new = traj_obs.shape[0]
        buffer_max = config["BUFFER_SIZE"]
        indices = (jnp.arange(num_new) + buffer_ptr) % buffer_max
        
        buffer_obs = buffer_obs.at[indices].set(traj_obs)
        buffer_actions = buffer_actions.at[indices].set(traj_actions)
        buffer_rewards = buffer_rewards.at[indices].set(traj_rewards)
        buffer_dones = buffer_dones.at[indices].set(traj_dones)
        
        new_ptr = (buffer_ptr + num_new) % buffer_max
        new_size = jnp.minimum(buffer_size + num_new, buffer_max)
        
        return buffer_obs, buffer_actions, buffer_rewards, buffer_dones, new_ptr, new_size
    
    return update_buffer


# =============================================================================
# Main Training Function
# =============================================================================

def run_mbrl(config):
    """Run M1 MBRL training (pure imagination, NOT Dyna)."""
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config["TOKENS_PER_BLOCK"] = 65
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"M1-MBRL-{config['ENV_NAME']}-{int(config['TOTAL_TIMESTEPS']//1e6)}M",
        )

    print("\n" + "="*60)
    print("M1 MBRL Training (Pure Imagination - NOT Dyna)")
    print("="*60)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Num envs: {config['NUM_ENVS']}")
    print(f"Num updates: {config['NUM_UPDATES']}")
    print(f"Background planning starts at: {config['BACKGROUND_PLANNING_START']:,}")
    print(f"Imagination rollout length: {config['TWM_ROLLOUT_LEN']}")
    print(f"Imagination batch size: {config['IMAGINATION_BATCH_SIZE']}")
    print("="*60)
    print("NOTE: Policy is trained ONLY on imagined data (after T_BP)")
    print("      Real data is used ONLY for world model training")
    print("="*60 + "\n")

    # =========================================================================
    # Setup Environment and Networks
    # =========================================================================
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params
    config["NUM_ACTIONS"] = env.action_space(env_params).n

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env, num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    # Create networks
    network = ActorCriticRNN(env.action_space(env_params).n, config=config)
    vqvae = create_vqvae(config)
    twm = create_twm(config)

    # =========================================================================
    # Initialize States
    # =========================================================================
    rng = jax.random.PRNGKey(config["SEED"])
    
    # Policy
    rng, net_rng = jax.random.split(rng)
    init_x = (jnp.zeros((1, config["NUM_ENVS"], 63, 63, 3)), jnp.zeros((1, config["NUM_ENVS"])))
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
    network_params = network.init(net_rng, init_hstate, init_x)
    
    param_count = sum(x.size for x in jax.tree.leaves(network_params))
    print(f"Policy parameter count: {param_count:,}")
    
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    policy_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    
    # VQ-VAE
    rng, vqvae_rng = jax.random.split(rng)
    vqvae_state = create_vqvae_train_state(config, vqvae_rng, jnp.zeros((1, 63, 63, 3)))
    
    # Load pre-trained VQ-VAE if specified
    if config["USE_PRETRAINED_VQVAE"] and config["VQVAE_CHECKPOINT"]:
        print(f"Loading pre-trained VQ-VAE from: {config['VQVAE_CHECKPOINT']}")
        with open(config["VQVAE_CHECKPOINT"], 'rb') as f:
            loaded_params = pickle.load(f)
        vqvae_state = vqvae_state.replace(params=loaded_params)
        print("VQ-VAE loaded successfully!")
    
    # TWM
    rng, twm_rng = jax.random.split(rng)
    twm_state = create_twm_train_state(config, twm_rng, jnp.zeros((1, config["TOKENS_PER_BLOCK"]), dtype=jnp.int32))
    
    # Environment
    rng, env_rng = jax.random.split(rng)
    obsv, env_state = env.reset(env_rng, env_params)
    hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
    last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
    
    # Buffer (for world model training only in M1)
    buffer_size = config["BUFFER_SIZE"]
    buffer_obs = jnp.zeros((buffer_size, 63, 63, 3))
    buffer_actions = jnp.zeros((buffer_size,), dtype=jnp.int32)
    buffer_rewards = jnp.zeros((buffer_size,))
    buffer_dones = jnp.zeros((buffer_size,))
    buffer_ptr = jnp.array(0, dtype=jnp.int32)
    buffer_count = jnp.array(0, dtype=jnp.int32)
    
    # Sequence buffer for TWM training (stores sequences)
    seq_buffer_size = config["SEQ_BUFFER_SIZE"]
    seq_len = config["TWM_SEQ_LEN"]
    seq_buffer_obs = jnp.zeros((seq_buffer_size, seq_len, 63, 63, 3))
    seq_buffer_actions = jnp.zeros((seq_buffer_size, seq_len), dtype=jnp.int32)
    seq_buffer_ptr = jnp.array(0, dtype=jnp.int32)
    seq_buffer_count = jnp.array(0, dtype=jnp.int32)

    # =========================================================================
    # Create JIT Functions
    # =========================================================================
    env_rollout = make_env_rollout_fn(env, env_params, network, config)
    vqvae_update = make_vqvae_update_fn(vqvae, config)
    twm_update = make_twm_update_fn(twm, vqvae, config)
    imagination_step = make_imagination_fn(network, vqvae, twm, config)
    update_buffer = make_buffer_update_fn(config)

    # =========================================================================
    # Training Loop (Python outer loop - not JIT traced!)
    # =========================================================================
    total_steps = 0
    t0 = time.time()
    imagination_started = False
    
    # Accumulators for sequence building
    current_sequences_obs = []
    current_sequences_actions = []
    
    pbar = tqdm(range(config["NUM_UPDATES"]), desc="Training")
    for update_idx in pbar:
        # ---------------------------------------------------------------------
        # Step 1: Environment Rollout (for data collection only)
        # ---------------------------------------------------------------------
        traj, env_state, obsv, last_done, hstate, initial_hstate, rng = env_rollout(
            policy_state, env_state, obsv, last_done, hstate, rng
        )
        total_steps += config["NUM_ENVS"] * config["NUM_STEPS"]
        
        # ---------------------------------------------------------------------
        # Step 2: Update Buffer (for world model training)
        # ---------------------------------------------------------------------
        traj_obs = traj.obs.reshape(-1, 63, 63, 3)
        traj_actions = traj.action.reshape(-1)
        traj_rewards = traj.reward.reshape(-1)
        traj_dones = traj.done.reshape(-1)
        
        buffer_obs, buffer_actions, buffer_rewards, buffer_dones, buffer_ptr, buffer_count = update_buffer(
            buffer_obs, buffer_actions, buffer_rewards, buffer_dones,
            buffer_ptr, buffer_count, traj_obs, traj_actions, traj_rewards, traj_dones
        )
        
        # Build sequences for TWM training
        # Reshape to (num_envs, num_steps, ...)
        env_obs = traj.obs.transpose(1, 0, 2, 3, 4)  # (num_envs, num_steps, H, W, C)
        env_actions = traj.action.transpose(1, 0)  # (num_envs, num_steps)
        
        # Add sequences to buffer if we have enough steps
        if config["NUM_STEPS"] >= seq_len:
            for i in range(config["NUM_ENVS"]):
                # Take first seq_len steps as a sequence
                seq_obs = env_obs[i, :seq_len]
                seq_actions = env_actions[i, :seq_len]
                
                # Add to sequence buffer
                idx = int(seq_buffer_ptr) % seq_buffer_size
                seq_buffer_obs = seq_buffer_obs.at[idx].set(seq_obs)
                seq_buffer_actions = seq_buffer_actions.at[idx].set(seq_actions)
                seq_buffer_ptr = (seq_buffer_ptr + 1) % seq_buffer_size
                seq_buffer_count = jnp.minimum(seq_buffer_count + 1, seq_buffer_size)
        
        # ---------------------------------------------------------------------
        # Step 3: VQ-VAE Update (always, for reconstruction)
        # ---------------------------------------------------------------------
        if not config["USE_PRETRAINED_VQVAE"]:
            vqvae_state, vqvae_loss = vqvae_update(vqvae_state, traj_obs, config["VQVAE_UPDATES_PER_ITER"])
        else:
            vqvae_loss = 0.0
        
        # ---------------------------------------------------------------------
        # Step 4: TWM Update (train world model on real data)
        # ---------------------------------------------------------------------
        twm_loss = 0.0
        if int(seq_buffer_count) >= config["TWM_BATCH_SIZE"]:
            # Sample batch of sequences
            rng, sample_rng = jax.random.split(rng)
            batch_idx = jax.random.randint(sample_rng, (config["TWM_BATCH_SIZE"],), 0, int(seq_buffer_count))
            batch_obs = seq_buffer_obs[batch_idx]
            batch_actions = seq_buffer_actions[batch_idx]
            
            twm_state, twm_loss = twm_update(twm_state, vqvae_state, batch_obs, batch_actions, rng)
        
        # ---------------------------------------------------------------------
        # Step 5: Imagination + Policy Update (M1: ONLY source of policy training)
        # ---------------------------------------------------------------------
        if total_steps >= config["BACKGROUND_PLANNING_START"]:
            if not imagination_started:
                print(f"\n*** Starting imagination-based policy training at step {total_steps:,} ***\n")
                imagination_started = True
            
            # M1: Policy is trained ONLY on imagined data
            policy_state, rng = imagination_step(
                policy_state, vqvae_state, twm_state, buffer_obs, buffer_count, rng
            )
        
        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------
        if update_idx % 10 == 0:
            # Compute metrics from real environment (for monitoring only)
            returned = traj.info["returned_episode"]
            if returned.sum() > 0:
                avg_return = (traj.info["returned_episode_returns"] * returned).sum() / returned.sum()
            else:
                avg_return = 0.0
            
            status = "WM-only" if not imagination_started else "Imagination"
            pbar.set_postfix({
                'mode': status,
                'steps': f'{total_steps:,}',
                'return': f'{float(avg_return):.2f}',
                'vq': f'{float(vqvae_loss):.3f}',
                'twm': f'{float(twm_loss):.3f}',
            })
            
            if config["USE_WANDB"]:
                log_dict = {
                    'step': total_steps,
                    'return': float(avg_return),
                    'vqvae_loss': float(vqvae_loss),
                    'twm_loss': float(twm_loss),
                    'buffer_size': int(buffer_count),
                    'imagination_active': imagination_started,
                }
                wandb.log(log_dict)
    
    t1 = time.time()
    print(f"\nTraining complete!")
    print(f"Time: {t1 - t0:.2f}s")
    print(f"SPS: {config['TOTAL_TIMESTEPS'] / (t1 - t0):.0f}")
    
    # Save checkpoints
    if config["SAVE_POLICY"]:
        save_dir = f"checkpoints/m1_mbrl_{config['ENV_NAME']}_{config['SEED']}"
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/policy_params.pkl", 'wb') as f:
            pickle.dump(policy_state.params, f)
        with open(f"{save_dir}/vqvae_params.pkl", 'wb') as f:
            pickle.dump(vqvae_state.params, f)
        with open(f"{save_dir}/twm_params.pkl", 'wb') as f:
            pickle.dump(twm_state.params, f)
        print(f"Saved checkpoints to {save_dir}/")
    
    return {
        'policy_state': policy_state,
        'vqvae_state': vqvae_state,
        'twm_state': twm_state,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 MBRL for Craftax (Pure Imagination)")
    
    # Environment
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    parser.add_argument("--num_envs", type=int, default=48)
    parser.add_argument("--num_steps", type=int, default=64)
    
    # Training
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e7)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--update_epochs_wm", type=int, default=2)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=False)
    
    # VQ-VAE
    parser.add_argument("--vqvae_codebook_size", type=int, default=512)
    parser.add_argument("--vqvae_embed_dim", type=int, default=128)
    parser.add_argument("--vqvae_lr", type=float, default=0.001)
    parser.add_argument("--vqvae_updates_per_iter", type=int, default=3)
    parser.add_argument("--use_pretrained_vqvae", action="store_true")
    parser.add_argument("--vqvae_checkpoint", type=str, default="")
    
    # TWM
    parser.add_argument("--twm_seq_len", type=int, default=20)
    parser.add_argument("--twm_num_layers", type=int, default=3)
    parser.add_argument("--twm_num_heads", type=int, default=8)
    parser.add_argument("--twm_embed_dim", type=int, default=128)
    parser.add_argument("--twm_dropout", type=float, default=0.1)
    parser.add_argument("--twm_lr", type=float, default=0.001)
    parser.add_argument("--twm_max_grad_norm", type=float, default=0.5)
    parser.add_argument("--twm_rollout_len", type=int, default=15)
    parser.add_argument("--twm_temperature", type=float, default=1.0)
    parser.add_argument("--twm_batch_size", type=int, default=16)
    
    # MBRL
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--seq_buffer_size", type=int, default=2000)
    parser.add_argument("--background_planning_start", type=int, default=200000)
    parser.add_argument("--imagination_batch_size", type=int, default=32)
    
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="craftax-mbrl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--save_policy", action="store_true")
    
    # Smoke test
    parser.add_argument("--smoke_test", action="store_true")
    
    args, rest = parser.parse_known_args()
    if rest:
        print(f"Warning: Unknown args: {rest}")
    
    if args.smoke_test:
        print("\n*** SMOKE TEST MODE ***\n")
        args.total_timesteps = 50000
        args.num_envs = 8
        args.num_steps = 32
        args.vqvae_updates_per_iter = 1
        args.background_planning_start = 10000
        args.buffer_size = 5000
        args.seq_buffer_size = 200
        args.twm_batch_size = 4
        args.imagination_batch_size = 8
        args.use_wandb = False
    
    if args.seed is None:
        args.seed = np.random.randint(2**31)
    
    run_mbrl(args)

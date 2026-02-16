"""
M1 Model-Based RL Implementation for Craftax
Based on "Improving Transformer World Models for Data-Efficient RL"

Uses the RNN-based IMPALA policy from ppo_m1_best.py (ActorCriticRNN)

PURE M1 (NOT Dyna):
- From step 0: Collect data, train VQ-VAE AND TWM
- After T_BP: Train policy ONLY on imagined data from TWM
- Real environment data is ONLY used to train the world model, NOT the policy

Paper's training counts per iteration:
- N_iters_tok = 500 (tokenizer updates)
- N_iters_TWM = 500 (TWM updates)
- N_mb_training_WM = 3 (minibatches per update)
- N_iters_AC = 150 (imagination policy updates after T_BP)

MEMORY OPTIMIZED VERSION:
- Uses separate JIT functions instead of one giant JIT
- Python outer loop to avoid tracing entire training
"""

import argparse
import os
import sys
import time
import pickle
import functools
import signal
import shutil
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

import flashbax as fbx
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
    """Create JIT-compiled VQ-VAE single update function."""
    
    def _vqvae_loss_fn(params, obs_batch):
        recon, tokens, total_loss, metrics = vqvae.apply(params, obs_batch, method=vqvae.get_vq_loss)
        return total_loss, metrics
    
    @jax.jit
    def vqvae_update_single(vqvae_state, obs_batch):
        """Single VQ-VAE update step."""
        grad_fn = jax.value_and_grad(_vqvae_loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(vqvae_state.params, obs_batch)
        vqvae_state = vqvae_state.apply_gradients(grads=grads)
        return vqvae_state, loss
    
    return vqvae_update_single


def make_twm_update_fn(twm, vqvae, config):
    """Create JIT-compiled TWM single update function.
    
    Uses WorldModel.compute_loss which trains ALL three heads:
      1. Observation prediction (cross-entropy on codebook tokens)
      2. Reward prediction (cross-entropy on {-1, 0, +1} → {0, 1, 2})
      3. Termination prediction (cross-entropy on {continue, end})
    """
    
    def _twm_loss_fn(params, obs_tokens, actions, rewards, dones, dropout_rng):
        """Compute TWM loss using proper compute_loss (obs + reward + termination)."""
        B, T, K = obs_tokens.shape
        batch = {
            'obs_tokens': obs_tokens,       # (B, T, 64)
            'actions': actions,              # (B, T)
            'rewards': rewards,              # (B, T)
            'ends': dones,                   # (B, T)
            'mask_padding': jnp.zeros((B, T), dtype=jnp.bool_),  # No padding
        }
        loss_output = twm.apply(
            params,
            batch,
            dropout_rng,
            method=twm.compute_loss,
            rngs={'dropout': dropout_rng},
        )
        return loss_output.total_loss, (loss_output.loss_obs, loss_output.loss_rewards, loss_output.loss_ends)
    
    @jax.jit
    def twm_update_single(twm_state, vqvae_params, obs_batch, actions_batch, rewards_batch, dones_batch, rng):
        """Single TWM update step with obs + reward + termination losses."""
        B, T = obs_batch.shape[:2]
        obs_flat = obs_batch.reshape(B * T, 63, 63, 3)
        tokens_flat = vqvae.apply(vqvae_params, obs_flat, method=vqvae.encode)
        obs_tokens = tokens_flat.reshape(B, T, -1)
        
        rng, dropout_rng = jax.random.split(rng)
        
        grad_fn = jax.value_and_grad(_twm_loss_fn, has_aux=True)
        (loss, (loss_obs, loss_rew, loss_ends)), grads = grad_fn(
            twm_state.params, obs_tokens, actions_batch, rewards_batch, dones_batch, dropout_rng
        )
        twm_state = twm_state.apply_gradients(grads=grads)
        
        return twm_state, loss, (loss_obs, loss_rew, loss_ends), rng
    
    return twm_update_single


def make_imagination_fn(network, vqvae, twm, config):
    """Create JIT-compiled imagination rollout + PPO function (M1 style)."""
    
    # Use paper's settings for imagination PPO
    n_mb_imagination = config.get("N_MB_IMAGINATION", 1)
    n_epoch_imagination = config.get("N_EPOCH_IMAGINATION", 1)
    
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
        N = traj_batch.obs.shape[1]
        permutation = jax.random.permutation(perm_rng, N)
        batch = (init_hstate, traj_batch, advantages, targets)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
        
        # Paper: N_mb_WM = 1 for imagination
        num_minibatches = n_mb_imagination
        minibatches = jax.tree.map(
            lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], num_minibatches, -1] + list(x.shape[2:])), 1, 0),
            shuffled_batch,
        )
        policy_state, losses = jax.lax.scan(_ppo_update_minbatch, policy_state, minibatches)
        return (policy_state, init_hstate, traj_batch, advantages, targets, rng), losses
    
    @jax.jit
    def imagination_step(policy_state, vqvae_params, twm_state, buffer_obs, buffer_size, rng):
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
        start_tokens = vqvae.apply(vqvae_params, start_obs, method=vqvae.encode)
        
        # Initialize KV cache
        # Must match transformer's internal causal mask size = TWM_SEQ_LEN * TOKENS_PER_BLOCK
        # The initial obs frame (64 tokens) + TWM_ROLLOUT_LEN * 65 must fit within this.
        # This is enforced by setting TWM_ROLLOUT_LEN = TWM_SEQ_LEN - 1.
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
            rng, action_rng, gen_rng, rew_rng, done_rng = jax.random.split(rng, 5)
            
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
            
            # Reward: sample from 3-class categorical {0,1,2} -> {-1,0,+1}
            sampled_rew_class = jax.random.categorical(rew_rng, rew_logits, axis=-1)
            reward = (sampled_rew_class - 1).astype(jnp.float32)
            
            # Done: sample from 2-class categorical (class 0=continue, class 1=done)
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
            next_obs = vqvae.apply(vqvae_params, all_next_tokens, method=vqvae.decode_tokens)
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
        # Paper: N_epoch_WM = 1 for imagination
        ppo_state, _ = jax.lax.scan(_ppo_update_epoch, ppo_state, None, n_epoch_imagination)
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
# Checkpointing
# =============================================================================

def save_checkpoint(ckpt_dir, step, policy_state, vqvae_state, twm_state, 
                   fbx_buffer_state, buffer_data, metadata, max_checkpoints=2):
    """Save training state to checkpoint directory and rotate old checkpoints."""
    step_dir = os.path.join(ckpt_dir, f"checkpoint_{step}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Save models
    # We use pickle for simplicity as it handles JAX arrays (pulling to host if needed)
    with open(os.path.join(step_dir, "policy_state.pkl"), "wb") as f:
        pickle.dump(policy_state, f)
    with open(os.path.join(step_dir, "vqvae_state.pkl"), "wb") as f:
        pickle.dump(vqvae_state, f)
    with open(os.path.join(step_dir, "twm_state.pkl"), "wb") as f:
        pickle.dump(twm_state, f)
        
    # Save buffers
    # buffer_data is a dict or tuple
    with open(os.path.join(step_dir, "flat_buffer.pkl"), "wb") as f:
        pickle.dump(buffer_data, f)
        
    # Save Flashbax buffer state
    with open(os.path.join(step_dir, "fbx_buffer.pkl"), "wb") as f:
        pickle.dump(fbx_buffer_state, f)
        
    # Save metadata
    with open(os.path.join(step_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"Saved checkpoint to {step_dir}")
    
    # Rotate checkpoints: keep only the most recent `max_checkpoints`
    try:
        checkpoints = []
        for d in os.listdir(ckpt_dir):
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(ckpt_dir, d)):
                try:
                    step_num = int(d.split("_")[1])
                    checkpoints.append((step_num, d))
                except ValueError:
                    continue
        
        # Sort by step number (ascending)
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove old checkpoints if we have more than max_checkpoints
        while len(checkpoints) > max_checkpoints:
            oldest_step, oldest_dir = checkpoints.pop(0)
            oldest_path = os.path.join(ckpt_dir, oldest_dir)
            print(f"Removing old checkpoint: {oldest_path}")
            shutil.rmtree(oldest_path)
            
    except Exception as e:
        print(f"Warning: Failed to rotate checkpoints: {e}")


def load_checkpoint(ckpt_path):
    """Load training state from checkpoint directory."""
    print(f"Loading checkpoint from {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path {ckpt_path} does not exist")

    # Load models
    with open(os.path.join(ckpt_path, "policy_state.pkl"), "rb") as f:
        policy_state = pickle.load(f)
    with open(os.path.join(ckpt_path, "vqvae_state.pkl"), "rb") as f:
        vqvae_state = pickle.load(f)
    with open(os.path.join(ckpt_path, "twm_state.pkl"), "rb") as f:
        twm_state = pickle.load(f)
        
    # Load buffers
    with open(os.path.join(ckpt_path, "flat_buffer.pkl"), "rb") as f:
        buffer_data = pickle.load(f)
        
    with open(os.path.join(ckpt_path, "fbx_buffer.pkl"), "rb") as f:
        fbx_buffer_state = pickle.load(f)
        
    # Load metadata
    with open(os.path.join(ckpt_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
        
    print(f"Loaded checkpoint from step {metadata['total_steps']}")
    return policy_state, vqvae_state, twm_state, fbx_buffer_state, buffer_data, metadata


# =============================================================================
# Main Training Function
# =============================================================================

def run_mbrl(config):
    """Run M1 MBRL training (pure imagination, NOT Dyna)."""
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config["TOKENS_PER_BLOCK"] = 65
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    # Validate: initial obs (64 tokens) + rollout_len * 65 must fit in TWM_SEQ_LEN * 65
    max_rollout = config["TWM_SEQ_LEN"] - 1  # initial frame uses ~1 block
    if config["TWM_ROLLOUT_LEN"] > max_rollout:
        print(f"WARNING: TWM_ROLLOUT_LEN={config['TWM_ROLLOUT_LEN']} too large for TWM_SEQ_LEN={config['TWM_SEQ_LEN']}. "
              f"Clamping to {max_rollout} (initial obs uses 64 of 65 tokens in first block).")
        config["TWM_ROLLOUT_LEN"] = max_rollout

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"M1-MBRL-{config['ENV_NAME']}-{int(config['TOTAL_TIMESTEPS']//1e6)}M",
        )

    print("\n" + "="*70)
    print("M1 MBRL Training (Pure Imagination - NOT Dyna)")
    print("="*70)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Num envs: {config['NUM_ENVS']}")
    print(f"Num updates: {config['NUM_UPDATES']}")
    print(f"Background planning starts at: {config['BACKGROUND_PLANNING_START']:,}")
    print("-"*70)
    print(f"Tokenizer iters per update: {config['N_ITERS_TOK']}")
    print(f"TWM iters per update: {config['N_ITERS_TWM']}")
    print(f"WM minibatches: {config['N_MB_WM']}")
    print(f"Imagination policy updates: {config['N_ITERS_AC']}")
    print(f"Imagination rollout length: {config['TWM_ROLLOUT_LEN']}")
    print(f"Imagination batch size: {config['IMAGINATION_BATCH_SIZE']}")
    print(f"Buffer: flashbax trajectory buffer (max {config['BUFFER_SIZE']:,} transitions)")
    print("-"*70)
    print(f"Pre-trained tokenizer: {config['USE_PRETRAINED_TOKENIZER']}")
    if config['USE_PRETRAINED_TOKENIZER']:
        print(f"  Path: {config['TOKENIZER_PATH']}")
    print("="*70)
    print("NOTE: Policy is trained ONLY on imagined data (after T_BP)")
    print("      Real data is used ONLY for world model training")
    print("="*70 + "\n")

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
    
    # Load pre-trained tokenizer if specified
    if config["USE_PRETRAINED_TOKENIZER"]:
        print(f"Loading pre-trained tokenizer from: {config['TOKENIZER_PATH']}")
        with open(config["TOKENIZER_PATH"], 'rb') as f:
            loaded_params = pickle.load(f)
        vqvae_state = vqvae_state.replace(params=loaded_params)
        print("Pre-trained tokenizer loaded successfully! (frozen, no gradient updates)")
    
    # TWM
    rng, twm_rng = jax.random.split(rng)
    twm_state = create_twm_train_state(config, twm_rng, jnp.zeros((1, config["TOKENS_PER_BLOCK"]), dtype=jnp.int32))
    
    # Environment
    rng, env_rng = jax.random.split(rng)
    obsv, env_state = env.reset(env_rng, env_params)
    hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
    last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
    
    # Flat buffer for VQ-VAE training and imagination starting states
    flat_buffer_size = config["BUFFER_SIZE"]
    buffer_obs = jnp.zeros((flat_buffer_size, 63, 63, 3))
    buffer_actions = jnp.zeros((flat_buffer_size,), dtype=jnp.int32)
    buffer_rewards = jnp.zeros((flat_buffer_size,))
    buffer_dones = jnp.zeros((flat_buffer_size,))
    buffer_ptr = jnp.array(0, dtype=jnp.int32)
    buffer_count = jnp.array(0, dtype=jnp.int32)
    
    # Flashbax trajectory buffer for TWM training (paper: flashbax with 128k max)
    # Stores (NUM_ENVS, time) and samples overlapping random windows of length TWM_SEQ_LEN
    fbx_buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
        min_length_time_axis=config["TWM_SEQ_LEN"] + 1,
        sample_batch_size=config["TWM_BATCH_SIZE"],
        sample_sequence_length=config["TWM_SEQ_LEN"],
        period=1,
        add_batch_size=config["NUM_ENVS"],
    )
    example_timestep = {
        'obs': jnp.zeros((63, 63, 3)),
        'action': jnp.zeros((), dtype=jnp.int32),
        'reward': jnp.zeros(()),
        'done': jnp.zeros(()),
    }
    fbx_buffer_state = fbx_buffer.init(example_timestep)

    # =========================================================================
    # Checkpoint Loading
    # =========================================================================
    start_step = 0
    imagination_started = False
    if config.get("RESUME_FROM"):
        print(f"\n*** Resuming from checkpoint: {config['RESUME_FROM']} ***\n")
        policy_state, vqvae_state, twm_state, fbx_buffer_state, buffer_data, metadata = load_checkpoint(config['RESUME_FROM'])
        
        # Unpack flat buffer
        buffer_obs, buffer_actions, buffer_rewards, buffer_dones, buffer_ptr, buffer_count = buffer_data
        
        start_step = metadata['total_steps']
        # rng = metadata.get('rng', rng) # Restore RNG if available (optional)
        imagination_started = metadata.get('imagination_started', False)
        print(f"Resumed at step {start_step:,}")

    # =========================================================================
    # Create JIT Functions
    # =========================================================================
    env_rollout = make_env_rollout_fn(env, env_params, network, config)
    vqvae_update_single = make_vqvae_update_fn(vqvae, config)
    twm_update_single = make_twm_update_fn(twm, vqvae, config)
    imagination_step = make_imagination_fn(network, vqvae, twm, config)
    update_buffer = make_buffer_update_fn(config)

    # =========================================================================
    # Training Loop (Python outer loop - not JIT traced!)
    # =========================================================================
    total_steps = start_step
    start_update = start_step // (config["NUM_ENVS"] * config["NUM_STEPS"])
    t0 = time.time()
    
    # Signal handler for graceful exit
    exit_requested = False
    def signal_handler(sig, frame):
        nonlocal exit_requested
        print("\n\n*** Signal received, requesting graceful exit... ***\n")
        exit_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    pbar = tqdm(range(start_update, config["NUM_UPDATES"]), desc="Training", initial=start_update, total=config["NUM_UPDATES"])
    for update_idx in pbar:
        # ---------------------------------------------------------------------
        # Step 1: Environment Rollout (for data collection only)
        # ---------------------------------------------------------------------
        traj, env_state, obsv, last_done, hstate, initial_hstate, rng = env_rollout(
            policy_state, env_state, obsv, last_done, hstate, rng
        )
        total_steps += config["NUM_ENVS"] * config["NUM_STEPS"]
        
        # ---------------------------------------------------------------------
        # Step 2: Update Buffers (flat for VQ-VAE/imagination, flashbax for TWM)
        # ---------------------------------------------------------------------
        # Flat buffer update for VQ-VAE and imagination starting states
        traj_obs = traj.obs.reshape(-1, 63, 63, 3)
        traj_actions = traj.action.reshape(-1)
        traj_rewards = traj.reward.reshape(-1)
        traj_dones = traj.done.reshape(-1)
        
        buffer_obs, buffer_actions, buffer_rewards, buffer_dones, buffer_ptr, buffer_count = update_buffer(
            buffer_obs, buffer_actions, buffer_rewards, buffer_dones,
            buffer_ptr, buffer_count, traj_obs, traj_actions, traj_rewards, traj_dones
        )
        
        # Flashbax trajectory buffer update for TWM training
        # Transpose from (NUM_STEPS, NUM_ENVS, ...) to (NUM_ENVS, NUM_STEPS, ...)
        # flashbax add expects (add_batch_size, sequence_length, ...)
        fbx_add_data = {
            'obs': traj.obs.transpose(1, 0, 2, 3, 4),        # (NUM_ENVS, NUM_STEPS, 63, 63, 3)
            'action': traj.action.transpose(1, 0).astype(jnp.int32),  # (NUM_ENVS, NUM_STEPS)
            'reward': traj.reward.transpose(1, 0),             # (NUM_ENVS, NUM_STEPS)
            'done': traj.done.transpose(1, 0).astype(jnp.float32),    # (NUM_ENVS, NUM_STEPS)
        }
        fbx_buffer_state = fbx_buffer.add(fbx_buffer_state, fbx_add_data)
        
        # ---------------------------------------------------------------------
        # Step 3: VQ-VAE Update (N_ITERS_TOK iterations, if not pretrained)
        # Paper: 500 iterations with 3 minibatches each
        # ---------------------------------------------------------------------
        vqvae_loss = 0.0
        if not config["USE_PRETRAINED_TOKENIZER"] and config["N_ITERS_TOK"] > 0:
            for tok_iter in range(config["N_ITERS_TOK"]):
                for mb in range(config["N_MB_WM"]):
                    # Sample minibatch from buffer
                    rng, sample_rng = jax.random.split(rng)
                    mb_size = min(config["VQVAE_BATCH_SIZE"], int(buffer_count))
                    if mb_size > 0:
                        mb_idx = jax.random.randint(sample_rng, (mb_size,), 0, int(buffer_count))
                        obs_mb = buffer_obs[mb_idx]
                        vqvae_state, vqvae_loss = vqvae_update_single(vqvae_state, obs_mb)
        
        # ---------------------------------------------------------------------
        # Step 4: TWM Update (N_ITERS_TWM iterations from step 0)
        # Paper: 500 iterations with 3 minibatches each
        # Flashbax samples overlapping random windows from the full buffer
        # ---------------------------------------------------------------------
        twm_loss = 0.0
        twm_loss_obs = 0.0
        twm_loss_rew = 0.0
        twm_loss_ends = 0.0
        can_sample_twm = fbx_buffer.can_sample(fbx_buffer_state)
        if can_sample_twm and config["N_ITERS_TWM"] > 0:
            for twm_iter in range(config["N_ITERS_TWM"]):
                for mb in range(config["N_MB_WM"]):
                    rng, sample_rng = jax.random.split(rng)
                    fbx_batch = fbx_buffer.sample(fbx_buffer_state, sample_rng)
                    # fbx_batch.experience is a dict with (TWM_BATCH_SIZE, TWM_SEQ_LEN, ...)
                    batch_obs = fbx_batch.experience['obs']
                    batch_actions = fbx_batch.experience['action']
                    batch_rewards = fbx_batch.experience['reward']
                    batch_dones = fbx_batch.experience['done']
                    
                    twm_state, twm_loss, (twm_loss_obs, twm_loss_rew, twm_loss_ends), rng = twm_update_single(
                        twm_state, vqvae_state.params, batch_obs, batch_actions, batch_rewards, batch_dones, rng
                    )
        
        # ---------------------------------------------------------------------
        # Step 5: Imagination + Policy Update (M1: ONLY source of policy training)
        # Paper: N_ITERS_AC = 150 policy updates per iteration after T_BP
        # ---------------------------------------------------------------------
        if total_steps >= config["BACKGROUND_PLANNING_START"]:
            if not imagination_started:
                print(f"\n*** Starting imagination-based policy training at step {total_steps:,} ***\n")
                imagination_started = True
            
            # M1: Policy is trained ONLY on imagined data
            for ac_iter in range(config["N_ITERS_AC"]):
                policy_state, rng = imagination_step(
                    policy_state, vqvae_state.params, twm_state, buffer_obs, buffer_count, rng
                )
        
        # ---------------------------------------------------------------------
        # Logging with achievements/score
        # ---------------------------------------------------------------------
        if True:
            status = "WM-only" if not imagination_started else "Imagination"
            
            # Compute episode-averaged metrics (like ppo_rnn.py)
            # This averages values over completed episodes only
            returned = traj.info["returned_episode"]
            num_returned = returned.sum()
            
            if num_returned > 0:
                # Average all info values over completed episodes
                metric = jax.tree.map(
                    lambda x: (x * returned).sum() / num_returned,
                    traj.info,
                )
                avg_return = float(metric["returned_episode_returns"])
            else:
                # No completed episodes, use zeros
                metric = jax.tree.map(lambda x: jnp.zeros_like(x).mean(), traj.info)
                avg_return = 0.0
            
            # Use create_log_dict with properly averaged metrics
            log_dict = create_log_dict(metric, config)
            score = log_dict.get("score", 0.0)
            
            pbar.set_postfix({
                'mode': status,
                'steps': f'{total_steps:,}',
                'return': f'{float(avg_return):.2f}',
                'score': f'{float(score):.2f}',
                'vq': f'{float(vqvae_loss):.3f}',
                'twm': f'{float(twm_loss):.3f}',
                'twm_r': f'{float(twm_loss_rew):.3f}',
                'twm_d': f'{float(twm_loss_ends):.3f}',
            })
            
            if config["USE_WANDB"]:
                log_dict.update({
                    'step': total_steps,
                    'vqvae_loss': float(vqvae_loss),
                    'twm_loss': float(twm_loss),
                    'twm_loss_obs': float(twm_loss_obs),
                    'twm_loss_rew': float(twm_loss_rew),
                    'twm_loss_ends': float(twm_loss_ends),
                    'buffer_size': int(buffer_count),
                    'imagination_active': imagination_started,
                })
                wandb.log(log_dict)
        
        # ---------------------------------------------------------------------
        # Checkpointing
        # ---------------------------------------------------------------------
        # Check if we crossed a checkpoint threshold
        prev_steps = total_steps - config["NUM_ENVS"] * config["NUM_STEPS"]
        checkpoint_threshold_crossed = (total_steps // config["CHECKPOINT_FREQ"]) > (prev_steps // config["CHECKPOINT_FREQ"])
        
        if (total_steps > 0 and checkpoint_threshold_crossed) or exit_requested:
            print(f"\nSaving checkpoint at step {total_steps:,}...")
            buffer_data = (buffer_obs, buffer_actions, buffer_rewards, buffer_dones, buffer_ptr, buffer_count)
            metadata = {
                'total_steps': total_steps,
                'update_idx': update_idx,
                'rng': rng, # Note: JAX PRNGKey is array, pickle handles it
                'imagination_started': imagination_started,
            }
            save_checkpoint(config["CHECKPOINT_DIR"], total_steps, policy_state, vqvae_state, twm_state, 
                           fbx_buffer_state, buffer_data, metadata, max_checkpoints=config["MAX_CHECKPOINTS"])
            
            if exit_requested:
                print(f"Exiting gracefully at step {total_steps:,}...")
                break
    
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
    parser.add_argument("--num_steps", type=int, default=96)
    
    # Training
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--lr", type=float, default=0.00045)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.925)
    parser.add_argument("--gae_lambda", type=float, default=0.625)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=False)
    
    # VQ-VAE / Tokenizer
    parser.add_argument("--vqvae_codebook_size", type=int, default=512)
    parser.add_argument("--vqvae_embed_dim", type=int, default=128)
    parser.add_argument("--vqvae_lr", type=float, default=0.001)
    parser.add_argument("--vqvae_batch_size", type=int, default=256)
    parser.add_argument("--use_pretrained_tokenizer", action="store_true", 
                        help="Use pre-trained VQ-VAE tokenizer (frozen)")
    parser.add_argument("--tokenizer_path", type=str, default="token_wm/tokenizer/vqvae_params.pkl",
                        help="Path to pre-trained tokenizer params")
    parser.add_argument("--n_iters_tok", type=int, default=500,
                        help="Number of tokenizer update iterations per step (paper: 500)")
    
    # TWM
    parser.add_argument("--twm_seq_len", type=int, default=20)
    parser.add_argument("--twm_num_layers", type=int, default=3)
    parser.add_argument("--twm_num_heads", type=int, default=8)
    parser.add_argument("--twm_embed_dim", type=int, default=128)
    parser.add_argument("--twm_dropout", type=float, default=0.1)
    parser.add_argument("--twm_lr", type=float, default=0.001)
    parser.add_argument("--twm_max_grad_norm", type=float, default=0.5)
    parser.add_argument("--twm_rollout_len", type=int, default=19,
                        help="Imagination rollout length. Must be <= twm_seq_len-1 (initial obs uses ~1 block)")
    parser.add_argument("--twm_temperature", type=float, default=1.0)
    parser.add_argument("--twm_batch_size", type=int, default=16)
    parser.add_argument("--n_iters_twm", type=int, default=500,
                        help="Number of TWM update iterations per step (paper: 500)")
    parser.add_argument("--n_mb_wm", type=int, default=3,
                        help="Number of minibatches for WM training (paper: 3)")
    
    # Imagination / MBRL
    parser.add_argument("--buffer_size", type=int, default=128000)
    parser.add_argument("--background_planning_start", type=int, default=200000)
    parser.add_argument("--imagination_batch_size", type=int, default=48)
    parser.add_argument("--n_iters_ac", type=int, default=150,
                        help="Number of imagination policy updates per step (paper: 150)")
    parser.add_argument("--n_mb_imagination", type=int, default=1,
                        help="Number of minibatches for imagination PPO (paper: 1)")
    parser.add_argument("--n_epoch_imagination", type=int, default=1,
                        help="Number of epochs for imagination PPO (paper: 1)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_size", type=int, default=2048)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="craftax-mbrl")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--save_policy", action="store_true")

    # Checkpointing
    parser.add_argument("--checkpoint_freq", type=int, default=50000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--max_checkpoints", type=int, default=2,
                        help="Maximum number of recent checkpoints to keep")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Path to checkpoint directory to resume from (e.g. checkpoints/checkpoint_100000)")
    
    # Smoke test
    parser.add_argument("--smoke_test", action="store_true")
    
    args, rest = parser.parse_known_args()
    if rest:
        print(f"Warning: Unknown args: {rest}")
    
    if args.smoke_test:
        print("\n*** SMOKE TEST MODE (with pre-trained tokenizer) ***\n")
        args.total_timesteps = 50000
        args.num_envs = 8
        args.num_steps = 32
        args.use_pretrained_tokenizer = True  # Use pre-trained tokenizer
        args.n_iters_tok = 0  # Skip tokenizer training
        args.n_iters_twm = 5  # Reduced TWM iterations
        args.n_iters_ac = 3   # Reduced imagination iterations
        args.n_mb_wm = 1      # Reduced minibatches
        args.background_planning_start = 5000
        args.buffer_size = 5000
        args.twm_batch_size = 4
        args.imagination_batch_size = 8
        args.use_wandb = False
    
    if args.seed is None:
        args.seed = np.random.randint(2**31)
    
    run_mbrl(args)

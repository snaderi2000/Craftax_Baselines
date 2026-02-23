"""
M2 Model-Based RL (Dyna) Implementation for Craftax
Based on "Improving Transformer World Models for Data-Efficient RL"

Uses the RNN-based IMPALA policy from ppo_m1_best.py (ActorCriticRNN)

M2 (M1 + Dyna):
- From step 0: Collect data, train VQ-VAE AND TWM
- Policy can be trained on real environment rollouts (PPO)
- After T_BP: Policy is additionally trained on imagined data from TWM

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
import errno
import gzip
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
        max_blocks=config.get("TWM_MAX_BLOCKS", config["TWM_SEQ_LEN"]),
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
        reward_num_classes=2 if config.get("USE_BINARY_REWARD_TARGET", True) else 3,
        binary_reward_threshold=config.get("BINARY_REWARD_THRESHOLD", 0.5),
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
      2. Reward prediction (cross-entropy; binary for Craftax M1 by default)
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


def make_real_policy_update_fn(network, config):
    """Create JIT-compiled PPO update on real environment trajectories."""
    num_minibatches = config["NUM_MINIBATCHES"]
    update_epochs = config["UPDATE_EPOCHS"]

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
        n_envs = traj_batch.obs.shape[1]
        permutation = jax.random.permutation(perm_rng, n_envs)
        batch = (init_hstate, traj_batch, advantages, targets)
        shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

        minibatches = jax.tree.map(
            lambda x: jnp.swapaxes(
                jnp.reshape(x, [x.shape[0], num_minibatches, -1] + list(x.shape[2:])),
                1,
                0,
            ),
            shuffled_batch,
        )
        policy_state, losses = jax.lax.scan(_ppo_update_minbatch, policy_state, minibatches)

        epoch_total_loss = losses[0].mean()
        epoch_value_loss = losses[1][0].mean()
        epoch_actor_loss = losses[1][1].mean()
        epoch_entropy = losses[1][2].mean()
        epoch_metrics = (epoch_total_loss, epoch_value_loss, epoch_actor_loss, epoch_entropy)
        return (policy_state, init_hstate, traj_batch, advantages, targets, rng), epoch_metrics

    @jax.jit
    def real_policy_step(policy_state, traj, initial_hstate, final_hstate, last_obs, last_done, rng):
        """Apply PPO updates on one real rollout trajectory."""
        traj_for_ppo = Transition(
            done=traj.done,
            action=traj.action,
            value=traj.value,
            reward=traj.reward,
            log_prob=traj.log_prob,
            obs=traj.obs,
            info=None,
        )

        ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
        _, _, last_val = network.apply(policy_state.params, final_hstate, ac_in)
        last_val = last_val.squeeze(0)

        advantages, targets = _calculate_gae(traj_for_ppo, last_val, last_done)
        init_hstate_batch = initial_hstate[None, :]

        rng, ppo_rng = jax.random.split(rng)
        ppo_state = (policy_state, init_hstate_batch, traj_for_ppo, advantages, targets, ppo_rng)
        ppo_state, epoch_metrics = jax.lax.scan(_ppo_update_epoch, ppo_state, None, update_epochs)
        policy_state = ppo_state[0]

        metrics = {
            "real_ppo_total_loss": epoch_metrics[0].mean(),
            "real_ppo_value_loss": epoch_metrics[1].mean(),
            "real_ppo_actor_loss": epoch_metrics[2].mean(),
            "real_ppo_entropy": epoch_metrics[3].mean(),
        }
        return policy_state, metrics, rng

    return real_policy_step


def make_imagination_fn(network, vqvae, twm, config):
    """Create JIT-compiled imagination rollout + PPO function."""
    
    # Use paper's settings for imagination PPO
    n_mb_imagination = config.get("N_MB_IMAGINATION", 1)
    n_epoch_imagination = config.get("N_EPOCH_IMAGINATION", 1)
    burnin_horizon = config.get("BURNIN_HORIZON", 5)
    use_binary_reward_target = config.get("USE_BINARY_REWARD_TARGET", True)
    
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
    def imagination_step(policy_state, vqvae_params, twm_state, start_obs, start_done, burnin_obs, burnin_done, burnin_actions, rng):
        """
        Single imagination rollout + PPO update.
        In M2, this branch complements real-data PPO updates.
        """
        N = config["IMAGINATION_BATCH_SIZE"]
        rng, imagine_rng = jax.random.split(rng)

        start_hstate = ScannedRNN.initialize_carry(N, 256)
        if burnin_horizon > 0:
            burnin_obs_t = jnp.swapaxes(burnin_obs, 0, 1)    # (M, N, ...)
            burnin_done_t = jnp.swapaxes(burnin_done, 0, 1)  # (M, N)
            start_hstate, _, _ = network.apply(
                policy_state.params,
                start_hstate,
                (burnin_obs_t, burnin_done_t),
            )
        
        # Tokenize starting observation
        start_tokens = vqvae.apply(vqvae_params, start_obs, method=vqvae.encode)
        
        # Initialize KV cache
        # Must match transformer's internal causal mask size = TWM_SEQ_LEN * TOKENS_PER_BLOCK
        # The initial obs frame (64 tokens) + TWM_ROLLOUT_LEN * 65 must fit within this.
        # This is enforced by setting TWM_ROLLOUT_LEN = TWM_SEQ_LEN - 1.
        cache = KeysValues.init(
            n=N,
            num_heads=config["TWM_NUM_HEADS"],
            max_tokens=config.get("TWM_MAX_BLOCKS", config["TWM_SEQ_LEN"]) * config["TOKENS_PER_BLOCK"],
            embed_dim=config["TWM_EMBED_DIM"],
            num_layers=config["TWM_NUM_LAYERS"],
        )

        # Burn in TWM cache with past (obs, action) context to match policy burn-in context.
        if burnin_horizon > 0:
            burnin_obs_flat = burnin_obs.reshape(N * burnin_horizon, 63, 63, 3)
            burnin_tokens_flat = vqvae.apply(vqvae_params, burnin_obs_flat, method=vqvae.encode)
            burnin_tokens = burnin_tokens_flat.reshape(N, burnin_horizon, -1)
            burnin_tokens_t = jnp.swapaxes(burnin_tokens, 0, 1)                     # (M, N, 64)
            burnin_actions_t = jnp.swapaxes(burnin_actions.astype(jnp.int32), 0, 1) # (M, N)

            def _burnin_twm_step(cache, step_inputs):
                obs_tok_t, act_t = step_inputs
                _, cache = twm.apply(twm_state.params, obs_tok_t, past_keys_values=cache, deterministic=True)
                action_tok_t = act_t.reshape(N, 1)
                _, cache = twm.apply(twm_state.params, action_tok_t, past_keys_values=cache, deterministic=True)
                return cache, None

            cache, _ = jax.lax.scan(_burnin_twm_step, cache, (burnin_tokens_t, burnin_actions_t))

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
            
            # Reward: binary targets (paper's Craftax setup) or legacy 3-class mapping.
            sampled_rew_class = jax.random.categorical(rew_rng, rew_logits, axis=-1)
            if use_binary_reward_target:
                reward = sampled_rew_class.astype(jnp.float32)
            else:
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

def _is_disk_quota_error(exc: OSError) -> bool:
    """Return True when an OSError indicates out-of-space/quota conditions."""
    return exc.errno in (errno.ENOSPC, errno.EDQUOT, 122)


def _rotate_checkpoints(ckpt_dir, max_checkpoints):
    """Keep only the newest max_checkpoints directories named checkpoint_<step>."""
    if max_checkpoints < 0:
        return

    checkpoints = []
    if not os.path.exists(ckpt_dir):
        return

    for d in os.listdir(ckpt_dir):
        if d.startswith("checkpoint_") and os.path.isdir(os.path.join(ckpt_dir, d)):
            try:
                step_num = int(d.split("_")[1])
                checkpoints.append((step_num, d))
            except ValueError:
                continue

    checkpoints.sort(key=lambda x: x[0])
    while len(checkpoints) > max_checkpoints:
        _, oldest_dir = checkpoints.pop(0)
        oldest_path = os.path.join(ckpt_dir, oldest_dir)
        print(f"Removing old checkpoint: {oldest_path}")
        shutil.rmtree(oldest_path, ignore_errors=True)


def _build_empty_flat_buffer(config):
    """Create an empty flat replay buffer matching training shapes."""
    flat_buffer_size = config["BUFFER_SIZE"]
    buffer_obs = jnp.zeros((flat_buffer_size, 63, 63, 3))
    buffer_actions = jnp.zeros((flat_buffer_size,), dtype=jnp.int32)
    buffer_rewards = jnp.zeros((flat_buffer_size,))
    buffer_dones = jnp.zeros((flat_buffer_size,))
    buffer_ptr = jnp.array(0, dtype=jnp.int32)
    buffer_count = jnp.array(0, dtype=jnp.int32)
    return (buffer_obs, buffer_actions, buffer_rewards, buffer_dones, buffer_ptr, buffer_count)


def _init_empty_fbx_buffer_state(config):
    """Initialize an empty flashbax trajectory buffer state."""
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
    return fbx_buffer.init(example_timestep)


def _to_jax_arrays(tree):
    """Convert NumPy arrays/scalars in a pytree to JAX arrays."""
    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, (np.ndarray, np.generic)) else x,
        tree,
    )


def save_checkpoint(ckpt_dir, step, policy_state, vqvae_state, twm_state, 
                   fbx_buffer_state, buffer_data, metadata, max_checkpoints=2,
                   save_buffers=False, compress_buffers=True, buffer_gzip_level=1):
    """
    Save training state to checkpoint directory.
    Returns True if checkpoint was written, False if skipped due to quota/disk pressure.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    # Free space before writing new checkpoint.
    try:
        _rotate_checkpoints(ckpt_dir, max(0, max_checkpoints - 1))
    except Exception as e:
        print(f"Warning: Failed to pre-rotate checkpoints: {e}")

    step_dir = os.path.join(ckpt_dir, f"checkpoint_{step}")
    os.makedirs(step_dir, exist_ok=True)

    def _dump_pickle(filename, payload, compress=False):
        payload = jax.device_get(payload)
        path = os.path.join(step_dir, filename)
        if compress:
            path = f"{path}.gz"
            with gzip.open(path, "wb", compresslevel=buffer_gzip_level) as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    try:
        # Save model params and optimizer state.
        _dump_pickle("policy_params.pkl", policy_state.params)
        _dump_pickle("policy_opt_state.pkl", policy_state.opt_state)
        _dump_pickle("vqvae_params.pkl", vqvae_state.params)
        _dump_pickle("vqvae_opt_state.pkl", vqvae_state.opt_state)
        _dump_pickle("twm_params.pkl", twm_state.params)
        _dump_pickle("twm_opt_state.pkl", twm_state.opt_state)
    except OSError as e:
        if _is_disk_quota_error(e):
            print(f"Warning: Skipping checkpoint at step {step:,} (disk quota exceeded).")
            shutil.rmtree(step_dir, ignore_errors=True)
            return False
        raise

    buffers_saved = {
        "flat_buffer": False,
        "fbx_buffer": False,
    }
    if save_buffers:
        flat_buffer_path = os.path.join(
            step_dir, "flat_buffer.pkl.gz" if compress_buffers else "flat_buffer.pkl"
        )
        fbx_buffer_path = os.path.join(
            step_dir, "fbx_buffer.pkl.gz" if compress_buffers else "fbx_buffer.pkl"
        )
        try:
            _dump_pickle("flat_buffer.pkl", buffer_data, compress=compress_buffers)
            buffers_saved["flat_buffer"] = True
        except OSError as e:
            if _is_disk_quota_error(e):
                print("Warning: Could not save flat replay buffer (disk quota exceeded).")
                try:
                    os.remove(flat_buffer_path)
                except OSError:
                    pass
            else:
                raise

        try:
            _dump_pickle("fbx_buffer.pkl", fbx_buffer_state, compress=compress_buffers)
            buffers_saved["fbx_buffer"] = True
        except OSError as e:
            if _is_disk_quota_error(e):
                print("Warning: Could not save flashbax buffer state (disk quota exceeded).")
                try:
                    os.remove(fbx_buffer_path)
                except OSError:
                    pass
            else:
                raise

    metadata_to_save = dict(metadata)
    metadata_to_save["buffers_saved"] = buffers_saved
    metadata_to_save["save_buffers_enabled"] = bool(save_buffers)
    metadata_to_save["compress_buffers_enabled"] = bool(compress_buffers)
    metadata_to_save["buffer_gzip_level"] = int(buffer_gzip_level)
    try:
        _dump_pickle("metadata.pkl", metadata_to_save)
    except OSError as e:
        if _is_disk_quota_error(e):
            print(f"Warning: Skipping checkpoint at step {step:,} (disk quota exceeded before metadata write).")
            shutil.rmtree(step_dir, ignore_errors=True)
            return False
        raise

    print(f"Saved checkpoint to {step_dir}")

    try:
        _rotate_checkpoints(ckpt_dir, max_checkpoints)
    except Exception as e:
        print(f"Warning: Failed to rotate checkpoints: {e}")

    return True


def load_checkpoint(ckpt_path, config, network, vqvae, twm):
    """Load training state from checkpoint directory."""
    print(f"Loading checkpoint from {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path {ckpt_path} does not exist")

    # Re-create TrainStates (needed to restore optimizer structure)
    # We need dummy data to init, but we'll overwrite params immediately
    rng = jax.random.PRNGKey(0)
    
    # Policy
    rng, net_rng = jax.random.split(rng)
    init_x = (jnp.zeros((1, config["NUM_ENVS"], 63, 63, 3)), jnp.zeros((1, config["NUM_ENVS"])))
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
    network_params = network.init(net_rng, init_hstate, init_x)
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    policy_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
    
    # VQ-VAE
    rng, vqvae_rng = jax.random.split(rng)
    vqvae_params = vqvae.init(vqvae_rng, jnp.zeros((1, 63, 63, 3)))
    tx_vq = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["VQVAE_LR"]))
    vqvae_state = TrainState.create(apply_fn=vqvae.apply, params=vqvae_params, tx=tx_vq)
    
    # TWM
    rng, twm_rng = jax.random.split(rng)
    twm_params = twm.init(twm_rng, jnp.zeros((1, config["TOKENS_PER_BLOCK"]), dtype=jnp.int32))
    tx_twm = optax.chain(optax.clip_by_global_norm(config["TWM_MAX_GRAD_NORM"]), optax.adam(config["TWM_LR"]))
    twm_state = TrainState.create(apply_fn=twm.apply, params=twm_params, tx=tx_twm)

    # Load params and opt_state
    with open(os.path.join(ckpt_path, "policy_params.pkl"), "rb") as f:
        policy_state = policy_state.replace(params=_to_jax_arrays(pickle.load(f)))
    with open(os.path.join(ckpt_path, "policy_opt_state.pkl"), "rb") as f:
        policy_state = policy_state.replace(opt_state=_to_jax_arrays(pickle.load(f)))
        
    with open(os.path.join(ckpt_path, "vqvae_params.pkl"), "rb") as f:
        vqvae_state = vqvae_state.replace(params=_to_jax_arrays(pickle.load(f)))
    with open(os.path.join(ckpt_path, "vqvae_opt_state.pkl"), "rb") as f:
        vqvae_state = vqvae_state.replace(opt_state=_to_jax_arrays(pickle.load(f)))
        
    with open(os.path.join(ckpt_path, "twm_params.pkl"), "rb") as f:
        twm_state = twm_state.replace(params=_to_jax_arrays(pickle.load(f)))
    with open(os.path.join(ckpt_path, "twm_opt_state.pkl"), "rb") as f:
        twm_state = twm_state.replace(opt_state=_to_jax_arrays(pickle.load(f)))
        
    # Load metadata
    with open(os.path.join(ckpt_path, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    # Load buffers (optional for lightweight checkpoints)
    flat_buffer_path = os.path.join(ckpt_path, "flat_buffer.pkl")
    flat_buffer_path_gz = f"{flat_buffer_path}.gz"
    if os.path.exists(flat_buffer_path):
        with open(flat_buffer_path, "rb") as f:
            buffer_data = _to_jax_arrays(pickle.load(f))
    elif os.path.exists(flat_buffer_path_gz):
        with gzip.open(flat_buffer_path_gz, "rb") as f:
            buffer_data = _to_jax_arrays(pickle.load(f))
    else:
        print("Warning: flat_buffer.pkl missing. Initializing empty flat buffer.")
        buffer_data = _build_empty_flat_buffer(config)

    fbx_buffer_path = os.path.join(ckpt_path, "fbx_buffer.pkl")
    fbx_buffer_path_gz = f"{fbx_buffer_path}.gz"
    if os.path.exists(fbx_buffer_path):
        with open(fbx_buffer_path, "rb") as f:
            fbx_buffer_state = _to_jax_arrays(pickle.load(f))
    elif os.path.exists(fbx_buffer_path_gz):
        with gzip.open(fbx_buffer_path_gz, "rb") as f:
            fbx_buffer_state = _to_jax_arrays(pickle.load(f))
    else:
        print("Warning: fbx_buffer.pkl missing. Initializing empty flashbax buffer.")
        fbx_buffer_state = _init_empty_fbx_buffer_state(config)
        
    print(f"Loaded checkpoint from step {metadata['total_steps']}")
    return policy_state, vqvae_state, twm_state, fbx_buffer_state, buffer_data, metadata


# =============================================================================
# Main Training Function
# =============================================================================

def run_mbrl(config):
    """Run M2 MBRL training (Dyna: real + imagination policy updates)."""
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
    if config["BURNIN_HORIZON"] > max_rollout:
        print(f"WARNING: BURNIN_HORIZON={config['BURNIN_HORIZON']} too large for TWM_SEQ_LEN={config['TWM_SEQ_LEN']}. "
              f"Clamping to {max_rollout}.")
        config["BURNIN_HORIZON"] = max_rollout
    config["N_ITERS_REAL_AC"] = max(0, int(config.get("N_ITERS_REAL_AC", 1)))
    config["BURNIN_OVERSAMPLE_FACTOR"] = max(1, int(config.get("BURNIN_OVERSAMPLE_FACTOR", 4)))
    config["BURNIN_MAX_SAMPLING_ATTEMPTS"] = max(1, int(config.get("BURNIN_MAX_SAMPLING_ATTEMPTS", 4)))
    config["TWM_MAX_BLOCKS"] = config["TWM_SEQ_LEN"] + config["BURNIN_HORIZON"]

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"M2-DYNA-{config['ENV_NAME']}-{int(config['TOTAL_TIMESTEPS']//1e6)}M",
        )

    print("\n" + "="*70)
    print("M2 MBRL Training (Dyna: Real + Imagination)")
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
    print(f"Real policy updates enabled: {config['DYNA_ENABLE_REAL_UPDATES']}")
    print(f"Real policy updates after T_BP only: {config['DYNA_REAL_UPDATES_AFTER_BP_ONLY']}")
    print(f"Real policy update repeats per outer update: {config['N_ITERS_REAL_AC']}")
    print(f"Real PPO epochs/minibatches: {config['UPDATE_EPOCHS']}/{config['NUM_MINIBATCHES']}")
    print(f"Imagination policy updates: {config['N_ITERS_AC']}")
    print(f"Imagination rollout length: {config['TWM_ROLLOUT_LEN']}")
    print(f"Imagination batch size: {config['IMAGINATION_BATCH_SIZE']}")
    print(f"Imagination burn-in horizon: {config['BURNIN_HORIZON']}")
    print(f"Burn-in oversample factor: {config['BURNIN_OVERSAMPLE_FACTOR']}")
    print(f"Burn-in max sampling attempts: {config['BURNIN_MAX_SAMPLING_ATTEMPTS']}")
    print(f"Burn-in require no-done context: {config['BURNIN_REQUIRE_NO_DONE_CONTEXT']}")
    print(f"Burn-in require start not-done: {config['BURNIN_REQUIRE_START_NOT_DONE']}")
    print(f"Burn-in fallback allowed: {config['BURNIN_ALLOW_FALLBACK']}")
    print(f"TWM max blocks at inference: {config['TWM_MAX_BLOCKS']}")
    print(f"Binary reward target: {config['USE_BINARY_REWARD_TARGET']} (threshold={config['BINARY_REWARD_THRESHOLD']})")
    print(f"Buffer: flashbax trajectory buffer (max {config['BUFFER_SIZE']:,} transitions)")
    print(f"Checkpoint includes replay buffers: {config['CHECKPOINT_SAVE_BUFFERS']}")
    print(f"Checkpoint compresses replay buffers: {config['CHECKPOINT_COMPRESS_BUFFERS']} (gzip level={config['CHECKPOINT_GZIP_LEVEL']})")
    print("-"*70)
    print(f"Pre-trained tokenizer: {config['USE_PRETRAINED_TOKENIZER']}")
    if config['USE_PRETRAINED_TOKENIZER']:
        print(f"  Path: {config['TOKENIZER_PATH']}")
    print("="*70)
    print("NOTE: Dyna mode can train policy on real rollouts + imagined rollouts.")
    print("      Real data is always used for world model training.")
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
    vqvae_param_count = sum(x.size for x in jax.tree.leaves(vqvae_state.params))
    print(f"VQ-VAE parameter count: {vqvae_param_count:,}")
    
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
    twm_param_count = sum(x.size for x in jax.tree.leaves(twm_state.params))
    print(f"TWM parameter count: {twm_param_count:,}")
    
    # Environment
    rng, env_rng = jax.random.split(rng)
    obsv, env_state = env.reset(env_rng, env_params)
    hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
    last_done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
    
    # Flat buffer for VQ-VAE training and imagination starting states
    buffer_obs, buffer_actions, buffer_rewards, buffer_dones, buffer_ptr, buffer_count = _build_empty_flat_buffer(config)
    
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
        policy_state, vqvae_state, twm_state, fbx_buffer_state, buffer_data, metadata = load_checkpoint(
            config['RESUME_FROM'], config, network, vqvae, twm
        )
        
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
    real_policy_step = make_real_policy_update_fn(network, config)
    imagination_step = make_imagination_fn(network, vqvae, twm, config)
    update_buffer = make_buffer_update_fn(config)

    def _sample_candidate_windows(fbx_state, rng, num_candidates):
        """Sample candidate contiguous windows from flashbax."""
        b = config["TWM_BATCH_SIZE"]
        chunks = max(1, (num_candidates + b - 1) // b)
        obs_chunks = []
        done_chunks = []
        action_chunks = []
        for _ in range(chunks):
            rng, sample_rng = jax.random.split(rng)
            fbx_batch = fbx_buffer.sample(fbx_state, sample_rng)
            obs_chunks.append(fbx_batch.experience["obs"])       # (B, T, 63, 63, 3)
            done_chunks.append(fbx_batch.experience["done"])     # (B, T)
            action_chunks.append(fbx_batch.experience["action"]) # (B, T)
        obs_all = jnp.concatenate(obs_chunks, axis=0)[:num_candidates]
        done_all = jnp.concatenate(done_chunks, axis=0)[:num_candidates].astype(jnp.float32)
        action_all = jnp.concatenate(action_chunks, axis=0)[:num_candidates].astype(jnp.int32)
        return obs_all, done_all, action_all, rng

    def sample_imagination_context(fbx_state, rng):
        """
        Sample contiguous (burn-in + start) windows from flashbax trajectories,
        filtering invalid windows that cross episode boundaries.
        """
        n = config["IMAGINATION_BATCH_SIZE"]
        m = config["BURNIN_HORIZON"]
        require_no_done_context = config["BURNIN_REQUIRE_NO_DONE_CONTEXT"]
        require_start_not_done = config["BURNIN_REQUIRE_START_NOT_DONE"]
        oversample_factor = config["BURNIN_OVERSAMPLE_FACTOR"]
        max_attempts = config["BURNIN_MAX_SAMPLING_ATTEMPTS"]
        allow_fallback = config["BURNIN_ALLOW_FALLBACK"]

        selected_obs = []
        selected_done = []
        selected_action = []
        selected_count = 0

        total_candidates = 0
        total_valid = 0
        total_done_in_context = 0
        total_start_done = 0

        for _ in range(max_attempts):
            if selected_count >= n:
                break

            remaining = n - selected_count
            num_candidates = max(remaining, remaining * oversample_factor)
            cand_obs, cand_done, cand_action, rng = _sample_candidate_windows(
                fbx_state, rng, num_candidates
            )

            start_done = cand_done[:, m]
            if m > 0:
                done_in_context = jnp.any(cand_done[:, :m] > 0.5, axis=1)
            else:
                done_in_context = jnp.zeros((num_candidates,), dtype=bool)
            start_is_done = start_done > 0.5

            valid_mask = jnp.ones((num_candidates,), dtype=bool)
            if require_no_done_context and m > 0:
                valid_mask = jnp.logical_and(valid_mask, ~done_in_context)
            if require_start_not_done:
                valid_mask = jnp.logical_and(valid_mask, ~start_is_done)

            valid_idx = np.where(np.asarray(valid_mask))[0]
            take = min(remaining, int(valid_idx.shape[0]))
            if take > 0:
                idx = jnp.asarray(valid_idx[:take], dtype=jnp.int32)
                selected_obs.append(cand_obs[idx])
                selected_done.append(cand_done[idx])
                selected_action.append(cand_action[idx])
                selected_count += take

            total_candidates += int(num_candidates)
            total_valid += int(valid_idx.shape[0])
            total_done_in_context += int(np.asarray(done_in_context).sum())
            total_start_done += int(np.asarray(start_is_done).sum())

        fallback_count = max(0, n - selected_count)
        if fallback_count > 0:
            if not allow_fallback:
                raise RuntimeError(
                    "Insufficient valid burn-in windows and fallback is disabled. "
                    "Try reducing burn-in constraints or increasing sampling attempts."
                )
            fb_obs, fb_done, fb_action, rng = _sample_candidate_windows(
                fbx_state, rng, fallback_count
            )
            selected_obs.append(fb_obs)
            selected_done.append(fb_done)
            selected_action.append(fb_action)

        obs_all = jnp.concatenate(selected_obs, axis=0)[:n]
        done_all = jnp.concatenate(selected_done, axis=0)[:n]
        action_all = jnp.concatenate(selected_action, axis=0)[:n]

        burnin_obs = obs_all[:, :m]
        burnin_done = done_all[:, :m]
        burnin_actions = action_all[:, :m]
        start_obs = obs_all[:, m]
        start_done = done_all[:, m]

        denom = max(1, total_candidates)
        diagnostics = {
            "burnin_valid_fraction": float(total_valid) / float(denom),
            "burnin_fallback_fraction": float(fallback_count) / float(max(1, n)),
            "burnin_done_in_context_fraction": float(total_done_in_context) / float(denom),
            "burnin_start_done_fraction": float(total_start_done) / float(denom),
        }
        return start_obs, start_done, burnin_obs, burnin_done, burnin_actions, diagnostics, rng

    # =========================================================================
    # Training Loop (Python outer loop - not JIT traced!)
    # =========================================================================
    total_steps = start_step
    start_update = start_step // (config["NUM_ENVS"] * config["NUM_STEPS"])
    t0 = time.time()
    last_update_time = t0
    last_update_steps = start_step
    
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
        # Step 5: Policy Update (M2 Dyna: real updates + imagination updates)
        # ---------------------------------------------------------------------
        real_policy_metrics = {
            "real_ppo_total_loss": 0.0,
            "real_ppo_value_loss": 0.0,
            "real_ppo_actor_loss": 0.0,
            "real_ppo_entropy": 0.0,
        }
        real_updates_active = False
        real_update_iters = 0
        if config["DYNA_ENABLE_REAL_UPDATES"]:
            real_updates_active = (
                (not config["DYNA_REAL_UPDATES_AFTER_BP_ONLY"])
                or (total_steps >= config["BACKGROUND_PLANNING_START"])
            )
        if real_updates_active:
            real_metric_sums = {k: 0.0 for k in real_policy_metrics.keys()}
            for _ in range(config["N_ITERS_REAL_AC"]):
                policy_state, real_diag, rng = real_policy_step(
                    policy_state, traj, initial_hstate, hstate, obsv, last_done, rng
                )
                for k in real_metric_sums.keys():
                    real_metric_sums[k] += float(real_diag[k])
                real_update_iters += 1
            real_policy_metrics = {
                k: v / float(max(1, real_update_iters))
                for k, v in real_metric_sums.items()
            }

        burnin_metrics = {
            "burnin_valid_fraction": 0.0,
            "burnin_fallback_fraction": 0.0,
            "burnin_done_in_context_fraction": 0.0,
            "burnin_start_done_fraction": 0.0,
        }
        imag_update_iters = 0
        if total_steps >= config["BACKGROUND_PLANNING_START"]:
            if not imagination_started:
                print(f"\n*** Starting imagination-based policy training at step {total_steps:,} ***\n")
                imagination_started = True
            
            if can_sample_twm:
                burnin_metric_sums = {k: 0.0 for k in burnin_metrics.keys()}
                # M2: Imagined policy updates (same as M1 branch)
                for ac_iter in range(config["N_ITERS_AC"]):
                    start_obs, start_done, burnin_obs, burnin_done, burnin_actions, burnin_diag, rng = sample_imagination_context(
                        fbx_buffer_state, rng
                    )
                    policy_state, rng = imagination_step(
                        policy_state, vqvae_state.params, twm_state,
                        start_obs, start_done, burnin_obs, burnin_done, burnin_actions, rng
                    )
                    for k in burnin_metric_sums.keys():
                        burnin_metric_sums[k] += float(burnin_diag[k])
                    imag_update_iters += 1
                burnin_metrics = {
                    k: v / float(max(1, imag_update_iters))
                    for k, v in burnin_metric_sums.items()
                }
        
        # ---------------------------------------------------------------------
        # Logging with achievements/score
        # ---------------------------------------------------------------------
        if True:
            now = time.time()
            update_time_sec = max(1e-9, now - last_update_time)
            steps_since_last = total_steps - last_update_steps
            sps_inst = float(steps_since_last) / update_time_sec
            elapsed_total = max(1e-9, now - t0)
            sps_avg = float(total_steps - start_step) / elapsed_total
            last_update_time = now
            last_update_steps = total_steps

            status = "WM-only"
            if real_updates_active and imagination_started:
                status = "Dyna"
            elif real_updates_active:
                status = "Real-only"
            elif imagination_started:
                status = "Imagination"
            
            # Compute episode-averaged metrics (like ppo_rnn.py)
            # This averages values over completed episodes only
            returned = traj.info["returned_episode"]
            num_returned = returned.sum()
            traj_reward_mean = float(traj.reward.mean())
            traj_done_mean = float(traj.done.mean())
            
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
                'sps': f'{float(sps_inst):.0f}',
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
                    'num_returned_episodes': int(num_returned),
                    'traj_reward_mean': traj_reward_mean,
                    'traj_done_mean': traj_done_mean,
                    'sps_inst': sps_inst,
                    'sps_avg': sps_avg,
                    'update_time_sec': float(update_time_sec),
                    'buffer_size': int(buffer_count),
                    'real_policy_updates_active': int(real_updates_active),
                    'real_policy_update_iters': int(real_update_iters),
                    'real_ppo_total_loss': real_policy_metrics["real_ppo_total_loss"],
                    'real_ppo_value_loss': real_policy_metrics["real_ppo_value_loss"],
                    'real_ppo_actor_loss': real_policy_metrics["real_ppo_actor_loss"],
                    'real_ppo_entropy': real_policy_metrics["real_ppo_entropy"],
                    'imagination_active': imagination_started,
                    'imagination_update_iters': int(imag_update_iters),
                    'burnin_valid_fraction': burnin_metrics["burnin_valid_fraction"],
                    'burnin_fallback_fraction': burnin_metrics["burnin_fallback_fraction"],
                    'burnin_done_in_context_fraction': burnin_metrics["burnin_done_in_context_fraction"],
                    'burnin_start_done_fraction': burnin_metrics["burnin_start_done_fraction"],
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
                           fbx_buffer_state, buffer_data, metadata,
                           max_checkpoints=config["MAX_CHECKPOINTS"],
                           save_buffers=config["CHECKPOINT_SAVE_BUFFERS"],
                           compress_buffers=config["CHECKPOINT_COMPRESS_BUFFERS"],
                           buffer_gzip_level=config["CHECKPOINT_GZIP_LEVEL"])
            
            if exit_requested:
                print(f"Exiting gracefully at step {total_steps:,}...")
                break
    
    t1 = time.time()
    print(f"\nTraining complete!")
    print(f"Time: {t1 - t0:.2f}s")
    print(f"SPS: {config['TOTAL_TIMESTEPS'] / (t1 - t0):.0f}")
    
    # Save checkpoints
    if config["SAVE_POLICY"]:
        save_dir = f"checkpoints/m2_mbrl_{config['ENV_NAME']}_{config['SEED']}"
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
    parser = argparse.ArgumentParser(description="M2 MBRL for Craftax (Dyna: Real + Imagination)")
    
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
    parser.add_argument("--use_binary_reward_target", action=argparse.BooleanOptionalAction, default=True,
                        help="Use binary TWM reward targets (1=achievement reward event, 0=otherwise)")
    parser.add_argument("--binary_reward_threshold", type=float, default=0.5,
                        help="Threshold for binary reward targets; rewards >= threshold map to class 1")
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
    parser.add_argument("--dyna_enable_real_updates", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable PPO policy updates on real environment trajectories (Dyna)")
    parser.add_argument("--dyna_real_updates_after_bp_only", action=argparse.BooleanOptionalAction, default=False,
                        help="If true, delay real policy updates until background_planning_start")
    parser.add_argument("--n_iters_real_ac", type=int, default=1,
                        help="Number of repeated real PPO updates per outer training iteration")
    parser.add_argument("--burnin_horizon", type=int, default=5,
                        help="RNN burn-in horizon for imagination rollout starts (paper: 5)")
    parser.add_argument("--burnin_oversample_factor", type=int, default=4,
                        help="Oversampling factor for candidate burn-in windows before filtering")
    parser.add_argument("--burnin_max_sampling_attempts", type=int, default=4,
                        help="Maximum retries to gather valid burn-in windows")
    parser.add_argument("--burnin_require_no_done_context", action=argparse.BooleanOptionalAction, default=True,
                        help="Require burn-in context frames to contain no done flags")
    parser.add_argument("--burnin_require_start_not_done", action=argparse.BooleanOptionalAction, default=True,
                        help="Require the sampled start frame to not be terminal")
    parser.add_argument("--burnin_allow_fallback", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow fallback to unfiltered windows if valid burn-in windows are insufficient")
    
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
    parser.add_argument("--checkpoint_save_buffers", action=argparse.BooleanOptionalAction, default=False,
                        help="Save replay buffers inside checkpoints (large files; may exceed quota)")
    parser.add_argument("--checkpoint_compress_buffers", action=argparse.BooleanOptionalAction, default=True,
                        help="Gzip-compress replay buffer checkpoint files (recommended for quota-limited jobs)")
    parser.add_argument("--checkpoint_gzip_level", type=int, default=1,
                        help="Gzip compression level for replay buffers (1=fastest, 9=smallest)")
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
        args.n_iters_real_ac = 1
        args.n_mb_wm = 1      # Reduced minibatches
        args.background_planning_start = 5000
        args.buffer_size = 5000
        args.twm_batch_size = 4
        args.imagination_batch_size = 8
        args.use_wandb = False
    
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    # Clamp gzip level to valid range.
    args.checkpoint_gzip_level = int(np.clip(args.checkpoint_gzip_level, 0, 9))
    
    run_mbrl(args)

"""
M1 Model-Based RL Implementation for Craftax
Based on "Improving Transformer World Models for Data-Efficient RL"

Uses the RNN-based IMPALA policy from ppo_m1_best.py (ActorCriticRNN)
and trains it on both real environment data AND imagined TWM rollouts.

Algorithm 1 from the paper:
1. Collect data from environment → replay buffer
2. Update policy on environment data (PPO)
3. Update world model (VQ-VAE + TWM)
4. Update policy on imagined data (after T_BP steps)
"""

import argparse
import os
import sys
import time
import pickle
import functools
from typing import NamedTuple, Dict, Any, Tuple

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
        obs, dones = x  # obs: (T, B, 63, 63, 3), dones: (T, B)

        # 1. IMPALA CNN Encoder
        x_enc = obs.astype(jnp.float32)
        for ch in (64, 64, 128):
            x_enc = ImpalaStack(ch)(x_enc)
        x_enc = nn.relu(x_enc)
        
        # Flatten CNN output -> z_t (8192 dims)
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
# Main Training Function
# =============================================================================

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["TOKENS_PER_BLOCK"] = 65

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params
    config["NUM_ACTIONS"] = env.action_space(env_params).n

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    # Pre-create modules
    vqvae = create_vqvae(config)
    twm = create_twm(config)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # =================================================================
        # Initialize Policy Network (RNN-based)
        # =================================================================
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, net_rng = jax.random.split(rng)
        
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], 63, 63, 3)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
        network_params = network.init(net_rng, init_hstate, init_x)
        
        param_count = sum(x.size for x in jax.tree.leaves(network_params))
        print(f"Policy parameter count: {param_count:,}")

        # Paper: no LR annealing for MBRL
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        policy_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        # =================================================================
        # Initialize World Model (VQ-VAE + TWM)
        # =================================================================
        rng, vqvae_rng, twm_rng = jax.random.split(rng, 3)
        sample_obs = jnp.zeros((1, 63, 63, 3))
        sample_tokens = jnp.zeros((1, config["TOKENS_PER_BLOCK"]), dtype=jnp.int32)
        
        vqvae_state = create_vqvae_train_state(config, vqvae_rng, sample_obs)
        twm_state = create_twm_train_state(config, twm_rng, sample_tokens)

        # Load pre-trained VQ-VAE if specified
        # (This happens outside JIT, handled in run_mbrl)

        # =================================================================
        # Initialize Environment
        # =================================================================
        rng, env_rng = jax.random.split(rng)
        obsv, env_state = env.reset(env_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)

        # =================================================================
        # Replay Buffer (simple circular buffer)
        # =================================================================
        # Memory: buffer_size * 63 * 63 * 3 * 4 bytes
        # 50k entries ≈ 2.4GB, 128k ≈ 6GB
        buffer_size = config["BUFFER_SIZE"]
        buffer = {
            'obs': jnp.zeros((buffer_size, 63, 63, 3)),
            'actions': jnp.zeros((buffer_size,), dtype=jnp.int32),
            'rewards': jnp.zeros((buffer_size,)),
            'dones': jnp.zeros((buffer_size,)),
            # Note: next_obs removed to save memory (not needed for imagination sampling)
            'ptr': jnp.array(0, dtype=jnp.int32),
            'size': jnp.array(0, dtype=jnp.int32),
        }

        # =================================================================
        # Training Helpers
        # =================================================================
        
        def _calculate_gae(traj_batch, last_val, last_done):
            def _get_advantages(carry, transition):
                gae, next_value, next_done = carry
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                return (gae, value, done), gae
            
            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val, last_done),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        def _ppo_loss_fn(params, init_hstate, traj_batch, gae, targets):
            _, pi, value = network.apply(
                params, init_hstate[0], (traj_batch.obs, traj_batch.done)
            )
            log_prob = pi.log_prob(traj_batch.action)
            
            # Value loss
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            
            # Actor loss
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
            
            # Normalize advantages for whole batch
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, config["NUM_ENVS"])
            batch = (init_hstate, traj_batch, advantages, targets)
            
            shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            minibatches = jax.tree.map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])),
                    1, 0,
                ),
                shuffled_batch,
            )
            
            policy_state, losses = jax.lax.scan(_ppo_update_minbatch, policy_state, minibatches)
            return (policy_state, init_hstate, traj_batch, advantages, targets, rng), losses

        def _vqvae_loss_fn(params, obs_batch):
            recon, tokens, total_loss, metrics = vqvae.apply(
                params, obs_batch, method=vqvae.get_vq_loss
            )
            return total_loss, metrics

        # =================================================================
        # Imagination Rollout Function
        # =================================================================
        
        def _imagine_rollout(
            rng, policy_state, vqvae_state, twm_state, 
            start_obs, start_hstate, start_done,
            num_steps,
        ):
            """
            Generate imagined trajectory using TWM.
            Policy acts on decoded observations from TWM.
            """
            N = start_obs.shape[0]
            
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
                
                # Policy takes action based on current observation
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
                reward = jax.nn.softmax(rew_logits)[:, 1]  # Probability of reward
                reward = jax.random.bernoulli(rew_rng, reward).astype(jnp.float32)
                
                new_done = jax.nn.softmax(done_logits)[:, 1]
                new_done = jax.random.bernoulli(done_rng, new_done).astype(jnp.float32)
                
                # Generate next observation tokens autoregressively
                obs_logits = output.logits_observations[:, -1, :]
                next_token = jax.random.categorical(gen_rng, obs_logits / config["TWM_TEMPERATURE"], axis=-1)
                next_tokens = [next_token]
                
                # Generate remaining 63 tokens and feed all 64 to cache
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
                
                # Feed the last token to complete the frame
                last_input = last_token.reshape(N, 1)
                _, cache = twm.apply(twm_state.params, last_input, past_keys_values=cache, deterministic=True)
                
                # Stack all 64 tokens
                all_next_tokens = jnp.concatenate([next_token.reshape(N, 1), generated_tokens.T], axis=1)
                
                # Decode to observation
                next_obs = vqvae.apply(vqvae_state.params, all_next_tokens, method=vqvae.decode_tokens)
                
                # Crop to 63x63 if needed
                next_obs = next_obs[:, :63, :63, :]
                
                transition = Transition(
                    done=current_done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=current_obs,
                    info=None,
                )
                
                return (next_obs, hstate, new_done, cache, rng), transition
            
            carry = (start_obs, start_hstate, start_done, cache, rng)
            carry, traj = jax.lax.scan(_imagine_step, carry, None, num_steps)
            final_obs, final_hstate, final_done, _, _ = carry
            
            # Get final value
            ac_in = (final_obs[np.newaxis, :], final_done[np.newaxis, :])
            _, _, last_val = network.apply(policy_state.params, final_hstate, ac_in)
            last_val = last_val.squeeze(0)
            
            return traj, last_val, final_done, final_hstate

        # =================================================================
        # Main Update Step
        # =================================================================
        
        def _update_step(runner_state, update_idx):
            (
                policy_state,
                vqvae_state,
                twm_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                buffer,
                rng,
                total_steps,
            ) = runner_state

            # -----------------------------------------------------------------
            # Step 1: Collect environment trajectories
            # -----------------------------------------------------------------
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
            carry, env_traj = jax.lax.scan(_env_step, carry, None, config["NUM_STEPS"])
            policy_state, env_state, last_obs, last_done, hstate, rng = carry
            
            total_steps = total_steps + config["NUM_ENVS"] * config["NUM_STEPS"]

            # Add to replay buffer
            traj_obs = env_traj.obs.reshape(-1, 63, 63, 3)
            traj_actions = env_traj.action.reshape(-1)
            traj_rewards = env_traj.reward.reshape(-1)
            traj_dones = env_traj.done.reshape(-1)
            
            num_new = traj_obs.shape[0]
            ptr = buffer['ptr']
            indices = (jnp.arange(num_new) + ptr) % buffer_size
            
            buffer = {
                'obs': buffer['obs'].at[indices].set(traj_obs),
                'actions': buffer['actions'].at[indices].set(traj_actions),
                'rewards': buffer['rewards'].at[indices].set(traj_rewards),
                'dones': buffer['dones'].at[indices].set(traj_dones),
                'ptr': (ptr + num_new) % buffer_size,
                'size': jnp.minimum(buffer['size'] + num_new, buffer_size),
            }

            # -----------------------------------------------------------------
            # Step 2: PPO update on environment data
            # -----------------------------------------------------------------
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(policy_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            
            advantages, targets = _calculate_gae(env_traj, last_val, last_done)
            
            init_hstate_batch = initial_hstate[None, :]
            ppo_state = (policy_state, init_hstate_batch, env_traj, advantages, targets, rng)
            ppo_state, _ = jax.lax.scan(_ppo_update_epoch, ppo_state, None, config["UPDATE_EPOCHS"])
            policy_state = ppo_state[0]
            rng = ppo_state[-1]

            # -----------------------------------------------------------------
            # Step 3: Update World Model
            # -----------------------------------------------------------------
            obs_batch = env_traj.obs.reshape(-1, 63, 63, 3)
            
            def _vqvae_update(state, _):
                grad_fn = jax.value_and_grad(_vqvae_loss_fn, has_aux=True)
                (loss, metrics), grads = grad_fn(state.params, obs_batch)
                state = state.apply_gradients(grads=grads)
                return state, loss
            
            vqvae_state, vqvae_losses = jax.lax.scan(
                _vqvae_update, vqvae_state, None, config["VQVAE_UPDATES_PER_ITER"]
            )

            # -----------------------------------------------------------------
            # Step 4: Imagination training (after T_BP)
            # -----------------------------------------------------------------
            # Paper Algorithm 1: Do imagination rollouts as part of the main loop
            # Instead of 150 iterations inside one update, we do 1 per update step.
            # Over many update steps, this accumulates to many imagination updates.
            do_imagination = total_steps >= config["BACKGROUND_PLANNING_START"]
            
            def _do_imagination(carry):
                policy_state, vqvae_state, twm_state, buffer, rng = carry
                rng, sample_rng, imagine_rng = jax.random.split(rng, 3)
                
                # Sample starting states from buffer
                sample_idx = jax.random.randint(
                    sample_rng, (config["NUM_ENVS"],), 0, jnp.maximum(buffer['size'], 1)
                )
                start_obs = buffer['obs'][sample_idx]
                start_done = jnp.zeros(config["NUM_ENVS"])
                start_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
                
                # Single imagination rollout (no nested scan!)
                imag_traj, imag_last_val, imag_last_done, _ = _imagine_rollout(
                    imagine_rng, policy_state, vqvae_state, twm_state,
                    start_obs, start_hstate, start_done,
                    config["TWM_ROLLOUT_LEN"],
                )
                
                # PPO on imagined data
                imag_advantages, imag_targets = _calculate_gae(imag_traj, imag_last_val, imag_last_done)
                
                init_hstate_batch = start_hstate[None, :]
                ppo_state = (policy_state, init_hstate_batch, imag_traj, imag_advantages, imag_targets, rng)
                ppo_state, _ = jax.lax.scan(_ppo_update_epoch, ppo_state, None, config["UPDATE_EPOCHS_WM"])
                policy_state = ppo_state[0]
                rng = ppo_state[-1]
                
                return policy_state, rng
            
            def _skip_imagination(carry):
                policy_state, _, _, _, rng = carry
                return policy_state, rng
            
            imag_carry = (policy_state, vqvae_state, twm_state, buffer, rng)
            policy_state, rng = jax.lax.cond(
                do_imagination, _do_imagination, _skip_imagination, imag_carry
            )

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            metric = jax.tree.map(
                lambda x: (x * env_traj.info["returned_episode"]).sum()
                / (env_traj.info["returned_episode"].sum() + 1e-8),
                env_traj.info,
            )
            metric["vqvae_loss"] = vqvae_losses[-1]
            metric["total_steps"] = total_steps
            metric["buffer_size"] = buffer['size']

            if config["DEBUG"] and config["USE_WANDB"]:
                def callback(metric, update_idx):
                    to_log = create_log_dict(metric, config)
                    to_log["vqvae_loss"] = float(metric["vqvae_loss"])
                    batch_log(update_idx, to_log, config)
                jax.debug.callback(callback, metric, update_idx)

            runner_state = (
                policy_state,
                vqvae_state,
                twm_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                buffer,
                rng,
                total_steps,
            )
            return runner_state, metric

        # =================================================================
        # Run Training
        # =================================================================
        rng, train_rng = jax.random.split(rng)
        runner_state = (
            policy_state,
            vqvae_state,
            twm_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            buffer,
            train_rng,
            0,
        )
        
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# =============================================================================
# Main
# =============================================================================

def run_mbrl(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"M1-MBRL-{config['ENV_NAME']}-{int(config['TOTAL_TIMESTEPS']//1e6)}M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    train_vmap = jax.vmap(train_jit)

    print("\n" + "="*60)
    print("M1 MBRL Training (RNN Policy)")
    print("="*60)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"Num envs: {config['NUM_ENVS']}")
    print(f"Background planning starts at: {config['BACKGROUND_PLANNING_START']:,}")
    print(f"Imagination rollout length: {config['TWM_ROLLOUT_LEN']}")
    print("="*60 + "\n")

    t0 = time.time()
    out = train_vmap(rngs)
    jax.block_until_ready(out)
    t1 = time.time()

    print(f"\nTraining complete!")
    print(f"Time: {t1 - t0:.2f}s")
    print(f"SPS: {config['TOTAL_TIMESTEPS'] / (t1 - t0):.0f}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1 MBRL for Craftax")
    
    # Environment
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    parser.add_argument("--num_envs", type=int, default=48)
    parser.add_argument("--num_steps", type=int, default=96)
    
    # Training
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--update_epochs_wm", type=int, default=1)
    parser.add_argument("--num_minibatches", type=int, default=8)
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
    parser.add_argument("--vqvae_updates_per_iter", type=int, default=5)
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
    parser.add_argument("--twm_rollout_len", type=int, default=20)
    parser.add_argument("--twm_temperature", type=float, default=1.0)
    
    # MBRL
    # Note: Buffer stores observations (63x63x3 float32 = ~48KB each)
    # 50k entries = ~2.4GB, 128k = ~6GB
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--background_planning_start", type=int, default=200000)
    # Note: imagination_iters is now 1 per update step (paper-style), this param is unused
    parser.add_argument("--imagination_iters", type=int, default=1)
    
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_repeats", type=int, default=1)
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
        args.total_timesteps = 100000
        args.num_envs = 8
        args.num_steps = 32
        args.vqvae_updates_per_iter = 1
        args.background_planning_start = 20000
        args.imagination_iters = 2
        args.buffer_size = 10000
        args.use_wandb = False
    
    if args.seed is None:
        args.seed = np.random.randint(2**31)
    
    run_mbrl(args)

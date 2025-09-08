import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax.core.frozen_dict import unfreeze


import flashbax as fbx
from flashbax.vault import Vault
from pathlib import Path
from flashbax.utils import get_tree_shape_prefix

from flax.training import train_state
from flax import serialization
from functools import partial
import os

# Import all our modules and the data loader
from world_model.sample import make_replay_samplers
from world_model.tokenizer import Tokenizer, EncoderDecoderConfig, compute_loss as compute_tokenizer_loss
from world_model.transformer import TransformerConfig
from world_model.world_model import WorldModel, compute_wm_loss



import wandb
from typing import Any, NamedTuple

from flax.training import orbax_utils
from flax.training import train_state
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import (
    ActorCriticConv,
    ActorCriticConvRNN,
)
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


from datetime import datetime


# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl

def write_entire_buffer_once(vault: Vault, fbx_state):
    """Write the entire buffer snapshot to the vault, handling wrap-around if full."""
    T = get_tree_shape_prefix(fbx_state.experience, n_axes=2)[1]  # time-axis length
    k = int(fbx_state.current_index)                              # next-write index (ring head)

    if bool(fbx_state.is_full):
        # Chronological order is [k:T) then [0:k)
        n1 = vault.write(fbx_state, source_interval=(k, T), dest_start=vault.vault_index)
        n2 = vault.write(fbx_state, source_interval=(0, k), dest_start=vault.vault_index)
        return n1 + n2
    else:
        # Not full yet: data is [0:k)
        return vault.write(fbx_state, source_interval=(0, k), dest_start=vault.vault_index)


class TrainState(train_state.TrainState):
    batch_stats: Any
    q_mean:     jnp.ndarray   # scalar  (running mean  μ_target)
    q_var:      jnp.ndarray   # scalar  (running var   σ²_target)

# train state for WM and Tokenizer 
class WMTrainState(train_state.TrainState):
    rng: jax.random.PRNGKey

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    h: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

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

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        else:
            network = ActorCriticConvRNN(
                action_dim   = env.action_space(env_params).n,
                head_width   = config["LAYER_SIZE"],   # 2048 in the paper
                rnn_hidden   = config.get("RNN_HIDDEN", 256),  # ← 0 = “no-GRU” ablation
                use_gru      = config.get("USE_GRU", True)     # optional explicit flag
            )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))


        variables = network.init(_rng, init_x)
        params = variables["params"]
        batch_stats = variables["batch_stats"]


        # Place the world model components 
        rng, tokenizer_key, wm_key = jax.random.split(rng, 3)

        # Initialize tokenizer
        tokenizer_config = EncoderDecoderConfig(
            resolution=config["OBS_RESOLUTION"], 
            in_channels=3, 
            z_channels=128, 
            ch=64, 
            ch_mult=[1, 2, 4, 8], 
            num_res_blocks=1, 
            attn_resolutions=[], 
            out_ch=3, 
            dropout=0.0
        )

        tokenizer = Tokenizer(
            vocab_size=config["VOCAB_SIZE"], 
            embed_dim=config["EMBED_DIM"], # The embedding dimension is 128 
            encoder_config=tokenizer_config, 
            decoder_config=tokenizer_config
        )

        dummy_obs = jnp.zeros((1, config["OBS_RESOLUTION"], config["OBS_RESOLUTION"], 3))
        tokenizer_params = tokenizer.init(tokenizer_key, dummy_obs, train=False)['params']

        #Initialize the world model 
        wm_config = TransformerConfig(
            tokens_per_block=config["TOKENS_PER_OBS"] + 1, 
            max_blocks=config["T_WM"], 
            attention='block_causal',
            num_layers=3, 
            num_heads=8, 
            embed_dim=config["EMBED_DIM"], # The embedding dimension is 128 
            embed_pdrop=0.1, 
            resid_pdrop=0.1, 
            attn_pdrop=0.1,
        )
        world_model = WorldModel(
            obs_vocab_size=config["VOCAB_SIZE"], 
            act_vocab_size=env.action_space(env_params).n, 
            config=wm_config
        )
        dummy_tokens = jnp.zeros((1, config["T_WM"] * (config["TOKENS_PER_OBS"] + 1)), dtype=jnp.int32)
        wm_params = world_model.init(wm_key, dummy_tokens, train=False)['params']

        def count_parameters(params):
            flat_params = jax.tree_util.tree_leaves(unfreeze(params))
            return sum(p.size for p in flat_params)

        print("Total parameters:", count_parameters(params))
        
        
        #network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-8),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-8),
            )


        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            q_mean=jnp.array(0.0),
            q_var=jnp.array(1.0),
        )

        

        # The world model uses a simple Adam optimizer with a learning rate of 1e-3 
        wm_tx = optax.adam(config["WM_LR"])

        tokenizer_state = WMTrainState.create(
            apply_fn=tokenizer.apply,
            params=tokenizer_params,
            tx=wm_tx,
            rng=tokenizer_key
        )
        
        wm_state = WMTrainState.create(
            apply_fn=world_model.apply,
            params=wm_params,
            tx=wm_tx,
            rng=wm_key
        )



        # --- ADD THIS BLOCK TO INITIALIZE THE BUFFER ---
        print("Initializing Flashbax replay buffer...")
        # Define the structure of what we want to save from each timestep
        example_item = {
            "obs": jnp.zeros(env.observation_space(env_params).shape, dtype=jnp.uint8),
            "actions": jnp.zeros((), dtype=jnp.int32),
            "rewards": jnp.zeros((), dtype=jnp.float32),
            "dones": jnp.zeros((), dtype=bool),
        }
        
        # Create the buffer function
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["T_WM"], # Can't sample a sequence until you have one
            add_batch_size=config["NUM_ENVS"],
            # --- ADD THE MISSING ARGUMENTS ---
            sample_batch_size=config.get("WM_BATCH_SIZE", 16), # configurable to control GPU memory
            sample_sequence_length=config["T_WM"], # This is T_WM from the paper
            period=1, # This is a standard value, allowing sampling to start at any valid step
        )
        # Initialize the buffer's state
        buffer_state = buffer.init(example_item)
        # --------------------------------------------------



        # JIT compiled training steps for world model and tokenizer
        @jax.jit
        def tokenizer_train_step(state, batch): 
            rng, dropout_rng = jax.random.split(state.rng)

            def loss_fn(params):
                obs_5d = batch['obs'] # Note: 'observations' was renamed to 'obs' in the buffer
                B, T, H, W, C = obs_5d.shape
                obs_4d = obs_5d.reshape(B * T, H, W, C)
                
                # Compute full loss object to extract components
                loss_object = compute_tokenizer_loss(
                    model=tokenizer,
                    params=params, 
                    batch={'observations': obs_4d},
                    rngs={'dropout': dropout_rng}
                )
                aux = (
                    loss_object.total_loss,
                    loss_object.reconstruction_loss,
                    loss_object.codebook_loss,
                    loss_object.commitment_loss,
                )
                return loss_object.total_loss, aux
            (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state.replace(rng=rng), aux

        @jax.jit
        def twm_train_step(wm_state, tokenizer_params, batch): 
            rng, dropout_rng = jax.random.split(wm_state.rng)
            
            def loss_fn(params):
                # Prepare batch for TWM loss function
                # This logic should be inside the loss function for purity
                batch_copy = batch.copy()
                batch_copy['ends'] = batch_copy.pop('dones')
                batch_copy['observations'] = batch_copy.pop('obs')
                batch_copy['mask_padding'] = jnp.ones_like(batch_copy['actions'], dtype=jnp.bool_)

                loss_object = compute_wm_loss(
                    world_model_params=params, 
                    tokenizer_params=tokenizer_params,
                    world_model=world_model,    
                    tokenizer=tokenizer,
                    batch=batch_copy, 
                    rngs={'dropout': dropout_rng}
                )
                # Debug print only (no W&B logging from JIT path)
                jax.debug.print("TWM Loss: {x}", x=loss_object.total_loss)
                return loss_object.total_loss
            
            loss_val, grads = jax.value_and_grad(loss_fn)(wm_state.params)
            new_wm_state = wm_state.apply_gradients(grads=grads)
            return new_wm_state.replace(rng=rng), loss_val
            
        @partial(jax.jit, static_argnames=['buffer'])
        def update_world_model(tok_state, wm_state, buffer_state, rng, buffer):
            """
            Samples data from the buffer and trains the tokenizer and TWM.
            """
            # --- Phase 1: Update Tokenizer ---
            def train_tok_body_fn(i, state):
                tok_state, rng, loss_sum = state
                rng, sample_key = jax.random.split(rng)
                
                # Sample a batch of trajectories from the live buffer
                batch = buffer.sample(buffer_state, sample_key).experience
                
                # Perform one gradient update step
                tok_state, losses = tokenizer_train_step(tok_state, batch)
                loss_sum = loss_sum + jnp.array(losses)
                return tok_state, rng, loss_sum

            # Run the tokenizer training loop and compute mean losses (total, rec, codebook, commitment)
            tok_init = (tok_state, rng, jnp.zeros((4,), dtype=jnp.float32))
            tok_state, rng, tok_loss_sum = jax.lax.fori_loop(0, config["N_TOK_ITERS"], train_tok_body_fn, tok_init)
            tok_loss_mean = tok_loss_sum / jnp.maximum(1, config["N_TOK_ITERS"])
            
            # Freeze tokenizer params for TWM training
            frozen_tokenizer_params = jax.lax.stop_gradient(tok_state.params)

            # --- Phase 2: Update World Model ---
            def train_wm_body_fn(i, state):
                wm_state, rng, loss_sum = state
                rng, sample_key = jax.random.split(rng)
                
                # Sample a new batch of trajectories
                batch = buffer.sample(buffer_state, sample_key).experience
                
                # Perform one gradient update step
                wm_state, loss_val = twm_train_step(wm_state, frozen_tokenizer_params, batch)
                loss_sum = loss_sum + loss_val
                return wm_state, rng, loss_sum
            
            # Run the world model training loop and compute mean loss
            init_state = (wm_state, rng, jnp.array(0.0, dtype=jnp.float32))
            wm_state, rng, wm_loss_sum = jax.lax.fori_loop(0, config["N_WM_ITERS"], train_wm_body_fn, init_state)
            wm_loss_mean = wm_loss_sum / jnp.maximum(1, config["N_WM_ITERS"])
            
            return tok_state, wm_state, rng, tok_loss_mean, wm_loss_mean


        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        
        
        
              # TRAIN LOOP
        @jax.jit
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            print(">>> JAX: EXECUTING a compiled training step.")
            train_state, tokenizer_state, wm_state, env_state, last_obs, rng, update_step, h, buffer_state = runner_state
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    rng,
                    update_step,
                    h
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                #pi, value = network.apply(train_state.params, last_obs)


                # bundle both params and batch_stats into a single “vars” dict
                vars = {
                    'params':      train_state.params,
                    'batch_stats': train_state.batch_stats,
                }

                # 3) run apply_fn in TRAIN mode, allowing batch_stats to mutate
                #    note how we pull out the updated stats in new_model_state
                ((pi, value, h_next), new_model_state) = train_state.apply_fn(
                    vars,
                    last_obs,
                    h,
                    mutable=['batch_stats']  # allow BatchNorm to write new running‐stats
                )



                # 4) stash the updated batch_stats back into your TrainState
                train_state = train_state.replace(batch_stats=new_model_state['batch_stats'])


                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                q_mean, q_var = train_state.q_mean, train_state.q_var
                value_raw = value * jnp.sqrt(q_var) + q_mean      # <- restore units

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                transition = Transition(
                    done=done,
                    action=action,
                    value=value_raw,     #  <<< here
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    h=h, #h_next
                    info=info,
                )

                # ——— Reset hidden where episodes have ended ———
                # `done` has shape (num_envs,), so we add a trailing axis to broadcast
                h_next = jnp.where(done[:, None], jnp.zeros_like(h_next), h_next)

                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    rng,
                    update_step,
                    h_next
                )
                return runner_state, transition

 
            runner_state_in = (train_state, env_state, last_obs, rng, update_step, h)
            runner_state_out, traj_batch = jax.lax.scan(
                _env_step, runner_state_in, None, config["NUM_STEPS"]
            )


            # --- ADD THIS BLOCK TO POPULATE THE BUFFER ---
            # Create a dictionary with the data we want to store

            transposed_traj = jax.tree.map(
                lambda x: jnp.swapaxes(x, 0, 1),
                traj_batch
            )
            data_to_add = {
                "obs": (transposed_traj.obs * 255).astype(jnp.uint8),
                "actions": transposed_traj.action,
                "rewards": transposed_traj.reward,
                "dones": transposed_traj.done,
            }
            

            # Add the data to the buffer
            buffer_state = buffer.add(buffer_state, data_to_add)
            # -----------------------------------------------
           

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                rng,
                update_step,
                h
            ) = runner_state_out
            #_, last_val = network.apply(train_state.params, last_obs)
            vars = {
                'params':      train_state.params,
                'batch_stats': train_state.batch_stats,
            }
            ((_, last_val, _), new_model_state) = train_state.apply_fn(
                vars,
                last_obs,
                h,
                mutable=['batch_stats'],   # allow BatchNorm to write its running stats
            )
            train_state = train_state.replace(
                batch_stats=new_model_state['batch_stats']
            )
            
            q_mean, q_var = train_state.q_mean, train_state.q_var
            last_val = last_val * jnp.sqrt(q_var) + q_mean     # undo standardisation


            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            #jax.debug.print("adv std={:.3f}", jnp.std(advantages))
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info


                    ema_decay = config["ALPHA"]   #add a command line argument for this
                    q_mean, q_var = train_state.q_mean, train_state.q_var

                    batch_mean = jax.lax.stop_gradient(jnp.mean(targets))
                    batch_var  = jax.lax.stop_gradient(jnp.var(targets))

                    q_mean_new = ema_decay * q_mean + (1 - ema_decay) * batch_mean
                    q_var_new  = ema_decay * q_var  + (1 - ema_decay) * batch_var

                    targets_std = jax.lax.stop_gradient(
                        (targets - q_mean_new) / jnp.sqrt(q_var_new + 1e-8)
                    )


                    # Policy/value network
                    def _loss_fn(params, batch_stats, traj_batch, advs, targets_std):
                        # RERUN NETWORK
                        #pi, value = network.apply(params, traj_batch.obs)

                        # 1) bundle params + BN state
                        vars = {'params':      params,
                                'batch_stats': batch_stats}

                        # 2) rerun the network in train mode, allow BN to update
                        ((pi, value, h_next), new_model_state) = train_state.apply_fn(
                            vars,
                            traj_batch.obs,
                            traj_batch.h,
                            mutable=['batch_stats']  # let BatchNorm write its running‐stats
                        )

                        
                        new_logp = pi.log_prob(traj_batch.action)
                        ratio   = jnp.exp(new_logp - traj_batch.log_prob)

                        unclipped = ratio * advs
                        clipped   = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * advs
                        loss_actor = -jnp.minimum(unclipped, clipped).mean()
                        
                        value_loss = 0.5 * jnp.square(value - targets_std).mean() # scalar

                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"]  * value_loss   # e.g. 0.5
                            - config["ENT_COEF"] * entropy      # e.g. 0.01
                        )

                        return total_loss, (new_model_state['batch_stats'],
                            value_loss.mean(),
                            loss_actor,
                            entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)



                    (total_loss, (new_batch_stats, value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params,
                        train_state.batch_stats,
                        traj_batch,
                        advantages,
                        targets_std,
                    )
                    jax.debug.print("value_loss={:.3f}", value_loss)
                    
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(batch_stats=new_batch_stats, q_mean=q_mean_new, q_var=q_var_new)

                    losses = (total_loss, value_loss, loss_actor, entropy)
                    return train_state,  (total_loss, value_loss, loss_actor, entropy)

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # minibatches = jax.tree.map(
                #     lambda x: jnp.reshape(
                #         x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                #     ),
                #     shuffled_batch,
                # )

                # This helper function explicitly calculates the minibatch size, avoiding the error.
                def create_minibatches(x):
                    minibatch_size = x.shape[0] // config["NUM_MINIBATCHES"]
                    return jnp.reshape(x, (config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:])

                # Use the new robust function to create the minibatches
                minibatches = jax.tree.map(create_minibatches, shuffled_batch)

                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            ) 

            # unpack runner_state to pull out your new hidden state h

            #rng = update_state[-1]

            # Update the world model and tokenizer using data from the buffer
            # tokenizer_state, wm_state, rng = update_world_model(
            #     tokenizer_state, 
            #     wm_state, 
            #     buffer_state, 
            #     rng,   
            #     buffer
            # )

            # # wandb logging (host callback to avoid tracer conversions)
            # if config["DEBUG"] and config["USE_WANDB"]:

            #     def callback(metric, update_step):
            #         to_log = create_log_dict(metric, config)
            #         batch_log(update_step, to_log, config)

            #     jax.debug.callback(
            #         callback,
            #         metric,
            #         update_step,
            #     )


            runner_state = (train_state, tokenizer_state, wm_state, env_state, last_obs, rng, update_step + 1, h, buffer_state)
            return runner_state, metric

        @partial(jax.jit, static_argnames=['buffer'])
        def rollout_imagined_trajectories(train_state, tokenizer_params, wm_params, buffer_state, rng, buffer):
            # Sample starting observations from the replay buffer
            rng, sample_key = jax.random.split(rng)
            sample = buffer.sample(buffer_state, sample_key).experience
            # Use the first frame as the starting observation; scale to [0, 1]
            initial_obs = sample['obs'][:, 0].astype(jnp.float32) / 255.0

            # Encode initial observations to tokens
            encode_out = tokenizer.apply({'params': tokenizer_params}, initial_obs, train=False, method=tokenizer.encode)
            current_obs_tokens = encode_out.tokens  # shape: (B, TOKENS_PER_OBS)

            # Initial policy hidden state
            B = initial_obs.shape[0]
            h0 = jnp.zeros((B, network.rnn_hidden))

            # Target spatial size to maintain (match env obs)
            target_h = initial_obs.shape[1]
            target_w = initial_obs.shape[2]

            # KV cache for the world model
            kv_cache0 = None

            def imagine_step(carry, unused):
                train_state, rng, current_obs, current_tokens, h, kv_cache = carry

                # Policy step
                rng, rng_pi = jax.random.split(rng)
                vars = {
                    'params':      train_state.params,
                    'batch_stats': train_state.batch_stats,
                }
                ((pi, value, h_next), _) = train_state.apply_fn(
                    vars,
                    current_obs,
                    h,
                    mutable=['batch_stats']
                )
                action = pi.sample(seed=rng_pi)
                log_prob = pi.log_prob(action)

                q_mean, q_var = train_state.q_mean, train_state.q_var
                value_raw = value * jnp.sqrt(q_var) + q_mean

                # World model step: predict next obs tokens, reward, and done
                action_token = action[:, None]  # (B, 1)
                token_input_flat = jnp.concatenate([current_tokens, action_token], axis=1)
                wm_outputs, new_kv = world_model.apply(
                    {'params': wm_params},
                    token_input_flat,
                    past_keys_values=kv_cache,
                    train=False
                )

                next_obs_tokens = jnp.argmax(wm_outputs.logits_observations, axis=-1)  # (B, TOKENS_PER_OBS)
                # Take last time index for rewards/ends (T=1 here)
                reward_class = jnp.argmax(wm_outputs.logits_rewards[:, -1, :], axis=-1)
                reward = reward_class.astype(jnp.int32) - 1  # map {0,1,2} -> {-1,0,1}
                done_class = jnp.argmax(wm_outputs.logits_ends[:, -1, :], axis=-1)
                done = (done_class == 1)

                # Decode next observation tokens to image in [0,1]
                next_obs = tokenizer.apply(
                    {'params': tokenizer_params},
                    next_obs_tokens,
                    method=Tokenizer.decode_from_tokens
                )
                # Crop decoder output to the env resolution to keep shapes static across steps
                next_obs = next_obs[:, :target_h, :target_w, :]

                # Reset hidden state on done
                h_next = jnp.where(done[:, None], jnp.zeros_like(h_next), h_next)

                transition = Transition(
                    done=done,
                    action=action,
                    value=value_raw,
                    reward=reward.astype(jnp.float32),
                    log_prob=log_prob,
                    obs=current_obs,
                    next_obs=next_obs,
                    h=h,
                    info=jnp.zeros((B,))
                )

                carry = (train_state, rng, next_obs, next_obs_tokens, h_next, new_kv)
                return carry, transition

            carry0 = (train_state, rng, initial_obs, current_obs_tokens, h0, kv_cache0)
            carry_final, traj = jax.lax.scan(
                imagine_step, carry0, None, length=config['T_WM']
            )

            # Unpack final carry
            train_state_out, rng_out, _, _, _, _ = carry_final
            imagined_traj_batch = traj
            return imagined_traj_batch, train_state_out, rng_out
        @jax.jit
        def update_on_imagined_trajectories(train_state, imagined_traj_batch, rng):
            # Compute bootstrap value for the final imagined state
            vars = {
                'params':      train_state.params,
                'batch_stats': train_state.batch_stats,
            }
            ((_, last_val, _), _) = train_state.apply_fn(
                vars,
                imagined_traj_batch.next_obs[-1],
                imagined_traj_batch.h[-1],
                mutable=['batch_stats']
            )
            q_mean, q_var = train_state.q_mean, train_state.q_var
            last_val = last_val * jnp.sqrt(q_var) + q_mean

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(imagined_traj_batch, last_val)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Additional stabilization for imagined advantages
            advantages = jnp.clip(advantages, -10.0, 10.0)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    ema_decay = config["ALPHA"]
                    q_mean, q_var = train_state.q_mean, train_state.q_var

                    batch_mean = jax.lax.stop_gradient(jnp.mean(targets))
                    batch_var  = jax.lax.stop_gradient(jnp.var(targets))

                    q_mean_new = ema_decay * q_mean + (1 - ema_decay) * batch_mean
                    q_var_new  = ema_decay * q_var  + (1 - ema_decay) * batch_var

                    targets_std = jax.lax.stop_gradient(
                        (targets - q_mean_new) / jnp.sqrt(q_var_new + 1e-8)
                    )
                    targets_std = jnp.clip(targets_std, -10.0, 10.0)

                    def _loss_fn(params, batch_stats, traj_batch, advs, targets_std):
                        vars = {
                            'params':      params,
                            'batch_stats': batch_stats
                        }
                        ((pi, value, h_next), _) = train_state.apply_fn(
                            vars,
                            traj_batch.obs,
                            traj_batch.h,
                            mutable=['batch_stats']
                        )
                        # Numeric guards (imagined PPO only)
                        advs = jnp.nan_to_num(advs, nan=0.0)
                        targets_std = jnp.nan_to_num(targets_std, nan=0.0)
                        new_logp = pi.log_prob(traj_batch.action)
                        log_ratio = new_logp - traj_batch.log_prob
                        log_ratio = jnp.nan_to_num(log_ratio, nan=0.0, posinf=0.0, neginf=0.0)
                        log_ratio = jnp.clip(log_ratio, -20.0, 20.0)
                        ratio   = jnp.exp(log_ratio)

                        unclipped = ratio * advs
                        clipped   = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * advs
                        loss_actor = -jnp.minimum(unclipped, clipped).mean()

                        value = jnp.nan_to_num(value, nan=0.0)
                        value_loss = 0.5 * jnp.square(value - targets_std).mean()
                        entropy = jnp.nan_to_num(pi.entropy(), nan=0.0).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"]  * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, (
                            value_loss.mean(),
                            loss_actor,
                            entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params,
                        train_state.batch_stats,
                        traj_batch,
                        advantages,
                        targets_std,
                    )
                    jax.debug.print("imag_value_loss={:.3f}", value_loss)
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, value_loss, loss_actor, entropy)
                    return train_state, (total_loss, value_loss, loss_actor, entropy)

                (train_state, traj_batch, advantages, targets, rng) = update_state
                rng, _rng = jax.random.split(rng)
                # Derive batch size from the imagined batch shape to allow T_WM != NUM_STEPS
                time_len = traj_batch.done.shape[0]
                num_envs = traj_batch.done.shape[1]
                batch_size = time_len * num_envs
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (imagined_traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                def create_minibatches(x):
                    minibatch_size = x.shape[0] // config["NUM_MINIBATCHES"]
                    return jnp.reshape(x, (config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:])
                minibatches = jax.tree.map(create_minibatches, shuffled_batch)
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    imagined_traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                imagined_traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.get("IMAG_UPDATE_EPOCHS", 1)
            )
            return update_state[0], update_state[-1]

        rng, _rng = jax.random.split(rng)

        h0 = jnp.zeros((config["NUM_ENVS"], network.rnn_hidden))
        # Add buffer_state to the runner_state tuple
        runner_state = (train_state, tokenizer_state, wm_state, env_state, obsv, rng, 0, h0, buffer_state)

        # runner_state, metric = jax.lax.scan(
        #     _update_step, runner_state, None, config["NUM_UPDATES"]
        # )

        # The main training loop
        print(">>> Starting training loop...")
        for update_num in range(1, config["NUM_UPDATES"] + 1):
            
            # Track real environment steps for warmup
            if update_num == 1:
                total_env_steps = 0
            
            # === Part 1: PPO Update on Real Data ===
            print(f"--- Step {int(update_num)}: Running PPO Update (real env data) ---")
            # This calls your modified _update_step (now PPO-only)
            runner_state, metric = _update_step(runner_state, None) 
            total_env_steps += config["NUM_ENVS"] * config["NUM_STEPS"]

            # === Part 2: World Model Update ===
            print(f"--- Step {int(update_num)}: Running World Model Update ---")
            
            # Unpack the states needed for the world model update
            train_state, tokenizer_state, wm_state, env_state, last_obs, rng, update_step_count, h, buffer_state = runner_state
            
            # Call the JIT'd world model update function
            tokenizer_state, wm_state, rng, tok_loss_mean, wm_loss_mean = update_world_model(
                tokenizer_state, wm_state, buffer_state, rng, buffer
            )
             
            # Re-assemble the runner_state for the next iteration
            runner_state = (train_state, tokenizer_state, wm_state, env_state, last_obs, rng, update_step_count, h, buffer_state)

            # === Part 3: PPO Update on Imagined Data (M2) ===
            if config.get("USE_IMAGINED_UPDATES", False) and (total_env_steps >= config["T_BP"]):
                print(f"--- Step {int(update_num)}: Running PPO Update (imagined data) ---")
                imagined_traj_batch, train_state, rng = rollout_imagined_trajectories(
                    train_state, tokenizer_state.params, wm_state.params, buffer_state, rng, buffer
                )
                train_state, rng = update_on_imagined_trajectories(
                    train_state, imagined_traj_batch, rng
                )
            # === Part 4: Unified logging via a single host callback ===
            if config["DEBUG"] and config["USE_WANDB"]:
                def unified_log_callback(metric_host, tok_losses_host, wm_loss_host, step_host):
                    import numpy as np
                    step_scalar = int(np.asarray(step_host).reshape(-1)[0])
                    to_log = create_log_dict(metric_host, config)
                    # tokenizer losses: (total, rec, codebook, commitment)
                    tok_losses = np.asarray(tok_losses_host)
                    if tok_losses.shape[0] == 4:
                        to_log["tokenizer/loss_total"] = float(tok_losses[0])
                        to_log["tokenizer/reconstruction_loss"] = float(tok_losses[1])
                        to_log["tokenizer/codebook_loss"] = float(tok_losses[2])
                        to_log["tokenizer/commitment_loss"] = float(tok_losses[3])
                    to_log["wm/loss_total"] = float(np.asarray(wm_loss_host))
                    batch_log(step_scalar, to_log, config)
                jax.debug.callback(
                    unified_log_callback,
                    metric,
                    tok_loss_mean,
                    wm_loss_mean,
                    update_step_count,
                )
                # Re-assemble runner_state with updated train_state and rng
                runner_state = (
                    train_state,
                    tokenizer_state,
                    wm_state,
                    env_state,
                    last_obs,
                    rng,
                    update_step_count,
                    h,
                    buffer_state,
                )


        return {"runner_state": runner_state}  # , "info": metric}

    return train

def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    # train_jit = jax.jit(make_train(config))
    # train_vmap = jax.vmap(train_jit)

    train = make_train(config)
    train_vmap = jax.vmap(train)

    print("\n>>> PYTHON: About to call the JIT-compiled function. The long pause is the one-time compilation.")
    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")

    if config["SAVE_BUFFER"]:
        print("\n--- Saving Final Replay Buffer ---")

        final_runner_state = jax.tree.map(lambda x: x[0], out["runner_state"])
        final_buffer_state = final_runner_state[-1]

        # Use a stable UID you control (CLI arg or timestamp). Example:
        VAULT_UID = config.get("VAULT_UID") or "my_first_buffer_run"
        # Prefer an absolute base dir to avoid CWD surprises:
        REL_DIR = "/home/synaderi/Craftax_Baselines"

        vault = Vault(
            vault_name="craftax_replay_buffer",
            experience_structure=final_buffer_state.experience,  # creating (or reusing) this UID
            rel_dir=REL_DIR,
            vault_uid=VAULT_UID
        )

        n_written = write_entire_buffer_once(vault, final_buffer_state)
        print(f"✅ Saved {n_written} timesteps to: {REL_DIR}/craftax_replay_buffer/{VAULT_UID}")           


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--update_epochs", type=int, default=250)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae_lambda", type=float, default=0.65)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=1024)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--use_gru", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=128000,
        help="Total replay buffer size. Should be at least num_envs * num_steps.",
    )
    parser.add_argument("--t_wm", type=int, default=20, help="Trajectory length for the TWM.")
    parser.add_argument("--save_buffer", action="store_true", help="Save the final replay buffer to disk.")
    parser.add_argument("--vault_uid", type=str, default=None)


    # World Model Hyperparameters from the paper
    parser.add_argument("--wm_lr", type=float, default=1e-3, help="Learning rate for the world model and tokenizer.")
    parser.add_argument("--vocab_size", type=int, default=512, help="Codebook size for the VQ-VAE tokenizer.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension for the models.")
    parser.add_argument("--tokens_per_obs", type=int, default=64, help="Number of tokens per observation (e.g., 8x8 feature map).")
    parser.add_argument("--obs_resolution", type=int, default=63, help="Resolution of the environment observations.")
    parser.add_argument("--n_tok_iters", type=int, default=500, help="Number of training iterations for the tokenizer per update.")
    parser.add_argument("--n_wm_iters", type=int, default=500, help="Number of training iterations for the world model per update.")
    parser.add_argument("--wm_batch_size", type=int, default=16, help="Batch size sampled from replay for WM/tokenizer training.")
    parser.add_argument("--use_imagined_updates", action=argparse.BooleanOptionalAction, default=False, help="Enable PPO updates on imagined trajectories (M2).")
    parser.add_argument("--t_bp", type=int, default=0, help="Warmup env steps before enabling imagined PPO updates.")




    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)

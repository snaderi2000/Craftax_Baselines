import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from craftax.craftax_env import make_craftax_env_from_name
from flax.core.frozen_dict import unfreeze

import flashbax as fbx
from flashbax.vault import Vault
from pathlib import Path
from flashbax.utils import get_tree_shape_prefix

import wandb
from typing import Any, NamedTuple

from flax.training import orbax_utils
from flax.training import train_state
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)


from flashbax.utils import get_tree_shape_prefix



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
            # network = ActorCriticConvRNN(
            #     env.action_space(env_params).n, config["LAYER_SIZE"]
            # )
            # network = ActorCriticConvRNN(
            #     action_dim   = env.action_space(env_params).n,
            #     head_width   = config["LAYER_SIZE"],   # 2048 in the paper
            #     rnn_hidden   = config.get("RNN_HIDDEN", 256),  # ← 0 = “no-GRU” ablation
            #     use_gru      = config.get("USE_GRU", True)     # optional explicit flag
            # )

            network_train = ActorCriticConvRNN(
                action_dim = env.action_space(env_params).n,
                head_width = config["LAYER_SIZE"],
                rnn_hidden = config.get("RNN_HIDDEN", 256),
                use_gru    = config.get("USE_GRU", True),
                train=True,          # 🔹 BN updates allowed
            )

            network_eval = ActorCriticConvRNN(
                action_dim = env.action_space(env_params).n,
                head_width = config["LAYER_SIZE"],
                rnn_hidden = config.get("RNN_HIDDEN", 256),
                use_gru    = config.get("USE_GRU", True),
                train=False,         # 🔹 BN frozen
            )

        jax.debug.print("DEBUG: config[USE_GRU] is {}", config.get("USE_GRU"))


        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        init_h = jnp.zeros((1, config.get("RNN_HIDDEN", 256)))


        variables = network_train.init(_rng, init_x, init_h)
        params = variables["params"]
        batch_stats = variables["batch_stats"]


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
        # train_state = TrainState.create(
        #     apply_fn=network.apply,
        #     params=network_params,
        # #     tx=tx,
        # )
             


        train_state = TrainState.create(
            apply_fn=network_train.apply,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            q_mean=jnp.array(0.0),
            q_var=jnp.array(1.0),
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
            sample_batch_size=32, # A reasonable default for how many trajectories to sample at once
            sample_sequence_length=config["T_WM"], # This is T_WM=20 from the paper
            period=1, # This is a standard value, allowing sampling to start at any valid step
        )
        # Initialize the buffer's state
        buffer_state = buffer.init(example_item)
        # --------------------------------------------------

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        
        
        
              # TRAIN LOOP
        def _update_step(runner_state, unused):

            # Unpack the new runner_state which includes the buffer_state
            train_state, env_state, last_obs, rng, update_step, h, buffer_state = runner_state

            # COLLECT TRAJECTORIES
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
                ((pi, value, h_next), new_model_state) = network_train.apply(
                    vars,
                    last_obs,
                    h,
                    mutable=["batch_stats"]  # allow BatchNorm to write new running‐stats
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

            # runner_state, traj_batch = jax.lax.scan(
            #     _env_step, runner_state, None, config["NUM_STEPS"]
            # )

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
                new_rng, # <-- This is the updated RNG key
                update_step,
                h
            ) = runner_state_out
            #_, last_val = network.apply(train_state.params, last_obs)
            vars = {
                'params':      train_state.params,
                'batch_stats': train_state.batch_stats,
            }
            ((_, last_val, _), new_model_state) = network_train.apply(
                vars,
                last_obs,
                h,
                mutable=["batch_stats"],   # allow BatchNorm to write its running stats
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
                    #traj_batch, advantages, targets = batch_info
                    mb_traj, mb_adv, targets_mb = batch_info


                    ema_decay = config["ALPHA"]   #add a command line argument for this
                    q_mean, q_var = train_state.q_mean, train_state.q_var

                    batch_mean = jax.lax.stop_gradient(jnp.mean(targets_mb))
                    batch_var  = jax.lax.stop_gradient(jnp.var(targets_mb))

                    q_mean_new = ema_decay * q_mean + (1 - ema_decay) * batch_mean
                    q_var_new  = ema_decay * q_var  + (1 - ema_decay) * batch_var

                    targets_std_mb = jax.lax.stop_gradient(
                        (targets_mb - q_mean_new) / jnp.sqrt(q_var_new + 1e-8)
                    )


                    # Policy/value network
                    def _loss_fn(params, batch_stats, mb_traj, mb_adv, targets_std_mb):
                        # mb_traj.obs:      [T, Bmb, ...]
                        # mb_traj.done:     [T, Bmb]
                        # mb_traj.action:   [T, Bmb]
                        # mb_traj.log_prob: [T, Bmb]

                        T, Bmb = mb_traj.obs.shape[:2]
                        
                        # 1) Batch CNN once on [T*Bmb, 63,63,3] using method=encode
                        obs_flat = mb_traj.obs.reshape((T * Bmb,) + mb_traj.obs.shape[2:])
                        z_flat = network_eval.apply(
                            {"params": params, "batch_stats": batch_stats},
                            obs_flat,
                            method=network_eval.encode,
                        )
                        z = z_flat.reshape((T, Bmb, -1))

                        # 2) Initialize hidden state h0 from stored h0
                        # jax.debug.print("mb_traj.h {}", mb_traj.h.shape)
                        if config["USE_GRU"]:
                            #h0 = jnp.zeros((Bmb, config.get("RNN_HIDDEN", 256)), dtype=jnp.float32) 
                            h0 = mb_traj.h[0]
                        else:
                            h0 = jnp.zeros((Bmb, config.get("RNN_HIDDEN", 256)), dtype=jnp.float32)

                        def step(carry, inp):
                            h, bn_state = carry
                            z_t, done_t, act_t = inp  # z_t: [Bmb, Dz]

                            logits_pi, value_t, h_next = network_eval.apply(
                                {"params": params, "batch_stats": bn_state},
                                z_t,
                                h,
                                method=network_eval.core,
                            )

                            # Use the distrax API for cleaner and safer probability math
                            new_logp_t = logits_pi.log_prob(act_t)  # [Bmb]
                            ent_t = logits_pi.entropy()             # [Bmb]

                            # Reset hidden state for the next step if this step was a 'done'
                            h_next = jnp.where(done_t[:, None], jnp.zeros_like(h_next), h_next)
                            return (h_next, bn_state), (value_t, new_logp_t, ent_t)

                        (hT, _), (value_T, new_logp_T, ent_T) = jax.lax.scan(
                            step,
                            init=(h0, batch_stats),
                            xs=(z, mb_traj.done, mb_traj.action),
                            length=T,
                        )

                        # PPO losses computed over [T, Bmb]
                        ratio = jnp.exp(new_logp_T - mb_traj.log_prob)

                        unclipped = ratio * mb_adv
                        clipped   = jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"]
                        ) * mb_adv

                        loss_actor = -jnp.mean(jnp.minimum(unclipped, clipped))

                        value_loss = 0.5 * jnp.mean(jnp.square(value_T - targets_std_mb))

                        entropy = jnp.mean(ent_T)

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"]  * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, (batch_stats, value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (_, value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params,
                        train_state.batch_stats,
                        mb_traj,
                        mb_adv,
                        targets_std_mb, 
                    )



                    
                    #jax.debug.print("value_loss={:.3f}", value_loss)
                    
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(q_mean=q_mean_new, q_var=q_var_new)

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

                T  = config["NUM_STEPS"]
                B  = config["NUM_ENVS"]
                MB = config["NUM_MINIBATCHES"]
                Bmb = B // MB
                assert B % MB == 0

                # Permute only across env dimension, NOT across time.
                perm_env = jax.random.permutation(_rng, B)

                def permute_env(x):
                    # x is [T, B, ...] or [T, B]
                    return jnp.take(x, perm_env, axis=1)

                traj_shuf = jax.tree_util.tree_map(permute_env, traj_batch)
                adv_shuf  = permute_env(advantages)
                tgt_shuf  = permute_env(targets)   # <-- (see note below)

                def split_env_minibatches(x):
                    # [T, B, ...] -> [MB, T, Bmb, ...]
                    x = x.reshape((T, MB, Bmb) + x.shape[2:])
                    return jnp.swapaxes(x, 0, 1)   # [MB, T, Bmb, ...]

                mb_traj = jax.tree_util.tree_map(split_env_minibatches, traj_shuf)
                mb_adv  = split_env_minibatches(adv_shuf)
                mb_tgt  = split_env_minibatches(tgt_shuf)

                # This is what jax.lax.scan will iterate over (MB steps)
                minibatches = (mb_traj, mb_adv, mb_tgt)

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
                rng, # <-- Use the correct, updated RNG
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            #jax.debug.print("upd {}/{} q̂ μ={:.3f} σ={:.3f}", update_step, config["NUM_UPDATES"], train_state.q_mean, jnp.sqrt(train_state.q_var))

            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            # unpack runner_state to pull out your new hidden state h

            rng = update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            # runner_state = (
            #     train_state,
            #     env_state,
            #     last_obs,
            #     rng,
            #     update_step + 1,
            #     h
            # )
            runner_state = (train_state, env_state, last_obs, rng, update_step + 1, h, buffer_state)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        # runner_state = (
        #     train_state,
        #     env_state,
        #     obsv,
        #     _rng,
        #     0,
        # )

        h0 = jnp.zeros((config["NUM_ENVS"], config["RNN_HIDDEN"]), dtype=jnp.float32)
        # Add buffer_state to the runner_state tuple
        runner_state = (train_state, env_state, obsv, rng, 0, h0, buffer_state)

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
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

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))



    # --- Corrected save block (stable UID + wrap-around safe) ---
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e6
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=0.00045)
    parser.add_argument("--num_steps", type=int, default=96)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.925)
    parser.add_argument("--gae_lambda", type=float, default=0.625)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
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
    parser.add_argument("--layer_size", type=int, default=2048)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--rnn_hidden", type=int, default=256)
    parser.add_argument("--use_gru", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--buffer_size", type=int, default=128000)
    parser.add_argument("--t_wm", type=int, default=20, help="Trajectory length for the TWM.")
    parser.add_argument("--save_buffer", action="store_true", help="Save the final replay buffer to disk.")
    parser.add_argument("--vault_uid", type=str, default=None)

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

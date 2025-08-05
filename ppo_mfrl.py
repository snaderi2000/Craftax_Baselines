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
            apply_fn=network.apply,
            params=params,
            batch_stats=batch_stats,
            tx=tx,
            q_mean=jnp.array(0.0),
            q_var=jnp.array(1.0),
        )


        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        
        
        
              # TRAIN LOOP
        def _update_step(runner_state, unused):
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
                    h=h_next,
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

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                rng,
                update_step,
                h
            ) = runner_state
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
                    #jax.debug.print("value_loss={:.3f}", value_loss)
                    
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
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
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

            runner_state = (
                train_state,
                env_state,
                last_obs,
                rng,
                update_step + 1,
                h
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        # runner_state = (
        #     train_state,
        #     env_state,
        #     obsv,
        #     _rng,
        #     0,
        # )

        h0 = jnp.zeros((config["NUM_ENVS"], network.rnn_hidden))
        runner_state = (train_state, env_state, obsv, rng, 0, h0)

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

import argparse
import os
import sys

import jax

# Compatibility for newer Optax/Flax code running on JAX versions before
# jax.tree was introduced, such as jax==0.4.23.
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)

    jax.tree = _JaxTreeCompat()

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time

from flax.training import orbax_utils
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name


#from models.actor_critic import ImpalaStack, DenseResBlock

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl

class PPOTrainState(TrainState):
    target_mean: jnp.ndarray
    target_mean_sq: jnp.ndarray
    target_debias: jnp.ndarray

class ImpalaResBlock(nn.Module):
    channels: int
    groups: int = 32
    @nn.compact
    def __call__(self, x):
        residual = x
        # (a) ReLU
        x = nn.relu(x)
        

        # GroupNorm (HERE)
        x = nn.GroupNorm(
            num_groups=self.groups,
            epsilon=1e-5
        )(x)

        # Conv
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )(x)

        return x + residual
class ImpalaStack(nn.Module):
    channels: int
    groups: int = 32

    @nn.compact
    def __call__(self, x):
        # (b) Conv 3x3 stride 1
        x = nn.Conv(self.channels, (3, 3), strides=(1, 1), padding="SAME")(x)

        x = nn.GroupNorm(
            num_groups=self.groups,
            epsilon=1e-5
        )(x) 
        # (c) MaxPool 3x3 stride 2
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        # (d) Two ResNet blocks
        x = ImpalaResBlock(self.channels)(x)
        x = ImpalaResBlock(self.channels)(x)
        return x

class AchDistImpalaResBlock(nn.Module):
    channels: int
    groups: int = 1

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.Conv(self.channels, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.Conv(self.channels, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        return residual + x

class AchDistImpalaStack(nn.Module):
    channels: int
    groups: int = 1
    first_conv_norm: bool = True
    post_pool_groups: int = 1

    @nn.compact
    def __call__(self, x):
        if self.first_conv_norm:
            x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(x)
        x = nn.Conv(self.channels, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = nn.GroupNorm(num_groups=self.post_pool_groups, epsilon=1e-5)(x)
        x = AchDistImpalaResBlock(self.channels, groups=self.groups)(x)
        x = AchDistImpalaResBlock(self.channels, groups=self.groups)(x)
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
        """Applies the module."""
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
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorCriticRNN(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # x is (Time, Batch, 63, 63, 3) and (Time, Batch)

        # 1. CNN encoder.
        if self.config["ARCH"] == "paper":
            x_enc = obs.astype(jnp.float32)
            for ch in (64, 64, 128):
                x_enc = ImpalaStack(ch)(x_enc)
            x_enc = nn.relu(x_enc)
            # Paper-style raw flattened IMPALA feature: [T, B, 8192].
            z_t = x_enc.reshape((*x_enc.shape[:2], -1))

            # RNN bridge: 8192 -> 256.
            rnn_input_features = nn.LayerNorm()(z_t)
            rnn_input_features = nn.Dense(256, kernel_init=orthogonal(2))(rnn_input_features)
            rnn_input_features = nn.relu(rnn_input_features)
            feedforward_features = z_t
        elif self.config["ARCH"] in ("achdist_strong", "achdist_baseline"):
            x_enc = obs.astype(jnp.float32)
            for i, ch in enumerate((64, 128, 128)):
                x_enc = AchDistImpalaStack(
                    ch,
                    groups=1,
                    first_conv_norm=i > 0,
                    post_pool_groups=1,
                )(x_enc)
            x_enc = x_enc.reshape((*x_enc.shape[:2], -1))

            # Match Achievement-Distillation PPOGRUStrong:
            # IMPALA dense outsize 256, then visual latent hidsize 1024.
            x_enc = nn.LayerNorm()(x_enc)
            x_enc = nn.Dense(
                self.config["ACHDIST_IMPALA_OUTSIZE"],
                kernel_init=orthogonal(2),
            )(x_enc)
            x_enc = nn.relu(x_enc)
            feedforward_features = nn.LayerNorm()(x_enc)
            feedforward_features = nn.Dense(
                self.config["ACHDIST_HIDSIZE"],
                kernel_init=orthogonal(2),
            )(feedforward_features)
            feedforward_features = nn.relu(feedforward_features)

            # RNN bridge: 1024 -> 256.
            rnn_input_features = nn.LayerNorm()(feedforward_features)
            rnn_input_features = nn.Dense(256, kernel_init=orthogonal(2))(rnn_input_features)
            rnn_input_features = nn.relu(rnn_input_features)
        else:
            raise ValueError(f"Unknown ARCH={self.config['ARCH']}")

        # 3. RNN Update (Paper calls output y_t) [cite: 630]
        if self.config["ARCH"] == "achdist_baseline":
            shared_input = feedforward_features
        elif self.config["USE_GRU"] or self.config["NO_GRU_MEMORY"] == "masked_gru":
            rnn_in = (rnn_input_features, dones)
            hidden, y_t = ScannedRNN()(hidden, rnn_in)
            if not self.config["USE_GRU"]:
                y_t = 0.0 * y_t
            y_t = nn.relu(y_t)
            shared_input = jnp.concatenate([feedforward_features, y_t], axis=-1)
        else:
            # No-GRU ablation: remove recurrent state updates while preserving
            # the original head input shape for a closer/stabler graph.
            if self.config["NO_GRU_MEMORY"] == "zeros":
                memory_features = 0.0 * rnn_input_features
            elif self.config["NO_GRU_MEMORY"] == "projection":
                memory_features = rnn_input_features
            else:
                raise ValueError(
                    f"Unknown NO_GRU_MEMORY={self.config['NO_GRU_MEMORY']}"
                )
            shared_input = jnp.concatenate([feedforward_features, memory_features], axis=-1)

        # 5. Actor Head
        if self.config["ARCH"] == "achdist_baseline":
            h_actor = shared_input
        elif self.config["ARCH"] == "achdist_strong":
            h_actor = nn.LayerNorm()(shared_input)
            h_actor = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h_actor)
            h_actor = nn.relu(h_actor)
        else:
            h_actor = nn.LayerNorm()(shared_input)
            h_actor = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h_actor)
            h_actor = nn.relu(h_actor)
            
            # These blocks are now linear (ReLU removed from inside the class)
            h_actor = DenseResBlock(self.config["LAYER_SIZE"])(h_actor)
            h_actor = DenseResBlock(self.config["LAYER_SIZE"])(h_actor)
            
            # NEW: Added ReLU here to act as the 'cap' after linear residual additions
            h_actor = nn.relu(h_actor) 
            
            h_actor = nn.LayerNorm()(h_actor)
        actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h_actor)
        pi = distrax.Categorical(logits=actor_logits)

        # 6. Critic Head
        if self.config["ARCH"] == "achdist_baseline":
            h_critic = shared_input
        elif self.config["ARCH"] == "achdist_strong":
            h_critic = nn.LayerNorm()(shared_input)
            h_critic = nn.Dense(
                self.config["ACHDIST_VF_HEAD_HIDSIZE"],
                kernel_init=orthogonal(2),
            )(h_critic)
            h_critic = nn.relu(h_critic)
        else:
            h_critic = nn.LayerNorm()(shared_input)
            h_critic = nn.Dense(self.config["LAYER_SIZE"], kernel_init=orthogonal(2))(h_critic)
            h_critic = nn.relu(h_critic)
            
            # These blocks are now linear
            h_critic = DenseResBlock(self.config["LAYER_SIZE"])(h_critic)
            h_critic = DenseResBlock(self.config["LAYER_SIZE"])(h_critic)
            
            # NEW: Added ReLU here to act as the 'cap' after linear residual additions
            h_critic = nn.relu(h_critic)
            
            h_critic = nn.LayerNorm()(h_critic)
        critic_value = nn.Dense(1, kernel_init=orthogonal(1.0))(h_critic)

        return hidden, pi, jnp.squeeze(critic_value, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    # Wrap with some extra logging
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
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
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], 63, 63, 3)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], 256 #config["LAYER_SIZE"]
        )
        network_params = network.init(_rng, init_hstate, init_x)

        # Print parameter size
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
        print(f"Model parameter count: {param_count:,}")

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
        train_state = PPOTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            target_mean=jnp.array(0.0),
            target_mean_sq=jnp.array(0.0),
            target_debias=jnp.array(0.0),
        )

        def _target_mean_var(train_state):
            debias = jnp.maximum(train_state.target_debias, config["TARGET_NORM_EPS"])
            mean = train_state.target_mean / debias
            mean_sq = train_state.target_mean_sq / debias
            var = jnp.maximum(mean_sq - mean**2, config["TARGET_NORM_MIN_VAR"])
            return mean, var

        def _normalize_value(train_state, value):
            mean, var = _target_mean_var(train_state)
            return (value - mean) / jnp.sqrt(var)

        def _denormalize_value(train_state, value):
            mean, var = _target_mean_var(train_state)
            return value * jnp.sqrt(var) + mean

        def _update_target_stats(train_state, targets):
            beta = config["TARGET_NORM_BETA"]
            batch_mean = jnp.mean(targets)
            batch_mean_sq = jnp.mean(jnp.square(targets))
            return train_state.replace(
                target_mean=beta * train_state.target_mean + (1.0 - beta) * batch_mean,
                target_mean_sq=beta * train_state.target_mean_sq
                + (1.0 - beta) * batch_mean_sq,
                target_debias=beta * train_state.target_debias + (1.0 - beta),
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], 256 #config["LAYER_SIZE"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                if config["NORMALIZE_VALUE_TARGETS"]:
                    value = _denormalize_value(train_state, value)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            if config["NORMALIZE_VALUE_TARGETS"]:
                last_val = _denormalize_value(train_state, last_val)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info
                    if config["NORMALIZE_VALUE_TARGETS"]:
                        train_state = _update_target_stats(train_state, targets)

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        if config["NORMALIZE_VALUE_TARGETS"]:
                            old_value = _normalize_value(train_state, traj_batch.value)
                            value_targets = _normalize_value(train_state, targets)
                        else:
                            old_value = traj_batch.value
                            value_targets = targets
                        value_pred_clipped = old_value + (
                            value - old_value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - value_targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        #gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                # This standardizes the GAEs for the WHOLE pool of data at once.
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree_util.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-PPO_RNN-"
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
            train_state = jax.tree_util.tree_map(lambda x: x[0], train_states)
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
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument(
        "--arch",
        type=str,
        choices=("paper", "achdist_strong", "achdist_baseline"),
        default="paper",
    )
    parser.add_argument("--achdist_impala_outsize", type=int, default=256)
    parser.add_argument("--achdist_hidsize", type=int, default=1024)
    parser.add_argument("--achdist_vf_head_hidsize", type=int, default=1280)
    parser.add_argument(
        "--normalize_value_targets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--target_norm_beta", type=float, default=0.99)
    parser.add_argument("--target_norm_eps", type=float, default=1e-2)
    parser.add_argument("--target_norm_min_var", type=float, default=1e-2)
    parser.add_argument("--use_gru", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no_gru_memory",
        type=str,
        choices=("zeros", "projection", "masked_gru"),
        default="zeros",
    )
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

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

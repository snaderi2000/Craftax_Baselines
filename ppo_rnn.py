# import argparse
# import os
# import sys

# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import numpy as np
# import optax
# import time

# from flax.training import orbax_utils
# from orbax.checkpoint import (
#     PyTreeCheckpointer,
#     CheckpointManagerOptions,
#     CheckpointManager,
# )

# import wandb
# from flax.linen.initializers import constant, orthogonal
# from typing import Sequence, NamedTuple, Dict, Union, Tuple, Optional
# from flax.training.train_state import TrainState
# import distrax
# import functools
# from dataclasses import field
# import math

# from wrappers import (
#     LogWrapper,
#     OptimisticResetVecEnvWrapper,
#     BatchEnvWrapper,
#     AutoResetEnvWrapper,
# )
# from logz.batch_logging import create_log_dict, batch_log

# from craftax.craftax_env import make_craftax_env_from_name

# # Code adapted from the original implementation made by Chris Lu
# # Original code located at https://github.com/luchris429/purejaxrl


# class ScannedRNN(nn.Module):
#     @functools.partial(
#         nn.scan,
#         variable_broadcast="params",
#         in_axes=0,
#         out_axes=0,
#         split_rngs={"params": False},
#     )
#     @nn.compact
#     def __call__(self, carry, x):
#         """Applies the module."""
#         rnn_state = carry
#         ins, resets = x
#         rnn_state = jnp.where(
#             resets[:, np.newaxis],
#             self.initialize_carry(ins.shape[0], ins.shape[1]),
#             rnn_state,
#         )
#         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
#         return new_rnn_state, y

#     @staticmethod
#     def initialize_carry(batch_size, hidden_size):
#         # Use a dummy key since the default state init fn is just zeros.
#         cell = nn.GRUCell(features=hidden_size)
#         return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# class FanInInitReLULayer(nn.Module):
#     inchan: int
#     outchan: int
#     layer_type: str = "conv"
#     init_scale: float = 1.0
#     kernel_size: Union[int, Sequence[int]] = 3
#     strides: Union[int, Sequence[int]] = 1
#     padding: Union[str, int, Sequence[int]] = "SAME"

#     # normalization flags…
#     batch_norm: bool = False
#     batch_norm_kwargs: Dict    = field(default_factory=dict)
#     group_norm_groups: Optional[int] = None
#     layer_norm: bool = False

#     # activation & training‐mode
#     use_activation: bool = True
#     train: bool          = True             # control BatchNorm mode

#     @nn.compact
#     def __call__(self, x):
#         # 1) Normalization (if any)
#         if self.batch_norm:
#             x = nn.BatchNorm(
#                 use_running_average=not self.train,
#                 **self.batch_norm_kwargs
#             )(x)
#         elif self.group_norm_groups is not None:
#             x = nn.GroupNorm(num_groups=self.group_norm_groups)(x)
#         elif self.layer_norm:
#             x = nn.LayerNorm()(x)

#         # 2) Core layer
#         if self.layer_type in ("conv", "conv2d", "conv3d"):
#             x = nn.Conv(
#                 features=self.outchan,
#                 kernel_size=self.kernel_size,
#                 strides=self.strides,
#                 padding=self.padding,
#                 use_bias=(not (self.batch_norm
#                            or self.group_norm_groups
#                            or self.layer_norm)),
#                 kernel_init=orthogonal(self.init_scale),
#                 bias_init=constant(0.0),
#             )(x)

#         elif self.layer_type == "linear":
#             x = nn.Dense(
#                 features=self.outchan,
#                 use_bias=True,
#                 kernel_init=orthogonal(self.init_scale),
#                 bias_init=constant(0.0),
#             )(x)
#         else:
#             raise ValueError(f"Unsupported layer_type: {self.layer_type}")

#         # 3) Activation
#         if self.use_activation:
#             x = nn.relu(x)

#         return x

# class CnnBasicBlock(nn.Module):
#     inchan: int
#     init_scale: float = 1.0
#     init_norm_kwargs: Dict = field(default_factory=dict)
#     train: bool = True

#     @nn.compact
#     def __call__(self, x):
#         conv0 = FanInInitReLULayer(
#             inchan=self.inchan,
#             outchan=self.inchan,
#             layer_type="conv",
#             kernel_size=3,
#             padding=1,
#             init_scale=math.sqrt(self.init_scale),
#             **self.init_norm_kwargs,
#             train=self.train
#         )
#         conv1 = FanInInitReLULayer(
#             inchan=self.inchan,
#             outchan=self.inchan,
#             layer_type="conv",
#             kernel_size=3,
#             padding=1,
#             init_scale=math.sqrt(self.init_scale),
#             **self.init_norm_kwargs,
#             train=self.train
#         )
#         return x + conv1(conv0(x))

# class CnnDownStack(nn.Module):
#     inchan: int
#     nblock: int
#     outchan: int
#     init_scale: float = 1.0
#     pool: bool = True
#     post_pool_groups: Optional[int] = None
#     init_norm_kwargs: Dict = field(default_factory=dict)
#     first_conv_norm: bool = False
#     train: bool = True

#     @nn.compact
#     def __call__(self, x):
#         # first convolution
#         first_norm_kwargs = dict(self.init_norm_kwargs)
#         if not self.first_conv_norm:
#             first_norm_kwargs.update({"batch_norm": False, "group_norm_groups": None})
#         x = FanInInitReLULayer(
#             inchan=self.inchan,
#             outchan=self.outchan,
#             layer_type="conv",
#             init_scale=1.0,
#             kernel_size=(3,3), padding="SAME",
#             **first_norm_kwargs,
#             train=self.train
#         )(x)
#         # optional pooling + post-norm
#         if self.pool:
#             x = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
#             if self.post_pool_groups is not None:
#                 x = nn.GroupNorm(num_groups=self.post_pool_groups)(x)
#         # residual blocks
#         for _ in range(self.nblock):
#             x = CnnBasicBlock(
#                 inchan=self.outchan,
#                 init_scale=self.init_scale / math.sqrt(self.nblock),
#                 init_norm_kwargs=self.init_norm_kwargs,
#                 train=self.train
#             )(x)
#         return x

# class ImpalaCNN_RNN(nn.Module):
#     inshape: Sequence[int]
#     chans: Sequence[int]
#     nblock: int
#     init_norm_kwargs: Dict = field(default_factory=lambda: {'batch_norm': True, 'batch_norm_kwargs': {'momentum': 0.99}})
#     first_conv_norm: bool = False
#     train: bool = True

#     post_pool_groups: Optional[int] = None

#     @nn.compact
#     def __call__(self, x):
#         # x: [B, H, W, C]  (NHWC)
#         c, h, w = self.inshape
#         # apply down stacks
#         for i, outchan in enumerate(self.chans):
#             use_first_norm = True if i > 0 else self.first_conv_norm
#             x = CnnDownStack(
#                 inchan=c,
#                 nblock=self.nblock,
#                 outchan=outchan,
#                 init_scale=1.0 / math.sqrt(len(self.chans)),
#                 pool=True,
#                 post_pool_groups=self.post_pool_groups,
#                 init_norm_kwargs=self.init_norm_kwargs,
#                 first_conv_norm=use_first_norm,
#                 train=self.train
#             )(x)
#             # update channel count for next stack
#             c = outchan
#         x = nn.relu(x)           # <<< before flatten
#         # flatten
#         x = x.reshape((x.shape[0], -1))
#         return x


# class ActorCriticRNN(nn.Module):
#     action_dim: Sequence[int]
#     config: Dict
#     # Added for ImpalaCNN_RNN
#     cnn_chans: Sequence[int] = (64, 64, 128)
#     rnn_hidden: int = 256
#     head_width: int = 2048
#     n_res_blocks: int = 2
#     train: bool = True

#     @nn.compact
#     def __call__(self, hidden, x):
#         obs, dones = x
#         # Impala CNN for observation processing (B.1)
#         # Permute to NHWC for CNN
#         obs_permuted = jnp.transpose(obs, (0, 1, 3, 4, 2)) # [T,B,H,W,C]
        
#         # Reshape for ImpalaCNN_RNN if necessary (remove time dim)
#         T, B, H, W, C = obs_permuted.shape
#         obs_reshaped = obs_permuted.reshape(T * B, H, W, C)

#         z = ImpalaCNN_RNN(
#             inshape=(C, H, W),
#             chans=self.cnn_chans,
#             nblock=self.n_res_blocks,
#             first_conv_norm=True,
#             post_pool_groups=None,
#             train=self.train
#         )(obs_reshaped)
        
#         # Reshape back to include time dim
#         z = z.reshape(T, B, -1)

#         # Pre-RNN projection to rnn_hidden
#         x_rnn_in = nn.LayerNorm()(z)
#         x_rnn_in = nn.Dense(self.rnn_hidden)(x_rnn_in)
#         x_rnn_in = nn.relu(x_rnn_in)

#         rnn_in = (x_rnn_in, dones)
#         hidden, embedding = ScannedRNN()(hidden, rnn_in)
#         embedding = nn.relu(embedding) # apply relu after GRU output, as in ActorCriticConvRNN

#         # Shared embedding for actor/critic heads
#         shared = jnp.concatenate([z, embedding], axis=-1)

#         # Actor head
#         actor_head = nn.LayerNorm()(shared)
#         actor_head = nn.Dense(self.head_width)(actor_head)
#         actor_head = nn.relu(actor_head)
#         for _ in range(self.n_res_blocks):
#             res = actor_head
#             actor_head = nn.Dense(self.head_width)(actor_head); actor_head = nn.relu(actor_head)
#             actor_head = nn.Dense(self.head_width)(actor_head)
#             actor_head = nn.relu(actor_head + res)
#         actor_head = nn.LayerNorm()(actor_head)
#         actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_head)
#         pi = distrax.Categorical(logits=actor_mean)

#         # Critic head
#         critic_head = nn.LayerNorm()(shared)
#         critic_head = nn.Dense(self.head_width)(critic_head)
#         critic_head = nn.relu(critic_head)
#         for _ in range(self.n_res_blocks):
#             res = critic_head
#             critic_head = nn.Dense(self.head_width)(critic_head); critic_head = nn.relu(critic_head)
#             critic_head = nn.Dense(self.head_width)(critic_head)
#             critic_head = nn.relu(critic_head + res)
#         critic_head = nn.LayerNorm()(critic_head)
#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic_head)

#         return hidden, pi, jnp.squeeze(critic, axis=-1)


# class Transition(NamedTuple):
#     done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     obs: jnp.ndarray
#     info: jnp.ndarray


# def make_train(config):
#     config["NUM_UPDATES"] = (
#         config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
#     )
#     config["MINIBATCH_SIZE"] = (
#         config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
#     )

#     # Create environment
#     env = make_craftax_env_from_name(
#         config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
#     )
#     env_params = env.default_params

#     # Wrap with some extra logging
#     env = LogWrapper(env)

#     # Wrap with a batcher, maybe using optimistic resets
#     if config["USE_OPTIMISTIC_RESETS"]:
#         env = OptimisticResetVecEnvWrapper(
#             env,
#             num_envs=config["NUM_ENVS"],
#             reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
#         )
#     else:
#         env = AutoResetEnvWrapper(env)
#         env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

#     def linear_schedule(count):
#         frac = (
#             1.0
#             - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
#             / config["NUM_UPDATES"]
#         )
#         return config["LR"] * frac

#     def train(rng):
#         # INIT NETWORK
#         network = ActorCriticRNN(env.action_space(env_params).n, config=config)
#         rng, _rng = jax.random.split(rng)
#         init_x = (
#             jnp.zeros(
#                 (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
#             ),
#             jnp.zeros((1, config["NUM_ENVS"])),
#         )
#         init_hstate = ScannedRNN.initialize_carry(
#             config["NUM_ENVS"], config["LAYER_SIZE"]
#         )
#         network_params = network.init(_rng, init_hstate, init_x)
#         if config["ANNEAL_LR"]:
#             tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(learning_rate=linear_schedule, eps=1e-5),
#             )
#         else:
#             tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(config["LR"], eps=1e-5),
#             )
#         train_state = TrainState.create(
#             apply_fn=network.apply,
#             params=network_params,
#             tx=tx,
#         )

#         # INIT ENV
#         rng, _rng = jax.random.split(rng)
#         obsv, env_state = env.reset(_rng, env_params)
#         init_hstate = ScannedRNN.initialize_carry(
#             config["NUM_ENVS"], config["LAYER_SIZE"]
#         )

#         # TRAIN LOOP
#         def _update_step(runner_state, unused):
#             # COLLECT TRAJECTORIES
#             def _env_step(runner_state, unused):
#                 (
#                     train_state,
#                     env_state,
#                     last_obs,
#                     last_done,
#                     hstate,
#                     rng,
#                     update_step,
#                 ) = runner_state
#                 rng, _rng = jax.random.split(rng)

#                 # SELECT ACTION
#                 ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
#                 hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
#                 action = pi.sample(seed=_rng)
#                 value, action = (
#                     value.squeeze(0),
#                     action.squeeze(0),
#                 )

#                 # STEP ENV
#                 rng, _rng = jax.random.split(rng)
#                 obsv, env_state, reward, done, info = env.step(
#                     _rng, env_state, action, env_params
#                 )
#                 transition = Transition(
#                     last_done, action, value, reward, last_obs, info
#                 )
#                 runner_state = (
#                     train_state,
#                     env_state,
#                     obsv,
#                     done,
#                     hstate,
#                     rng,
#                     update_step,
#                 )
#                 return runner_state, transition

#             initial_hstate = runner_state[-3]
#             runner_state, traj_batch = jax.lax.scan(
#                 _env_step, runner_state, None, config["NUM_STEPS"]
#             )

#             # CALCULATE ADVANTAGE
#             (
#                 train_state,
#                 env_state,
#                 last_obs,
#                 last_done,
#                 hstate,
#                 rng,
#                 update_step,
#             ) = runner_state
#             ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
#             _, _, last_val = network.apply(train_state.params, hstate, ac_in)
#             last_val = last_val.squeeze(0)

#             def _calculate_gae(traj_batch, last_val, last_done):
#                 def _get_advantages(carry, transition):
#                     gae, next_value, next_done = carry
#                     done, value, reward = (
#                         transition.done,
#                         transition.value,
#                         transition.reward,
#                     )
#                     delta = (
#                         reward + config["GAMMA"] * next_value * (1 - next_done) - value
#                     )
#                     gae = (
#                         delta
#                         + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
#                     )
#                     return (gae, value, done), gae

#                 _, advantages = jax.lax.scan(
#                     _get_advantages,
#                     (jnp.zeros_like(last_val), last_val, last_done),
#                     traj_batch,
#                     reverse=True,
#                     unroll=16,
#                 )
#                 return advantages, advantages + traj_batch.value

#             advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

#             # UPDATE NETWORK
#             def _update_epoch(update_state, unused):
#                 def _update_minbatch(train_state, batch_info):
#                     init_hstate, traj_batch, advantages, targets, logp_old_mb = batch_info # B.2 - added logp_old_mb

#                     def _loss_fn(params, init_hstate, traj_batch, gae, targets):
#                         # RERUN NETWORK
#                         _, pi_new_seq, value = network.apply(
#                             params, init_hstate[0], (traj_batch.obs, traj_batch.done)
#                         )
#                         # Compute log_prob and entropy directly from logits (A.2)
#                         logits = pi_new_seq.logits
#                         ls = jax.nn.log_softmax(logits, axis=-1)
#                         logp_new = jnp.take_along_axis(ls, traj_batch.action[..., None], axis=-1).squeeze(-1)
#                         p = jnp.exp(ls)
#                         entropy = (-p * ls).sum(axis=-1).mean()

#                         # CALCULATE VALUE LOSS
#                         value_pred_clipped = traj_batch.value + (
#                             value - traj_batch.value
#                         ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
#                         value_losses = jnp.square(value - targets)
#                         value_losses_clipped = jnp.square(value_pred_clipped - targets)
#                         value_loss = (
#                             0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
#                         )

#                         # CALCULATE ACTOR LOSS
#                         # Use logp_old from the frozen policy (B.2)
#                         ratio = jnp.exp(logp_new - logp_old_mb)
#                         gae = (gae - gae.mean()) / (gae.std() + 1e-8)
#                         loss_actor1 = ratio * gae
#                         loss_actor2 = (
#                             jnp.clip(
#                                 ratio,
#                                 1.0 - config["CLIP_EPS"],
#                                 1.0 + config["CLIP_EPS"],
#                             )
#                             * gae
#                         )
#                         loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
#                         loss_actor = loss_actor.mean()
#                         # entropy = pi_new_seq.entropy().mean() # A.2 - Removed, already computed

#                         total_loss = (
#                             loss_actor
#                             + config["VF_COEF"] * value_loss
#                             - config["ENT_COEF"] * entropy
#                         )
#                         return total_loss, (value_loss, loss_actor, entropy)

#                     grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
#                     total_loss, grads = grad_fn(
#                         train_state.params, init_hstate, traj_batch, advantages, targets
#                     )
#                     train_state = train_state.apply_gradients(grads=grads)
#                     return train_state, total_loss

#                 (
#                     train_state,
#                     init_hstate,
#                     traj_batch,
#                     advantages,
#                     targets,
#                     rng,
#                 ) = update_state

#                 # Freeze old params and calculate logp_old_seq once per epoch (B.2)
#                 params_old = train_state.params
#                 # Make sure the input shape matches the network's __call__ method for evaluation
#                 # It expects (hidden_state, (obs, dones))
#                 # traj_batch.obs is (NUM_STEPS, NUM_ENVS, *OBS_SHAPE)
#                 # traj_batch.done is (NUM_STEPS, NUM_ENVS)
                
#                 # Need to handle the time and batch dimensions for evaluation
#                 # The network expects (T, B, ...) for obs and (T, B) for dones
#                 # But apply expects (B, ...) for obs for initial shape inference.
#                 # So, we pass a dummy batch size of 1 for init and then use the full batch for apply.

#                 # Reshape traj_batch.obs and traj_batch.done for network.apply to simulate (1, NUM_ENVS, ...)
#                 # Since network.apply expects (HSTATE, (OBS, DONES)), where OBS is (B, H, W, C) and DONES is (B)
#                 # and we have (T, B, H, W, C) and (T, B) for traj_batch

#                 # We need to pass the full (T*B, ...) for the CNN part and then reshape for the RNN part.
#                 # The network's __call__ expects (hidden, x) where x = (obs, dones)
#                 # and obs is (T, B, H, W, C) and dones is (T, B)

#                 # For the evaluation pass to calculate logp_old_seq, we need to pass the full trajectory batch to the network.
#                 # The network.apply expects (hidden_state, (obs_batch, dones_batch))
#                 # hidden_state is (NUM_ENVS, LAYER_SIZE)
#                 # obs_batch is (NUM_STEPS, NUM_ENVS, H, W, C)
#                 # dones_batch is (NUM_STEPS, NUM_ENVS)

#                 _, pi_old_seq, _ = network.apply(
#                     params_old, init_hstate[0], (traj_batch.obs, traj_batch.done)
#                 )
#                 ls_old = jax.nn.log_softmax(pi_old_seq.logits, axis=-1)
#                 logp_old_seq = jnp.take_along_axis(
#                     ls_old, traj_batch.action[..., None], axis=-1
#                 ).squeeze(-1)  # [T,B]

#                 rng, _rng = jax.random.split(rng)
#                 permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
#                 batch = (init_hstate, traj_batch, advantages, targets, logp_old_seq) # B.2 - added logp_old_seq

#                 shuffled_batch = jax.tree.map(
#                     lambda x: jnp.take(x, permutation, axis=1), batch
#                 )

#                 minibatches = jax.tree.map(
#                     lambda x: jnp.swapaxes(
#                         jnp.reshape(
#                             x,
#                             [x.shape[0], config["NUM_MINIBATCHES"], -1]
#                             + list(x.shape[2:]),
#                         ),
#                         1,
#                         0,
#                     ),
#                     shuffled_batch,
#                 )

#                 train_state, total_loss = jax.lax.scan(
#                     _update_minbatch, train_state, minibatches
#                 )
#                 update_state = (
#                     train_state,
#                     init_hstate,
#                     traj_batch,
#                     advantages,
#                     targets,
#                     rng,
#                 )
#                 return update_state, total_loss

#             init_hstate = initial_hstate[None, :]  # TBH
#             update_state = (
#                 train_state,
#                 init_hstate,
#                 traj_batch,
#                 advantages,
#                 targets,
#                 rng,
#             )
#             update_state, loss_info = jax.lax.scan(
#                 _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
#             )
#             train_state = update_state[0]
#             metric = jax.tree.map(
#                 lambda x: (x * traj_batch.info["returned_episode"]).sum()
#                 / traj_batch.info["returned_episode"].sum(),
#                 traj_batch.info,
#             )
#             rng = update_state[-1]
#             if config["DEBUG"] and config["USE_WANDB"]:

#                 def callback(metric, update_step):
#                     to_log = create_log_dict(metric, config)
#                     batch_log(update_step, to_log, config)

#                 if (update_step % 20) == 0: # D.1
#                     jax.debug.callback(callback, metric, update_step)

#             runner_state = (
#                 train_state,
#                 env_state,
#                 last_obs,
#                 last_done,
#                 hstate,
#                 rng,
#                 update_step + 1,
#             )
#             return runner_state, metric

#         rng, _rng = jax.random.split(rng)
#         runner_state = (
#             train_state,
#             env_state,
#             obsv,
#             jnp.zeros((config["NUM_ENVS"]), dtype=bool),
#             init_hstate,
#             _rng,
#             0,
#         )
#         runner_state, metric = jax.lax.scan(
#             _update_step, runner_state, None, config["NUM_UPDATES"]
#         )
#         return {"runner_state": runner_state, "metric": metric}

#     return train


# def run_ppo(config):
#     config = {k.upper(): v for k, v in config.__dict__.items()}

#     if config["USE_WANDB"]:
#         wandb.init(
#             project=config["WANDB_PROJECT"],
#             entity=config["WANDB_ENTITY"],
#             config=config,
#             name=config["ENV_NAME"]
#             + "-PPO_RNN-"
#             + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
#             + "M",
#         )

#     rng = jax.random.PRNGKey(config["SEED"])
#     rngs = jax.random.split(rng, config["NUM_REPEATS"])

#     train_jit = jax.jit(make_train(config))
#     train_vmap = jax.vmap(train_jit)

#     t0 = time.time()
#     out = train_vmap(rngs)
#     t1 = time.time()
#     print("Time to run experiment", t1 - t0)
#     print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

#     if config["USE_WANDB"]:

#         def _save_network(rs_index, dir_name):
#             train_states = out["runner_state"][rs_index]
#             train_state = jax.tree.map(lambda x: x[0], train_states)
#             orbax_checkpointer = PyTreeCheckpointer()
#             options = CheckpointManagerOptions(max_to_keep=1, create=True)
#             path = os.path.join(wandb.run.dir, dir_name)
#             checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
#             print(f"saved runner state to {path}")
#             save_args = orbax_utils.save_args_from_target(train_state)
#             checkpoint_manager.save(
#                 config["TOTAL_TIMESTEPS"],
#                 train_state,
#                 save_kwargs={"save_args": save_args},
#             )

#         if config["SAVE_POLICY"]:
#             _save_network(0, "policies")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
#     parser.add_argument(
#         "--num_envs",
#         type=int,
#         default=1024,
#     )
#     parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
#     parser.add_argument("--lr", type=float, default=2e-4)
#     parser.add_argument("--num_steps", type=int, default=64)
#     parser.add_argument("--update_epochs", type=int, default=4)
#     parser.add_argument("--num_minibatches", type=int, default=8)
#     parser.add_argument("--gamma", type=float, default=0.99)
#     parser.add_argument("--gae_lambda", type=float, default=0.8)
#     parser.add_argument("--clip_eps", type=float, default=0.15) # A.3
#     parser.add_argument("--ent_coef", type=float, default=0.01)
#     parser.add_argument("--vf_coef", type=float, default=0.5)
#     parser.add_argument("--max_grad_norm", type=float, default=1.0)
#     parser.add_argument("--activation", type=str, default="tanh")
#     parser.add_argument(
#         "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
#     )
#     parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
#     parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
#     parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
#     parser.add_argument(
#         "--use_wandb", action=argparse.BooleanOptionalAction, default=True
#     )
#     parser.add_argument(
#         "--save_policy", action=argparse.BooleanOptionalAction, default=False
#     )
#     parser.add_argument("--num_repeats", type=int, default=1)
#     parser.add_argument("--layer_size", type=int, default=512)
#     parser.add_argument("--wandb_project", type=str)
#     parser.add_argument("--wandb_entity", type=str)
#     parser.add_argument(
#         "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
#     )
#     parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

#     args, rest_args = parser.parse_known_args(sys.argv[1:])
#     if rest_args:
#         raise ValueError(f"Unknown args {rest_args}")

#     if args.seed is None:
#         args.seed = np.random.randint(2**31)

#     if args.jit:
#         run_ppo(args)
#     else:
#         with jax.disable_jit():
#             run_ppo(args)


import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
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
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
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
            params=network_params,
            tx=tx,
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
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    rng,
                    update_step,
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
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

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
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, advs, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        
                        new_logp = pi.log_prob(traj_batch.action)
                        ratio   = jnp.exp(new_logp - traj_batch.log_prob)

                        unclipped = ratio * advs
                        clipped   = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * advs
                        loss_actor = -jnp.minimum(unclipped, clipped).mean()
                        
                        value_loss = 0.5 * jnp.square(value - targets).mean() # scalar

                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"]  * value_loss   # e.g. 0.5
                            - config["ENT_COEF"] * entropy      # e.g. 0.01
                        )

                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, value_loss, loss_actor, entropy)
                    return train_state, losses

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

            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

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
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            _rng,
            0,
        )
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

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
import threading

# Compatibility for newer Optax/Flax code running on JAX versions before
# `jax.tree` was introduced.
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)

    jax.tree = _JaxTreeCompat()
if not hasattr(jax.tree, 'structure'):
    jax.tree.structure = jax.tree_util.tree_structure

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
    ClassicDeathPenaltyWrapper,
)
from logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name
from ppo_data_parallel import active as parallel_active, shard_batch, replicate

_CHECKPOINT_WRITE_LOCK = threading.Lock()

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class ScannedRNN(nn.Module):
    spatial_gate: bool = False

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
        if self.spatial_gate:
            from models.sru_gru import spatial_multiplier
            scale = spatial_multiplier(ins)
            # Reuse the stock Flax projections and reset convention. Only the
            # candidate preactivation gets the SRU's multiplicative transform.
            activation = lambda candidate: jnp.tanh(scale * candidate)
            return nn.GRUCell(features=ins.shape[1], activation_fn=activation)(rnn_state, ins)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


def initialize_recurrent_carry(batch_size, config):
    if config.get('NEURAL_MAP_CHANNELS', 0):
        from models.actor_critic_neural_map import initialize_carry
        return initialize_carry(batch_size, config)
    return ScannedRNN.initialize_carry(batch_size, config['LAYER_SIZE'])


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x, *, return_memory=False):
        if self.config.get('NEURAL_MAP_CHANNELS', 0):
            if return_memory:
                raise ValueError('Recurrent sequence readout requires the standard GRU/SRU.')
            from models.actor_critic_neural_map import NeuralMapActorCritic
            return NeuralMapActorCritic(self.action_dim, self.config)(hidden, x)
        obs, dones = x
        world = None
        if self.config.get('WORLD_VIEW'):
            from symbolic_world_view import LOCAL, WorldReader
            world = WorldReader(self.config['LAYER_SIZE'],name='world_reader')(obs[...,LOCAL:])
            obs = obs[...,:LOCAL]
        encoder = self.config.get('SYMBOLIC_ENCODER', 'mlp')
        if encoder == 'map_mlp':
            cells=self.config.get('MAP_VIEW_SIZE',33)**2
            from symbolic_survival_map import FEATURE_SIZE
            extra=FEATURE_SIZE if self.config.get('MAP_SURVIVAL_FEATURES','none')!='none' else 0
            if obs.shape[-1]!=1345+2*cells+extra:raise ValueError('Map-GRU observation shape mismatch')
            terrain=nn.Embed(18,self.config.get('MAP_EMBEDDING_SIZE',4),name='terrain_embedding')(
                obs[...,1345:1345+cells].astype(jnp.int32))
            obs=jnp.concatenate((obs[...,:1345],terrain.reshape((*obs.shape[:-1],-1)),obs[...,1345+cells:]),axis=-1)
        if encoder in ('spatial', 'spatial_map'):
            from models.conditioned_symbolic import symbolic_spatial_features, symbolic_map_features
            # Put the partitioned environment dimension before time while the
            # per-observation CNN flattens its leading dimensions. Restore the
            # time-major layout before the GRU; parameter names/shapes match.
            env_major = parallel_active(self.config)
            feature_obs = jnp.swapaxes(obs, 0, 1) if env_major else obs
            features, shape = (symbolic_map_features(feature_obs, self.config.get('MAP_VIEW_SIZE', 33))
                               if encoder == 'spatial_map' else symbolic_spatial_features(feature_obs))
            obs = features.reshape((*shape, features.shape[-1]))
            if env_major:
                obs = jnp.swapaxes(obs, 0, 1)
        embedding = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        if world is not None:
            embedding = embedding + world
        embedding = nn.relu(embedding)
        observation_embedding = embedding
        residual_heads = self.config.get('SRU_RESIDUAL_ACTOR_CRITIC', False)
        if residual_heads:
            embedding = nn.relu(nn.Dense(self.config['LAYER_SIZE'], name='memory_input')(
                nn.LayerNorm(name='memory_input_norm')(embedding)))

        # Apply identically during rollout, PPO sequence replay and inference.
        # Keep the exact same GRU parameters for the no-history control.
        if self.config.get('RESET_GRU_EVERY_STEP', False):
            dones = jnp.ones_like(dones, dtype=bool)
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(spatial_gate=self.config.get('SRU_GRU', False))(hidden, rnn_in)
        memory_sequence = embedding
        if residual_heads:
            from models.sru_gru import ResidualDecisionHead
            features = jnp.concatenate((observation_embedding, nn.relu(embedding)), axis=-1)
            width = self.config['LAYER_SIZE']
            logits = ResidualDecisionHead(width, self.action_dim, .01, name='actor_head')(features)
            value = ResidualDecisionHead(width, 1, 1., name='critic_head')(features)
            outputs = hidden, distrax.Categorical(logits=logits), value[..., 0]
            return (*outputs, memory_sequence) if return_memory else outputs
        if self.config.get('GRU_CONCAT_OBSERVATION', False):
            embedding = jnp.concatenate((observation_embedding, embedding), axis=-1)

        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        outputs = (hidden, pi, jnp.squeeze(critic, axis=-1))
        return (*outputs, memory_sequence) if return_memory else outputs


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def recurrent_ppo_loss(params, network, init_hstate, traj_batch, gae, targets, config, decoder=None):
    """PPO loss over a GLOBAL recurrent minibatch, including normalization."""
    outputs = network.apply(params, init_hstate, (traj_batch.obs, traj_batch.done),
                            **({'return_memory': True} if decoder is not None else {}))
    _, pi, value = outputs[:3]
    log_prob = pi.log_prob(traj_batch.action)
    value_pred_clipped = traj_batch.value + (value-traj_batch.value).clip(
        -config['CLIP_EPS'], config['CLIP_EPS'])
    value_losses = jnp.square(value-targets)
    value_losses_clipped = jnp.square(value_pred_clipped-targets)
    value_loss = .5*jnp.maximum(value_losses, value_losses_clipped).mean()
    ratio = jnp.exp(log_prob-traj_batch.log_prob)
    gae = (gae-gae.mean())/(gae.std()+1e-8)
    loss_actor1 = ratio*gae
    loss_actor2 = jnp.clip(ratio, 1.-config['CLIP_EPS'], 1.+config['CLIP_EPS'])*gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    total_loss = loss_actor+config['VF_COEF']*value_loss-config['ENT_COEF']*entropy
    if config.get('SRU_GRU') and config.get('SRU_IDENTITY_COEF', 0.):
        from models.sru_gru import gate_penalty
        total_loss += config['SRU_IDENTITY_COEF'] * gate_penalty(params)
    metrics = (value_loss, loss_actor, entropy)
    if decoder is not None:
        from models.sru_map_reconstruction import STRIDE, reconstruction_loss
        loss, recall = reconstruction_loss(decoder, outputs[3][::STRIDE], traj_batch.info['map_queries'][::STRIDE])
        total_loss += config['SRU_MAP_COEF'] * loss
        metrics += (loss, *recall)
    return total_loss, metrics


def make_train(config):
    if config.get('DATA_PARALLEL_DEVICES', 1) > 1 and not parallel_active(config):
        raise ValueError('Use ppo_data_parallel.compile_train/execute_train for multi-device PPO')
    config.setdefault('START_TIMESTEPS',0)
    config["TOTAL_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    lr_horizon=config.get('LR_HORIZON_TIMESTEPS') or config['TOTAL_TIMESTEPS']
    if lr_horizon<config['TOTAL_TIMESTEPS']:raise ValueError('LR horizon must cover the training target')
    config['LR_HORIZON_UPDATES']=lr_horizon//config['NUM_STEPS']//config['NUM_ENVS']
    config['NUM_UPDATES']=(config['TOTAL_TIMESTEPS']-config['START_TIMESTEPS'])//config['NUM_STEPS']//config['NUM_ENVS']
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], False if config.get('ALL22_TERMINAL_BONUS') is not None or config.get('MOTION_INPUTS',False) or config.get('SYMBOLIC_ENCODER') in ('spatial_map','map_mlp') or config.get("DEATH_PENALTY", 0.) or config.get('LAVA_PENALTY',0.) or config.get('DIAMOND_REWARD', 1.) != 1. or config.get('PLANT_REWARD', 1.) != 1. else not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    if config.get("DEATH_PENALTY", 0.):
        env = ClassicDeathPenaltyWrapper(env, config["DEATH_PENALTY"])
    if config.get('LAVA_PENALTY',0.):
        from wrappers import ClassicLavaPenaltyWrapper
        env=ClassicLavaPenaltyWrapper(env,config['LAVA_PENALTY'])
    if config.get('DIAMOND_REWARD', 1.) != 1. or config.get('PLANT_REWARD', 1.) != 1.:
        from wrappers import ClassicAchievementRewardWrapper
        env = ClassicAchievementRewardWrapper(env, config.get('DIAMOND_REWARD', 1.), config.get('PLANT_REWARD', 1.))

    if config.get('ALL22_TERMINAL_BONUS') is not None:
        from wrappers import ClassicCompletionWrapper
        if config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1":
            raise ValueError("Completion bonus requires Craftax Classic Symbolic")
        env = ClassicCompletionWrapper(env, config['ALL22_TERMINAL_BONUS'])

    from symbolic_map_memory import with_map_memory
    env = with_map_memory(env, config)
    from symbolic_motion_inputs import with_motion_inputs
    env = with_motion_inputs(env, config)
    use_map = config.get('SRU_MAP_COEF', 0.) > 0
    if use_map:
        from models.sru_map_reconstruction import TargetWrapper, MapTrainState, init_decoder, sample_queries, STRIDE, QUERY_COUNT
        if not config.get('SRU_GRU') or config.get('USE_OPTIMISTIC_RESETS') or config.get('CTEC_COEF'):
            raise ValueError('SRU map reconstruction requires the standard-reset SRU')
        env = TargetWrapper(env)

    # Wrap with some extra logging
    env = LogWrapper(env)
    use_replay = config.get('SRU_FAILURE_REPLAY', False)
    if use_replay and (not config.get('SRU_GRU') or use_map
            or config['USE_OPTIMISTIC_RESETS'] or config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1'):
        raise ValueError('Failure seed replay requires the regular, standard-reset Classic SRU')

    # Wrap with a batcher, maybe using optimistic resets
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        if use_replay:
            from failure_seed_replay import FailureSeedAutoReset
            env = FailureSeedAutoReset(env)
        else:
            env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["LR_HORIZON_UPDATES"]
        )
        return config["LR"] * frac

    use_sru = config.get('SRU_GRU', False)
    use_sru_schedule = config.get('SRU_LR_ANNEAL_FROM') is not None or config.get('SRU_LR_ANNEAL_UNTIL') is not None
    if use_sru_schedule:
        from models.sru_gru import apply_scaled_updates, learning_rate_scale
        start,end = config.get('SRU_LR_ANNEAL_FROM'),config.get('SRU_LR_ANNEAL_UNTIL')
        batch = config['NUM_STEPS'] * config['NUM_ENVS']
        if (not use_sru or config['ANNEAL_LR'] or type(start) is not int or type(end) is not int
                or not 0 <= start < end or start % batch or end % batch
                or not 0 < config['SRU_LR_FINAL_FRACTION'] <= 1):
            raise ValueError('SRU learning-rate schedule requires aligned start/end steps, a positive floor, and fixed base rates')

    def optimizer():
        lr = linear_schedule if config['ANNEAL_LR'] else config['LR']
        if use_sru:
            from models.sru_gru import make_optimizer
            if config.get('SRU_RESIDUAL_ACTOR_CRITIC') and config['ANNEAL_LR']:
                gate_lr = lambda count: linear_schedule(count) * (config['SRU_GATE_LR'] / config['LR'])
                return make_optimizer(config, lr, gate_lr=gate_lr)
            return make_optimizer(config, lr)
        return optax.chain(optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
                           optax.adam(lr, eps=1e-5))

    use_ctec=config.get('CTEC_COEF',0.)>0
    if use_ctec:
        from ctec import ExplorationTrainState, initialize, update_curiosity
        state_class=ExplorationTrainState
    elif use_replay:
        from failure_seed_replay import ReplayTrainState, SLOTS
        state_class = ReplayTrainState
    else:state_class=MapTrainState if use_map else TrainState

    def extra_state(key):
        if use_replay:
            shape = (config['NUM_ENVS'], SLOTS)
            return dict(replay_keys=jnp.zeros((*shape, 2), jnp.uint32), replay_failures=jnp.zeros(shape, jnp.int32))
        if use_map:
            return {'decoder': init_decoder(config['LAYER_SIZE'], jax.random.fold_in(key, 149))}
        return {'curiosity': initialize(config, key)} if use_ctec else {}
    initial_local_params=None
    if config.get('LOCAL_POLICY_INIT') and not config.get('RESUME_FROM'):
        from symbolic_world_view import source_params
        initial_local_params=source_params(config['LOCAL_POLICY_INIT'])
    restored_train_state=None
    if config.get('RESUME_FROM'):
        if config.get('MEMORY_TYPE','gru')!='gru':raise ValueError('Recurrent resume currently supports GRU')
        restore_net=ActorCriticRNN(env.action_space(env_params).n,config=config)
        restore_params=restore_net.init(jax.random.PRNGKey(0),
            initialize_recurrent_carry(1,config),
            (jnp.zeros((1,1,*env.observation_space(env_params).shape)),jnp.zeros((1,1),bool)))
        template=state_class.create(apply_fn=restore_net.apply,params=restore_params,tx=optimizer(),
            **extra_state(jax.random.PRNGKey(0)))
        checkpoint_path=os.path.join(config['RESUME_FROM'],str(config['RESUME_STEP']))
        if os.path.isdir(os.path.join(checkpoint_path,'default')):
            checkpoint_path=os.path.join(checkpoint_path,'default')
        # Legacy CheckpointManager.restore silently returns None for a direct
        # PyTree payload, which would otherwise fall back to fresh weights.
        checkpointer = PyTreeCheckpointer()
        if use_sru:
            from models.sru_gru import has_spatial_gate, transplant_gru
            payload = checkpointer.restore(checkpoint_path)
            source = payload['params']
            if not has_spatial_gate(source):
                restored_train_state = template.replace(params=transplant_gru(restore_params, source))
                print(f'Transferred pretrained GRU weights at {config["START_TIMESTEPS"]:,}; '
                      'initialized the SRU gate to identity and restarted Adam.', flush=True)
            elif (use_map and 'decoder' not in payload) or (use_replay and 'replay_keys' not in payload):
                base = TrainState.create(apply_fn=restore_net.apply, params=restore_params, tx=optimizer())
                base = checkpointer.restore(checkpoint_path, item=base)
                restored_train_state = template.replace(params=base.params, opt_state=base.opt_state, step=base.step)
            else:
                restored_train_state = checkpointer.restore(checkpoint_path, item=template)
        else:
            restored_train_state = checkpointer.restore(checkpoint_path, item=template)
        if restored_train_state is None:raise ValueError('Checkpoint did not contain a recurrent train state')
        print(f'Loaded recurrent weights at {config["START_TIMESTEPS"]:,}; environment and hidden state reset.',flush=True)

    def train(rng):
        # INIT NETWORK
        if config.get('MEMORY_TYPE','gru') == 'transformer':
            from models.actor_critic_transformer import ActorCriticTransformer
            network = ActorCriticTransformer(env.action_space(env_params).n,config=config)
            initialize_memory = lambda: ActorCriticTransformer.initialize_carry(
                config['NUM_ENVS'],config.get('TRANSFORMER_CONTEXT',64),env.observation_space(env_params).shape[0])
        else:
            network = ActorCriticRNN(env.action_space(env_params).n, config=config)
            initialize_memory = lambda: initialize_recurrent_carry(config['NUM_ENVS'],config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = initialize_memory()
        network_params = network.init(_rng, init_hstate, init_x)
        if initial_local_params is not None:
            from symbolic_world_view import transplant
            network_params=transplant(network_params,initial_local_params)
        train_state = state_class.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimizer(),
            **extra_state(jax.random.fold_in(rng,8123)),
        )
        if restored_train_state is not None:train_state=restored_train_state
        train_state = replicate(train_state, config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        if use_replay:
            env_state = env_state.replace(keys=train_state.replay_keys, failures=train_state.replay_failures)
        init_hstate = initialize_memory()
        obsv, env_state, init_hstate = shard_batch((obsv, env_state, init_hstate), 0, config)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, rollout_index):
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
                if use_map:
                    # Label the pre-action observation, before an autoreset can replace its world.
                    def label(_):
                        keys = jax.random.split(jax.random.fold_in(_rng, 7291), config['NUM_ENVS'])
                        return jax.vmap(sample_queries)(env_state.env_state, keys)
                    map_queries = jax.lax.cond(rollout_index % STRIDE == 0, label,
                        lambda _: jnp.zeros((config['NUM_ENVS'], QUERY_COUNT, 5), jnp.int16), None)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
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
                obsv, env_state, reward, done, info = shard_batch(
                    (obsv, env_state, reward, done, info), 0, config)
                if use_map:
                    info = dict(info, map_queries=shard_batch(map_queries, 0, config))
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
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"]) if use_map else None, config["NUM_STEPS"]
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
            last_val = last_val.squeeze(0)

            curiosity_metrics={}
            if use_ctec:
                curiosity,bonus,curiosity_metrics=update_curiosity(train_state.curiosity,
                    traj_batch.obs,traj_batch.action,traj_batch.done,traj_batch.reward,
                    jax.random.fold_in(rng,9917),update_step,config)
                train_state=train_state.replace(curiosity=curiosity)
                traj_batch=traj_batch._replace(reward=traj_batch.reward+bonus)

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
            if use_sru_schedule:
                lr_scale = learning_rate_scale(config, update_step)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    batch_info = shard_batch(batch_info, 1, config)
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, decoder=None):
                        return recurrent_ppo_loss(params, network,
                            jax.tree_util.tree_map(lambda a:a[0],init_hstate),
                            traj_batch, advantages, targets, config, decoder)

                    if use_map:
                        grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
                        total_loss, (grads, decoder_grads) = grad_fn(train_state.params, train_state.decoder.params)
                        decoder = train_state.decoder.apply_gradients(grads=replicate(decoder_grads, config))
                        train_state = train_state.replace(decoder=decoder)
                    else:
                        total_loss, grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                    if use_sru:
                        from models.sru_gru import mask_warmup_gradients
                        grads = mask_warmup_gradients(grads, train_state.step, config)
                    # This constraint makes the complete global gradient
                    # available before Optax's global-norm clip and Adam.
                    grads = replicate(grads, config)
                    if use_sru_schedule:
                        train_state = apply_scaled_updates(train_state, grads, lr_scale)
                    else:
                        train_state = train_state.apply_gradients(grads=grads)
                    train_state = replicate(train_state, config)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
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

            init_hstate = jax.tree_util.tree_map(lambda a:a[None,...],initial_hstate)
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
            if use_replay:
                train_state = train_state.replace(replay_keys=env_state.keys, replay_failures=env_state.failures)
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                {k:v for k,v in traj_batch.info.items() if k != 'map_queries'},
            )
            if use_replay:
                ended = traj_batch.info['returned_episode']
                replayed = traj_batch.info['curriculum/replayed_episode'].astype(bool)
                for label, selected in (('replay', replayed), ('fresh', ~replayed)):
                    count = (ended & selected).sum()
                    event = 'replayed_all22' if label == 'replay' else 'fresh_all22'
                    hits = (traj_batch.info['curriculum/'+event] * ended).sum()
                    metric['curriculum/'+label+'_completed_count'] = count
                    metric['curriculum/'+label+'_all22_count'] = hits
            if use_map:
                map_loss, accuracy, long_hits, long_count = loss_info[1][3:]
                metric['training/map_reconstruction_loss'] = map_loss.mean()
                metric['training/map_recall_accuracy'] = accuracy.mean()
                metric['training/map_recall_age100_accuracy'] = long_hits.sum() / jnp.maximum(long_count.sum(), 1)
                metric['training/map_recall_age100_queries'] = long_count.sum() / config['UPDATE_EPOCHS']
            metric.update(curiosity_metrics)
            if use_sru:
                from models.sru_gru import gate_penalty, warmup_minibatches
                metric['training/sru_identity_penalty'] = gate_penalty(train_state.params)
                metric['training/sru_pretrained_frozen'] = (train_state.step <= warmup_minibatches(config)).astype(jnp.float32)
            if use_sru_schedule:
                metric['training/sru_gate_learning_rate'] = config['SRU_GATE_LR'] * lr_scale
                metric['training/sru_pretrained_learning_rate'] = config['LR'] * lr_scale
            elif config.get('SRU_RESIDUAL_ACTOR_CRITIC') and config['ANNEAL_LR']:
                count = train_state.step - config['NUM_MINIBATCHES'] * config['UPDATE_EPOCHS']
                metric['training/learning_rate'] = linear_schedule(count)
                metric['training/sru_gate_learning_rate'] = linear_schedule(count) * (config['SRU_GATE_LR'] / config['LR'])
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            if config["SAVE_POLICY"] and config["USE_WANDB"] and config.get("SAVE_POLICY_MILESTONES"):
                batch_size = config["NUM_ENVS"] * config["NUM_STEPS"]
                milestone_updates = jnp.asarray(tuple(
                    (step-config['START_TIMESTEPS']+batch_size-1)//batch_size for step in config["SAVE_POLICY_MILESTONES"]), jnp.int32)
                current_update = update_step+1

                def save_checkpoint(ts, completed_update):
                    # Python integer arithmetic keeps 10B labels safe with JAX int32.
                    step = config['START_TIMESTEPS']+int(np.asarray(completed_update))*batch_size
                    with _CHECKPOINT_WRITE_LOCK:
                        if parallel_active(config):
                            from ppo_data_parallel import save_checkpoint_from_callback
                            save_checkpoint_from_callback(os.path.join(wandb.run.dir, 'policies'),
                                step, ts, config['POLICY_MAX_TO_KEEP'])
                        else:
                            manager = CheckpointManager(os.path.join(wandb.run.dir, 'policies'),
                                PyTreeCheckpointer(), CheckpointManagerOptions(
                                    max_to_keep=config["POLICY_MAX_TO_KEEP"], create=True))
                            manager.save(step, ts, save_kwargs={"save_args": orbax_utils.save_args_from_target(ts)})
                            manager.wait_until_finished()
                    print(f'Saved RNN checkpoint at {step:,} steps', flush=True)

                def save_branch(unused):
                    jax.debug.callback(save_checkpoint, train_state, current_update,
                                       ordered=not parallel_active(config))
                    return jnp.asarray(0)

                jax.lax.cond(jnp.any(current_update == milestone_updates), save_branch,
                             lambda _: jnp.asarray(0), operand=None)

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
        from training_time_limit import run_update_loop
        runner_state, metric = run_update_loop(
            _update_step,runner_state,config['NUM_UPDATES'],config.get('TRAINING_DEADLINE',0.))
        return {"runner_state": runner_state, "metric": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    if (config.get('RESET_GRU_EVERY_STEP',False) or config.get('GRU_CONCAT_OBSERVATION',False)) and config.get('MEMORY_TYPE','gru')!='gru':
        raise ValueError('GRU reset/concatenation options require GRU memory')
    if config.get('MOTION_INPUTS',False) and config.get('MEMORY_TYPE','gru')!='gru':
        raise ValueError('Motion inputs currently support GRU memory only')
    from ppo_data_parallel import compile_train, validate_data_parallel
    parallel_devices = validate_data_parallel(config, jax.local_device_count())
    from ppo import _resolve_policy_checkpoint
    config['RESUME_FROM'],config['RESUME_STEP']=_resolve_policy_checkpoint(config.get('RESUME_FROM'))
    config['START_TIMESTEPS']=config['RESUME_STEP'] or 0
    if config.get('SRU_RESIDUAL_ACTOR_CRITIC') and not config.get('SRU_GRU'):
        raise ValueError('Residual SRU heads require --sru_gru')
    if config.get('SRU_GRU'):
        if (config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1' or config['MEMORY_TYPE'] != 'gru'
                or config['SYMBOLIC_ENCODER'] != 'mlp' or not config['GRU_CONCAT_OBSERVATION']
                or not config['MOTION_INPUTS'] or not (config['RESUME_FROM'] or config.get('SRU_RESIDUAL_ACTOR_CRITIC'))
                or config.get('WORLD_VIEW') or config.get('RESET_GRU_EVERY_STEP')):
            raise ValueError('SRU continuation requires the regular motion-input GRU-concat policy')
        if (not np.isfinite(config['SRU_GATE_LR']) or config['SRU_GATE_LR'] <= 0
                or not np.isfinite(config['SRU_IDENTITY_COEF']) or config['SRU_IDENTITY_COEF'] < 0
                or config['SRU_WARMUP_STEPS'] < 0):
            raise ValueError('Invalid SRU gate learning rate, identity penalty, or warmup')
    penalty = config.get('DEATH_PENALTY', 0.)
    lava_penalty=config.get('LAVA_PENALTY',0.)
    if not np.isfinite(lava_penalty) or lava_penalty<0:raise ValueError('lava_penalty must be finite and nonnegative')
    if lava_penalty and config['ENV_NAME']!='Craftax-Classic-Symbolic-v1':raise ValueError('Lava penalty requires Classic symbolic observations')
    if penalty and lava_penalty:raise ValueError('Choose either a general death penalty or a lava-only penalty')
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError('death_penalty must be finite and nonnegative')
    if penalty and config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1':
        raise ValueError('Death-penalty RNN experiment supports Classic symbolic observations')
    diamond_reward = config.get('DIAMOND_REWARD', 1.)
    if not np.isfinite(diamond_reward) or diamond_reward < 0:
        raise ValueError('diamond_reward must be finite and nonnegative')
    if diamond_reward != 1. and config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1':
        raise ValueError('Diamond-reward wrapper supports Classic symbolic observations')
    plant_reward = config.get('PLANT_REWARD', 1.)
    if not np.isfinite(plant_reward) or plant_reward < 0:
        raise ValueError('plant_reward must be finite and nonnegative')
    if (plant_reward != 1. or config.get('SYMBOLIC_ENCODER', 'mlp') == 'spatial') and config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1':
        raise ValueError('Plant reward and spatial encoder require Classic symbolic observations')
    if config.get('SYMBOLIC_ENCODER', 'mlp') == 'spatial' and config.get('MEMORY_TYPE', 'gru') != 'gru':
        raise ValueError('Spatial recurrent encoder currently supports GRU memory')
    if config['NUM_ENVS'] % config['NUM_MINIBATCHES']:
        raise ValueError('Recurrent minibatches must contain whole environment sequences')
    batch_size = config['NUM_ENVS']*config['NUM_STEPS']
    if config['TOTAL_TIMESTEPS'] < batch_size:
        raise ValueError('Training budget must contain at least one PPO update')
    config['SAVE_POLICY_MILESTONES'] = tuple(sorted(set(
        int(float(item)) for item in str(config.get('SAVE_POLICY_MILESTONES', '')).split(',')
        if item.strip() and config['START_TIMESTEPS'] < int(float(item)) < config['TOTAL_TIMESTEPS'])))
    config['POLICY_MAX_TO_KEEP'] = max(1, len(config['SAVE_POLICY_MILESTONES'])+1)
    if config['RESUME_FROM']:
        existing=sum(name.isdigit() for name in os.listdir(config['RESUME_FROM']))
        config['POLICY_MAX_TO_KEEP']=max(config['POLICY_MAX_TO_KEEP'],existing+len(config['SAVE_POLICY_MILESTONES'])+1)

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config.get("RUN_NAME") or config["ENV_NAME"]
            + "-PPO_RNN-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = compile_train(config) if parallel_devices > 1 else jax.jit(make_train(config))
    t0 = time.time()
    if parallel_devices > 1:
        out = jax.tree.map(lambda x:x[None],train_jit(rngs[0]))
    elif config.get('TRAINING_DEADLINE',0.):
        if config['NUM_REPEATS']!=1:raise ValueError('Wall-clock capped training supports one repeat per process')
        out=jax.tree.map(lambda x:x[None],train_jit(rngs[0]))
    else:
        out = jax.vmap(train_jit)(rngs)
    jax.block_until_ready(out)
    if parallel_devices > 1:
        jax.effects_barrier()
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    completed_updates=int(np.asarray(out['runner_state'][-1][0]))
    executed_steps=completed_updates*config['NUM_ENVS']*config['NUM_STEPS']
    print('New environment steps:',executed_steps)
    print("SPS: ", executed_steps / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            if parallel_devices > 1:
                from ppo_data_parallel import checkpoint_train_state
                train_state = checkpoint_train_state(train_state)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=config['POLICY_MAX_TO_KEEP'], create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config['START_TIMESTEPS']+executed_steps,
                train_state,
                save_kwargs={"save_args": save_args},
            )
            checkpoint_manager.wait_until_finished()

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")
        wandb.summary['actual_training_steps']=config['START_TIMESTEPS']+executed_steps
        wandb.summary['wall_clock_limit_reached']=completed_updates<config['NUM_UPDATES']


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
    parser.add_argument('--lr_horizon_timesteps',type=lambda x:int(float(x)),default=None,
        help='Optional annealing horizon, allowing continuation to stop without changing the original LR schedule.')
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument('--data_parallel_devices', type=int, default=1,
                        help='Shard one recurrent PPO policy across 1, 2, or 4 visible devices; batch sizes remain global.')
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument('--death_penalty', type=float, default=0.,
                        help='Additional death cost, including lava; excludes time-limit-only endings.')
    parser.add_argument('--diamond_reward', type=float, default=1.,
                        help='Total achievement reward for the FIRST diamond, replacing +1; other rewards unchanged.')
    parser.add_argument('--all22_terminal_bonus', type=float, help='End training episodes on all 22 and add this one-time bonus.')
    parser.add_argument('--plant_reward', type=float, default=1., help='Total FIRST eat-plant achievement reward.')
    parser.add_argument('--symbolic_encoder', choices=('mlp', 'spatial', 'spatial_map', 'map_mlp'), default='mlp')
    from ctec import DEFAULTS as CTEC_DEFAULTS
    parser.add_argument('--neural_map_channels',type=int,default=0)
    parser.add_argument('--neural_map_reset_every_step',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--map_survival_features',choices=('none','masked','enabled'),default='none')
    for name,value in CTEC_DEFAULTS.items():parser.add_argument('--'+name,type=type(value),default=value)
    parser.add_argument('--map_embedding_size',type=int,default=4)
    parser.add_argument('--map_view_size', type=int, default=33,
        help='Odd side length of the agent-centered remembered-terrain view; explicit actual odometry.')
    parser.add_argument('--training_deadline',type=float,default=0.,help='Unix-time deadline; stop after a PPO update and save actual progress.')
    parser.add_argument('--resume_from',type=str,default=None,help='Restore recurrent weights and optimizer; start fresh environment episodes.')
    parser.add_argument('--lava_penalty',type=float,default=0.,help='Additional cost only on terminal lava entry.')
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
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_policy_milestones', type=str, default='')
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument('--memory_type',choices=['gru','transformer'],default='gru')
    parser.add_argument('--reset_gru_every_step',action=argparse.BooleanOptionalAction,default=False,
                        help='Matched no-history control: reset GRU before every observation in rollout and PPO replay.')
    parser.add_argument('--gru_concat_observation',action=argparse.BooleanOptionalAction,default=False,
                        help='Give actor/critic the current observation embedding concatenated with the GRU output.')
    parser.add_argument('--world_view',choices=('terrain','entities','local'))
    parser.add_argument('--local_policy_init')
    parser.add_argument('--motion_inputs',action=argparse.BooleanOptionalAction,default=False,
                        help='Append previous action (17-way one-hot) and actual dx,dy; clear on episode resets.')
    parser.add_argument('--sru_gru', action='store_true', help='Add an identity-initialized spatial gate to the pretrained GRU.')
    parser.add_argument('--sru_failure_replay', action='store_true', help='Weight and retry failed training worlds until an all-22 episode.')
    parser.add_argument('--sru_warmup_steps', type=int, default=0, help='Additional environment steps training only the SRU gate.')
    parser.add_argument('--sru_gate_lr', type=float, default=3e-5)
    parser.add_argument('--sru_residual_actor_critic', action='store_true')
    parser.add_argument('--sru_identity_coef', type=float, default=0.)
    parser.add_argument('--sru_map_coef', type=float, default=0.)
    parser.add_argument('--sru_lr_anneal_from', type=int, help='Absolute environment step where SRU rate decay starts.')
    parser.add_argument('--sru_lr_anneal_until', type=int, help='Absolute environment step where both SRU learning rates reach their floor.')
    parser.add_argument('--sru_lr_final_fraction', type=float, default=.1)
    parser.add_argument('--transformer_dim',type=int,default=256)
    parser.add_argument('--transformer_heads',type=int,default=4)
    parser.add_argument('--transformer_layers',type=int,default=2)
    parser.add_argument('--transformer_context',type=int,default=64)
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

import argparse
import json
import os
import sys
import time
import threading

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

# Compatibility for newer Optax/Flax code running on JAX versions before
# jax.tree was introduced, such as jax==0.4.23.
if not hasattr(jax, "tree"):
    class _JaxTreeCompat:
        map = staticmethod(jax.tree_util.tree_map)
        leaves = staticmethod(jax.tree_util.tree_leaves)
        reduce = staticmethod(jax.tree_util.tree_reduce)

    jax.tree = _JaxTreeCompat()

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
from models.icm import ICMEncoder, ICMForward, ICMInverse
from wrappers import (
    ClassicDeathPenaltyWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from ppo_data_parallel import active as parallel_active, replicate, shard_batch

_CHECKPOINT_WRITE_LOCK = threading.Lock()

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


_SYMBOLIC_ACTOR_MODULES = frozenset(f"Dense_{i}" for i in range(4))
_SYMBOLIC_CRITIC_MODULES = frozenset(f"Dense_{i}" for i in range(4, 8))


def _symbolic_component_modules(params):
    """Return actor/critic module names and validate checkpoint compatibility."""
    modules = set(params["params"].keys())
    expected = _SYMBOLIC_ACTOR_MODULES | _SYMBOLIC_CRITIC_MODULES
    if modules != expected:
        raise ValueError(
            "Component transfer currently supports the symbolic ActorCritic "
            f"with modules {sorted(expected)}; found {sorted(modules)}."
        )
    return _SYMBOLIC_ACTOR_MODULES, _SYMBOLIC_CRITIC_MODULES


def _replace_param_component(random_params, source_params, source_component):
    """Copy one actor/critic branch from source into freshly initialized params."""
    actor_modules, critic_modules = _symbolic_component_modules(random_params)
    _symbolic_component_modules(source_params)
    source_modules = actor_modules if source_component == "actor" else critic_modules
    mixed = dict(random_params)
    mixed_root = dict(random_params["params"])
    for module_name in source_modules:
        mixed_root[module_name] = source_params["params"][module_name]
    mixed["params"] = mixed_root
    return mixed


def _mask_component_gradients(grads, train_component):
    """Zero gradients outside the selected actor/critic branch."""
    if train_component == "both":
        return grads
    actor_modules, critic_modules = _symbolic_component_modules(grads)
    if train_component == "actor":
        trainable_modules = actor_modules
    elif train_component == "critic":
        trainable_modules = critic_modules
    elif train_component == "actor_critic_head":
        # Preserve the pretrained critic representation (Dense_4--Dense_6)
        # while allowing its scalar output calibration (Dense_7) to adapt.
        trainable_modules = actor_modules | {"Dense_7"}
    else:
        raise ValueError(f"Unknown train component: {train_component}")
    masked = dict(grads)
    masked_root = dict(grads["params"])
    for module_name, module_grads in grads["params"].items():
        if module_name not in trainable_modules:
            masked_root[module_name] = jax.tree.map(jnp.zeros_like, module_grads)
    masked["params"] = masked_root
    return masked


def feedforward_ppo_loss(params, network, traj_batch, gae, targets, config):
    """Vanilla global-minibatch PPO objective used by data-parallel map MLPs."""
    from policy_temperature import tempered_policy
    pi, value = network.apply(params, traj_batch.obs)
    pi = tempered_policy(pi, config.get('POLICY_TEMPERATURE', 1.))
    log_prob = pi.log_prob(traj_batch.action)
    value_pred_clipped = traj_batch.value + (value-traj_batch.value).clip(
        -config['CLIP_EPS'], config['CLIP_EPS'])
    value_losses = jnp.square(value-targets)
    value_losses_clipped = jnp.square(value_pred_clipped-targets)
    value_loss = 0.5*jnp.maximum(value_losses,value_losses_clipped).mean()
    ratio = jnp.exp(log_prob-traj_batch.log_prob)
    gae = (gae-gae.mean())/(gae.std()+1e-8)
    loss_actor1 = ratio*gae
    loss_actor2 = jnp.clip(ratio,1.-config['CLIP_EPS'],1.+config['CLIP_EPS'])*gae
    loss_actor = (-jnp.minimum(loss_actor1,loss_actor2)).mean()
    entropy = pi.entropy().mean()
    total_loss = loss_actor+config['VF_COEF']*value_loss-config['ENT_COEF']*entropy
    approx_kl = ((ratio-1)-(log_prob-traj_batch.log_prob)).mean()
    clip_fraction = (jnp.abs(ratio-1)>config['CLIP_EPS']).mean()
    return total_loss,(value_loss,loss_actor,entropy,approx_kl,clip_fraction)


def make_train(config):
    if config.get('DATA_PARALLEL_DEVICES', 1)>1 and not parallel_active(config):
        raise ValueError('Use ppo_data_parallel.compile_train/execute_train for multi-device PPO')
    from policy_temperature import tempered_policy
    temperature=config.get('POLICY_TEMPERATURE',1.)
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('POLICY_TEMPERATURE must be finite and strictly positive')
    learning_program = config.get('LEARNING_PROGRAM_GRAPH')
    config["START_TIMESTEPS"] = config.get("START_TIMESTEPS", 0)
    config["TOTAL_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    lr_horizon=config.get('LR_HORIZON_TIMESTEPS') or config['TOTAL_TIMESTEPS']
    if lr_horizon<config['TOTAL_TIMESTEPS']:raise ValueError('LR horizon must cover the training target')
    config['LR_HORIZON_UPDATES']=lr_horizon//config['NUM_STEPS']//config['NUM_ENVS']
    config["NUM_UPDATES"] = (
        (config["TOTAL_TIMESTEPS"] - config["START_TIMESTEPS"])
        // config["NUM_STEPS"]
        // config["NUM_ENVS"]
    )
    if config["NUM_UPDATES"] < 0:
        raise ValueError(
            "TOTAL_TIMESTEPS must be >= START_TIMESTEPS when resuming. "
            f"Got total={config['TOTAL_TIMESTEPS']} start={config['START_TIMESTEPS']}."
        )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    death_penalty = config.get("DEATH_PENALTY", 0.0)
    shaped_achievements = config.get('DIAMOND_REWARD', 1.) != 1. or config.get('PLANT_REWARD', 1.) != 1.
    env = make_craftax_env_from_name(
        config["ENV_NAME"],
        False if config.get('ALL22_TERMINAL_BONUS') is not None or config.get('SYMBOLIC_ENCODER')=='map_mlp' or config.get('RESET_SEED_BANK') or death_penalty or config.get('LAVA_PENALTY',0.) or shaped_achievements or config.get("CLUSTER_FREE_PRACTICE") or config.get("ID_PRACTICE") or config.get("CULTURE_DIAMOND_ORACLE") or config.get("DIAMOND_DRILLS") or config.get("STONE_PROBE") or config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0 or config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0 else not config["USE_OPTIMISTIC_RESETS"],
    )
    env_params = env.default_params

    if death_penalty:
        if config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1":
            raise ValueError("--death_penalty supports Craftax-Classic-Symbolic-v1")
        env = ClassicDeathPenaltyWrapper(env, death_penalty)
    if config.get('LAVA_PENALTY',0.):
        from wrappers import ClassicLavaPenaltyWrapper
        env=ClassicLavaPenaltyWrapper(env,config['LAVA_PENALTY'])
    if shaped_achievements:
        from wrappers import ClassicAchievementRewardWrapper
        env = ClassicAchievementRewardWrapper(env, config.get('DIAMOND_REWARD', 1.), config.get('PLANT_REWARD', 1.))

    if config.get('ALL22_TERMINAL_BONUS') is not None:
        from wrappers import ClassicCompletionWrapper
        if config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1":
            raise ValueError("Completion bonus requires Craftax Classic Symbolic")
        env = ClassicCompletionWrapper(env, config['ALL22_TERMINAL_BONUS'])

    from fixed_world_seeds import with_fixed_world_seeds
    env=with_fixed_world_seeds(env,config)
    from symbolic_map_memory import map_discovery_reward, with_map_memory
    coverage_bonus = config.get('MAP_COVERAGE_BONUS') or 0.
    if not np.isfinite(coverage_bonus) or coverage_bonus < 0 or (coverage_bonus and (
            config.get('SYMBOLIC_ENCODER') != 'map_mlp' or config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1')):
        raise ValueError('Map coverage bonus requires a nonnegative coefficient and a Classic symbolic map policy')
    env=with_map_memory(env,config,track_coverage=bool(coverage_bonus))
    if config.get('RECOVERY_BANK'):
        from recovery_resets import RecoveryResetWrapper
        env=RecoveryResetWrapper(env,config['RECOVERY_BANK'],config['RECOVERY_PROBABILITY'])
    if config.get("CULTURE_DIAMOND_ORACLE"):
        from rare_event_culture import DiamondOracleWrapper
        env = DiamondOracleWrapper(env)
    if config.get("STONE_PROBE"):
        from stone_outcome import StoneOutcomeWrapper
        env = StoneOutcomeWrapper(env)
    if config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0:
        from multi_outcome import MultiOutcomeWrapper
        env = MultiOutcomeWrapper(env)
    if config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0:
        from targeted_outcome import TargetedOutcomeWrapper
        env = TargetedOutcomeWrapper(env)
    if config.get("CLUSTER_FREE_PRACTICE"):
        from cluster_free_practice import ClusterFreePractice
        env = ClusterFreePractice(env, config['NUM_ENVS'], config['CLUSTER_FREE_PRACTICE'],
            config['CF_PRACTICE_ENVS'], config['CF_CAPACITY'], config['CF_HORIZON'],
            config['CF_WARMUP'], config['CF_MAX_REUSE'])
    elif config.get("ID_PRACTICE"):
        from id_practice import IDPractice
        if config.get("ID_PRACTICE_RARE_FOUR"):
            from rare_four_practice import RareFourPractice
            if config.get("ID_PRACTICE_BACKWARD"):
                from backward_practice import BackwardPractice
                practice_class = BackwardPractice
            else:
                practice_class = RareFourPractice
            env = practice_class(env,config["NUM_ENVS"],config["ID_PRACTICE_ENVS"],
                config["ID_PRACTICE_CAPACITY"],config["ID_PRACTICE_HORIZON"],
                config["ID_PRACTICE_RARITY_WINDOW"],config["ID_PRACTICE_RARITY_REFRESH"],
                config["ID_PRACTICE_RARITY_WARMUP"],config["ID_PRACTICE_MAX_REUSE"],
                config.get("ID_PRACTICE_FULL_EPISODE",False))
        else:
            env = IDPractice(env,config["NUM_ENVS"],config["ID_PRACTICE_ENVS"],
                             config["ID_PRACTICE_CAPACITY"],config["ID_PRACTICE_HORIZON"],
                             config.get("ID_PRACTICE_FULL_EPISODE",False))
    elif config.get("DIAMOND_DRILLS"):
        from diamond_skill_drill import MixedDiamondDrills
        env = MixedDiamondDrills(env, config["NUM_ENVS"], config["DIAMOND_DRILL_ENVS"], config["DIAMOND_DRILL_HORIZON"])
    elif config["USE_OPTIMISTIC_RESETS"]:
        env = LogWrapper(env)
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = LogWrapper(env)
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    if config.get("DIAMOND_CURRICULUM"):
        from diamond_density_curriculum import DiamondDensityCurriculum
        env = DiamondDensityCurriculum(env, config["DIAMOND_CURRICULUM_START"],
                                      config["DIAMOND_DENSE_PROBABILITY"],
                                      config["DIAMOND_DENSITY_MULTIPLIER"])

    def linear_schedule(count):
        schedule_updates = config["NUM_UPDATES"] if config.get("RESUME_RESET_OPTIMIZER") and not config.get('LR_HORIZON_TIMESTEPS') else config["LR_HORIZON_UPDATES"]
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / schedule_updates
        )
        return config["LR"] * frac

    # INIT NETWORK
    if "Symbolic" in config["ENV_NAME"]:
        if config.get('SYMBOLIC_ENCODER')=='map_mlp':
            from models.actor_critic_map_mlp import map_mlp_network
            network=map_mlp_network(config,env.action_space(env_params).n)
        elif config.get('SYMBOLIC_ENCODER', 'mlp') == 'spatial':
            from models.conditioned_symbolic import ConditionedSymbolicActorCritic
            network = ConditionedSymbolicActorCritic(env.action_space(env_params).n)
        else:
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
    else:
        network = ActorCriticConv(env.action_space(env_params).n, config["LAYER_SIZE"])

    def init_train_state(rng):
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
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
        if config.get('MAP_V1_ATTENTION_ADAPTER'):
            from models.actor_critic_board_transfer import adapter_only_optimizer
            tx = adapter_only_optimizer(tx)
        if config.get("STONE_PROBE"):
            from stone_outcome import initialize_head
            network_params = initialize_head(network_params,jax.random.fold_in(rng,4001),config["LAYER_SIZE"])
        if config.get("REWARD_SIGN_AUX_COEF", 0) > 0:
            from reward_sign_outcome import initialize_head
            network_params = initialize_head(network_params, jax.random.fold_in(rng,5001), config["LAYER_SIZE"])
        if config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0:
            from multi_outcome import initialize_head
            network_params = initialize_head(network_params, jax.random.fold_in(rng,6001), config["LAYER_SIZE"])
        if config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0:
            from targeted_outcome import initialize_head
            network_params = initialize_head(network_params,jax.random.fold_in(rng,7001),config["LAYER_SIZE"])
        if config.get("AUX_TASK", "none") != "none" and config.get("AUX_COEF", 0) > 0:
            from aux_prediction import initialize_head
            network_params = initialize_head(network_params, jax.random.fold_in(rng, 9001),
                                             config["LAYER_SIZE"], config["AUX_TASK"])
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        return train_state

    restored_train_state = None
    if config.get('MAP_V1_IMITATION_INIT'):
        from pathlib import Path
        from models.actor_critic_map_mlp import transfer_map_mlp_params
        if not config.get('MAP_V1_ATTENTION_ADAPTER') or config.get('RESUME_FROM'):
            raise ValueError('Use the imitation initializer only for the first frozen-V1 PPO segment')
        checkpoint = Path(config['MAP_V1_IMITATION_INIT']).resolve()
        source = PyTreeCheckpointer().restore(str(checkpoint))['params']
        initial = init_train_state(jax.random.PRNGKey(config.get('SEED') or 0))
        restored_train_state = initial.replace(params=transfer_map_mlp_params(initial.params, source))
        print(f'Initialized board adapter and frozen V1 from {checkpoint}. Starting PPO and Adam at zero.', flush=True)
    if config["RESUME_FROM"] is not None:
        from pathlib import Path
        checkpoint = Path(config['RESUME_FROM']) / str(config['RESUME_STEP'])
        seed = 0 if config["SEED"] is None else config["SEED"]
        restore_item = init_train_state(jax.random.PRNGKey(seed))
        orbax_checkpointer = PyTreeCheckpointer()
        print(
            "Restoring policy checkpoint "
            f"step {config['RESUME_STEP']} from {config['RESUME_FROM']}"
        )
        old_crop = config.get('MAP_WARMSTART_CROP_FROM')
        if config.get('MAP_FINE_SHORTCUT'):
            from models.actor_critic_map_mlp import restore_map_shortcut_state
            payload = checkpoint/'default' if (checkpoint/'default').is_dir() else checkpoint
            restored_train_state = restore_map_shortcut_state(orbax_checkpointer, payload.resolve(), restore_item)
            if config.get('RESUME_RESET_OPTIMIZER'):
                restored_train_state = restore_item.replace(params=restored_train_state.params)
        elif (config.get('MAP_GLOBAL_READOUT') or old_crop) and config.get('RESUME_RESET_OPTIMIZER'):
            from models.actor_critic_map_mlp import transfer_map_mlp_params, widen_map_mlp_params
            payload = checkpoint/'default' if (checkpoint/'default').is_dir() else checkpoint
            source = orbax_checkpointer.restore(str(payload))['params']
            if old_crop:
                params = widen_map_mlp_params(restore_item.params, source, old_crop, config['MAP_VIEW_SIZE'])
                print(f'Centered the trained {old_crop}x{old_crop} kernels in a {config["MAP_VIEW_SIZE"]}x{config["MAP_VIEW_SIZE"]} crop', flush=True)
            else:
                params = transfer_map_mlp_params(restore_item.params, source)
            restored_train_state = restore_item.replace(params=params)
        else:
            if (checkpoint / '_METADATA').is_file() and not (checkpoint / 'default').is_dir():
                restored_train_state = orbax_checkpointer.restore(str(checkpoint.resolve()), item=restore_item)
            else:
                options = CheckpointManagerOptions(max_to_keep=1, create=False)
                manager = CheckpointManager(config["RESUME_FROM"], orbax_checkpointer, options)
                try:
                    restored_train_state = manager.restore(config["RESUME_STEP"], items=restore_item)
                finally:
                    manager.close()
            if config.get("RESUME_RESET_OPTIMIZER"):
                restored_train_state = restore_item.replace(params=restored_train_state.params)
        if config.get("RESUME_RESET_OPTIMIZER"):
            print('Restored actor/critic weights; restarted optimizer and continuation LR schedule',flush=True)

    initial_practice_archive = None
    if config.get("ID_PRACTICE_ARCHIVE_FROM"):
        from practice_continuation import read_archive
        initial_practice_archive = read_archive(config["ID_PRACTICE_ARCHIVE_FROM"],
            config["ID_PRACTICE_CAPACITY"],config["ID_PRACTICE_RARITY_WINDOW"],config["ID_PRACTICE_MAX_REUSE"])

    component_source_train_state = None
    if config["COMPONENT_INIT_FROM"] is not None:
        seed = 0 if config["SEED"] is None else config["SEED"]
        restore_item = init_train_state(jax.random.PRNGKey(seed))
        orbax_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=False)
        checkpoint_manager = CheckpointManager(
            config["COMPONENT_INIT_FROM"], orbax_checkpointer, options
        )
        print(
            "Loading frozen component from checkpoint "
            f"{config['COMPONENT_INIT_STEP']} at {config['COMPONENT_INIT_FROM']}"
        )
        component_source_train_state = checkpoint_manager.restore(
            config["COMPONENT_INIT_STEP"], items=restore_item
        )

    culture_memory = None
    culture_encoder_params = None
    if config.get("CULTURE_PATH"):
        from culture import CultureMemory, encode, build_policy_target
        culture_memory = CultureMemory.load(config["CULTURE_PATH"])
        encoder_path = culture_memory.metadata["encoder_path"]
        with np.load(encoder_path) as data:
            culture_encoder_params = {"params": {
                f"Dense_{i}": {leaf: jnp.asarray(data[f"Dense_{i}_{leaf}"])
                               for leaf in ["kernel", "bias"]} for i in range(8)
            }}
        culture_mode = culture_memory.metadata["representation"]
        shuffled_actions = None
        if config.get("CULTURE_SHUFFLE_ACTIONS"):
            shuffled_actions = np.random.default_rng(config["SEED"]).permutation(
                culture_memory.arrays["actions"][:, 0]
            )

    def bc_pretrain(train_state):
        """Pretrain only actor; reset optimizer before on-policy PPO starts."""
        # Match the expert action labels available to online retrieval.
        observations = culture_memory.arrays["observations"]
        actions = culture_memory.arrays["actions"][:, 0]
        observations, actions = jnp.asarray(observations), jnp.asarray(actions)
        def bc_step(carry, step):
            state = carry
            key = jax.random.fold_in(jax.random.PRNGKey(config["SEED"]+1000), step)
            ix = jax.random.randint(key, (config["BC_BATCH_SIZE"],), 0, len(actions))
            def loss(params):
                pi, _ = network.apply(params, observations[ix])
                return -pi.log_prob(actions[ix]).mean()
            value, grads = jax.value_and_grad(loss)(state.params)
            grads = _mask_component_gradients(grads, "actor")
            return state.apply_gradients(grads=grads), value
        updated, losses = jax.lax.scan(bc_step, train_state, jnp.arange(config["BC_STEPS"]))
        return updated.replace(step=train_state.step, opt_state=train_state.opt_state)

    def train(rng):
        if restored_train_state is None:
            train_state = init_train_state(rng)
        else:
            train_state = restored_train_state

        if component_source_train_state is not None:
            source_component = (
                "critic"
                if config["TRAIN_COMPONENT"] in ("actor", "actor_critic_head")
                else "actor"
            )
            mixed_params = _replace_param_component(
                train_state.params,
                component_source_train_state.params,
                source_component,
            )
            # Keep the freshly initialized optimizer state: only parameters from
            # the requested source branch are transferred.
            train_state = train_state.replace(params=mixed_params)

        if config.get("BC_STEPS", 0):
            train_state = bc_pretrain(train_state)
        train_state = replicate(train_state, config)

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }
        if learning_program is not None:
            ex_state['learning_memory'] = jnp.zeros(4, dtype=jnp.float32)
        if config.get("AUX_TASK", "none") != "none" and config.get("AUX_COEF", 0) > 0:
            from aux_prediction import projection
            ex_state['aux_projection'] = projection(env.observation_space(env_params).shape[0],
                                                     config["AUX_TARGET_SEED"])
        if config.get("STONE_PROBE"):
            from stone_outcome import empty_replay
            ex_state['stone_replay'] = empty_replay(config["STONE_REPLAY_CAPACITY"],env.observation_space(env_params).shape[0])
        if config.get("REWARD_SIGN_AUX_COEF", 0) > 0:
            from reward_sign_outcome import empty_replay
            ex_state['reward_sign_replay'] = empty_replay(config["REWARD_SIGN_REPLAY_CAPACITY"], env.observation_space(env_params).shape[0])
        if config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0:
            from multi_outcome import empty_replay
            ex_state['multi_outcome_replay'] = empty_replay(config["MULTI_OUTCOME_REPLAY_CAPACITY"], env.observation_space(env_params).shape[0])
        if config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0:
            from targeted_outcome import empty_replay
            ex_state['targeted_outcome_replay'] = empty_replay(config["TARGETED_OUTCOME_REPLAY_CAPACITY"],env.observation_space(env_params).shape[0])

        if config["TRAIN_ICM"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Encoder
            icm_encoder_network = ICMEncoder(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_encoder"] = TrainState.create(
                apply_fn=icm_encoder_network.apply,
                params=icm_encoder_network_params,
                tx=tx,
            )

            # Forward
            icm_forward_network = ICMForward(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
                num_actions=env.num_actions,
            )
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(
                _rng, jnp.zeros((1, config["ICM_LATENT_SIZE"])), jnp.zeros((1,))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_forward"] = TrainState.create(
                apply_fn=icm_forward_network.apply,
                params=icm_forward_network_params,
                tx=tx,
            )

            # Inverse
            icm_inverse_network = ICMInverse(
                num_layers=3,
                output_dim=env.num_actions,
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_inverse_network_params = icm_inverse_network.init(
                _rng,
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_inverse"] = TrainState.create(
                apply_fn=icm_inverse_network.apply,
                params=icm_inverse_network_params,
                tx=tx,
            )

            if config["USE_E3B"]:
                ex_state["e3b_matrix"] = (
                    jnp.repeat(
                        jnp.expand_dims(
                            jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                        ),
                        config["NUM_ENVS"],
                        axis=0,
                    )
                    / config["E3B_LAMBDA"]
                )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        obsv, env_state = shard_batch((obsv,env_state),0,config)
        if initial_practice_archive is not None:
            from practice_continuation import seed_archive
            env_state = seed_archive(env_state,initial_practice_archive)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                pi = tempered_policy(pi,temperature)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                obsv, env_state, reward_e, done, info = shard_batch(
                    (obsv,env_state,reward_e,done,info),0,config)

                reward_i = jnp.zeros(config["NUM_ENVS"])

                if config["TRAIN_ICM"]:
                    latent_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, last_obs
                    )
                    latent_next_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, obsv
                    )

                    latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                        ex_state["icm_forward"].params, latent_obs, action
                    )
                    error = (latent_next_obs - latent_next_obs_pred) * (
                        1 - done[:, None]
                    )
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["ICM_REWARD_COEFF"]

                    if config["USE_E3B"]:
                        # Embedding is (NUM_ENVS, 128)
                        # e3b_matrix is (NUM_ENVS, 128, 128)
                        us = jax.vmap(jnp.matmul)(ex_state["e3b_matrix"], latent_obs)
                        bs = jax.vmap(jnp.dot)(latent_obs, us)

                        def update_c(c, b, u):
                            return c - (1.0 / (1 + b)) * jnp.outer(u, u)

                        updated_cs = jax.vmap(update_c)(ex_state["e3b_matrix"], bs, us)
                        new_cs = (
                            jnp.repeat(
                                jnp.expand_dims(
                                    jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                                ),
                                config["NUM_ENVS"],
                                axis=0,
                            )
                            / config["E3B_LAMBDA"]
                        )
                        ex_state["e3b_matrix"] = jnp.where(
                            done[:, None, None], new_cs, updated_cs
                        )

                        e3b_bonus = jnp.where(
                            done, jnp.zeros((config["NUM_ENVS"],)), bs
                        )

                        reward_i = e3b_bonus * config["E3B_REWARD_COEFF"]

                if coverage_bonus:
                    reward_i += map_discovery_reward(info['map_new_cells'], done, coverage_bonus)
                reward = reward_e + reward_i
                if config.get('CLUSTER_FREE_PRACTICE'):
                    _, successor_value = network.apply(train_state.params, info['cf/bootstrap_obs'])
                    env_state, error = env.record_prediction(env_state, info, value,
                                                             successor_value, config['GAMMA'])
                    reward += config['GAMMA'] * jax.lax.stop_gradient(successor_value) * info['cf/truncated']
                    info['cf/error'] = error
                    info = {k: v for k, v in info.items() if k != 'cf/bootstrap_obs'}
                if config.get("ID_PRACTICE"):
                    if not config.get("ID_PRACTICE_FULL_EPISODE"):
                        from practice_continuation import bootstrap_reward
                        _, timeout_value = network.apply(train_state.params,info['practice/bootstrap_obs'])
                        reward,bootstrap_mask,bootstrap_addition = bootstrap_reward(reward,timeout_value,info,
                            config["GAMMA"],config.get("ID_PRACTICE_SUCCESS_BOOTSTRAP",False))
                        info['practice/bootstrap_applied'] = bootstrap_mask
                        info['practice/bootstrap_addition'] = bootstrap_addition
                    info = {k:v for k,v in info.items() if k != 'practice/bootstrap_obs'}
                if config.get("DIAMOND_DRILLS"):
                    n_drills = config["DIAMOND_DRILL_ENVS"]
                    _, drill_timeout_value = network.apply(train_state.params, info['drill/bootstrap_obs'][-n_drills:])
                    timeout_value = jnp.concatenate([jnp.zeros(config["NUM_ENVS"]-n_drills), drill_timeout_value])
                    reward += config["GAMMA"] * jax.lax.stop_gradient(timeout_value) * info['drill/truncated']
                    # Bootstrap timeout's pre-reset state, while done still cuts GAE across resets.
                    info = {k:v for k,v in info.items() if k != 'drill/bootstrap_obs'}

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
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
                ex_state,
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

            if learning_program is None:
                advantages, targets = _calculate_gae(traj_batch, last_val)
            else:
                from learning_rule.runtime import credit_targets, update_memory
                program_progress = update_step / config['NUM_UPDATES']
                program_memory = ex_state['learning_memory']
                advantages, targets = credit_targets(learning_program, traj_batch.reward,
                    traj_batch.value, traj_batch.done, last_val, program_memory,
                    config['GAMMA'], config['GAE_LAMBDA'], program_progress)
                next_program_memory = update_memory(learning_program, program_memory,
                    traj_batch.reward, traj_batch.value, traj_batch.done, advantages,
                    config['GAMMA'], config['GAE_LAMBDA'], program_progress)
            stone_batch = None
            reward_sign_batch = None
            multi_outcome_batch = None
            targeted_outcome_batch = None
            aux_batch = None
            aux_valid_fraction = jnp.asarray(0.)
            aux_positive_fraction = jnp.asarray(0.)
            if config.get("AUX_TASK", "none") != "none" and config.get("AUX_COEF", 0) > 0:
                from aux_prediction import targets as prediction_targets, sample_batch
                data = prediction_targets(traj_batch.obs, traj_batch.next_obs, traj_batch.action,
                    traj_batch.reward_e, traj_batch.done, config["AUX_TASK"],
                    config["AUX_HORIZON"], ex_state['aux_projection'])
                aux_valid_fraction = data[-1].mean()
                if config["AUX_TASK"] == 'event':
                    aux_positive_fraction = (data[3][..., 0] * data[-1]).sum() / jnp.maximum(data[-1].sum(), 1)
                aux_batch = sample_batch(data, jax.random.fold_in(
                    jax.random.PRNGKey(config["SEED"] + 9002), update_step), config["AUX_BATCH"])
                aux_batch = jax.tree.map(lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]), aux_batch)
            if config.get("STONE_PROBE"):
                from stone_outcome import append_replay,sample_replay,outcome_loss
                flat_obs = traj_batch.obs.reshape((-1,traj_batch.obs.shape[-1]))
                ex_state['stone_replay'] = append_replay(ex_state['stone_replay'],flat_obs,
                    traj_batch.action.reshape(-1),traj_batch.info['stone_attempt'].reshape(-1),
                    traj_batch.info['stone_success'].reshape(-1))
                stone_batch = sample_replay(ex_state['stone_replay'],
                    jax.random.fold_in(jax.random.PRNGKey(config["SEED"]+4000),update_step),config["STONE_AUX_BATCH"])
                stone_batch = jax.tree.map(lambda x:x.reshape((config["NUM_MINIBATCHES"],-1)+x.shape[1:]),stone_batch)
            if config.get("REWARD_SIGN_AUX_COEF", 0) > 0:
                from reward_sign_outcome import append_replay, sample_replay
                ex_state['reward_sign_replay'] = append_replay(ex_state['reward_sign_replay'],
                    traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1])),
                    traj_batch.action.reshape(-1), traj_batch.reward_e.reshape(-1))
                reward_sign_batch = sample_replay(ex_state['reward_sign_replay'],
                    jax.random.fold_in(jax.random.PRNGKey(config["SEED"] + 5000), update_step),
                    config["REWARD_SIGN_AUX_BATCH"])
                reward_sign_batch = jax.tree.map(lambda x:x.reshape((config["NUM_MINIBATCHES"],-1)+x.shape[1:]),reward_sign_batch)
            if config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0:
                from multi_outcome import OUTCOMES, append_replay, sample_replay
                labels = jnp.stack([traj_batch.info['outcome/' + name] for name in OUTCOMES], axis=-1)
                ex_state['multi_outcome_replay'] = append_replay(ex_state['multi_outcome_replay'],
                    traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1])),
                    traj_batch.action.reshape(-1), labels.reshape((-1, len(OUTCOMES))))
                multi_outcome_batch = sample_replay(ex_state['multi_outcome_replay'],
                    jax.random.fold_in(jax.random.PRNGKey(config["SEED"] + 6000), update_step),
                    config["MULTI_OUTCOME_AUX_BATCH"])
                multi_outcome_batch = jax.tree.map(lambda x:x.reshape((config["NUM_MINIBATCHES"],-1)+x.shape[1:]),multi_outcome_batch)
            if config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0:
                from targeted_outcome import TASKS, append_replay, sample_replay
                eligible = jnp.stack([traj_batch.info['targeted/eligible/'+name] for name in TASKS],axis=-1)
                labels = jnp.stack([traj_batch.info['targeted/success/'+name] for name in TASKS],axis=-1)
                ex_state['targeted_outcome_replay'] = append_replay(ex_state['targeted_outcome_replay'],
                    traj_batch.obs.reshape((-1,traj_batch.obs.shape[-1])),traj_batch.action.reshape(-1),
                    labels.reshape((-1,len(TASKS))),eligible.reshape((-1,len(TASKS))))
                targeted_outcome_batch = sample_replay(ex_state['targeted_outcome_replay'],
                    jax.random.fold_in(jax.random.PRNGKey(config["SEED"]+7000),update_step),
                    config["TARGETED_OUTCOME_AUX_BATCH"])
                targeted_outcome_batch = jax.tree.map(lambda x:x.reshape((config["NUM_MINIBATCHES"],-1)+x.shape[1:]),targeted_outcome_batch)

            # Retrieve once per rollout, outside the differentiable PPO loss.
            # Independent RNG preserves the factual PPO sampling RNG stream.
            culture_stats = {}
            n_culture = config.get("CULTURE_QUERY_BATCH", 512)
            if culture_memory is not None and config.get("CULTURE_COEF", 0) > 0:
                flat_obs = traj_batch.obs.reshape((-1, traj_batch.obs.shape[-1]))
                query_key = jax.random.fold_in(jax.random.PRNGKey(config["SEED"]+2000), update_step)
                ix = jax.random.randint(query_key, (n_culture,), 0, flat_obs.shape[0])
                culture_obs = flat_obs[ix]
                embeddings = encode(culture_encoder_params, culture_obs, culture_mode)
                similarities, indices = culture_memory.retrieve(embeddings, config["CULTURE_K"])
                if config.get("CULTURE_RANDOM_RETRIEVAL"):
                    random_key = jax.random.fold_in(query_key, 1)
                    indices = jax.random.randint(random_key, indices.shape, 0, len(culture_memory.arrays["actions"]))
                    sampled = jnp.asarray(culture_memory.arrays["embeddings"])[indices]
                    similarities = (sampled * embeddings[:, None, :]).sum(axis=-1)
                    similarities, order = jax.lax.top_k(similarities, similarities.shape[-1])
                    indices = jnp.take_along_axis(indices, order, axis=-1)
                culture_q, culture_gate, culture_stats = build_policy_target(
                    culture_memory, similarities, indices,
                    action_dim=env.action_space(env_params).n,
                    threshold=config["CULTURE_THRESHOLD"], temperature=config["CULTURE_TEMPERATURE"],
                    weighting=config["CULTURE_WEIGHTING"], shuffled_actions=shuffled_actions,
                )
                if config.get("CULTURE_DIAMOND_ORACLE"):
                    opportunity = traj_batch.info['diamond_oracle'].reshape(-1)[ix]
                    culture_gate = culture_gate * opportunity
                    culture_stats['oracle_opportunity_rate'] = traj_batch.info['diamond_oracle'].mean()
                    culture_stats['oracle_gated_match_rate'] = (culture_gate > 0).mean()
                    # Half the auxiliary batch rehearses successful rare-event states.
                    # This is supervised expert replay, not PPO/off-policy value learning.
                    half = n_culture // 2
                    replay_key = jax.random.fold_in(query_key, 2)
                    replay_ix = jax.random.randint(replay_key, (half,), 0, len(culture_memory.arrays['actions']))
                    replay_actions = jnp.asarray(culture_memory.arrays['actions'][:, 0]
                                                 if shuffled_actions is None else shuffled_actions)[replay_ix]
                    culture_obs = culture_obs.at[:half].set(jnp.asarray(culture_memory.arrays['observations'])[replay_ix])
                    culture_q = culture_q.at[:half].set(jax.nn.one_hot(replay_actions, env.action_space(env_params).n))
                    culture_gate = culture_gate.at[:half].set(1.)
            else:
                culture_obs = jnp.zeros((n_culture, traj_batch.obs.shape[-1]))
                culture_q = jnp.zeros((n_culture, env.action_space(env_params).n))
                culture_gate = jnp.zeros(n_culture)
            culture_batches = jax.tree.map(
                lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                (culture_obs, culture_q, culture_gate),
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets, cultural_batch, stone_minibatch, reward_sign_minibatch, multi_outcome_minibatch, targeted_outcome_minibatch, aux_minibatch = batch_info
                    traj_batch, advantages, targets = shard_batch(
                        (traj_batch,advantages,targets),0,config)

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        if parallel_active(config):
                            loss,terms = feedforward_ppo_loss(params,network,traj_batch,gae,targets,config)
                            return loss,terms[:3]+(jnp.asarray(0.),)*6+terms[3:]
                        program_advantage = gae
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        pi = tempered_policy(pi,temperature)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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

                        culture_loss = jnp.asarray(0.)
                        if culture_memory is not None and config.get("CULTURE_COEF", 0) > 0:
                            c_obs, c_target, c_gate = cultural_batch
                            c_pi, _ = network.apply(params, c_obs)
                            ce = -(c_target * jax.nn.log_softmax(c_pi.logits_parameter(), axis=-1)).sum(axis=-1)
                            culture_loss = (c_gate * ce).mean()

                        stone_loss = jnp.asarray(0.)
                        if config.get("STONE_PROBE"):
                            stone_loss = outcome_loss(params,stone_minibatch)
                        reward_sign_loss = jnp.asarray(0.)
                        if config.get("REWARD_SIGN_AUX_COEF", 0) > 0:
                            from reward_sign_outcome import outcome_loss as reward_loss
                            reward_sign_loss = reward_loss(params, reward_sign_minibatch)
                        multi_outcome_loss = jnp.asarray(0.)
                        if config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0:
                            from multi_outcome import outcome_loss as multi_loss
                            multi_outcome_loss = multi_loss(params, multi_outcome_minibatch)
                        targeted_outcome_loss = jnp.asarray(0.)
                        if config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0:
                            from targeted_outcome import outcome_loss as targeted_loss
                            targeted_outcome_loss = targeted_loss(params,targeted_outcome_minibatch)
                        prediction_loss = jnp.asarray(0.)
                        if config.get("AUX_TASK", "none") != "none" and config["AUX_COEF"] > 0:
                            from aux_prediction import loss as prediction_loss_fn
                            prediction_loss = prediction_loss_fn(params, aux_minibatch, config["AUX_TASK"])
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config.get("CULTURE_COEF", 0.) * culture_loss
                            + config.get("STONE_AUX_COEF",0.) * stone_loss
                            + config.get("REWARD_SIGN_AUX_COEF",0.) * reward_sign_loss
                            + config.get("MULTI_OUTCOME_AUX_COEF",0.) * multi_outcome_loss
                            + config.get("TARGETED_OUTCOME_AUX_COEF",0.) * targeted_outcome_loss
                            + config.get("AUX_COEF", 0.) * prediction_loss
                        )
                        if learning_program is not None:
                            from learning_rule.runtime import program_loss
                            total_loss = program_loss(learning_program, log_prob,
                                traj_batch.log_prob, value, traj_batch.value,
                                program_advantage, targets, pi.entropy(), program_memory,
                                config['GAMMA'], config['GAE_LAMBDA'], program_progress,
                                config['CLIP_EPS'], config['VF_COEF'], config['ENT_COEF'])
                        approx_kl = ((ratio - 1) - (log_prob - traj_batch.log_prob)).mean()
                        clip_fraction = (jnp.abs(ratio - 1) > config["CLIP_EPS"]).mean()
                        return total_loss, (value_loss, loss_actor, entropy, culture_loss,stone_loss,reward_sign_loss,multi_outcome_loss,targeted_outcome_loss,prediction_loss,approx_kl,clip_fraction)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, auxiliary), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    gradient_ratio = jnp.asarray(0.)
                    if config.get("AUX_TASK", "none") != "none" and config["AUX_COEF"] > 0:
                        from aux_prediction import loss as prediction_loss_fn
                        aux_grads = jax.grad(prediction_loss_fn)(train_state.params, aux_minibatch, config["AUX_TASK"])
                        ppo_grads = jax.tree.map(lambda g, a: g - config["AUX_COEF"] * a, grads, aux_grads)
                        gradient_ratio = config["AUX_COEF"] * optax.global_norm(aux_grads) / jnp.maximum(optax.global_norm(ppo_grads), 1e-8)
                    grads = _mask_component_gradients(
                        grads, config["TRAIN_COMPONENT"]
                    )
                    grads = replicate(grads, config)
                    train_state = replicate(train_state.apply_gradients(grads=grads),config)

                    losses = (total_loss, auxiliary + (gradient_ratio,))
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
                    _update_minbatch, train_state, (*minibatches, culture_batches,stone_batch,reward_sign_batch,multi_outcome_batch,targeted_outcome_batch,aux_batch)
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
            if config.get('RECOVERY_BANK'):
                metric['practice/recovery_step']=traj_batch.info['practice/recovery_step'].mean()
                metric['practice/start']=traj_batch.info['practice/start'].sum()
                metric['practice/recovery_start']=traj_batch.info['practice/recovery_start'].sum()
            if coverage_bonus:
                discoveries = traj_batch.info['map_new_cells']
                metric['training/new_map_cells_per_step'] = discoveries.mean()
                metric['training/map_coverage_bonus_per_step'] = map_discovery_reward(discoveries, traj_batch.done, coverage_bonus).mean()
            if config.get('CLUSTER_FREE_PRACTICE'):
                ended = traj_batch.info['cf/ended'].sum()
                metric.update({
                    'cf/interaction_fraction': traj_batch.info['cf/is_practice'].mean(),
                    'cf/mean_absolute_return_error': traj_batch.info['cf/error'].sum()/jnp.maximum(ended, 1),
                    'cf/practices_ended': ended, 'cf/starts': env_state.starts,
                    'cf/unique_starts': env_state.unique_starts, 'cf/bank_insertions': env_state.bank_total,
                    'cf/bank_size': (env_state.bank_info['generation'] > 0).sum(),
                    'cf/scored_snapshots': (env_state.bank_info['updates'] > 0).sum(),
                    'cf/max_reuse': env_state.bank_info['reuse'].max(),
                    'cf/total_practice_steps': env_state.practice_steps,
                    'cf/normal_episodes': env_state.normal_episodes})
            if config.get("ID_PRACTICE"):
                from id_practice import TASKS
                metric['practice/interaction_fraction'] = traj_batch.info['practice/is_practice'].mean()
                metric['practice/target_earned_events'] = traj_batch.info['practice/target_earned_now'].sum()
                metric['practice/success_rate'] = traj_batch.info['practice/success'].sum()/jnp.maximum(1,traj_batch.info['practice/ended'].sum())
                metric['practice/timeout_rate'] = traj_batch.info['practice/truncated'].sum()/jnp.maximum(1,traj_batch.info['practice/ended'].sum())
                metric['practice/archive_covered_ids'] = (env_state.archive_counts>0).sum()
                if not config.get("ID_PRACTICE_FULL_EPISODE"):
                    success_bootstraps = traj_batch.info['practice/bootstrap_applied'] & traj_batch.info['practice/success']
                    metric['practice/success_bootstrap_count'] = success_bootstraps.sum()
                    metric['practice/success_bootstrap_mean_addition'] = (traj_batch.info['practice/bootstrap_addition']*success_bootstraps).sum()/jnp.maximum(1,success_bootstraps.sum())
                for i,name in enumerate(TASKS):
                    metric['practice/steps/'+name] = env_state.practice_steps_by_id[i]
                    metric['practice/archive_count/'+name] = env_state.archive_counts[i]
                    metric['practice/archive_inserted/'+name] = env_state.archive_totals[i]
                    metric['practice/attempts/'+name] = traj_batch.info['practice/attempt_end/'+name].sum()
                    metric['practice/success_rate/'+name] = traj_batch.info['practice/success_end/'+name].sum()/jnp.maximum(1,traj_batch.info['practice/attempt_end/'+name].sum())
                if config.get("ID_PRACTICE_RARE_FOUR"):
                    metric['rare_four/normal_episodes'] = env_state.rarity_total
                    metric['rare_four/selected_count'] = env_state.rarity_selected.sum()
                    usable = (jnp.arange(config['ID_PRACTICE_CAPACITY'])[None,:] < env_state.archive_counts[:,None]) & (env_state.reuse_counts < config['ID_PRACTICE_MAX_REUSE'])
                    metric['rare_four/usable_ids'] = usable.any(axis=1).sum()
                    metric['rare_four/selected_usable_ids'] = (usable.any(axis=1) & env_state.rarity_selected).sum()
                    metric['rare_four/max_current_reuse'] = env_state.reuse_counts.max()
                    for i,name in enumerate(TASKS):
                        metric['rare_four/selected/'+name] = env_state.rarity_selected[i].astype(jnp.float32)
                        metric['rare_four/normal_rate/'+name] = env_state.rarity_sum[i]/jnp.maximum(1,jnp.minimum(env_state.rarity_total,config['ID_PRACTICE_RARITY_WINDOW']))
                        metric['rare_four/starts/'+name] = env_state.rarity_starts[i]
                        metric['rare_four/unique_snapshot_starts/'+name] = env_state.rarity_unique_starts[i]
                if config.get("ID_PRACTICE_BACKWARD"):
                    for i,name in enumerate(TASKS):
                        metric['backward/level/'+name] = env_state.backward_level[i]
                        for level,lag in enumerate((1,5,20,50)):
                            count = env_state.backward_attempt_counts[i,level]
                            metric[f'backward/attempts/{name}/lag{lag}'] = count
                            metric[f'backward/recent_success/{name}/lag{lag}'] = env_state.backward_outcomes[i,level].sum()/jnp.maximum(1,jnp.minimum(count,128))
            for name, value in zip(["value_loss", "policy_loss", "entropy", "culture_loss","stone_prediction_loss","reward_sign_prediction_loss","multi_outcome_prediction_loss","targeted_outcome_prediction_loss","aux_prediction_loss","approx_kl","clip_fraction","aux_to_ppo_gradient_ratio"], loss_info[1]):
                metric["training/" + name] = value.mean()
            metric['aux/valid_fraction'] = aux_valid_fraction
            metric['aux/event_positive_fraction'] = aux_positive_fraction
            metric['training/optimized_loss'] = loss_info[0].mean()
            if learning_program is not None:
                metric['program/credit_mean'] = advantages.mean()
                metric['program/credit_std'] = advantages.std()
                for i in range(4):
                    metric[f'program/memory_{i}'] = program_memory[i]
                ex_state['learning_memory'] = next_program_memory
            if config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0:
                from targeted_outcome import TASKS
                for i,name in enumerate(TASKS):
                    metric['targeted/positive_replay/'+name] = ex_state['targeted_outcome_replay']['counts'][2*i+1]
                    metric['targeted/negative_replay/'+name] = ex_state['targeted_outcome_replay']['counts'][2*i]
                    metric['targeted/attempts/'+name] = traj_batch.info['targeted/eligible/'+name].sum()
            if config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0:
                from multi_outcome import OUTCOMES
                for i, name in enumerate(OUTCOMES):
                    metric['multi_outcome/positive_replay/' + name] = ex_state['multi_outcome_replay']['counts'][2*i+1]
            if config.get("REWARD_SIGN_AUX_COEF", 0) > 0:
                for label, name in enumerate(('negative', 'zero', 'positive')):
                    metric['reward_sign/replay_' + name] = ex_state['reward_sign_replay']['counts'][label]
            if config.get("STONE_PROBE"):
                metric['stone/attempts'] = traj_batch.info['stone_attempt'].sum()
                metric['stone/successful_attempts'] = traj_batch.info['stone_success'].sum()
                metric['stone/replay_failures'] = ex_state['stone_replay']['counts'][0]
                metric['stone/replay_successes'] = ex_state['stone_replay']['counts'][1]
            if config.get("DIAMOND_CURRICULUM"):
                metric['curriculum/dense_transition_fraction'] = traj_batch.info['curriculum/dense_transition'].mean()
            if config.get("DIAMOND_DRILLS"):
                endings = traj_batch.done & (traj_batch.info['drill/is_drill'] > 0)
                count = jnp.maximum(endings.sum(), 1)
                metric['drill/success_rate'] = (traj_batch.info['drill/success']*endings).sum()/count
                metric['drill/timeout_rate'] = (traj_batch.info['drill/truncated']*endings).sum()/count
                metric['drill/interaction_fraction'] = traj_batch.info['drill/is_drill'].mean()
                normal_endings = traj_batch.done & (traj_batch.info['drill/is_drill'] == 0)
                normal_count = jnp.maximum(normal_endings.sum(), 1)
                metric['drill/normal_episode_count'] = normal_endings.sum()
                metric['drill/completed_drill_count'] = endings.sum()
                metric['drill/normal_episode_return'] = (traj_batch.info['returned_episode_returns']*normal_endings).sum()/normal_count
            if culture_memory is not None:
                metric["culture/size"] = jnp.asarray(len(culture_memory.arrays["actions"]))
                metric["culture/generation_id"] = jnp.asarray(config.get("GENERATION_ID", 1))
                metric["culture/queries"] = jnp.asarray(n_culture if culture_stats else 0)
                metric["culture/alpha"] = config.get("CULTURE_COEF", 0.) * culture_gate.mean()
                for key, value in culture_stats.items():
                    metric["culture/" + key] = value

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _inverse_loss_fn(
                        icm_encoder_params, icm_inverse_params, traj_batch
                    ):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.next_obs
                        )

                        action_pred_logits = ex_state["icm_inverse"].apply_fn(
                            icm_inverse_params, latent_obs, latent_next_obs
                        )
                        true_action = jax.nn.one_hot(
                            traj_batch.action, num_classes=action_pred_logits.shape[-1]
                        )

                        bce = -jnp.mean(
                            jnp.sum(
                                action_pred_logits
                                * true_action
                                * (1 - traj_batch.done[:, None]),
                                axis=1,
                            )
                        )

                        return bce * config["ICM_INVERSE_LOSS_COEF"]

                    inverse_grad_fn = jax.value_and_grad(
                        _inverse_loss_fn,
                        has_aux=False,
                        argnums=(
                            0,
                            1,
                        ),
                    )
                    inverse_loss, grads = inverse_grad_fn(
                        ex_state["icm_encoder"].params,
                        ex_state["icm_inverse"].params,
                        traj_batch,
                    )
                    icm_encoder_grad, icm_inverse_grad = grads
                    ex_state["icm_encoder"] = ex_state["icm_encoder"].apply_gradients(
                        grads=icm_encoder_grad
                    )
                    ex_state["icm_inverse"] = ex_state["icm_inverse"].apply_gradients(
                        grads=icm_inverse_grad
                    )

                    def _forward_loss_fn(icm_forward_params, traj_batch):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.next_obs
                        )

                        latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                            icm_forward_params, latent_obs, traj_batch.action
                        )

                        error = (latent_next_obs - latent_next_obs_pred) * (
                            1 - traj_batch.done[:, None]
                        )
                        return (
                            jnp.square(error).mean() * config["ICM_FORWARD_LOSS_COEF"]
                        )

                    forward_grad_fn = jax.value_and_grad(
                        _forward_loss_fn, has_aux=False
                    )
                    forward_loss, icm_forward_grad = forward_grad_fn(
                        ex_state["icm_forward"].params, traj_batch
                    )
                    ex_state["icm_forward"] = ex_state["icm_forward"].apply_gradients(
                        grads=icm_forward_grad
                    )

                    losses = (inverse_loss, forward_loss)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
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
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ICM"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

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
            if config.get("LOCAL_OUTPUT_DIR"):
                def local_callback(metric, update_step):
                    row = {k: float(np.asarray(v)) for k, v in metric.items()
                           if np.asarray(v).ndim == 0}
                    row['update'] = int(np.asarray(update_step))
                    with open(os.path.join(config["LOCAL_OUTPUT_DIR"], 'metrics.jsonl'), 'a') as file:
                        file.write(json.dumps(row, allow_nan=False) + '\n')
                jax.debug.callback(local_callback, metric, update_step)

            if (
                config["SAVE_POLICY"]
                and config["USE_WANDB"]
                and len(config["SAVE_POLICY_MILESTONES"]) > 0
            ):
                # Compare update counts so step totals above 2**31 remain safe
                # with JAX's default 32-bit integers. Compute checkpoint labels
                # using Python integers in the host callback.
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                milestone_updates = jnp.array(tuple(
                    (step - config["START_TIMESTEPS"] + batch_size - 1) // batch_size
                    for step in config["SAVE_POLICY_MILESTONES"]
                ), dtype=jnp.int32)
                current_update = update_step + 1
                should_save = jnp.any(current_update == milestone_updates)

                def _save_checkpoint(ts, cs):
                    import numpy as _np

                    step = config["START_TIMESTEPS"] + int(_np.asarray(cs)) * batch_size
                    if parallel_active(config):
                        from ppo_data_parallel import save_checkpoint_from_callback
                        with _CHECKPOINT_WRITE_LOCK:
                            save_checkpoint_from_callback(os.path.join(wandb.run.dir,'policies'),
                                step,ts,config['POLICY_MAX_TO_KEEP'])
                        print(f'Saved feedforward checkpoint at {step:,} steps',flush=True)
                        return
                    orbax_checkpointer = PyTreeCheckpointer()
                    options = CheckpointManagerOptions(
                        max_to_keep=config["POLICY_MAX_TO_KEEP"],
                        create=True,
                    )
                    path = os.path.join(wandb.run.dir, "policies")
                    checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
                    print(f"saving intermediate policy checkpoint to {path} at step {step}")
                    save_args = orbax_utils.save_args_from_target(ts)
                    checkpoint_manager.save(
                        step,
                        ts,
                        save_kwargs={"save_args": save_args},
                    )

                def _do_save(_):
                    jax.debug.callback(_save_checkpoint, train_state, current_update)
                    return _

                def _no_save(_):
                    return _

                _ = jax.lax.cond(should_save, _do_save, _no_save, operand=0)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            ex_state,
            _rng,
            0,
        )
        from training_time_limit import run_update_loop
        runner_state, metric = run_update_loop(
            _update_step, runner_state, config["NUM_UPDATES"],config.get('TRAINING_DEADLINE',0.))
        return {"runner_state": runner_state}  # , "info": metric}

    return train


def _resolve_policy_checkpoint(resume_from):
    if resume_from is None:
        return None, None

    checkpoint_path = resume_from
    basename = os.path.basename(os.path.normpath(resume_from))
    if basename.isdigit():
        return os.path.dirname(os.path.normpath(resume_from)), int(basename)

    policies_path = os.path.join(resume_from, "policies")
    if os.path.isdir(policies_path):
        checkpoint_path = policies_path

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=False)
    checkpoint_manager = CheckpointManager(
        checkpoint_path, orbax_checkpointer, options
    )
    latest_step = checkpoint_manager.latest_step()
    if latest_step is None:
        raise ValueError(f"No checkpoint steps found in {checkpoint_path}")

    return checkpoint_path, latest_step


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config['RECURRENT'] = False
    if config.get('MAP_WARMSTART_CROP_FROM') is not None and not (
            config.get('RESUME_FROM') and config.get('RESUME_RESET_OPTIMIZER')
            and config['SYMBOLIC_ENCODER']=='map_mlp' and not config.get('MAP_GLOBAL_READOUT')):
        raise ValueError('Crop widening requires a plain map_mlp checkpoint and a fresh optimizer')
    if config.get('MAP_PATCH_MLP') and not config.get('MAP_SINGLE_READOUT'):
        raise ValueError('The dense patch reader requires a single observed map')
    if config.get('MAP_BOARD_ONLY') and not config.get('MAP_SINGLE_READOUT'):
        raise ValueError('Board-only input requires the single observed world map')
    if config.get('MAP_BOARD_ATTENTION') and not config.get('MAP_BOARD_ONLY'):
        raise ValueError('Player-query attention requires the board-only policy')
    if config.get('MAP_V1_ATTENTION_ADAPTER') and not (config.get('MAP_V1_IMITATION_INIT') or config.get('RESUME_FROM')):
        raise ValueError('The frozen V1 decision layers need an imitation or PPO checkpoint')
    if config.get('MAP_SINGLE_READOUT') and (not config.get('MAP_GLOBAL_READOUT') or config.get('MAP_FULL_TERRAIN')):
        raise ValueError('Single-map policy requires observed-only global map input')
    if config.get('MAP_FULL_TERRAIN') and not config.get('MAP_GLOBAL_READOUT'):
        raise ValueError('Full terrain requires the global map readout')
    if config.get('MAP_GLOBAL_READOUT') and (config['SYMBOLIC_ENCODER'] != 'map_mlp'
            or config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1'):
        raise ValueError('Global map readout requires Classic symbolic Map + MLP')
    from ppo_data_parallel import validate_data_parallel,compile_train
    parallel_devices = validate_data_parallel(config,jax.local_device_count())
    for key, default in [('DEATH_PENALTY', 0.), ('LAVA_PENALTY',0.), ('DIAMOND_REWARD', 1.), ('PLANT_REWARD', 1.)]:
        value = config.get(key, default)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if (config.get('SYMBOLIC_ENCODER', 'mlp') == 'spatial' or config.get('DIAMOND_REWARD', 1.) != 1.
            or config.get('PLANT_REWARD', 1.) != 1. or config.get('LAVA_PENALTY',0.)) and config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1':
        raise ValueError('Spatial encoder and achievement reward adjustments require Classic symbolic observations')
    if config.get('DEATH_PENALTY',0.) and config.get('LAVA_PENALTY',0.):
        raise ValueError('Choose either a general death penalty or a lava-only penalty')
    if config.get('LEARNING_PROGRAM'):
        from learning_rule.graphs import Program
        if (config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1' or config.get('RESUME_FROM')
                or config.get('COMPONENT_INIT_FROM') or config['TRAIN_COMPONENT'] != 'both'
                or config.get('AUX_TASK', 'none') != 'none'
                or any(config.get(k) for k in ('TRAIN_ICM', 'USE_RND', 'USE_OPTIMISTIC_RESETS',
                    'CULTURE_PATH', 'STONE_PROBE', 'MULTI_OUTCOME_AUX_COEF', 'TARGETED_OUTCOME_AUX_COEF',
                    'REWARD_SIGN_AUX_COEF', 'DIAMOND_DRILLS', 'DIAMOND_CURRICULUM', 'DEATH_PENALTY',
                    'ID_PRACTICE', 'CLUSTER_FREE_PRACTICE'))):
            raise ValueError('Learning-rule search requires fresh ordinary Classic agents without other interventions')
        with open(config['LEARNING_PROGRAM']) as file:
            config['LEARNING_PROGRAM_GRAPH'] = Program.from_dict(json.load(file)).to_dict()
    if config.get("AUX_TASK", "none") != "none":
        if (config["ENV_NAME"] != 'Craftax-Classic-Symbolic-v1' or config.get("RESUME_FROM")
                or config.get("COMPONENT_INIT_FROM") or config["TRAIN_COMPONENT"] != 'both'
                or any(config.get(k) for k in ('TRAIN_ICM', 'USE_RND', 'USE_OPTIMISTIC_RESETS',
                    'CULTURE_PATH', 'STONE_PROBE', 'MULTI_OUTCOME_AUX_COEF', 'TARGETED_OUTCOME_AUX_COEF',
                    'REWARD_SIGN_AUX_COEF', 'DIAMOND_DRILLS', 'DIAMOND_CURRICULUM', 'DEATH_PENALTY',
                    'ID_PRACTICE', 'CLUSTER_FREE_PRACTICE'))):
            raise ValueError('Auxiliary search requires fresh ordinary Classic PPO without other interventions')
        if (config['AUX_COEF'] < 0 or not np.isfinite(config['AUX_COEF'])
                or config['AUX_HORIZON'] not in (1, 4, 16)
                or config['AUX_HORIZON'] > config['NUM_STEPS']
                or (config['AUX_TASK'] == 'inverse' and config['AUX_HORIZON'] != 1)
                or config['AUX_BATCH'] < config['NUM_MINIBATCHES']
                or config['AUX_BATCH'] % config['NUM_MINIBATCHES']):
            raise ValueError('Invalid auxiliary recipe or batch size')
    elif config.get('AUX_COEF', 0) != 0:
        raise ValueError('An auxiliary coefficient requires --aux_task')
    if config.get('LOCAL_OUTPUT_DIR'):
        config['LOCAL_OUTPUT_DIR'] = os.path.abspath(config['LOCAL_OUTPUT_DIR'])
        os.makedirs(config['LOCAL_OUTPUT_DIR'], exist_ok=True)
    config["RESUME_FROM"], config["RESUME_STEP"] = _resolve_policy_checkpoint(
        config["RESUME_FROM"]
    )
    (
        config["COMPONENT_INIT_FROM"],
        config["COMPONENT_INIT_STEP"],
    ) = _resolve_policy_checkpoint(config["COMPONENT_INIT_FROM"])
    if config["RESUME_FROM"] is not None and config["COMPONENT_INIT_FROM"] is not None:
        raise ValueError("--resume_from and --component_init_from are mutually exclusive")
    if config["TRAIN_COMPONENT"] != "both" and config["COMPONENT_INIT_FROM"] is None:
        raise ValueError(
            "A branch-specific --train_component requires --component_init_from"
        )
    if config["COMPONENT_INIT_FROM"] is not None:
        if config["TRAIN_COMPONENT"] == "both":
            raise ValueError(
                "--component_init_from requires --train_component actor or critic"
            )
        if "Symbolic" not in config["ENV_NAME"]:
            raise ValueError("Component transfer currently supports symbolic models only")
    config["START_TIMESTEPS"] = config["RESUME_STEP"] or 0
    if config.get("RESUME_RESET_OPTIMIZER") and config["RESUME_FROM"] is None:
        raise ValueError('Optimizer restart requires --resume_from')
    if config.get('CLUSTER_FREE_PRACTICE'):
        if (config['ENV_NAME'] != 'Craftax-Classic-Symbolic-v1' or config['USE_OPTIMISTIC_RESETS']
                or config['RESUME_FROM'] or config['COMPONENT_INIT_FROM'] or config['TRAIN_COMPONENT'] != 'both'
                or config.get('ID_PRACTICE') or config.get('DIAMOND_DRILLS') or config.get('DIAMOND_CURRICULUM')
                or config.get('DEATH_PENALTY') or config.get('CULTURE_PATH') or config.get('TRAIN_ICM')
                or config.get('USE_RND') or config.get('STONE_PROBE') or config.get('STONE_AUX_COEF')
                or config.get('TARGETED_OUTCOME_AUX_COEF') or config.get('MULTI_OUTCOME_AUX_COEF')
                or config.get('REWARD_SIGN_AUX_COEF')):
            raise ValueError('Cluster-free pilot requires fresh, standard-reset Classic PPO without other interventions')
    if config.get("ID_PRACTICE_SUCCESS_BOOTSTRAP") and (not config.get("ID_PRACTICE") or config.get("ID_PRACTICE_FULL_EPISODE")):
        raise ValueError('Success bootstrap requires short ID practice')
    if config.get("ID_PRACTICE_ARCHIVE_FROM") and (not config.get("ID_PRACTICE_RARE_FOUR") or config["RESUME_FROM"] is None or config.get("ID_PRACTICE_BACKWARD")):
        raise ValueError('Archive continuation requires resumed ordinary rare-four practice')
    if config.get("ID_PRACTICE_BACKWARD"):
        if not config.get("ID_PRACTICE_RARE_FOUR") or config.get("ID_PRACTICE_FULL_EPISODE"):
            raise ValueError("Backward practice requires short rare-four mode")
        if config["ID_PRACTICE_CAPACITY"]<4 or config["ID_PRACTICE_CAPACITY"]%4:
            raise ValueError("Backward archive capacity must be divisible by four")
    if config.get("ID_PRACTICE_FULL_EPISODE") and not config.get("ID_PRACTICE"):
        raise ValueError("Full-episode restarts require --id_practice")
    if config.get("ID_PRACTICE_RARE_FOUR"):
        if not config.get("ID_PRACTICE") or config["ID_PRACTICE_ENVS"]<1:
            raise ValueError("Rare-four mode requires nonzero ID practice slots")
        if min(config["ID_PRACTICE_RARITY_WINDOW"],config["ID_PRACTICE_RARITY_REFRESH"],
               config["ID_PRACTICE_RARITY_WARMUP"],config["ID_PRACTICE_MAX_REUSE"])<1:
            raise ValueError("Invalid rare-four rolling window/reuse configuration")
    if config.get("ID_PRACTICE"):
        if (config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1" or (config["RESUME_FROM"] is not None and not config.get("ID_PRACTICE_ARCHIVE_FROM"))
                or config["COMPONENT_INIT_FROM"] is not None or config["TRAIN_COMPONENT"] != 'both'
                or config.get("CULTURE_PATH") or config.get("DIAMOND_DRILLS") or config.get("DIAMOND_CURRICULUM")
                or config.get("DEATH_PENALTY") or config.get("TRAIN_ICM") or config.get("USE_RND")
                or config.get("STONE_PROBE") or config.get("STONE_AUX_COEF",0)
                or config.get("MULTI_OUTCOME_AUX_COEF",0) or config.get("TARGETED_OUTCOME_AUX_COEF",0)
                or config.get("REWARD_SIGN_AUX_COEF",0) or config["USE_OPTIMISTIC_RESETS"]):
            raise ValueError("ID practice requires ordinary Classic PPO, standard resets, and an explicit own-run archive when resuming")
        if not (0 <= config["ID_PRACTICE_ENVS"] < config["NUM_ENVS"]) or config["ID_PRACTICE_CAPACITY"] < 1 or config["ID_PRACTICE_HORIZON"] < 1:
            raise ValueError("Invalid ID-practice slots/archive/horizon")
    if config.get("TARGETED_OUTCOME_AUX_COEF", 0) < 0:
        raise ValueError("Targeted-outcome coefficient must be nonnegative")
    if config.get("TARGETED_OUTCOME_AUX_COEF", 0) > 0:
        if (config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1" or config["RESUME_FROM"] is not None
                or config["TRAIN_COMPONENT"] != 'both' or config.get("CULTURE_PATH")
                or config.get("DIAMOND_DRILLS") or config.get("DIAMOND_CURRICULUM")
                or config.get("DEATH_PENALTY") or config.get("TRAIN_ICM") or config.get("USE_RND")
                or config.get("STONE_AUX_COEF",0) != 0 or config.get("REWARD_SIGN_AUX_COEF",0) != 0
                or config.get("MULTI_OUTCOME_AUX_COEF",0) != 0):
            raise ValueError("Targeted-outcome pilot requires fresh ordinary Classic PPO without other auxiliary losses")
        if (config["TARGETED_OUTCOME_AUX_BATCH"] < config["NUM_MINIBATCHES"]
                or config["TARGETED_OUTCOME_AUX_BATCH"] % config["NUM_MINIBATCHES"]
                or config["TARGETED_OUTCOME_AUX_BATCH"] % 2 or config["TARGETED_OUTCOME_REPLAY_CAPACITY"] < 1):
            raise ValueError("Invalid targeted-outcome per-task batch or replay configuration")
    if config.get("MULTI_OUTCOME_AUX_COEF", 0) < 0:
        raise ValueError("Multi-outcome coefficient must be nonnegative")
    if config.get("MULTI_OUTCOME_AUX_COEF", 0) > 0:
        if (config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1" or config["RESUME_FROM"] is not None
                or config["TRAIN_COMPONENT"] != 'both' or config.get("CULTURE_PATH")
                or config.get("DIAMOND_DRILLS") or config.get("DIAMOND_CURRICULUM")
                or config.get("DEATH_PENALTY") or config.get("TRAIN_ICM") or config.get("USE_RND")
                or config.get("STONE_AUX_COEF", 0) != 0 or config.get("REWARD_SIGN_AUX_COEF", 0) != 0):
            raise ValueError("Multi-outcome pilot requires fresh ordinary Classic PPO without other auxiliary losses")
        if (config["MULTI_OUTCOME_AUX_BATCH"] < config["NUM_MINIBATCHES"]
                or config["MULTI_OUTCOME_AUX_BATCH"] % config["NUM_MINIBATCHES"]
                or config["MULTI_OUTCOME_REPLAY_CAPACITY"] < 1):
            raise ValueError("Invalid multi-outcome replay configuration")
    if config.get("REWARD_SIGN_AUX_COEF", 0) < 0:
        raise ValueError("Reward-sign coefficient must be nonnegative")
    if config.get("REWARD_SIGN_AUX_COEF", 0) > 0:
        if (config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1" or config["RESUME_FROM"] is not None
                or config["TRAIN_COMPONENT"] != 'both' or config.get("CULTURE_PATH")
                or config.get("DIAMOND_DRILLS") or config.get("DIAMOND_CURRICULUM")
                or config.get("DEATH_PENALTY") or config.get("TRAIN_ICM") or config.get("USE_RND")
                or config.get("STONE_AUX_COEF", 0) != 0):
            raise ValueError("Reward-sign pilot requires fresh ordinary Classic PPO without other auxiliary losses")
        if (config["REWARD_SIGN_AUX_BATCH"] < config["NUM_MINIBATCHES"]
                or config["REWARD_SIGN_AUX_BATCH"] % config["NUM_MINIBATCHES"]
                or config["REWARD_SIGN_REPLAY_CAPACITY"] < 1):
            raise ValueError("Invalid reward-sign replay configuration")
    if config.get("STONE_PROBE"):
        if (config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1" or config["RESUME_FROM"] is not None
                or config["TRAIN_COMPONENT"] != 'both' or config.get("CULTURE_PATH") or config.get("DIAMOND_DRILLS")
                or config.get("DIAMOND_CURRICULUM") or config.get("DEATH_PENALTY") or config.get("TRAIN_ICM") or config.get("USE_RND")):
            raise ValueError("Stone pilot requires fresh ordinary Classic PPO")
        if config["STONE_AUX_BATCH"] % config["NUM_MINIBATCHES"] or config["STONE_REPLAY_CAPACITY"] < 1 or config["STONE_AUX_COEF"] < 0:
            raise ValueError("Invalid stone auxiliary configuration")
    if config.get("DIAMOND_DRILLS"):
        if (config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1" or config["RESUME_FROM"] is not None
                or config.get("CULTURE_PATH") or config.get("DIAMOND_CURRICULUM") or config.get("DEATH_PENALTY")
                or config.get("TRAIN_ICM") or config.get("USE_RND")):
            raise ValueError("Diamond drills require fresh normal-reward Classic PPO without other interventions")
        if not 0 < config["DIAMOND_DRILL_ENVS"] < config["NUM_ENVS"] or config["DIAMOND_DRILL_HORIZON"] < 1:
            raise ValueError("Invalid drill environment count or horizon")
    if config.get("DIAMOND_CURRICULUM"):
        if config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1" or config["RESUME_FROM"] is not None:
            raise ValueError("Diamond curriculum supports fresh Classic symbolic runs only")
        if not 0 <= config["DIAMOND_DENSE_PROBABILITY"] <= 1 or config["DIAMOND_DENSITY_MULTIPLIER"] < 1:
            raise ValueError("Invalid diamond curriculum probability or multiplier")
        if config["DIAMOND_CURRICULUM_START"] < 0:
            raise ValueError("Curriculum start cannot be negative")
    if config.get("CULTURE_PATH"):
        if config["TRAIN_COMPONENT"] != "both" or config["COMPONENT_INIT_FROM"] is not None or config["RESUME_FROM"] is not None:
            raise ValueError("Culture pilot requires fresh actor and current-policy critic")
        if config["ENV_NAME"] != "Craftax-Classic-Symbolic-v1":
            raise ValueError("Culture pilot currently supports Classic symbolic observations")
    if (config.get("BC_STEPS", 0) or config.get("CULTURE_COEF", 0)) and not config.get("CULTURE_PATH"):
        raise ValueError("BC pretraining and cultural loss require --culture_path")
    if config.get("CULTURE_QUERY_BATCH", 512) % config["NUM_MINIBATCHES"]:
        raise ValueError("culture_query_batch must divide into PPO minibatches")
    if config.get("CULTURE_DIAMOND_ORACLE") and not config.get("CULTURE_PATH"):
        raise ValueError("Diamond oracle requires a successful-event --culture_path")
    milestones = []
    if config["SAVE_POLICY_INTERVAL"] and config["SAVE_POLICY_INTERVAL"] > 0:
        next_step = (
            (config["START_TIMESTEPS"] // config["SAVE_POLICY_INTERVAL"]) + 1
        ) * config["SAVE_POLICY_INTERVAL"]
        while next_step < config["TOTAL_TIMESTEPS"]:
            milestones.append(int(next_step))
            next_step += config["SAVE_POLICY_INTERVAL"]
    if config["SAVE_POLICY_MILESTONES"]:
        for raw in str(config["SAVE_POLICY_MILESTONES"]).split(","):
            raw = raw.strip()
            if raw:
                step = int(float(raw))
                if config["START_TIMESTEPS"] < step < config["TOTAL_TIMESTEPS"]:
                    milestones.append(step)
    config["SAVE_POLICY_MILESTONES"] = tuple(sorted(set(milestones)))
    config["POLICY_MAX_TO_KEEP"] = max(1, len(config["SAVE_POLICY_MILESTONES"]) + 1)
    if config['RESUME_FROM']:
        existing=sum(name.isdigit() for name in os.listdir(config['RESUME_FROM']))
        config['POLICY_MAX_TO_KEEP']=max(config['POLICY_MAX_TO_KEEP'],existing+len(config['SAVE_POLICY_MILESTONES'])+1)
    if config["RESUME_FROM"] is not None:
        print(
            "Resuming from "
            f"{config['RESUME_FROM']} at step {config['RESUME_STEP']:,}; "
            f"training until {config['TOTAL_TIMESTEPS']:,} total timesteps."
        )
    if config["COMPONENT_INIT_FROM"] is not None:
        if config["TRAIN_COMPONENT"] == "actor":
            transfer_description = "loading and freezing the critic"
        elif config["TRAIN_COMPONENT"] == "critic":
            transfer_description = "loading and freezing the actor"
        else:
            transfer_description = (
                "loading the critic, freezing its representation, and training "
                "only its scalar head"
            )
        print(
            f"Training {config['TRAIN_COMPONENT']} from scratch for "
            f"{config['TOTAL_TIMESTEPS']:,} timesteps; {transfer_description} "
            "from checkpoint "
            f"{config['COMPONENT_INIT_STEP']:,}."
        )
    if config["SAVE_POLICY"] and config["SAVE_POLICY_MILESTONES"]:
        print(
            "Intermediate policy checkpoints: "
            + ", ".join(f"{step:,}" for step in config["SAVE_POLICY_MILESTONES"])
        )

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=(
                config["RUN_NAME"]
                or config["ENV_NAME"]
                + "-"
                + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
                + "M"
            ),
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = compile_train(config) if parallel_devices>1 else jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit) if parallel_devices==1 else None

    t0 = time.time()
    if parallel_devices>1:
        out=jax.tree.map(lambda x:x[None],train_jit(rngs[0]))
    elif config.get('TRAINING_DEADLINE',0.):
        if config['NUM_REPEATS']!=1:raise ValueError('Wall-clock capped training supports one repeat per process')
        out=jax.tree.map(lambda x:x[None],train_jit(rngs[0]))
    else:
        out = train_vmap(rngs)
    jax.block_until_ready(out)
    if parallel_devices>1:jax.effects_barrier()
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    completed_updates=int(np.asarray(out['runner_state'][-1][0]))
    executed_steps = completed_updates*config["NUM_STEPS"]*config["NUM_ENVS"]
    print("New environment steps:", executed_steps)
    print("SPS: ", executed_steps / (t1 - t0))

    if config["USE_WANDB"] or config.get("LOCAL_OUTPUT_DIR"):
        save_root = config.get("LOCAL_OUTPUT_DIR") or wandb.run.dir
        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            if parallel_devices>1:
                from ppo_data_parallel import checkpoint_train_state
                train_state=checkpoint_train_state(train_state)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(
                max_to_keep=config["POLICY_MAX_TO_KEEP"],
                create=True,
            )
            path = os.path.join(save_root, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            save_step = (
                config["START_TIMESTEPS"]
                + executed_steps
            )
            checkpoint_manager.save(
                save_step,
                train_state,
                save_kwargs={"save_args": save_args},
            )
            checkpoint_manager.wait_until_finished()

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")
        if config.get("ID_PRACTICE") and config["ID_PRACTICE_ENVS"]:
            from id_practice import save_archive
            final_environment = jax.tree.map(lambda x:x[0],out['runner_state'][1])
            save_archive(final_environment,os.path.join(save_root,'practice_archive'))
        if config.get('CLUSTER_FREE_PRACTICE'):
            from cluster_free_practice import save_cluster_free_archive
            final_environment = jax.tree.map(lambda x: x[0], out['runner_state'][1])
            save_cluster_free_archive(final_environment, os.path.join(save_root, 'cluster_free_bank'))
    if config['USE_WANDB']:
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
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument('--lr_horizon_timesteps',type=lambda x:int(float(x)),default=None,
        help='Optional LR annealing horizon independent of the stopping step target.')
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument(
        "--death_penalty", type=float, default=0.0,
        help="Additional cost on death (including lava), excluding time-limit-only termination.",
    )
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument('--diamond_reward', type=float, default=1., help='Total first-diamond achievement reward.')
    parser.add_argument('--all22_terminal_bonus', type=float, help='End training episodes on all 22 and add this one-time bonus.')
    parser.add_argument('--plant_reward', type=float, default=1., help='Total first-eat-plant achievement reward.')
    parser.add_argument('--symbolic_encoder', choices=('mlp', 'spatial', 'map_mlp'), default='mlp')
    parser.add_argument('--map_view_size',type=int,default=33)
    parser.add_argument('--map_warmstart_crop_from',type=int,help='Center the trained source crop weights inside this wider crop.')
    parser.add_argument('--map_embedding_size',type=int,default=4)
    parser.add_argument('--map_global_readout', action='store_true',
                        help='Add the full observed terrain/age map to the local Map + MLP input.')
    parser.add_argument('--map_single_readout', action='store_true',
                        help='Supply the live observation, one dated world map and position.')
    parser.add_argument('--map_internal_crop', action='store_true',
                        help='Read the original Map-MLP neighborhood directly from the supplied world map.')
    parser.add_argument('--map_patch_mlp', action='store_true', help='Use a non-convolutional single-map reader.')
    parser.add_argument('--map_spatial_regions', action='store_true', help='Read the full live view and an ordered 8x8x32 regional terrain grid.')
    parser.add_argument('--map_board_only', action='store_true', help='Read one fixed observed world board with visible entities and a player marker.')
    parser.add_argument('--map_board_attention', action='store_true', help='Add a player-conditioned attention read of the fixed board.')
    parser.add_argument('--map_v1_attention_adapter', action='store_true', help='Train only the board adapter feeding frozen V1 decision layers.')
    parser.add_argument('--map_v1_imitation_init', help='Initialize PPO weights from a frozen-V1 imitation checkpoint with fresh Adam state.')
    parser.add_argument('--map_fine_shortcut', action='store_true', help='Add a gated, ordered path from the 16x16 map features.')
    parser.add_argument('--map_landmarks',choices=('masked','locations','events'))
    parser.add_argument('--map_full_terrain', action='store_true',
                        help='Expose every current terrain tile in the global map channel.')
    parser.add_argument('--training_deadline',type=float,default=0.,help='Unix-time deadline; stop after a PPO update and save actual progress.')
    parser.add_argument('--data_parallel_devices',type=int,default=1)
    parser.add_argument('--lava_penalty',type=float,default=0.,help='Additional cost only on terminal lava entry; ordinary deaths retain default rewards.')
    parser.add_argument('--map_coverage_bonus', type=float, default=0., help='Intrinsic PPO reward per newly observed in-bounds tile on a nonterminal step.')
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
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
    parser.add_argument(
        "--save_policy_interval",
        type=lambda x: int(float(x)),
        default=0,
        help=(
            "Save intermediate policy checkpoints every N environment steps. "
            "The final checkpoint is still saved by --save_policy."
        ),
    )
    parser.add_argument(
        "--save_policy_milestones",
        type=str,
        default="",
        help=(
            "Comma-separated intermediate policy checkpoint steps, e.g. "
            "1e9,2e9,5e9. Final TOTAL_TIMESTEPS is saved separately."
        ),
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Path to a saved wandb files directory or policies checkpoint "
            "directory to continue training from."
        ),
    )
    parser.add_argument('--recovery_bank',type=str)
    parser.add_argument('--recovery_probability',type=float,default=0.)
    parser.add_argument('--reset_seed_bank',type=str,default=None,
        help='Optional JSON world_seeds bank; uniformly sample original-generator worlds on reset.')
    parser.add_argument(
        "--component_init_from",
        type=str,
        default=None,
        help=(
            "Checkpoint directory used to initialize the frozen actor or critic "
            "while the trainable branch and optimizer start from scratch."
        ),
    )
    parser.add_argument(
        "--train_component",
        choices=("both", "actor", "critic", "actor_critic_head"),
        default="both",
        help=(
            "Parameters to update. actor_critic_head trains a fresh actor plus "
            "the pretrained critic's scalar head while freezing its trunk."
        ),
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument('--local_output_dir', type=str, default=None,
                        help='Save final checkpoints and scalar metrics locally, including with --no-use_wandb')
    parser.add_argument('--aux_task', choices=('none', 'inverse', 'observation_delta', 'stats_delta', 'event'), default='none')
    parser.add_argument('--aux_horizon', type=int, default=1)
    parser.add_argument('--aux_coef', type=float, default=0.)
    parser.add_argument('--aux_batch', type=int, default=256)
    parser.add_argument('--aux_target_seed', type=int, default=2026)
    parser.add_argument('--learning_program', type=str, default=None,
                        help='Validated JSON arithmetic program replacing credit assignment and the complete optimized loss')
    parser.add_argument("--culture_path", type=str, default=None)
    parser.add_argument("--culture_diamond_oracle", action="store_true")
    parser.add_argument("--diamond_curriculum", action="store_true")
    parser.add_argument("--diamond_drills", action="store_true")
    parser.add_argument("--stone_probe", action="store_true")
    parser.add_argument("--stone_aux_coef", type=float, default=0.)
    parser.add_argument("--stone_aux_batch", type=int, default=256)
    parser.add_argument("--stone_replay_capacity", type=int, default=1024)
    parser.add_argument("--reward_sign_aux_coef", type=float, default=0.)
    parser.add_argument("--reward_sign_aux_batch", type=int, default=256)
    parser.add_argument("--reward_sign_replay_capacity", type=int, default=1024)
    parser.add_argument("--multi_outcome_aux_coef", type=float, default=0.)
    parser.add_argument("--multi_outcome_aux_batch", type=int, default=256)
    parser.add_argument("--multi_outcome_replay_capacity", type=int, default=1024)
    parser.add_argument("--targeted_outcome_aux_coef", type=float, default=0.)
    parser.add_argument("--targeted_outcome_aux_batch", type=int, default=256,help="Replay examples PER task per rollout (22 tasks)")
    parser.add_argument("--targeted_outcome_replay_capacity", type=int, default=1024)
    parser.add_argument("--id_practice", action="store_true")
    parser.add_argument('--cluster_free_practice', choices=['baseline', 'uniform', 'prioritized'])
    parser.add_argument('--cf_practice_envs', type=int, default=13)
    parser.add_argument('--cf_capacity', type=int, default=2048)
    parser.add_argument('--cf_horizon', type=int, default=30)
    parser.add_argument('--cf_warmup', type=int, default=1007616)
    parser.add_argument('--cf_max_reuse', type=int, default=30)
    parser.add_argument("--id_practice_envs", type=int, default=32)
    parser.add_argument("--id_practice_capacity", type=int, default=64)
    parser.add_argument("--id_practice_horizon", type=int, default=64)
    parser.add_argument("--id_practice_rare_four", action="store_true")
    parser.add_argument("--id_practice_rarity_window", type=int, default=512)
    parser.add_argument("--id_practice_rarity_refresh", type=int, default=128)
    parser.add_argument("--id_practice_rarity_warmup", type=int, default=512)
    parser.add_argument("--id_practice_max_reuse", type=int, default=10)
    parser.add_argument("--id_practice_full_episode", action="store_true")
    parser.add_argument("--id_practice_backward", action="store_true")
    parser.add_argument("--id_practice_success_bootstrap", action="store_true")
    parser.add_argument("--id_practice_archive_from", type=str, default=None,
                        help="Trusted own-run rare-four archive directory; preserves snapshot reuse quotas.")
    parser.add_argument("--resume_reset_optimizer", action="store_true",
                        help="Keep restored weights, restart Adam and annealing over the continuation budget.")
    parser.add_argument('--policy_temperature',type=float,default=1.,
        help='Strictly positive PPO action temperature, used for collection, old/new likelihoods and entropy.')
    parser.add_argument("--diamond_drill_envs", type=int, default=32)
    parser.add_argument("--diamond_drill_horizon", type=int, default=64)
    parser.add_argument("--diamond_curriculum_start", type=int, default=50000000)
    parser.add_argument("--diamond_dense_probability", type=float, default=.25)
    parser.add_argument("--diamond_density_multiplier", type=int, default=10)
    parser.add_argument("--culture_coef", type=float, default=0.)
    parser.add_argument("--culture_k", type=int, default=4)
    parser.add_argument("--culture_threshold", type=float, default=.95)
    parser.add_argument("--culture_temperature", type=float, default=.1)
    parser.add_argument("--culture_query_batch", type=int, default=512)
    parser.add_argument("--culture_weighting", choices=["similarity_return", "similarity", "return"], default="similarity_return")
    parser.add_argument("--culture_shuffle_actions", action="store_true")
    parser.add_argument("--culture_random_retrieval", action="store_true")
    parser.add_argument("--bc_steps", type=int, default=0)
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--generation_id", type=int, default=1)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)

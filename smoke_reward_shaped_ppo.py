"""Exercise two full-size PPO updates and save finite trained parameters."""
import argparse,json,time
from pathlib import Path
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp


def verify_global_map_transfer(cfg, source):
    from craftax.craftax_env import make_craftax_env_from_name
    from models.actor_critic_map_mlp import map_mlp_network, transfer_map_mlp_params
    from symbolic_map_memory import with_map_memory

    raw = make_craftax_env_from_name(cfg['ENV_NAME'], False)
    config_local = dict(cfg, MAP_GLOBAL_READOUT=False, MAP_FULL_TERRAIN=False)
    local_env = with_map_memory(raw, config_local)
    global_env = with_map_memory(raw, cfg)
    key = jax.random.PRNGKey(16610)
    obs, state = global_env.reset(key, raw.default_params)
    local_obs, local_state = local_env.reset(key, raw.default_params)
    local_net, global_net = map_mlp_network(config_local), map_mlp_network(cfg)
    params = transfer_map_mlp_params(global_net.init(key, obs[None]), source)
    observations = []
    for step in range(12):
        np.testing.assert_array_equal(local_obs, obs[:local_obs.size])
        cells = 64**2
        terrain = np.asarray(obs[local_obs.size:local_obs.size+cells]).reshape(64, 64)
        ages = np.asarray(obs[local_obs.size+cells:-2]).reshape(64, 64)
        seen = np.asarray(state.last_seen) >= 0
        if cfg.get('MAP_FULL_TERRAIN'):
            np.testing.assert_array_equal(terrain, np.asarray(state.env_state.map) + 1)
            assert np.any(~seen)
        else:
            np.testing.assert_array_equal(terrain > 0, seen)
        np.testing.assert_allclose(ages[seen], np.log1p(step-np.asarray(state.last_seen)[seen])/np.log(10001.), atol=1e-6)
        np.testing.assert_array_equal(ages[~seen], 1.)
        np.testing.assert_array_equal(np.rint(np.asarray(obs[-2:])*63), state.position)
        observations.append(obs)
        action = jnp.asarray(1+step%4, jnp.int32)
        step_key = jax.random.fold_in(key, step)
        obs, state, *_ = global_env.step(step_key, state, action, raw.default_params)
        local_obs, local_state, *_ = local_env.step(step_key, local_state, action, raw.default_params)
    batch = jnp.stack(observations)
    local_pi, local_value = local_net.apply(source, batch[:, :local_obs.size])
    global_pi, global_value = global_net.apply(params, batch)
    np.testing.assert_array_equal(global_pi.probs, local_pi.probs)
    np.testing.assert_array_equal(global_value, local_value)
    print('Verified exact transferred action probabilities and values on 12 factual observations', flush=True)


def verify_widened_crop(cfg, source):
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.renderer import render_craftax_symbolic
    from models.actor_critic_map_mlp import map_mlp_network, widen_map_mlp_params
    from symbolic_map_memory import LOCAL_SIZE, with_map_memory

    old_size, size = cfg['MAP_WARMSTART_CROP_FROM'], cfg['MAP_VIEW_SIZE']
    old_cfg = dict(cfg, MAP_VIEW_SIZE=old_size)
    raw = make_craftax_env_from_name(cfg['ENV_NAME'], False)
    narrow, wide = with_map_memory(raw, old_cfg), with_map_memory(raw, cfg)
    key = jax.random.PRNGKey(17081)
    _, state = narrow.reset(key, raw.default_params)
    before, after = [], []
    def record(memory):
        live = render_craftax_symbolic(memory.env_state)
        before.append(narrow.observe(live, memory)[0])
        after.append(wide.observe(live, memory)[0])
    for step in range(20):
        record(state)
        step_key = jax.random.fold_in(key, step)
        _, state, *_ = narrow.step(step_key, state, jnp.asarray(1+step//5), raw.default_params)
    for point in ((0,0), (63,63), (2,60), (61,3)):
        pos = jnp.array(point)
        record(state.replace(position=pos, last_position=pos, env_state=state.env_state.replace(player_position=pos)))
    before, after = jnp.stack(before), np.stack(after)
    pad = (size-old_size)//2
    outer = np.ones((size, size), bool)
    outer[pad:pad+old_size, pad:pad+old_size] = False
    grid = after[:, LOCAL_SIZE:].reshape(len(after), 2, size, size)
    np.testing.assert_array_equal(after[:, :LOCAL_SIZE], before[:, :LOCAL_SIZE])
    np.testing.assert_array_equal(grid[:,:,pad:pad+old_size,pad:pad+old_size].reshape(len(after),-1), before[:, LOCAL_SIZE:])
    old_net, new_net = map_mlp_network(old_cfg), map_mlp_network(cfg)
    template = jax.eval_shape(new_net.init, key, jnp.asarray(after[:1]))
    params = widen_map_mlp_params(template, source, old_size, size)
    old_pi, old_value = old_net.apply(source, before)
    def check(batch):
        pi, value = new_net.apply(params, batch)
        np.testing.assert_allclose(pi.probs, old_pi.probs, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(value, old_value, atol=3e-5, rtol=2e-5)
        return float(np.max(np.abs(pi.probs-old_pi.probs))), float(np.max(np.abs(value-old_value)))
    prob_error, value_error = check(after)
    altered = grid.copy()
    rng = np.random.default_rng(17081)
    altered[:,0,outer] = rng.integers(0, 18, size=(len(after),outer.sum()))
    altered[:,1,outer] = rng.random((len(after),outer.sum()))
    check(np.concatenate((after[:,:LOCAL_SIZE], altered.reshape(len(after),-1)), axis=1))
    return dict(crop_warmstart_from=old_size, transfer_states=len(after),
                maximum_probability_error=prob_error, maximum_value_error=value_error,
                perturbed_outer_ring_checked=True)


def verify_sru_transfer(config, source):
    from craftax.craftax_env import make_craftax_env_from_name
    from models.sru_gru import transplant_gru
    from ppo_rnn_og import ActorCriticRNN, initialize_recurrent_carry
    from symbolic_motion_inputs import with_motion_inputs

    env = with_motion_inputs(make_craftax_env_from_name(config['ENV_NAME'], False), config)
    key = jax.random.PRNGKey(12319)
    observations, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.default_params)
    sequence = [observations]
    for step in range(15):
        keys = jax.random.split(jax.random.fold_in(key, step), 4)
        actions = 1 + (jnp.arange(4) + step//4) % 4
        observations, state, *_ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            keys, state, actions, env.default_params)
        sequence.append(observations)
    obs = jnp.stack(sequence)
    dones = jnp.zeros(obs.shape[:2], dtype=bool).at[8, 0].set(True)
    hidden = initialize_recurrent_carry(4, config)
    baseline = ActorCriticRNN(17, dict(config, SRU_GRU=False))
    augmented = ActorCriticRNN(17, config)
    params = transplant_gru(augmented.init(key, hidden, (obs[:1], dones[:1])), source)
    old_h, old_pi, old_v = baseline.apply(source, hidden, (obs, dones))
    new_h, new_pi, new_v = augmented.apply(params, hidden, (obs, dones))
    maximum = 0.
    for old, new in [(old_h, new_h), (old_pi.probs, new_pi.probs), (old_v, new_v)]:
        np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-6)
        maximum = max(maximum, float(np.max(np.abs(new-old))))
    return dict(sru_transfer_observations=obs.shape[0]*obs.shape[1], sru_maximum_output_error=maximum)


def main():
    p=argparse.ArgumentParser();p.add_argument('--architecture',choices=('mlp','gru','spatial','spatial_gru','spatial_map_gru','map_mlp','map_mlp_gru','neural_map_gru'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--sru-smoke-unfreeze',action='store_true',
                   help='Exercise the joint-training phase from a short SRU smoke checkpoint.')
    p.add_argument('--deadline-check',action='store_true')
    p.add_argument('--lava-only',action='store_true')
    p.add_argument('--config-json',type=Path);args=p.parse_args()
    recurrent=args.architecture in ('gru','spatial_gru','spatial_map_gru','map_mlp_gru','neural_map_gru')
    cfg=dict(ENV_NAME='Craftax-Classic-Symbolic-v1',NUM_ENVS=1024,NUM_STEPS=64,
        TOTAL_TIMESTEPS=131072,NUM_MINIBATCHES=8,UPDATE_EPOCHS=4,GAMMA=.99,GAE_LAMBDA=.8,
        CLIP_EPS=.2,VF_COEF=.5,ENT_COEF=.01,LR=.0002,MAX_GRAD_NORM=1.,ANNEAL_LR=True,
        LAYER_SIZE=512,SYMBOLIC_ENCODER=('map_mlp' if args.architecture in ('map_mlp','map_mlp_gru','neural_map_gru') else 'spatial_map' if args.architecture=='spatial_map_gru' else 'spatial' if args.architecture.startswith('spatial') else 'mlp'),
        MEMORY_TYPE='gru',RECURRENT=recurrent,DEATH_PENALTY=.5,DIAMOND_REWARD=5.,PLANT_REWARD=5.,
        USE_OPTIMISTIC_RESETS=True,OPTIMISTIC_RESET_RATIO=16,DEBUG=False,USE_WANDB=False,
        SAVE_POLICY=False,NUM_REPEATS=1,TRAIN_ICM=False,TRAIN_COMPONENT='both',
        RESUME_FROM=None,COMPONENT_INIT_FROM=None,SEED=50)
    if args.config_json:
        supplied={k.upper():v for k,v in json.loads(args.config_json.read_text()).items()}
        allowed=('ENV_NAME','NUM_ENVS','NUM_STEPS','NUM_MINIBATCHES','UPDATE_EPOCHS','LAYER_SIZE',
            'GAMMA','GAE_LAMBDA','CLIP_EPS','VF_COEF','ENT_COEF','LR','MAX_GRAD_NORM','ANNEAL_LR',
            'USE_OPTIMISTIC_RESETS','OPTIMISTIC_RESET_RATIO','DEATH_PENALTY','LAVA_PENALTY',
            'DIAMOND_REWARD','PLANT_REWARD','ALL22_TERMINAL_BONUS','SEED','LR_HORIZON_TIMESTEPS','MAP_VIEW_SIZE',
            'RECOVERY_BANK','RECOVERY_PROBABILITY','RESET_SEED_BANK','RESUME_RESET_OPTIMIZER','POLICY_TEMPERATURE','MAP_EMBEDDING_SIZE','MAP_GLOBAL_READOUT',
            'MAP_LANDMARKS','MAP_WARMSTART_CROP_FROM','MAP_FULL_TERRAIN','MAP_SINGLE_READOUT','MAP_INTERNAL_CROP','MAP_PATCH_MLP','MAP_SPATIAL_REGIONS','DATA_PARALLEL_DEVICES','RESET_GRU_EVERY_STEP','GRU_CONCAT_OBSERVATION','WORLD_VIEW','LOCAL_POLICY_INIT','MOTION_INPUTS','SRU_EPISODE_MEMORY','POSITION_TIME_INPUTS','MAP_SURVIVAL_FEATURES','NEURAL_MAP_CHANNELS','NEURAL_MAP_RESET_EVERY_STEP')
        cfg.update({k:supplied[k] for k in allowed if k in supplied})
        cfg['MAP_FINE_SHORTCUT'] = supplied.get('MAP_FINE_SHORTCUT', False)
        cfg['MAP_BOARD_ONLY'] = supplied.get('MAP_BOARD_ONLY', False)
        cfg['MAP_COVERAGE_BONUS'] = supplied.get('MAP_COVERAGE_BONUS', 0.)
        cfg['MAP_BOARD_ATTENTION'] = supplied.get('MAP_BOARD_ATTENTION', False)
        cfg['MAP_V1_ATTENTION_ADAPTER'] = supplied.get('MAP_V1_ATTENTION_ADAPTER', False)
        cfg['MAP_V1_IMITATION_INIT'] = supplied.get('MAP_V1_IMITATION_INIT')
        cfg.update({k:v for k,v in supplied.items() if k.startswith('CTEC_')})
        cfg.update({k:v for k,v in supplied.items() if k.startswith('SRU_')})
        cfg['TOTAL_TIMESTEPS']=2*cfg['NUM_ENVS']*cfg['NUM_STEPS']
    if args.lava_only:cfg.update(DEATH_PENALTY=0.,LAVA_PENALTY=2.,DIAMOND_REWARD=1.,PLANT_REWARD=1.)
    original_optimizer_steps=0
    memory_report={}
    original_curiosity_updates=0
    original_curiosity_optimizer_steps=0
    if args.resume_checkpoint:
        checkpoint=args.resume_checkpoint.resolve()
        payload=checkpoint/'default' if (checkpoint/'default').exists() else checkpoint
        source_state=ocp.PyTreeCheckpointer().restore(str(payload))
        if cfg.get('MAP_FINE_SHORTCUT') and 'fine_gate' not in source_state['params']['params']['global_reader']:
            import optax
            from flax.training.train_state import TrainState
            from craftax.craftax_env import make_craftax_env_from_name
            from models.actor_critic_map_mlp import map_mlp_network, restore_map_shortcut_state
            from symbolic_map_memory import with_map_memory
            env = with_map_memory(make_craftax_env_from_name(cfg['ENV_NAME'], False), cfg)
            obs, _ = jax.vmap(env.reset, in_axes=(0,None))(jax.random.split(jax.random.PRNGKey(709), 16), env.default_params)
            plain = map_mlp_network(dict(cfg, MAP_FINE_SHORTCUT=False))
            shortcut = map_mlp_network(cfg)
            tx = optax.chain(optax.clip_by_global_norm(cfg['MAX_GRAD_NORM']), optax.adam(cfg['LR'], eps=1e-5))
            template = TrainState.create(apply_fn=shortcut.apply, params=shortcut.init(jax.random.PRNGKey(710), obs[:1]), tx=tx)
            restored = restore_map_shortcut_state(ocp.PyTreeCheckpointer(), payload, template)
            pi, value = shortcut.apply(restored.params, obs)
            old_pi, old_value = plain.apply(source_state['params'], obs)
            np.testing.assert_array_equal(pi.logits, old_pi.logits)
            np.testing.assert_array_equal(value, old_value)
            memory_report['zero_gate_equivalent_worlds'] = len(obs)
        if cfg.get('MAP_LANDMARKS') and 'landmark_reader' not in source_state['params']['params']:
            from models.actor_critic_map_mlp import map_mlp_network, transfer_map_mlp_params
            from symbolic_map_memory import with_map_memory
            from symbolic_landmarks import SLOTS, FIELDS
            from craftax.craftax_env import make_craftax_env_from_name
            env = with_map_memory(make_craftax_env_from_name(cfg['ENV_NAME'],False),cfg)
            obs, _ = jax.vmap(env.reset,in_axes=(0,None))(jax.random.split(jax.random.PRNGKey(71),8),env.default_params)
            net = map_mlp_network(cfg)
            old = map_mlp_network(dict(cfg,MAP_LANDMARKS=None))
            source = source_state['params']
            params = transfer_map_mlp_params(net.init(jax.random.PRNGKey(18),obs),source)
            pi,v = net.apply(params,obs)
            old_pi,old_v = old.apply(source,obs[:,:-SLOTS*FIELDS])
            np.testing.assert_allclose(pi.probs,old_pi.probs,atol=2e-5,rtol=1e-5)
            np.testing.assert_allclose(v,old_v,atol=2e-5,rtol=1e-5)
            memory_report['landmark_transfer_states'] = 8
        if cfg.get('MAP_WARMSTART_CROP_FROM'):
            memory_report.update(verify_widened_crop(cfg, source_state['params']))
            print('Widened crop transfer verified:', memory_report, flush=True)
        if cfg.get('MAP_GLOBAL_READOUT') and not cfg.get('MAP_BOARD_ONLY') and 'global_reader' not in source_state['params']['params']:
            verify_global_map_transfer(cfg, source_state['params'])
        original_optimizer_steps=int(source_state['step'])
        if cfg.get('SRU_GRU'):
            from models.sru_gru import has_spatial_gate
            source_sru = has_spatial_gate(source_state['params'])
            if not source_sru:
                original_optimizer_steps = 0
                memory_report.update(verify_sru_transfer(cfg, source_state['params']))
            if args.sru_smoke_unfreeze:
                if not source_sru:
                    raise ValueError('Joint-training smoke requires a trained SRU smoke checkpoint')
                updates = original_optimizer_steps // (cfg['NUM_MINIBATCHES'] * cfg['UPDATE_EPOCHS'])
                cfg['SRU_WARMUP_STEPS'] = updates * cfg['NUM_ENVS'] * cfg['NUM_STEPS']
        if cfg.get('CTEC_COEF',0.)>0:
            original_curiosity_updates=int(source_state['curiosity']['updates'])
            original_curiosity_optimizer_steps=int(source_state['curiosity']['learner']['step'])
        if cfg.get('RESUME_RESET_OPTIMIZER'):original_optimizer_steps=0
        cfg.update(RESUME_FROM=str(checkpoint.parent),RESUME_STEP=int(checkpoint.name),START_TIMESTEPS=int(checkpoint.name))
        cfg['TOTAL_TIMESTEPS']=int(checkpoint.name)+2*cfg['NUM_ENVS']*cfg['NUM_STEPS']
    if args.deadline_check:
        cfg.update(TOTAL_TIMESTEPS=10000000000,TRAINING_DEADLINE=time.time()-1)
    if recurrent:
        from ppo_rnn_og import make_train
    else:
        from ppo import make_train
    began=time.time()
    keys=jax.random.split(jax.random.PRNGKey(cfg['SEED']),1)
    if cfg.get('DATA_PARALLEL_DEVICES',1)>1:
        from ppo_data_parallel import execute_train
        result=jax.tree_util.tree_map(lambda a:a[None],execute_train(cfg,keys[0]))
    elif args.deadline_check:
        train=jax.jit(make_train(cfg))
        result=jax.tree_util.tree_map(lambda a:a[None],train(keys[0]))
    else:
        train=jax.jit(make_train(cfg))
        result=jax.vmap(train)(keys)
    jax.block_until_ready(result)
    state=jax.tree_util.tree_map(lambda x:x[0],result['runner_state'][0])
    completed_updates=int(np.asarray(result['runner_state'][-1][0]))
    expected_updates=16 if args.deadline_check else 2
    assert completed_updates==expected_updates, f'Incorrect number of training updates: {completed_updates}'
    assert int(state.step)==original_optimizer_steps+expected_updates*cfg['UPDATE_EPOCHS']*cfg['NUM_MINIBATCHES'], f'Incorrect optimizer resume: {state.step}'
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state.params))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state.opt_state))
    if cfg.get('SRU_GRU') and args.resume_checkpoint:
        from flax.core import unfreeze
        from flax.traverse_util import flatten_dict
        from models.sru_gru import GATE_PATH, warmup_minibatches
        before = flatten_dict(unfreeze(source_state['params']))
        after = flatten_dict(unfreeze(state.params))
        gate = {k for k in after if k[:-1] == GATE_PATH}
        gate_changed = (not np.array_equal(before[GATE_PATH+('kernel',)], after[GATE_PATH+('kernel',)])
                        if source_sru else np.any(np.asarray(after[GATE_PATH+('kernel',)]) != 0))
        assert gate_changed, 'The SRU gate did not train'
        updated = {k[1] for k in before if k not in gate and not np.array_equal(before[k], after[k])}
        if int(state.step) <= warmup_minibatches(cfg):
            assert not updated, 'Pretrained parameters changed during SRU warmup'
        else:
            heads = {'actor_head','critic_head','memory_input'} if cfg.get('SRU_RESIDUAL_ACTOR_CRITIC') else {'Dense_1','Dense_4'}
            assert {'Dense_0', 'ScannedRNN_0'} | heads <= updated, updated
        memory_report.update(sru_gate_updated=True, sru_updated_pretrained_groups=sorted(updated))
        if cfg.get('SRU_MAP_COEF', 0.) > 0:
            decoder_steps = int(source_state.get('decoder', {}).get('step', 0))
            assert int(state.decoder.step) == decoder_steps + expected_updates * cfg['NUM_MINIBATCHES'] * cfg['UPDATE_EPOCHS']
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state.decoder))
            loss = np.asarray(result['metric']['training/map_reconstruction_loss'])
            assert np.isfinite(loss).all() and np.all(loss > 0), loss
            memory_report.update(decoder_steps=int(state.decoder.step), map_reconstruction_loss=float(loss.mean()))
    if cfg.get('MAP_WARMSTART_CROP_FROM'):
        from symbolic_map_memory import LOCAL_SIZE
        size, old_size = cfg['MAP_VIEW_SIZE'], cfg['MAP_WARMSTART_CROP_FROM']
        pad = (size-old_size)//2
        outer = np.ones((size,size), bool)
        outer[pad:pad+old_size,pad:pad+old_size] = False
        for name in ('Dense_0','Dense_4'):
            kernel = np.asarray(state.params['params']['actor_critic'][name]['kernel'])
            for group in (kernel[LOCAL_SIZE:-size**2],kernel[-size**2:]):
                assert np.any(group.reshape(size,size,-1)[outer] != 0), f'Outer weights were not trained in {name}'
        memory_report['actor_and_critic_outer_weights_updated'] = True
    if cfg.get('MAP_LANDMARKS') and 'landmark_transfer_states' in memory_report:
        weights = np.asarray(state.params['params']['landmark_reader']['residual']['kernel'])
        assert bool(np.any(weights)) == (cfg['MAP_LANDMARKS'] != 'masked')
    if cfg.get('MAP_INTERNAL_CROP'):
        from symbolic_map_memory import GLOBAL_MAP_FEATURES, LOCAL_SIZE
        obs_size = result['runner_state'][2].shape[-1]
        assert obs_size == LOCAL_SIZE + GLOBAL_MAP_FEATURES
        variables = state.params['params']
        assert 'terrain_embedding' in variables and 'global_reader' in variables
        if args.resume_checkpoint:
            for path in [('actor_critic', 'Dense_0', 'kernel'), ('global_reader', 'residual', 'kernel')]:
                old, new = source_state['params']['params'], variables
                for name in path:
                    old, new = old[name], new[name]
                assert not np.array_equal(old, new), f'No fine-tuning update to {path}'
        memory_report.update(single_map=True, observation_size=obs_size, internal_crop_size=cfg['MAP_VIEW_SIZE'])
    elif cfg.get('MAP_V1_ATTENTION_ADAPTER'):
        from symbolic_map_memory import BOARD_OBS_SIZE
        assert result['runner_state'][2].shape[-1] == BOARD_OBS_SIZE
        if not args.resume_checkpoint:
            source_state = ocp.PyTreeCheckpointer().restore(str(Path(cfg['MAP_V1_IMITATION_INIT']).resolve()))
        before = source_state['params']['params']
        after = state.params['params']
        assert set(before) == set(after) == {'actor_critic', 'terrain_embedding', 'board_adapter'}
        frozen = []
        for name in ('actor_critic', 'terrain_embedding'):
            leaves = jax.tree_util.tree_leaves(after[name])
            old = jax.tree_util.tree_leaves(before[name])
            assert len(leaves) == len(old)
            for a, b in zip(leaves, old):
                np.testing.assert_array_equal(a, b)
            frozen.extend(leaves)
        adapter = jax.tree_util.tree_leaves(after['board_adapter'])
        changed = sum(not np.array_equal(a, b) for a, b in zip(adapter, jax.tree_util.tree_leaves(before['board_adapter'])))
        assert changed > 0, 'PPO did not update the board adapter'
        memory_report.update(board_only=True, observation_size=BOARD_OBS_SIZE, frozen_arrays=len(frozen),
                             frozen_parameters=sum(x.size for x in frozen), trainable_parameters=sum(x.size for x in adapter),
                             changed_adapter_arrays=changed)
    elif cfg.get('MAP_BOARD_ONLY'):
        from symbolic_map_memory import BOARD_OBS_SIZE, GLOBAL_MAP_SIZE
        variables = state.params['params']
        assert result['runner_state'][2].shape[-1] == BOARD_OBS_SIZE
        assert variables['world_positions']['kernel'].shape[0] == GLOBAL_MAP_SIZE**2
        memory_report.update(board_only=True, observation_size=BOARD_OBS_SIZE, fixed_cells=GLOBAL_MAP_SIZE**2)
        if cfg.get('MAP_BOARD_ATTENTION'):
            assert np.any(np.asarray(variables['board_attention']['output']['kernel']) != 0)
            memory_report['board_attention'] = True
    elif cfg.get('MAP_SPATIAL_REGIONS'):
        from symbolic_map_memory import GLOBAL_MAP_FEATURES, LOCAL_SIZE
        variables = state.params['params']
        assert result['runner_state'][2].shape[-1] == LOCAL_SIZE + GLOBAL_MAP_FEATURES
        assert 'terrain_embedding' not in variables and 'live_encoder' not in variables
        for name in ('Dense_0', 'Dense_4'):
            assert variables['actor_critic'][name]['kernel'].shape[0] == LOCAL_SIZE + 2048
        regions = variables['global_reader']['regions']['kernel']
        assert regions.shape == (2, 2, 8, 32)
        if cfg.get('MAP_FINE_SHORTCUT'):
            gate = variables['global_reader']['fine_gate']
            old_gate = source_state['params']['params']['global_reader'].get('fine_gate', np.zeros(32)) if args.resume_checkpoint else np.zeros(32)
            assert gate.shape == (32,) and np.any(np.asarray(gate) != old_gate)
            memory_report['fine_gate_max_update'] = float(np.max(np.abs(np.asarray(gate) - old_gate)))
        if args.resume_checkpoint:
            assert not np.array_equal(source_state['params']['params']['global_reader']['regions']['kernel'], regions)
        memory_report.update(single_map=True, observation_size=LOCAL_SIZE+GLOBAL_MAP_FEATURES,
                             region_shape=[8,8,32], decision_features=LOCAL_SIZE+2048)
    elif cfg.get('MAP_SINGLE_READOUT'):
        variables = state.params['params']
        assert variables['actor_critic']['Dense_0']['kernel'].shape[0] == 192
        assert 'terrain_embedding' not in variables
        if cfg.get('MAP_PATCH_MLP'):
            reader = variables['global_reader']
            assert 'patches' not in reader and 'context' not in reader
            assert reader['patch_weights'].shape == (64, 384, 4)
            if args.resume_checkpoint and 'patch_weights' in source_state['params']['params']['global_reader']:
                assert not np.array_equal(source_state['params']['params']['global_reader']['patch_weights'], reader['patch_weights'])
            memory_report['map_patch_mlp'] = True
        if args.resume_checkpoint:
            before = source_state['params']['params']['global_reader']['features']['kernel']
            assert not np.array_equal(before, variables['global_reader']['features']['kernel'])
        memory_report.update(single_map=True, observation_size=9539, decision_features=192)
    elif cfg.get('MAP_GLOBAL_READOUT'):
        residual = state.params['params']['global_reader']['residual']['kernel']
        assert np.any(np.asarray(residual)!=0), 'Global map reader was not trained'
        memory_report['global_map_readout'] = True
    if cfg.get('CTEC_COEF',0.)>0:
        curiosity=state.curiosity
        assert int(curiosity.updates)==original_curiosity_updates+completed_updates
        assert int(curiosity.learner.step)==original_curiosity_optimizer_steps+completed_updates*cfg.get('CTEC_UPDATES',2)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(curiosity))
        memory_report['ctec_optimizer_updates']=int(curiosity.learner.step)
        memory_report['ctec_parameters']=sum(x.size for x in jax.tree_util.tree_leaves(curiosity.learner.params))
    if cfg.get('MOTION_INPUTS',False):
        from symbolic_motion_inputs import POSITION_TIME_INPUT_SIZE
        variables=state.params.get('params',state.params)
        extra_pt = POSITION_TIME_INPUT_SIZE if cfg.get('POSITION_TIME_INPUTS', False) else 0
        from symbolic_episode_memory import FEATURES
        expected_in = 1364 + extra_pt + (FEATURES if cfg.get('SRU_EPISODE_MEMORY') else 0)
        assert variables['Dense_0']['kernel'].shape[0]==expected_in, 'Motion suffix did not reach the network'
        assert recurrent, 'Motion inputs require the recurrent trainer'
        if extra_pt:
            new_rows = np.asarray(variables['Dense_0']['kernel'][1364:, :])
            assert np.any(new_rows != 0), 'Position-time input rows did not receive gradient updates'
        memory_report.update(motion_inputs=True,position_time_inputs=bool(extra_pt),
            observation_size=5462 if cfg.get('WORLD_VIEW') else expected_in,explicit_odometry=True)
        if cfg.get('SRU_FAILURE_REPLAY'):
            replay = result['runner_state'][1]
            np.testing.assert_array_equal(replay.failures[0], state.replay_failures)
            np.testing.assert_array_equal(replay.keys[0], state.replay_keys)
            memory_report['pending_failed_seed_count'] = int((state.replay_failures > 0).sum())
        if cfg.get('WORLD_VIEW'):
            assert np.any(np.asarray(variables['world_reader']['residual']['kernel'])!=0), 'World reader was not trained'
    if args.architecture in ('spatial_map_gru','map_mlp','map_mlp_gru','neural_map_gru'):
        map_state=result['runner_state'][1].env_state
        while not hasattr(map_state,'terrain'):map_state=map_state.env_state
        elapsed=np.asarray(map_state.elapsed)
        assert elapsed.max()>cfg['NUM_STEPS'], 'Map was reset at a PPO rollout boundary'
        assert np.asarray(map_state.terrain).max()<=17
        memory_report.update(dict(explicit_odometry=True,map_view_size=None if cfg.get('MAP_SINGLE_READOUT') else cfg.get('MAP_VIEW_SIZE',33),
            memory_survived_rollout_boundary=True,max_memory_age_steps=int(elapsed.max())))
    if cfg.get('RECOVERY_BANK'):
        slots=np.asarray(result['runner_state'][1].env_state.recovery)
        assert slots.any() and (cfg['RECOVERY_PROBABILITY']==1 or not slots.all())
        memory_report['recovery_lanes_after_two_updates']=int(slots.sum())
    if cfg.get('NEURAL_MAP_CHANNELS',0):
        neural_carry=result['runner_state'][-3]
        assert float(np.abs(np.asarray(neural_carry[1])).max())>0
        assert int(np.asarray(neural_carry[2]).sum())>0
        memory_report.update(neural_map_channels=cfg['NEURAL_MAP_CHANNELS'],neural_map_nonzero=True)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg['ARCHITECTURE']=args.architecture
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    actual_steps=cfg.get('START_TIMESTEPS',0)+completed_updates*cfg['NUM_ENVS']*cfg['NUM_STEPS']
    full_checkpoint = cfg.get('DATA_PARALLEL_DEVICES',1)>1 or cfg.get('SRU_GRU',False)
    if full_checkpoint:
        from ppo_data_parallel import checkpoint_train_state
        checkpoint_value=checkpoint_train_state(state)
    else:checkpoint_value={'params':state.params}
    ocp.PyTreeCheckpointer().save(str(out/str(actual_steps)),checkpoint_value)
    if full_checkpoint:
        restored=ocp.PyTreeCheckpointer().restore(str(out/str(actual_steps)),item=state)
        for expected,actual in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(np.asarray(expected),np.asarray(actual))
    report=dict(architecture=args.architecture,full_size_updates=completed_updates,num_envs=cfg['NUM_ENVS'],steps_per_env=cfg['NUM_STEPS'],
        optimizer_updates=int(state.step),parameters=sum(x.size for x in jax.tree_util.tree_leaves(state.params)),
        parameters_and_optimizer_finite=True,seconds=time.time()-began,
        actual_training_steps=actual_steps,deadline_stopped_early=args.deadline_check,
        resumed_optimizer_steps=original_optimizer_steps,
        learning_rate_horizon_timesteps=cfg.get('LR_HORIZON_TIMESTEPS') or (cfg['TOTAL_TIMESTEPS']-cfg.get('START_TIMESTEPS',0) if cfg.get('RESUME_RESET_OPTIMIZER') else cfg['TOTAL_TIMESTEPS']),
        policy_temperature=cfg.get('POLICY_TEMPERATURE',1.),
        reset_gru_every_step=cfg.get('RESET_GRU_EVERY_STEP',False),
        gru_concat_observation=cfg.get('GRU_CONCAT_OBSERVATION',False),
        data_parallel_devices=cfg.get('DATA_PARALLEL_DEVICES',1),
        full_optimizer_checkpoint_roundtrip=full_checkpoint,
        **memory_report)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

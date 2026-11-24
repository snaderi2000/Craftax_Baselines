"""Configuration invariants that prevent mismatched training/evaluation runs."""
import json,tempfile,unittest
from pathlib import Path
from ppo_experiment_config import Experiment,Rewards,PPO,Budget,Evaluation,Architecture
from run_reward_shaped_architectures import inference_config


class ExperimentConfigTests(unittest.TestCase):
    def test_neural_map_pilot_keeps_continuation_schedule_and_config(self):
        spec=Experiment.load('experiments/ppo_presets/lava_only_neural_map_gru_dp2_1b_seed50.json')
        metadata=spec.metadata(spec.architectures[0],Path.cwd(),'/python')
        cfg=inference_config(metadata)
        self.assertEqual(spec.budget.max_steps,1_000_000_000)
        self.assertEqual(cfg['LR_HORIZON_TIMESTEPS'],10_000_000_000)
        self.assertEqual(cfg['NEURAL_MAP_CHANNELS'],16)
        self.assertTrue(cfg['RECURRENT'])
        self.assertEqual(cfg['SYMBOLIC_ENCODER'],'map_mlp')
        self.assertIn('--neural_map_channels',metadata['command'])
        self.assertIn('--no-neural_map_reset_every_step',metadata['command'])
        for arch,channels in (('neural_map_gru',0),('map_mlp_gru',16)):
            with self.assertRaises(ValueError):
                Experiment('bad',architectures=(Architecture(arch,4,7),),ppo=PPO(neural_map_channels=channels)).validate()
        legacy=Experiment('default').metadata(Architecture('mlp',4,7),Path.cwd(),'/python')
        self.assertNotIn('--neural_map_channels',legacy['command'])

    def test_motion_inputs_propagate_and_reject_unsupported_encoders(self):
        spec=Experiment('motion',architectures=(Architecture('gru',5,1,(5,7)),),
            ppo=PPO(gru_concat_observation=True,motion_inputs=True))
        spec.validate();metadata=spec.metadata(spec.architectures[0],Path.cwd(),'/python')
        self.assertIn('--motion_inputs',metadata['command'])
        self.assertTrue(inference_config(metadata)['MOTION_INPUTS'])
        self.assertTrue(metadata['explicit_odometry'])
        self.assertEqual(metadata['local_num_envs'],512)
        default=Experiment('default').metadata(Architecture('gru',5,1),Path.cwd(),'/python')
        self.assertFalse(inference_config(default)['MOTION_INPUTS'])
        self.assertIn('--no-motion_inputs',default['command'])
        for name in ('mlp','spatial_gru','spatial_map_gru','map_mlp'):
            with self.assertRaisesRegex(ValueError,'flat symbolic GRU'):
                Experiment('bad',architectures=(Architecture(name,5,1),),ppo=PPO(motion_inputs=True)).validate()
        with self.assertRaisesRegex(ValueError,'boolean'):PPO(motion_inputs='yes').validate()

    def test_gru_concatenation_propagates_to_training_and_inference(self):
        spec=Experiment('concat',architectures=(Architecture('gru',5,7),),ppo=PPO(gru_concat_observation=True))
        spec.validate();metadata=spec.metadata(spec.architectures[0],Path.cwd(),'/python')
        self.assertIn('--gru_concat_observation',metadata['command'])
        self.assertTrue(inference_config(metadata)['GRU_CONCAT_OBSERVATION'])
        self.assertFalse(inference_config(Experiment('default').metadata(Architecture('gru',5,7),Path.cwd(),'/python'))['GRU_CONCAT_OBSERVATION'])
        with self.assertRaises(ValueError):Experiment('bad',architectures=(Architecture('mlp',5,7),),ppo=PPO(gru_concat_observation=True)).validate()

    def test_memory_control_and_deferred_evaluation_propagate(self):
        spec=Experiment('memory_control',architectures=(Architecture('gru',5,5),),
            ppo=PPO(reset_gru_every_step=True),evaluation=Evaluation(defer_until_training_complete=True))
        spec.validate();metadata=spec.metadata(spec.architectures[0],Path.cwd(),'/python')
        self.assertIn('--reset_gru_every_step',metadata['command'])
        self.assertTrue(inference_config(metadata)['RESET_GRU_EVERY_STEP'])
        self.assertTrue(metadata['evaluation']['defer_until_training_complete'])
        with self.assertRaises(ValueError):Experiment('bad',architectures=(Architecture('gru',5,5),)).validate()
        with self.assertRaises(ValueError):Experiment('bad',architectures=(Architecture('mlp',5,7),),ppo=PPO(reset_gru_every_step=True)).validate()

    def test_existing_presets_have_exact_reward_semantics_in_train_and_eval(self):
        cases={'lava_only.json':(0,2,1,1),'diamond_plant_bonus.json':(.5,0,5,5),'default_rewards.json':(0,0,1,1)}
        for filename,expected in cases.items():
            spec=Experiment.load(Path('experiments/ppo_presets')/filename)
            for job in spec.architectures:
                metadata=spec.metadata(job,Path.cwd(),'/python')
                command=metadata['command'];cfg=inference_config(metadata)
                train=tuple(float(command[command.index('--'+key)+1]) for key in ('death_penalty','lava_penalty','diamond_reward','plant_reward'))
                evaluation=tuple(cfg[key] for key in ('DEATH_PENALTY','LAVA_PENALTY','DIAMOND_REWARD','PLANT_REWARD'))
                self.assertEqual(train,expected);self.assertEqual(evaluation,expected)

    def test_changed_seed_ppo_and_evaluation_settings_propagate(self):
        spec=Experiment('small',seed=77,architectures=(Architecture('gru',2,6),),
            ppo=PPO(num_envs=128,num_steps=32,num_minibatches=4,lr=.001,layer_size=64,optimistic_resets=False),
            budget=Budget(max_steps=1000000),evaluation=Evaluation(common_checkpoints={'early':12345},common_episodes=64,final_episodes=32,batch=16,seed=999))
        spec.validate();metadata=spec.metadata(spec.architectures[0],Path.cwd(),'/python');cfg=inference_config(metadata)
        self.assertEqual(cfg['SEED'],77);self.assertEqual(cfg['NUM_ENVS'],128);self.assertEqual(cfg['NUM_STEPS'],32)
        self.assertEqual(cfg['LR'],.001);self.assertEqual(cfg['LAYER_SIZE'],64);self.assertFalse(cfg['USE_OPTIMISTIC_RESETS'])
        self.assertIn('--no-use_optimistic_resets',metadata['command'])
        self.assertEqual(metadata['evaluation']['seed'],999)
        self.assertEqual(metadata['evaluation']['common_checkpoints'],[('early',16384)])
        self.assertIn(12345,metadata['requested_checkpoints'])

    def test_rejects_ambiguous_rewards_gpu_collisions_and_invalid_cap(self):
        with self.assertRaises(ValueError):Rewards(death_penalty=.5,lava_penalty=2).validate()
        with self.assertRaises(ValueError):Experiment('bad',architectures=(Architecture('gru',1,1),)).validate()
        with self.assertRaises(ValueError):Budget(max_hours=1,evaluation_reserve_minutes=60).validate(65536)
        with self.assertRaises(ValueError):Experiment('bad',ppo=PPO(num_envs=7,num_minibatches=2)).validate()

    def test_typos_fail_and_normalized_config_round_trips(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json';spec=Experiment('roundtrip')
            path.write_text(json.dumps(spec.to_dict()))
            self.assertEqual(Experiment.load(path).fingerprint(),spec.fingerprint())
            path.write_text(json.dumps({'name':'typo','rewards':{'lava_penality':2}}))
            with self.assertRaisesRegex(ValueError,'Unknown Rewards'):Experiment.load(path)

    def test_launcher_requires_matching_smoke_validation_before_gpu_access(self):
        from run_ppo_study import prepare,launch
        with tempfile.TemporaryDirectory() as folder:
            study=Path(folder)/'study';spec=Experiment('unlaunched')
            prepare(spec,study)
            with self.assertRaisesRegex(RuntimeError,'Run --smoke'):launch(spec,study)
            self.assertFalse((study/'launch_manifest.json').exists())

    def test_prepare_cannot_overwrite_a_launched_study(self):
        from run_ppo_study import prepare
        with tempfile.TemporaryDirectory() as folder:
            study=Path(folder);manifest=study/'launch_manifest.json'
            manifest.write_text('{"existing": true}')
            with self.assertRaises(FileExistsError):prepare(Experiment('existing'),study)
            self.assertEqual(json.loads(manifest.read_text()),{'existing':True})

    def test_continuation_keeps_original_lr_horizon_and_explicit_source(self):
        spec=Experiment.load('experiments/ppo_presets/lava_only_spatial_gru_continue500m.json')
        metadata=spec.metadata(spec.architectures[0],Path.cwd(),'/python');command=metadata['command']
        cfg=inference_config(metadata)
        self.assertFalse(metadata['fresh_training'])
        self.assertEqual(metadata['start_timesteps'],581959680)
        self.assertEqual(spec.budget.max_steps-metadata['start_timesteps'],500000000)
        self.assertEqual(command[command.index('--lr_horizon_timesteps')+1],'10000000000')
        self.assertEqual(command[command.index('--total_timesteps')+1],'1081959680')
        self.assertTrue(command[command.index('--resume_from')+1].endswith('/581959680'))
        self.assertEqual(cfg['LR_HORIZON_TIMESTEPS'],10000000000)
        self.assertEqual(spec.budget.max_steps//spec.batch_steps*spec.batch_steps,1081933824)

    def test_map_architecture_propagates_odometry_and_view_size(self):
        spec=Experiment('map',architectures=(Architecture('spatial_map_gru',1,5),),
                        ppo=PPO(map_view_size=49))
        spec.validate()
        metadata=spec.metadata(spec.architectures[0],Path.cwd(),'/python')
        command=metadata['command'];cfg=inference_config(metadata)
        self.assertTrue(command[2].endswith('ppo_rnn_og.py'))
        self.assertEqual(command[command.index('--symbolic_encoder')+1],'spatial_map')
        self.assertEqual(command[command.index('--map_view_size')+1],'49')
        self.assertTrue(metadata['explicit_odometry']);self.assertTrue(cfg['RECURRENT'])
        self.assertEqual(cfg['SYMBOLIC_ENCODER'],'spatial_map');self.assertEqual(cfg['MAP_VIEW_SIZE'],49)
        self.assertEqual(len(Experiment('unchanged').architectures),4)
        with self.assertRaises(ValueError):PPO(map_view_size=32).validate()

    def test_fixed_world_finetuning_and_optimizer_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            bank=Path(folder)/'seeds.json';bank.write_text('{"world_seeds": [1, 2]}')
            source=9999941632
            spec=Experiment('memorize',resume_from=f'/policies/{source}',resume_reset_optimizer=True,
                architectures=(Architecture('mlp',3,7),),
                ppo=PPO(lr=.00002,reset_seed_bank=str(bank),optimistic_resets=False),
                budget=Budget(max_steps=source+50000000),
                evaluation=Evaluation(common_checkpoints={'1m':source+1000000}))
            spec.validate();meta=spec.metadata(spec.architectures[0],Path.cwd(),'/python')
            cfg=inference_config(meta);before=spec.fingerprint()
            self.assertIn('--resume_reset_optimizer',meta['command'])
            self.assertTrue(cfg['RESUME_RESET_OPTIMIZER']);self.assertEqual(cfg['RESET_SEED_BANK'],str(bank))
            self.assertEqual(meta['learning_rate_horizon_steps'],50000000)
            self.assertIsNone(cfg['LR_HORIZON_TIMESTEPS'])
            bank.write_text('{"world_seeds": [3, 4]}')
            self.assertNotEqual(before,spec.fingerprint())

    def test_training_temperature_and_paired_evaluation_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            bank=Path(folder)/'worlds.json';bank.write_text('{"world_seeds": [7, 8]}')
            spec=Experiment('temperature',architectures=(Architecture('mlp',2,7),),
                ppo=PPO(policy_temperature=.25,ent_coef=.001),
                evaluation=Evaluation(common_checkpoints={},common_episodes=6,final_episodes=6,batch=2,
                    temperatures=(1.,.25),world_seeds_file=str(bank),repeats=3))
            spec.validate();meta=spec.metadata(spec.architectures[0],Path.cwd(),'/python');cfg=inference_config(meta)
            self.assertEqual(cfg['POLICY_TEMPERATURE'],.25);self.assertEqual(cfg['ENT_COEF'],.001)
            self.assertEqual(meta['command'][meta['command'].index('--policy_temperature')+1],'0.25')
            self.assertEqual(meta['evaluation']['temperatures'],(1.,.25))
            self.assertEqual(meta['evaluation']['world_seeds_file'],str(bank))
            old=spec.fingerprint();bank.write_text('{"world_seeds": [9, 10]}')
            self.assertNotEqual(old,spec.fingerprint())
        with self.assertRaises(ValueError):PPO(policy_temperature=0).validate()

    def test_feedforward_map_configuration(self):
        spec=Experiment('map_mlp',architectures=(Architecture('map_mlp',2,7),),
            ppo=PPO(map_view_size=33,map_embedding_size=4))
        spec.validate();meta=spec.metadata(spec.architectures[0],Path.cwd(),'/python');cfg=inference_config(meta)
        self.assertFalse(cfg['RECURRENT']);self.assertTrue(meta['explicit_odometry'])
        self.assertEqual(cfg['SYMBOLIC_ENCODER'],'map_mlp');self.assertEqual(cfg['MAP_EMBEDDING_SIZE'],4)
        command=meta['command']
        self.assertTrue(command[2].endswith('ppo.py'))
        self.assertEqual(command[command.index('--symbolic_encoder')+1],'map_mlp')
        self.assertEqual(command[command.index('--map_view_size')+1],'33')
        self.assertEqual(command[command.index('--map_embedding_size')+1],'4')
        baseline=Experiment('baseline').metadata(Architecture('mlp',0,4),Path.cwd(),'/python')['command']
        self.assertNotIn('--map_embedding_size',baseline)

    def test_data_parallel_gpu_group_preserves_global_training_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            path.write_text(json.dumps(dict(name='dp2',architectures=[dict(
                name='spatial_map_gru',train_gpus=[5,6],evaluation_gpu=3)])))
            spec=Experiment.load(path);job=spec.architectures[0]
            self.assertEqual(job.train_gpu,5)
            self.assertEqual(job.training_devices,(5,6))
            meta=spec.metadata(job,Path.cwd(),'/python');cfg=inference_config(meta)
            self.assertEqual(meta['train_gpus'],[5,6]);self.assertEqual(meta['local_num_envs'],512)
            self.assertEqual(cfg['DATA_PARALLEL_DEVICES'],2)
            self.assertEqual((cfg['NUM_ENVS'],cfg['NUM_STEPS'],cfg['UPDATE_EPOCHS'],cfg['NUM_MINIBATCHES']),
                             (1024,64,4,8))
            command=meta['command']
            self.assertEqual(command[command.index('--data_parallel_devices')+1],'2')
            from run_reward_shaped_architectures import environment
            self.assertEqual(environment(job.training_devices)['CUDA_VISIBLE_DEVICES'],'5,6')
            self.assertEqual(environment(5)['CUDA_VISIBLE_DEVICES'],'5')

    def test_data_parallel_rejects_conflicts_unsupported_models_and_bad_splits(self):
        invalid=[Architecture('spatial_map_gru',5,6,(5,6)),
                 Architecture('spatial_map_gru',5,3,(5,5)),
                 Architecture('spatial_map_gru',5,3,(6,7)),
                 Architecture('mlp',5,3,(5,6)),
                 Architecture('gru',0,7,(0,1,2))]
        for job in invalid:
            with self.subTest(job=job),self.assertRaises(ValueError):
                Experiment('invalid',architectures=(job,)).validate()
        with self.assertRaises(ValueError):
            Experiment('bad_split',architectures=(Architecture('gru',5,3,(5,6)),),
                ppo=PPO(num_envs=24,num_minibatches=8,optimistic_resets=False)).validate()
        with self.assertRaises(ValueError):
            Experiment('cross_job_collision',architectures=(Architecture('gru',5,3,(5,6)),
                Architecture('spatial_gru',6,7))).validate()

    def test_four_device_configuration_and_legacy_single_device(self):
        job=Architecture('spatial_map_gru',0,7,(0,1,2,3))
        spec=Experiment('dp4',architectures=(job,));spec.validate()
        meta=spec.metadata(job,Path.cwd(),'/python')
        self.assertEqual(meta['data_parallel_devices'],4);self.assertEqual(meta['local_num_envs'],256)
        legacy=Experiment('legacy').metadata(Architecture('gru',1,5),Path.cwd(),'/python')
        self.assertEqual(legacy['train_gpus'],[1]);self.assertEqual(legacy['data_parallel_devices'],1)
        self.assertNotIn('--data_parallel_devices',legacy['command'])

    def test_four_gpu_feedforward_map_preserves_batch_and_resume_horizon(self):
        job=Architecture('map_mlp',0,7,(0,3,4,6))
        spec=Experiment('map_dp4',architectures=(job,),
            ppo=PPO(lr_horizon_timesteps=10000000000),budget=Budget(max_steps=5000003584,max_hours=48))
        spec.validate();meta=spec.metadata(job,Path.cwd(),'/python');cfg=inference_config(meta)
        self.assertFalse(cfg['RECURRENT']);self.assertEqual(cfg['DATA_PARALLEL_DEVICES'],4)
        self.assertEqual(meta['local_num_envs'],256);self.assertEqual(cfg['NUM_ENVS']*cfg['NUM_STEPS'],65536)
        self.assertEqual(cfg['LR_HORIZON_TIMESTEPS'],10000000000)
        self.assertTrue(meta['fresh_training']);self.assertFalse(meta['resume_reset_optimizer'])
        self.assertTrue(meta['command'][2].endswith('ppo.py'))
        self.assertEqual(meta['command'][meta['command'].index('--data_parallel_devices')+1],'4')
        # Feedforward batches can divide across GPUs without whole recurrent sequences.
        Experiment('flat_split',architectures=(job,),
            ppo=PPO(num_envs=12,num_steps=4,num_minibatches=6,optimistic_resets=False)).validate()
        with self.assertRaisesRegex(ValueError,'environments'):
            Experiment('bad_env_split',architectures=(job,),
                ppo=PPO(num_envs=10,num_steps=4,num_minibatches=2,optimistic_resets=False)).validate()


if __name__=='__main__':unittest.main()

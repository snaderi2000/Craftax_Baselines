"""Matched 10B PPO architecture screen: death -.5, diamond +5, plant +5."""
import argparse,csv,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
STUDY=ROOT/'concept_mapping/runs/reward_shaped_architectures_10b_seed50'
ARCHITECTURES=('mlp','gru','spatial','spatial_gru')
BUDGET=10000000000
BATCH=65536
FINAL=BUDGET//BATCH*BATCH
MILESTONES=(65536,100000,150000,220000,330000,500000,750000,1000000,1500000,
    2200000,3300000,5000000,7500000,10000000,15000000,22000000,33000000,50000000,
    75000000,100000000,150000000,220000000,330000000,500000000,750000000,1000000000,
    1500000000,2200000000,3300000000,5000000000,7500000000)
EVALUATIONS=(('100m',(100000000+BATCH-1)//BATCH*BATCH),
    ('1b',(1000000000+BATCH-1)//BATCH*BATCH),('10b',FINAL))
SOURCES=('ppo.py','ppo_rnn_og.py','wrappers.py','models/actor_critic.py',
    'models/conditioned_symbolic.py','logz/batch_logging.py','evaluate_reward_shaped_ppo.py',
    'run_reward_shaped_architectures.py','smoke_reward_shaped_ppo.py','test_reward_shaped_spatial.py',
    'symbolic_motion_inputs.py','symbolic_map_memory.py')


def write_json(path,data):
    path=Path(path);temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def command_for(architecture):
    recurrent=architecture in ('gru','spatial_gru')
    return [PYTHON,'-u',str(ROOT/('ppo_rnn_og.py' if recurrent else 'ppo.py')),
        '--env_name','Craftax-Classic-Symbolic-v1','--num_envs','1024',
        '--total_timesteps',str(BUDGET),'--num_steps','64','--update_epochs','4',
        '--num_minibatches','8','--gamma','.99','--gae_lambda','.8','--clip_eps','.2',
        '--vf_coef','.5','--ent_coef','.01','--lr','.0002','--max_grad_norm','1.',
        '--anneal_lr','--seed','50','--layer_size','512','--num_repeats','1',
        '--symbolic_encoder','spatial' if architecture.startswith('spatial') else 'mlp',
        '--death_penalty','.5','--diamond_reward','5','--plant_reward','5',
        '--use_optimistic_resets','--optimistic_reset_ratio','16','--save_policy',
        '--save_policy_milestones',','.join(map(str,MILESTONES)),'--use_wandb',
        '--wandb_project','Craftax_Baselines','--wandb_entity','synaderi-uc-davis',
        '--run_name',f'classic-d0p5-diamond5-plant5-{architecture}-10b-seed50']


def inference_config(plan):
    return dict(ENV_NAME=plan['environment'],LAYER_SIZE=plan.get('layer_size',512),
        SYMBOLIC_ENCODER=plan['symbolic_encoder'],RECURRENT=plan['recurrent'],
        MAP_VIEW_SIZE=plan.get('map_view_size',33),
        MAP_WARMSTART_CROP_FROM=plan.get('map_warmstart_crop_from'),
        MAP_GLOBAL_READOUT=plan.get('map_global_readout',False),
        MAP_LANDMARKS=plan.get('map_landmarks'),
        MAP_SINGLE_READOUT=plan.get('map_single_readout',False),
        MAP_BOARD_ONLY=plan.get('map_board_only',False),
        MAP_BOARD_ATTENTION=plan.get('map_board_attention',False),
        MAP_V1_ATTENTION_ADAPTER=plan.get('map_v1_attention_adapter',False),
        MAP_V1_IMITATION_INIT=plan.get('map_v1_imitation_init'),
        MAP_INTERNAL_CROP=plan.get('map_internal_crop',False),
        MAP_PATCH_MLP=plan.get('map_patch_mlp',False),
        MAP_SPATIAL_REGIONS=plan.get('map_spatial_regions',False),
        MAP_FINE_SHORTCUT=plan.get('map_fine_shortcut',False),
        MAP_FULL_TERRAIN=plan.get('map_full_terrain',False),
        MAP_EMBEDDING_SIZE=plan.get('map_embedding_size',4),
        **{key.upper():value for key,value in plan.items() if key.startswith('ctec_')},
        **{key.upper():value for key,value in plan.items() if key.startswith('sru_')},
        POLICY_TEMPERATURE=plan.get('policy_temperature',1.),
        RESET_GRU_EVERY_STEP=plan.get('reset_gru_every_step',False),
        GRU_CONCAT_OBSERVATION=plan.get('gru_concat_observation',False),
        WORLD_VIEW=plan.get('world_view'),LOCAL_POLICY_INIT=plan.get('local_policy_init'),
        MOTION_INPUTS=plan.get('motion_inputs',False),
        MAP_SURVIVAL_FEATURES=plan.get('map_survival_features','none'),
        NEURAL_MAP_CHANNELS=plan.get('neural_map_channels',0),
        NEURAL_MAP_RESET_EVERY_STEP=plan.get('neural_map_reset_every_step',False),
        MEMORY_TYPE='gru' if plan['recurrent'] else None,SEED=plan['seed'],
        DATA_PARALLEL_DEVICES=plan.get('data_parallel_devices',1),
        RECOVERY_BANK=plan.get('recovery_bank'),RECOVERY_PROBABILITY=plan.get('recovery_probability',0.),
        RESET_SEED_BANK=plan.get('reset_seed_bank'),RESUME_RESET_OPTIMIZER=plan.get('resume_reset_optimizer',False),
        TOTAL_TIMESTEPS=plan['requested_training_steps'],DEATH_PENALTY=abs(plan.get('additional_death_cost',0.)),
        LR_HORIZON_TIMESTEPS=(plan.get('lr_horizon_timesteps') if plan.get('resume_reset_optimizer') else plan.get('learning_rate_horizon_steps',plan['requested_training_steps'])),
        LAVA_PENALTY=abs(plan.get('additional_lava_cost',0.)),
        ALL22_TERMINAL_BONUS=plan.get('all22_terminal_bonus'),
        MAP_COVERAGE_BONUS=plan.get('map_coverage_bonus') or 0.,
        DIAMOND_REWARD=plan.get('first_diamond_total_reward',1.),PLANT_REWARD=plan.get('first_eat_plant_total_reward',1.),
        LR=plan.get('lr',.0002),MAX_GRAD_NORM=plan.get('max_grad_norm',1.),
        GAMMA=plan.get('gamma',.99),GAE_LAMBDA=plan.get('gae_lambda',.8),
        NUM_ENVS=plan.get('num_envs',1024),NUM_STEPS=plan.get('num_steps',64),
        NUM_MINIBATCHES=plan.get('num_minibatches',8),UPDATE_EPOCHS=plan.get('update_epochs',4),
        CLIP_EPS=plan.get('clip_eps',.2),VF_COEF=plan.get('vf_coef',.5),ENT_COEF=plan.get('ent_coef',.01),
        ANNEAL_LR=plan.get('anneal_lr',True),
        USE_OPTIMISTIC_RESETS=plan.get('optimistic_resets',plan.get('use_optimistic_resets',True)),
        OPTIMISTIC_RESET_RATIO=plan.get('optimistic_reset_ratio',16))


def prepare(study):
    study.mkdir(parents=True,exist_ok=True)
    for arch in ARCHITECTURES:
        out=study/arch;out.mkdir(exist_ok=True)
        smoke_path=study/'smoke'/arch/'summary.json'
        smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        plan=dict(architecture=arch,seed=50,train_gpu=ARCHITECTURES.index(arch),
            evaluation_gpu=ARCHITECTURES.index(arch)+4,command=command_for(arch),
            requested_training_steps=BUDGET,actual_training_steps=FINAL,fresh_training=True,
            additional_death_cost=-.5,first_diamond_total_reward=5,first_eat_plant_total_reward=5,
            other_achievement_rewards=1,health_reward_scale=.1,time_limit_only_death_cost=0,
            environment='Craftax-Classic-Symbolic-v1',max_episode_steps=10000,
            symbolic_encoder='spatial' if arch.startswith('spatial') else 'mlp',
            recurrent=arch in ('gru','spatial_gru'),trainable_parameters=smoke.get('parameters'),
            num_envs=1024,num_steps=64,update_epochs=4,num_minibatches=8,gamma=.99,
            gae_lambda=.8,clip_eps=.2,vf_coef=.5,ent_coef=.01,lr=.0002,
            anneal_lr=True,max_grad_norm=1.,training_reset='optimistic ratio 16',
            requested_checkpoints=list(MILESTONES),actual_checkpoints=sorted({(s+BATCH-1)//BATCH*BATCH for s in MILESTONES}|{FINAL}),
            evaluation=dict(checkpoints=list(EVALUATIONS),episodes_per_checkpoint=1024,
                seed=18050,maps='original maps',resets='standard independent',
                metrics='All 22 achievement rates, all-22 completion, original/shaped return, lava, episode length'),
            source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
        write_json(out/'experiment_metadata.json',plan)
        write_json(out/'inference_config.json',inference_config(plan))
    (study/'README.md').write_text('\n'.join([
        '# PPO reward and architecture comparison','',
        'Four fresh seed-50 runs, each trained for 10B requested interactions (9,999,941,632 completed).',
        'Death adds −0.5, including lava; time-limit-only endings have no extra cost. The FIRST diamond and FIRST eat-plant achievements each give +5 total (+4 added to the usual +1). Other rewards and mechanics are unchanged.','',
        '| Architecture | Training GPU | Evaluation GPU | Parameters |',
        '| --- | ---: | ---: | ---: |',
        *[f'| {arch} | {i} | {i+4} | {json.loads((study/arch/"experiment_metadata.json").read_text())["trainable_parameters"]:,} |' for i,arch in enumerate(ARCHITECTURES)],
        '', 'The spatial model is the existing internal-state-conditioned residual CNN over the 7×9×21 symbolic map. CNN + GRU adds the same spatial encoder before the 512-unit GRU. MLP/GRU use the original flat observation encoders. Model capacity and parameter sharing differ; this is an architecture screen, not a capacity-controlled study.','',
        'Shared PPO settings: 1,024 environments, 64-step rollouts, four epochs, eight minibatches, γ=0.99, λ=0.8, clipping 0.2, value coefficient 0.5, entropy coefficient 0.01, Adam learning rate 2e−4 linearly annealed, global gradient clip 1.0. All training runs use optimistic resets with ratio 16, matching the historical runs. Recurrent minibatches retain complete 64-step sequences and reset hidden state at episode boundaries.','',
        'Weights and optimizer states are saved at logarithmic milestones. W&B records training curves for all 22 achievements. Separate frozen-policy evaluations at 100M, 1B, and 10B use 1,024 common-seed episodes on original maps with independent standard resets and categorical action sampling. Evaluation interactions are counted separately.','',
        'The goal is high success on every achievement and completion of all 22 within an episode. The reported metrics include each event, the minimum individual achievement rate, and joint all-22 completion. Observing 100% on a finite sample cannot guarantee success on every possible map. This initial screen uses one training seed per architecture.','',
        'Smoke checks cover real diamond/plant rewards, lava and time-limit endings, recurrent reset and sequence consistency, two full-size training updates per architecture with finite parameters/optimizer state, and checkpoint reload plus frozen evaluation.','',
        'Plans and commands: `<architecture>/experiment_metadata.json`. Status: `<architecture>/progress.json`. Training logs: `<architecture>/training.log`. Frozen evaluations: `<architecture>/evaluation_{100m,1b,10b}/`. Aggregated results appear in `results.md` and `per_checkpoint_results.csv`.','',
        'Launch with `python run_reward_shaped_architectures.py --launch`. The launcher refuses to overwrite an existing launch and checks GPU availability. Source snapshots are retained in `source_snapshot/`.','']))


def summarize(study):
    with (study/'.summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);rows=[]
        for arch in (*ARCHITECTURES,'spatial_map_gru','map_mlp','map_mlp_gru','neural_map_gru'):
            for path in sorted((study/arch).glob('evaluation_*/summary.json')):
                label=path.parent.name.removeprefix('evaluation_')
                d=json.loads(path.read_text())
                rows.append(dict(architecture=arch,budget=label,steps=d['training_steps'],
                    episodes=d['episodes'],mean_achievements=d['mean_achievements'],
                    all_22_pct=d['all_22_success_pct'],mean_original_return=d['mean_original_return'],
                    lava_pct=d['lava_pct'],nonlava_death_pct=d.get('nonlava_death_pct'),
                    timeout_pct=d.get('timeout_pct'),**d['achievement_success_pct']))
        if rows:
            with (study/'per_checkpoint_results.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        lines=['# Frozen-policy evaluation results','',
            'One training seed per architecture; episode counts are retained in the CSV. Frozen evaluations use common seeds and independent normal-map episodes. Original return is reported separately.','',
            '| Budget | Architecture | Mean achievements | Diamond (%) | Eat plant (%) | All 22 (%) | Original return |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:lines.append(f'| {r["budget"]} | {r["architecture"]} | {r["mean_achievements"]:.3f} | {r["collect_diamond"]:.2f} | {r["eat_plant"]:.2f} | {r["all_22_pct"]:.2f} | {r["mean_original_return"]:.3f} |')
        if not rows:lines.extend(['','No scheduled frozen evaluations have completed yet.'])
        lines.extend(['','All 22 individual achievement rates are retained in the CSV and each evaluation summary.'])
        (study/'results.md').write_text('\n'.join(lines)+'\n')


def environment(gpu):
    visible=str(gpu) if isinstance(gpu,int) else ','.join(map(str,gpu))
    return dict(os.environ,CUDA_VISIBLE_DEVICES=visible,XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',WANDB_MODE=os.environ.get('WANDB_MODE','online'),WANDB_INIT_TIMEOUT='300')


def checkpoint_ready(path):
    return (path/'_METADATA').exists() or (path/'default'/'_METADATA').exists()


def worker(args):
    study=args.study;arch=args.architecture;out=study/arch
    plan=json.loads((out/'experiment_metadata.json').read_text())
    if (out/'training.log').exists():raise FileExistsError(f'Existing training attempt: {out}')
    progress=dict(state='starting',architecture=arch,seed=50,train_gpu=plan['train_gpu'],evaluation_gpu=plan['evaluation_gpu'])
    def status(**kw):
        progress.update(kw,time=time.time());write_json(out/'progress.json',progress)
    train_log=(out/'training.log').open('w')
    train=subprocess.Popen(plan['command'],cwd=ROOT,env=environment(plan['train_gpu']),
        stdout=train_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    status(state='training',pid=train.pid)
    run=None;eval_proc=None;eval_log=None;completed=set();failed=[];current=None
    while True:
        if run is None:
            log=re.sub(r'\x1b\[[0-9;]*m','',(out/'training.log').read_text(errors='replace'))
            matches=re.findall(r'Run data is saved locally in (\S+)',log)
            if matches:
                run=Path(matches[-1]);run=run if run.is_absolute() else ROOT/run
                write_json(out/'run_manifest.json',dict(**plan,local_run=str(run),
                    wandb_url='https://wandb.ai/synaderi-uc-davis/Craftax_Baselines/runs/'+run.name.rsplit('-',1)[-1]))
                write_json(run/'files/study_config.json',inference_config(plan))
                status(local_run=str(run))
        if eval_proc is not None and eval_proc.poll() is not None:
            code=eval_proc.returncode;eval_log.close()
            if code:failed.append(dict(budget=current,exit_code=code))
            completed.add(current);status(evaluations_completed=sorted(completed),evaluation_failures=failed)
            summarize(study);eval_proc=None
        if run is not None and eval_proc is None:
            for label,step in EVALUATIONS:
                checkpoint=run/'files/policies'/str(step)
                if label in completed or not checkpoint_ready(checkpoint):continue
                current=label;eval_log=(out/f'evaluation_{label}.log').open('w')
                command=[PYTHON,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),
                    '--run-path',str(run),'--checkpoint',str(checkpoint),'--out',str(out/f'evaluation_{label}'),
                    '--episodes','1024','--batch','32','--seed','18050']
                eval_proc=subprocess.Popen(command,cwd=ROOT,env=environment(plan['evaluation_gpu']),
                    stdout=eval_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                status(evaluating=label,evaluation_pid=eval_proc.pid);break
        code=train.poll()
        if code is not None:
            if code:
                status(state='training_failed',exit_code=code);train_log.close()
                if eval_proc is not None:eval_proc.wait();eval_log.close()
                raise RuntimeError(f'{arch} training failed; see training.log')
            if run is None:raise RuntimeError('Training completed without a W&B run path')
            if not checkpoint_ready(run/'files/policies'/str(FINAL)):
                raise RuntimeError('Final checkpoint is missing')
            status(state='training_complete_evaluations_pending')
            if len(completed)==len(EVALUATIONS) and eval_proc is None:break
        time.sleep(20)
    train_log.close();status(state='complete' if not failed else 'complete_with_evaluation_failures',final_checkpoint=FINAL)
    summarize(study)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=STUDY)
    p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--architecture',choices=ARCHITECTURES);p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args();args.study=args.study.resolve()
    if args.worker:worker(args);return
    if args.summarize_only:summarize(args.study);return
    if (args.study/'launch_manifest.json').exists():raise FileExistsError('Study already launched')
    prepare(args.study);summarize(args.study)
    if not args.launch:print(f'Prepared {args.study}');return
    for arch in ARCHITECTURES:
        smoke=json.loads((args.study/'smoke'/arch/'summary.json').read_text())
        ev=json.loads((args.study/'smoke'/arch/'evaluation/summary.json').read_text())
        if not smoke['parameters_and_optimizer_finite'] or ev['episodes']!=8:raise RuntimeError(f'{arch}: smoke gate failed')
    for gpu in range(8):
        free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        if free<14000:raise RuntimeError(f'GPU {gpu} is busy')
    if shutil.disk_usage(args.study).free<20*1024**3:raise RuntimeError('Less than 20 GiB disk headroom')
    snapshot=args.study/'source_snapshot'
    for name in SOURCES:
        destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    jobs=[]
    for arch in ARCHITECTURES:
        command=[PYTHON,'-u',str(Path(__file__).resolve()),'--worker','--architecture',arch,'--study',str(args.study)]
        with (args.study/arch/'worker.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        jobs.append(dict(architecture=arch,pid=process.pid,train_gpu=ARCHITECTURES.index(arch),
            evaluation_gpu=ARCHITECTURES.index(arch)+4,command=command))
    write_json(args.study/'launch_manifest.json',dict(time=time.time(),jobs=jobs,
        total_requested_training_steps=4*BUDGET,total_actual_training_steps=4*FINAL))
    print(json.dumps(jobs,indent=2),flush=True)


if __name__=='__main__':main()

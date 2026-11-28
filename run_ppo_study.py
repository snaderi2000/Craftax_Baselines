"""Configure once, then prepare, smoke-test, or launch a Classic PPO study."""
import argparse,hashlib,json,secrets,shutil,subprocess,time
from pathlib import Path
from ppo_experiment_config import Experiment
from run_reward_shaped_architectures import ROOT,PYTHON,write_json,inference_config,environment,summarize,checkpoint_ready

SOURCES=('symbolic_episode_memory.py','fused_sru.py','sparse_resets.py','ppo_experiment_config.py','run_ppo_study.py','ppo.py','ppo_rnn_og.py','wrappers.py',
    'models/actor_critic.py','models/conditioned_symbolic.py','training_time_limit.py',
    'cap_reward_shaped_runs.py','run_reward_shaped_architectures.py','evaluate_reward_shaped_ppo.py',
    'smoke_reward_shaped_ppo.py','test_reward_shaped_spatial.py','test_rnn_death_penalty.py',
    'symbolic_landmarks.py','symbolic_map_memory.py','test_symbolic_map_memory.py','fixed_world_seeds.py','test_fixed_world_seeds.py',
    'policy_temperature.py','test_training_temperature.py','test_evaluation_scheduling.py',
    'models/actor_critic_map_mlp.py','models/actor_critic_board_transfer.py','test_map_mlp.py','experiments/distill_single_map.py',
    'ppo_data_parallel.py','test_ppo_data_parallel.py','test_ppo_feedforward_data_parallel.py',
    'test_ppo_continuation_queue.py','test_gru_memory_control.py','validate_gru_delayed_cue.py',
    'symbolic_world_view.py','validate_world_view_transfer.py','symbolic_motion_inputs.py','test_symbolic_motion_inputs.py','ctec.py','test_ctec_map_gru.py',
    'models/sru_gru.py','test_sru_gru.py','models/sru_map_reconstruction.py','test_sru_map_reconstruction.py',
    'symbolic_survival_map.py','test_symbolic_survival_map.py','recovery_resets.py','collect_recovery_bank.py',
    'models/actor_critic_neural_map.py','symbolic_neural_map.py','test_neural_map.py',
    'failure_seed_replay.py','test_failure_seed_replay.py')


def source_hashes():return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def validation_key(spec):
    return hashlib.sha256(json.dumps(dict(config=spec.fingerprint(),sources=source_hashes()),sort_keys=True).encode()).hexdigest()


def prepare(spec,study):
    if spec.resume_after:raise ValueError('Use --queue for a deferred resume_after study')
    if (study/'launch_manifest.json').exists():raise FileExistsError(f'This study was already launched: {study}')
    if spec.resume_from and not checkpoint_ready(Path(spec.resume_from).resolve()):
        raise FileNotFoundError(f'Resume checkpoint is missing or incomplete: {spec.resume_from}')
    study.mkdir(parents=True,exist_ok=True)
    write_json(study/'experiment_config.json',spec.to_dict())
    for job in spec.architectures:
        out=study/job.name;out.mkdir(exist_ok=True)
        metadata=spec.metadata(job,ROOT,PYTHON)
        write_json(out/'experiment_metadata.json',metadata)
        write_json(out/'inference_config.json',inference_config(metadata))
    write_json(study/'prepared.json',dict(config_sha256=spec.fingerprint(),validation_key=validation_key(spec),sources=source_hashes()))
    (study/'README.md').write_text('\n'.join([
        '# '+spec.name,'',
        f'Training seed {spec.seed}; at most {spec.budget.max_hours:g} hours per model, including a {spec.budget.evaluation_reserve_minutes:g}-minute final evaluation reserve. The step ceiling is {spec.budget.max_steps:,}. Training temperature {spec.ppo.policy_temperature:g}; entropy coefficient {spec.ppo.ent_coef:g}.','',
        f'Initial weights: {spec.resume_from or spec.ppo.local_policy_init or spec.ppo.map_v1_imitation_init or "fresh initialization"}. Fresh optimizer: {not spec.resume_from or spec.resume_reset_optimizer}. Environment episodes/hidden state start fresh.','',
        'All settings and exact commands are in `experiment_config.json` and each model’s `experiment_metadata.json`. Costs are positive magnitudes subtracted from the base reward; diamond/plant values replace their first-achievement +1.','',
        '| Model | Train GPUs | Evaluation GPU |','| --- | --- | ---: |',
        *[f'| {j.name} | {", ".join(map(str,j.training_devices))} | {j.evaluation_gpu} |' for j in spec.architectures],
        '',f'Rewards: {json.dumps(spec.to_dict()["rewards"],sort_keys=True)}','',
        'Training and evaluation both use the original Classic map generator. Training reset mode is specified in the config; frozen evaluation uses independent standard resets. Evaluation reports all 22 achievements, joint completion, original/modified return, lava, other deaths, and time limits. Intermediate and final episode counts are explicit in the config.','',
        'Checkpoint milestones are logarithmic, with requested evaluation steps added automatically. The final checkpoint uses actual completed interactions. W&B histories, explicit inference configs, source snapshots, progress, and manifests are retained.','',
        'Preparation does not start GPU jobs. Launch requires smoke validation matching both this configuration and the current source hashes. For a new reward parameter or architecture, edit the config and repeat prepare/smoke/launch. Live studies are never overwritten.','']))


def require_free(gpu):
    if not isinstance(gpu,int):
        for device in gpu:require_free(device)
        return
    free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:raise RuntimeError(f'GPU {gpu} is occupied; choose a free device or explicitly replace its job')


def call_logged(command,gpu,path):
    with path.open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Failed with exit {result.returncode}; see {path}')


def smoke(spec,study,gpu_override=None,gpus_override=None):
    key=validation_key(spec);cache=study/'smoke'/key
    if (cache/'validation.json').exists():
        print(f'Current config/source validation already passed: {cache}',flush=True);return
    attempt=cache/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
    def devices_for(job):
        selected=tuple(gpus_override) if gpus_override is not None else ((gpu_override,) if gpu_override is not None else job.training_devices)
        if len(selected)!=len(job.training_devices) or len(set(selected))!=len(selected):
            raise ValueError('Smoke devices must match the configured number of training GPUs without duplicates')
        return selected
    gpu=devices_for(spec.architectures[0])[0];require_free(gpu)
    call_logged([PYTHON,'-m','unittest','-v','test_reward_shaped_spatial','test_rnn_death_penalty'],
        gpu,attempt/'reward_tests.log')
    call_logged([PYTHON,'-m','unittest','-v','test_training_temperature','test_evaluation_scheduling'],gpu,attempt/'temperature_tests.log')
    if any(job.name in ('gru','spatial_gru','spatial_map_gru','map_mlp_gru','neural_map_gru') for job in spec.architectures):
        call_logged([PYTHON,'-m','unittest','-v','test_gru_memory_control'],gpu,attempt/'gru_memory_tests.log')
    if spec.ppo.motion_inputs:
        call_logged([PYTHON,'-m','unittest','-v','test_symbolic_motion_inputs'],gpu,attempt/'motion_inputs_tests.log')
    if spec.ppo.sru_gru:
        call_logged([PYTHON,'-m','unittest','-v','test_sru_gru'],gpu,attempt/'sru_gru_tests.log')
    if spec.ppo.sru_failure_replay:
        call_logged([PYTHON,'-m','unittest','-v','test_failure_seed_replay'],gpu,attempt/'failure_seed_tests.log')
    if spec.ppo.sru_map_coef:
        call_logged([PYTHON,'-m','unittest','-v','test_sru_map_reconstruction'],gpu,attempt/'sru_map_tests.log')
    if any(job.name in ('spatial_map_gru','map_mlp','map_mlp_gru','neural_map_gru') for job in spec.architectures):
        call_logged([PYTHON,'-m','unittest','-v','test_symbolic_map_memory'],
            gpu,attempt/'map_memory_tests.log')
    if any(job.name=='map_mlp' for job in spec.architectures):
        call_logged([PYTHON,'-m','unittest','-v','test_map_mlp'],gpu,attempt/'map_mlp_tests.log')
    if spec.ppo.reset_seed_bank:
        call_logged([PYTHON,'-m','unittest','-v','test_fixed_world_seeds'],
            gpu,attempt/'fixed_world_tests.log')
    if any(job.name=='map_mlp_gru' for job in spec.architectures):
        call_logged([PYTHON,'-m','unittest','-v','test_ctec_map_gru'],gpu,attempt/'ctec_map_gru_tests.log')
    if spec.ppo.map_survival_features!='none':
        call_logged([PYTHON,'-m','unittest','-v','test_symbolic_survival_map'],gpu,attempt/'survival_map_tests.log')
    if spec.ppo.neural_map_channels:
        call_logged([PYTHON,'-m','unittest','-v','test_neural_map'],devices_for(spec.architectures[0]),attempt/'neural_map_tests.log')
    outcomes=[]
    for job in spec.architectures:
        smoke_gpus=devices_for(job);smoke_gpu=smoke_gpus[0]
        require_free(smoke_gpus)
        if len(smoke_gpus)>1:
            parallel_tests='test_ppo_feedforward_data_parallel' if job.name=='map_mlp' else 'test_ppo_data_parallel'
            call_logged([PYTHON,'-m','unittest','-v',parallel_tests],
                smoke_gpus,attempt/f'{job.name}_data_parallel_tests.log')
        destination=attempt/job.name
        command=[PYTHON,'-u',str(ROOT/'smoke_reward_shaped_ppo.py'),'--architecture',job.name,
            '--config-json',str(study/job.name/'inference_config.json'),'--out',str(destination)]
        if spec.resume_from:command.extend(['--resume-checkpoint',str(Path(spec.resume_from).resolve())])
        call_logged(command,smoke_gpus,attempt/f'{job.name}_training.log')
        if spec.ppo.neural_map_channels and not spec.resume_from:
            call_logged(command[:-2]+['--out',str(attempt/(job.name+'_resume')),
                '--resume-checkpoint',str(destination/str(2*spec.batch_steps))],
                smoke_gpus,attempt/f'{job.name}_resume.log')
        step=(int(Path(spec.resume_from).name) if spec.resume_from else 0)+2*spec.batch_steps
        evaluation_source = destination
        if spec.ppo.sru_gru:
            joint = attempt/f'{job.name}_joint'
            call_logged([PYTHON,'-u',str(ROOT/'smoke_reward_shaped_ppo.py'),'--architecture',job.name,
                '--config-json',str(study/job.name/'inference_config.json'),'--out',str(joint),
                '--resume-checkpoint',str(destination/str(step)),'--sru-smoke-unfreeze'],
                smoke_gpus,attempt/f'{job.name}_sru_joint.log')
            evaluation_source = joint
            step += 2*spec.batch_steps
        call_logged([PYTHON,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),
            '--checkpoint',str(evaluation_source/str(step)),'--config-json',str(evaluation_source/'config.json'),
            '--out',str(destination/'evaluation'),'--episodes',str(spec.evaluation.batch),
            '--batch',str(spec.evaluation.batch),'--seed',str(spec.evaluation.seed),
            '--temperature',str(spec.ppo.policy_temperature)],
            smoke_gpu,attempt/f'{job.name}_evaluation.log')
        result=json.loads((destination/'summary.json').read_text())
        if not result['parameters_and_optimizer_finite']:raise RuntimeError('Nonfinite smoke result')
        outcomes.append(dict(architecture=job.name,parameters=result['parameters'],
            trainable_parameters=result.get('trainable_parameters', result['parameters']),artifact_dir=str(destination),
            validation_gpu=smoke_gpu,validation_gpus=list(smoke_gpus)))
    if validation_key(spec)!=key:raise RuntimeError('Source changed during smoke tests; validation was not locked')
    write_json(cache/'validation.json',dict(validation_key=key,config_sha256=spec.fingerprint(),
        sources=source_hashes(),passed=True,outcomes=outcomes,time=time.time()))
    print(f'Smoke validation passed: {cache}',flush=True)


def launch(spec,study):
    key=validation_key(spec);validation_path=study/'smoke'/key/'validation.json'
    if not validation_path.exists():raise RuntimeError('Run --smoke for this config/source before --launch')
    validation=json.loads(validation_path.read_text())
    if validation['validation_key']!=key or not validation['passed']:raise RuntimeError('Invalid smoke validation')
    for job in spec.architectures:
        require_free(job.training_devices);require_free(job.evaluation_gpu)
    if shutil.disk_usage(study).free<20*1024**3:raise RuntimeError('Less than 20 GiB checkpoint headroom')
    start=time.time();deadline=start+spec.budget.max_hours*3600
    train_deadline=deadline-spec.budget.evaluation_reserve_minutes*60
    parameters={r['architecture']:r.get('trainable_parameters',r['parameters']) for r in validation['outcomes']}
    jobs=[]
    # Complete files and snapshots before starting any worker.
    for job in spec.architectures:
        out=study/job.name;metadata=spec.metadata(job,ROOT,PYTHON)
        command=metadata['command']+['--training_deadline',str(train_deadline)]
        metadata.update(command=command,trainable_parameters=parameters[job.name],
            original_start_time=start,overall_deadline=deadline,training_deadline=train_deadline,
            wall_clock_budget_seconds=deadline-start,source_sha256=source_hashes(),config_sha256=spec.fingerprint())
        write_json(out/'experiment_metadata.json',metadata)
        checkpoint=Path(spec.resume_from).resolve() if spec.resume_from else None
        previous_run=str(checkpoint.parents[2]) if checkpoint else None
        write_json(out/'time_budget_plan.json',dict(architecture=job.name,original_start_time=start,
            overall_deadline=deadline,training_deadline=train_deadline,resume_checkpoint=str(checkpoint) if checkpoint else None,
            previous_local_run=previous_run,wandb_run_id=secrets.token_hex(4),wandb_resume='allow',
            command=command,fresh_training=metadata["fresh_training"],local_policy_init=metadata.get("local_policy_init")))
    for name in SOURCES:
        target=study/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    for job in spec.architectures:
        out=study/job.name
        with (out/'worker.log').open('w') as log:
            process=subprocess.Popen([PYTHON,'-u',str(ROOT/'cap_reward_shaped_runs.py'),
                '--worker',job.name,'--study',str(study)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        jobs.append(dict(architecture=job.name,controller_pid=process.pid,
            train_gpu=job.train_gpu,train_gpus=list(job.training_devices),evaluation_gpu=job.evaluation_gpu))
    write_json(study/'launch_manifest.json',dict(time=start,jobs=jobs,
        training_deadline=train_deadline,overall_deadline=deadline,config_sha256=spec.fingerprint()))
    summarize(study);print(json.dumps(jobs,indent=2),flush=True)


def resolve_queued_resume(spec):
    """Return None until a successful source segment has a complete final save."""
    if spec.resume_from:
        checkpoint=Path(spec.resume_from).resolve()
        if not checkpoint_ready(checkpoint):raise RuntimeError('Explicit resume checkpoint is incomplete')
        if int(checkpoint.name)>=spec.budget.max_steps:
            raise ValueError('Source already meets or exceeds continuation target')
        return spec.to_dict()
    if not spec.resume_after:raise ValueError('No resume_after dependency configured')
    progress=json.loads((Path(spec.resume_after)/'progress.json').read_text())
    if progress.get('state') in ('failed','error'):
        raise RuntimeError('Source failed; no continuation was launched')
    terminal=('complete','complete_with_failures','wall_clock_complete')
    if progress.get('state') not in terminal:return None
    if progress.get('training_exit_code')!=0 or not progress.get('final_checkpoint'):
        raise RuntimeError('Source did not finish with a successful final checkpoint; queue preserved')
    checkpoint=Path(progress['final_checkpoint']).resolve()
    if not checkpoint_ready(checkpoint):raise RuntimeError('Source final checkpoint is incomplete')
    start=int(checkpoint.name)
    if start>=spec.budget.max_steps:raise ValueError('Source already meets or exceeds continuation target')
    data=spec.to_dict();data.update(resume_after=None,resume_from=str(checkpoint))
    data['checkpoint_milestones']=sorted({s for s in spec.checkpoint_milestones if start<s<spec.budget.max_steps}
        |{start+offset for offset in (1000000,5000000,20000000) if start+offset<spec.budget.max_steps})
    return data


def queue_worker(study):
    spec=Experiment.load(study/'queue_request.json')
    def status(state,**extra):write_json(study/'queue_progress.json',dict(state=state,time=time.time(),**extra))
    try:
        resolved=None
        while resolved is None:
            resolved=resolve_queued_resume(spec)
            if resolved is None:
                progress=json.loads((Path(spec.resume_after)/'progress.json').read_text())
                status('waiting_for_source',source_state=progress.get('state'),
                    latest_source_checkpoint=progress.get('latest_checkpoint'),target_total_steps=spec.budget.max_steps)
                time.sleep(10)
        config=study/'resolved_continuation.json';write_json(config,resolved)
        continuation=Experiment.load(config)
        source=resolved['resume_from']
        while True:
            try:
                for job in continuation.architectures:
                    require_free(job.training_devices);require_free(job.evaluation_gpu)
                break
            except RuntimeError as error:
                status('waiting_for_devices',resume_checkpoint=source,detail=str(error));time.sleep(10)
        for action in ('smoke','launch'):
            status('validating' if action=='smoke' else 'launching',resume_checkpoint=source,
                   resolved_config=str(config),target_total_steps=continuation.budget.max_steps)
            command=[PYTHON,'-u',str(ROOT/'run_ppo_study.py'),'--config',str(config),'--out',str(study),'--'+action]
            with (study/f'queue_{action}.log').open('a') as log:
                result=subprocess.run(command,cwd=ROOT,env=environment(continuation.architectures[0].training_devices),
                    stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Continuation {action} failed; see queue_{action}.log')
        job=continuation.architectures[0]
        plan=json.loads((study/job.name/'time_budget_plan.json').read_text())
        status('launched',resume_checkpoint=source,target_total_steps=continuation.budget.max_steps,
            launch_manifest=str(study/'launch_manifest.json'),
            wandb_url=f'https://wandb.ai/{continuation.wandb_entity}/{continuation.wandb_project}/runs/{plan["wandb_run_id"]}')
    except Exception as error:
        status('failed',error=str(error));raise


def queue(spec,study):
    if not (spec.resume_after or spec.resume_from):raise ValueError('--queue requires resume_after or resume_from')
    if spec.resume_from:resolve_queued_resume(spec)
    if (study/'queue_manifest.json').exists() or (study/'launch_manifest.json').exists():
        raise FileExistsError(f'Continuation already queued or launched: {study}')
    study.mkdir(parents=True,exist_ok=True)
    write_json(study/'queue_request.json',spec.to_dict())
    write_json(study/'queue_progress.json',dict(state='waiting_for_source' if spec.resume_after else 'waiting_for_devices',time=time.time(),
        source=spec.resume_after or spec.resume_from,target_total_steps=spec.budget.max_steps))
    with (study/'queue_worker.log').open('w') as log:
        process=subprocess.Popen([PYTHON,'-u',str(ROOT/'run_ppo_study.py'),
            '--config',str(study/'queue_request.json'),'--out',str(study),'--queue-worker'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write_json(study/'queue_manifest.json',dict(pid=process.pid,time=time.time(),
        source=spec.resume_after or spec.resume_from,target_total_steps=spec.budget.max_steps,max_hours=spec.budget.max_hours,
        config_sha256=spec.fingerprint()))
    print(f'Queued continuation {process.pid}: {study}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path)
    smoke_devices=p.add_mutually_exclusive_group()
    smoke_devices.add_argument('--smoke-gpu',type=int,help='Validate a single-device study on an idle GPU.')
    smoke_devices.add_argument('--smoke-gpus',help='Comma-separated validation GPUs; count must match the planned training devices.')
    action=p.add_mutually_exclusive_group();action.add_argument('--prepare',action='store_true')
    action.add_argument('--smoke',action='store_true');action.add_argument('--launch',action='store_true')
    action.add_argument('--queue',action='store_true');action.add_argument('--queue-worker',action='store_true')
    args=p.parse_args();spec=Experiment.load(args.config)
    study=(args.out or ROOT/'concept_mapping/runs'/spec.name).resolve()
    if args.queue:queue(spec,study);return
    if args.queue_worker:queue_worker(study);return
    prepare(spec,study)
    if args.smoke:smoke(spec,study,args.smoke_gpu,tuple(map(int,args.smoke_gpus.split(','))) if args.smoke_gpus else None)
    elif args.launch:launch(spec,study)
    else:print(f'Prepared only; no jobs launched: {study}',flush=True)


if __name__=='__main__':main()

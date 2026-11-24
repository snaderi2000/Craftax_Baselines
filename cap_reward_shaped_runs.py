"""Resume the architecture screen with a nine-hour end-to-end time budget."""
import argparse,hashlib,json,os,re,shutil,signal,subprocess,time,fcntl
from pathlib import Path
from run_reward_shaped_architectures import (ROOT,PYTHON,STUDY,ARCHITECTURES,EVALUATIONS,
    write_json,environment,checkpoint_ready,inference_config,summarize)


def checkpoints(roots):
    result={}
    for root in roots:
        for path in (Path(root)/'files/policies').glob('*'):
            if path.name.isdigit() and checkpoint_ready(path):result[int(path.name)]=path
    return result


def terminate_verified(pid,needle):
    source=Path(f'/proc/{pid}/cmdline')
    if not source.exists():return
    cmd=source.read_bytes().replace(b'\0',b' ').decode()
    if needle not in cmd:raise RuntimeError(f'PID {pid} is not the expected process')
    os.kill(pid,signal.SIGTERM)


class AdoptedTrainer:
    """Monitor an existing trainer without restarting its device computation."""
    def __init__(self,pid):
        self.pid=pid;self.signature=self._stat()[19]
    def _stat(self):
        return Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()
    def poll(self):
        try:
            fields=self._stat()
            return None if fields[19]==self.signature and fields[0] not in ('Z','X') else 0
        except FileNotFoundError:return 0
    def terminate(self):
        if self.poll() is None:os.kill(self.pid,signal.SIGTERM)
    def kill(self):
        if self.poll() is None:os.kill(self.pid,signal.SIGKILL)
    def wait(self,timeout=None):
        end=None if timeout is None else time.monotonic()+timeout
        while self.poll() is None:
            if end is not None and time.monotonic()>=end:raise subprocess.TimeoutExpired(str(self.pid),timeout)
            time.sleep(.1)
        return 0


def evaluation_jobs(evaluation):
    temperatures=evaluation.get('temperatures',[1.])
    def label_for(label,t):return label if t==temperatures[0] else label+'_t'+str(t).replace('.','p')
    targets=[(label_for(label,t),step,t) for label,step in
        evaluation.get('common_checkpoints',[(label,step) for label,step in EVALUATIONS if label!='10b']) for t in temperatures]
    return targets,[(label_for('wallclock',t),t) for t in temperatures]


def pending_common_evaluations(targets,done,available):
    return [(label,available[step],temperature) for label,step,temperature in targets
            if label not in done and step in available]


def try_evaluation_lock(gpu):
    lock=Path(f'/tmp/craftax-ppo-evaluation-gpu{gpu}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();return None
    return lock


def controller(study,arch,adopt_pid=None):
    out=study/arch
    plan_path=out/'time_budget_plan.json' if (out/'time_budget_plan.json').exists() else out/'wall_clock_resume.json'
    plan=json.loads(plan_path.read_text())
    original=json.loads((out/'experiment_metadata.json').read_text())
    evaluation=original.get('evaluation',{})
    targets,final_targets=evaluation_jobs(evaluation)
    deadline=plan['overall_deadline'];train_deadline=plan['training_deadline']
    train_gpus=original.get('train_gpus') or [original['train_gpu']]
    training_locks=[]
    for gpu in sorted(train_gpus):
        lock=try_evaluation_lock(gpu)
        if lock is None:raise RuntimeError(f'Training GPU {gpu} is reserved by another job')
        training_locks.append(lock)
    env=environment(train_gpus)
    env.update(WANDB_RUN_ID=plan['wandb_run_id'],WANDB_RESUME=plan.get('wandb_resume','must' if plan.get('resume_checkpoint') else 'allow'))
    log=(out/'training.log').open('a' if adopt_pid else 'w')
    if adopt_pid:
        cmd=Path(f'/proc/{adopt_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if '--run_name '+original['command'][original['command'].index('--run_name')+1] not in cmd:
            raise RuntimeError('Adopted PID does not match this experiment')
        train=AdoptedTrainer(adopt_pid)
    else:
        train=subprocess.Popen(plan['command'],cwd=ROOT,env=env,stdout=log,
            stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    progress=dict(state='training',architecture=arch,pid=train.pid,
        train_gpu=original['train_gpu'],train_gpus=train_gpus,evaluation_gpu=original['evaluation_gpu'],
        original_start_time=plan['original_start_time'],training_deadline=train_deadline,
        overall_deadline=deadline,resumed_from=plan.get('resume_checkpoint'))
    def status(**kw):
        progress.update(kw,time=time.time());write_json(out/'progress.json',progress)
    status()
    roots=[plan['previous_local_run']] if plan.get('previous_local_run') else []
    run=None;current=None;eval_proc=None;eval_log=None;eval_lock=None
    done={label for label in [x[0] for x in targets]+[x[0] for x in final_targets]
          if (out/f'evaluation_{label}/summary.json').exists()}
    failures=[];final_checkpoint=None;train_closed=False;train_failed=False
    def start_evaluation(label,checkpoint,episodes,temperature):
        nonlocal current,eval_proc,eval_log,eval_lock
        lock=try_evaluation_lock(original['evaluation_gpu'])
        if lock is None:
            status(waiting_for_evaluation_gpu=True);return
        eval_lock=lock
        current=label;eval_log=(out/f'evaluation_{label}.log').open('w')
        command=[PYTHON,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),
            '--config-json',str(out/'inference_config.json'),'--checkpoint',str(checkpoint),
            '--out',str(out/f'evaluation_{label}'),'--episodes',str(episodes),
            '--batch',str(evaluation.get('batch',32)),'--seed',str(evaluation.get('seed',18050)),
            '--temperature',str(temperature)]
        if evaluation.get('world_seeds_file'):
            command.extend(['--world-seeds-file',evaluation['world_seeds_file'],'--repeats',str(evaluation.get('repeats',1))])
        eval_proc=subprocess.Popen(command,cwd=ROOT,env=environment(original['evaluation_gpu']),
            stdout=eval_log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        status(evaluating=label,evaluation_pid=eval_proc.pid,waiting_for_evaluation_gpu=False)
    while True:
        now=time.time()
        if now>=deadline:
            if train.poll() is None:train.terminate()
            if eval_proc is not None and eval_proc.poll() is None:eval_proc.terminate()
            status(state='wall_clock_complete',hard_deadline_reached=True,
                final_evaluation_complete=(out/'evaluation_wallclock/summary.json').exists())
            break
        if run is None:
            text=re.sub(r'\x1b\[[0-9;]*m','',(out/'training.log').read_text(errors='replace'))
            matches=re.findall(r'Run data is saved locally in (\S+)',text)
            if matches:
                run=Path(matches[-1]);run=run if run.is_absolute() else ROOT/run
                if str(run) not in roots:roots.append(str(run))
                write_json(run/'files/study_config.json',inference_config(original))
                write_json(out/'run_manifest.json',{**original,'local_run':str(run),
                    'previous_local_runs':[plan['previous_local_run']] if plan.get('previous_local_run') else [],'checkpoint_roots':roots,
                    'wall_clock_budget_seconds':deadline-plan['original_start_time'],'original_start_time':plan['original_start_time'],
                    'overall_deadline':deadline,'training_deadline':train_deadline,
                    'wandb_url':None if env.get('WANDB_MODE')=='offline' else 'https://wandb.ai/synaderi-uc-davis/Craftax_Baselines/runs/'+plan['wandb_run_id']})
                status(local_run=str(run))
        available=checkpoints(roots)
        if available:status(latest_checkpoint=max(available))
        if eval_proc is not None and eval_proc.poll() is not None:
            code=eval_proc.returncode;eval_log.close()
            if eval_lock is not None:eval_lock.close();eval_lock=None
            if code:failures.append(dict(evaluation=current,exit_code=code))
            done.add(current);eval_proc=None
            status(evaluations_completed=sorted(done),evaluation_failures=failures)
            summarize(study)
        if train.poll() is None and now>train_deadline+300:
            train.terminate();status(graceful_stop_overrun=True)
        code=train.poll()
        if code is not None and not train_closed:
            if adopt_pid:
                text=(out/'training.log').read_text(errors='replace')
                if 'New environment steps:' not in text or 'saved runner state to' not in text or 'Traceback (most recent call last)' in text:code=1
            train_closed=True;log.close();train_failed=code!=0
            for lock in training_locks:lock.close()
            training_locks=[]
            available=checkpoints(roots)
            if not available:raise RuntimeError('No saved state available after training')
            final_checkpoint=available[max(available)]
            status(state='evaluating_final',training_exit_code=code,
                final_checkpoint=str(final_checkpoint),actual_training_steps=int(final_checkpoint.name))
        if eval_proc is None:
            remaining_final=[(label,t) for label,t in final_targets if label not in done]
            deferred_common=(pending_common_evaluations(targets,done,available)
                             if final_checkpoint is not None and evaluation.get('defer_until_training_complete',False) else [])
            if deferred_common:
                label,checkpoint,t=deferred_common[0]
                start_evaluation(label,checkpoint,evaluation.get('episodes_per_common_checkpoint',1024),t)
            elif final_checkpoint is not None and remaining_final:
                label,t=remaining_final[0]
                start_evaluation(label,final_checkpoint,evaluation.get('final_episodes',256),t)
            elif final_checkpoint is not None:
                status(state='complete' if not train_failed and not failures else 'complete_with_failures',
                    elapsed_hours=(time.time()-plan['original_start_time'])/3600)
                break
            elif not evaluation.get('defer_until_training_complete',False):
                for label,step,temperature in targets:
                    if label=='10b' or label in done or step not in available:continue
                    start_evaluation(label,available[step],evaluation.get('episodes_per_common_checkpoint',evaluation.get('episodes_per_checkpoint',1024)),temperature);break
        time.sleep(10)
    if train.poll() is None:
        try:train.wait(timeout=10)
        except subprocess.TimeoutExpired:train.kill();train.wait()
    if eval_proc is not None and eval_proc.poll() is None:
        try:eval_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:eval_proc.kill();eval_proc.wait()
    if not log.closed:log.close()
    if eval_log is not None and not eval_log.closed:eval_log.close()
    if eval_lock is not None:eval_lock.close()
    for lock in training_locks:lock.close()
    summarize(study)


def apply(study):
    if (study/'nine_hour_budget.json').exists():raise FileExistsError('Nine-hour cap already applied')
    launch=json.loads((study/'launch_manifest.json').read_text())
    start=launch['time'];deadline=start+9*3600;training_deadline=deadline-15*60
    if time.time()>=training_deadline:raise RuntimeError('Too late to resume within the original nine-hour window')
    jobs=[]
    for arch in ARCHITECTURES:
        smoke=json.loads((study/'smoke'/arch/'cap_resume/summary.json').read_text())
        if not (smoke['deadline_stopped_early'] and smoke['parameters_and_optimizer_finite'] and smoke['resumed_optimizer_steps']>0):
            raise RuntimeError(f'{arch}: cap/resume smoke gate failed')
        out=study/arch
        original=json.loads((out/'experiment_metadata.json').read_text())
        old_manifest=json.loads((out/'run_manifest.json').read_text())
        progress=json.loads((out/'progress.json').read_text())
        terminate_verified(progress['pid'],'--run_name classic-d0p5-diamond5-plant5-')
        jobs.append((arch,original,old_manifest,progress))
    # Wait for device contexts to close before starting replacement trainers.
    time.sleep(8)
    report=[]
    for arch,original,old_manifest,old_progress in jobs:
        out=study/arch;old_run=old_manifest['local_run'];available=checkpoints([old_run])
        step=max(available);checkpoint=available[step]
        q=sorted((k,v.stat().st_mtime) for k,v in available.items())[-5:]
        rates=sorted((b[0]-a[0])/(b[1]-a[1]) for a,b in zip(q,q[1:]))
        sps=rates[len(rates)//2]
        command=original['command']+['--resume_from',str(checkpoint),'--training_deadline',str(training_deadline)]
        plan=dict(architecture=arch,original_start_time=start,overall_deadline=deadline,
            training_deadline=training_deadline,final_evaluation_reserve_seconds=900,
            resume_checkpoint=str(checkpoint),resume_steps=step,previous_local_run=old_run,
            wandb_run_id=Path(old_run).name.rsplit('-',1)[-1],command=command,
            optimizer_restored=True,environment_and_hidden_state_reset=True,
            learning_rate_horizon_steps=10000000000,smoke=smoke,
            estimated_sps=sps,estimated_9h_steps=sps*9*3600,
            estimated_capped_steps=step+sps*max(0,training_deadline-time.time()))
        write_json(out/'wall_clock_resume.json',plan)
        (out/'training.log').rename(out/'training_segment1.log')
        if (out/'worker.log').exists():(out/'worker.log').rename(out/'worker_segment1.log')
        with (out/'worker.log').open('w') as log:
            process=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'--worker',arch,'--study',str(study)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        report.append(dict(architecture=arch,controller_pid=process.pid,sps=sps,
            resume_steps=step,original_start_time=start,training_deadline=training_deadline,
            overall_deadline=deadline,estimated_9h_steps=sps*9*3600))
        print(f'{arch}: resumed controller {process.pid} from {step:,}, measured {sps:,.0f} SPS',flush=True)
    write_json(study/'nine_hour_budget.json',dict(original_start_time=start,
        training_deadline=training_deadline,overall_deadline=deadline,seconds=32400,jobs=report,
        comparison='Common 100M and 1B checkpoints where reached, plus each model at its wall-clock limit.',
        final_evaluation_episodes=256,notes='The original 10B annealing horizon is preserved. Wall-clock results therefore have unequal interaction budgets. Existing checkpoints and W&B run IDs are retained.'))
    changed=['training_time_limit.py','ppo.py','ppo_rnn_og.py','cap_reward_shaped_runs.py','smoke_reward_shaped_ppo.py','run_reward_shaped_architectures.py']
    for name in changed:
        target=study/'wall_clock_cap_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    write_json(study/'wall_clock_cap_snapshot/source_hashes.json',
        {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in changed})
    with (study/'README.md').open('a') as f:
        f.write('\n## Nine-hour wall-clock cap\n\nTraining stops gracefully 15 minutes before nine hours from the original launch, saving the actual completed update. Controllers enforce the nine-hour outer deadline. The reserved time is used for a final 256-episode frozen-policy evaluation. Weights and optimizer states resume from the latest saved checkpoints; environment episodes and recurrent hidden state reset once. W&B run IDs are resumed, and the original 10B learning-rate horizon is preserved. Common-step evaluations at 100M and 1B remain available where reached. `nine_hour_budget.json` and each architecture’s `wall_clock_resume.json` record the deadline, checkpoint, and measured SPS.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=STUDY)
    p.add_argument('--apply',action='store_true');p.add_argument('--worker',choices=(*ARCHITECTURES,'spatial_map_gru','map_mlp','map_mlp_gru','neural_map_gru'))
    p.add_argument('--adopt-pid',type=int)
    args=p.parse_args();study=args.study.resolve()
    if args.worker:controller(study,args.worker,args.adopt_pid)
    elif args.apply:apply(study)
    else:p.error('Provide --apply or --worker')


if __name__=='__main__':main()

"""Three fresh seed-matched spatial/MLP PPO pairs at a 100M training budget."""
import argparse,csv,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
STUDY=ROOT/'concept_mapping/runs/spatial_ppo_vs_mlp_100m'
STEPS=99999744
MILESTONES=[65536,100000,150000,220000,262144,330000,500000,750000,1000000,1048576,
    1500000,2200000,3300000,5000000,7500000,10000000,15000000,20000000,
    22000000,33000000,50000000,75000000]
EVALUATIONS=[('1m',1048576),('5m',5000000),('10m',10000000),('20m',20000000),('50m',50000000),('100m',STEPS)]


def write_json(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def worker(args):
    free=int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:raise RuntimeError(f'GPU{args.gpu} is occupied; no existing job will be stopped')
    out=args.study/args.condition/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=False)
    architecture='conditioned' if args.condition=='spatial' else 'mlp'
    command=[PYTHON,'-u',str(ROOT/'train_ppo_leaps.py'),'--out',str(out/'training'),
        '--architecture',architecture,'--seed',str(args.seed),'--steps',str(STEPS),
        '--lr-horizon','100000000','--sil-coef','0','--gae-lambda','.8',
        '--checkpoint-milestones',','.join(map(str,MILESTONES)),
        '--run-name',f'ppo-spatial-comparison-100m-{args.condition}-seed{args.seed}','--wandb']
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        WANDB_MODE='online',WANDB_INIT_TIMEOUT='300')
    write_json(out/'experiment_metadata.json',dict(command=command,architecture=architecture,condition=args.condition,
        seed=args.seed,gpu=args.gpu,requested_interactions=100000000,actual_interactions=STEPS,
        learning_rate_horizon=100000000,pretrained_weights=False,ordinary_rewards=True,
        num_envs=128,rollout=64,epochs=4,minibatches=8,gamma=.99,gae_lambda=.8,
        clip=.2,value_coefficient=.5,entropy_coefficient=.01,initial_lr=.0002,max_grad_norm=1.,
        reset='optimistic ratio16',self_imitation=False,extra_simulator_calls=0,
        evaluations=EVALUATIONS,evaluation_episodes=256,evaluation_seed=17050,
        requested_checkpoint_milestones=MILESTONES+[STEPS],
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ('train_ppo_leaps.py','models/conditioned_symbolic.py','models/actor_critic.py',
             'models/oracle_search.py','evaluate_ppo_leaps.py')}))
    def status(**kw):write_json(out/'progress.json',dict(seed=args.seed,condition=args.condition,time=time.time(),**kw))
    status(state='training')
    with (out/'training.log').open('w') as f:
        proc=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT)
        status(state='training',pid=proc.pid);code=proc.wait()
    if code:status(state='failed',exit_code=code);raise RuntimeError('See training.log')
    progress=json.loads((out/'training/progress.json').read_text())
    assert progress['state']=='training_complete' and progress['env_step']==STEPS
    checkpoints=sorted(int(p.name) for p in (out/'training/checkpoints').iterdir() if p.name.isdigit())
    for label,requested in EVALUATIONS:
        step=min(s for s in checkpoints if s>=requested)
        status(state='evaluating',checkpoint=step,budget=label)
        with (out/f'evaluation_{label}.log').open('w') as f:
            code=subprocess.call([PYTHON,'-u',str(ROOT/'evaluate_ppo_leaps.py'),
                '--checkpoint',str(out/'training/checkpoints'/str(step)),
                '--out',str(out/f'evaluation_{label}'),'--episodes','256','--seed','17050'],
                cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT)
        if code:status(state='evaluation_failed',checkpoint=step,exit_code=code);raise RuntimeError('See evaluation log')
        summarize(args.study)
    status(state='complete',training_seconds=progress['seconds'],checkpoint=STEPS)


def summarize(study):
    # Workers and parent may finish together; serialize aggregate writes.
    import fcntl
    with (study/'.summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for condition in ('baseline','spatial'):
            for seed in (50,51,52):
                for label,_ in EVALUATIONS:
                    p=study/condition/f'seed{seed}'/f'evaluation_{label}'/'summary.json'
                    if not p.exists():continue
                    d=json.loads(p.read_text())
                    rows.append(dict(condition=condition,seed=seed,budget=label,steps=d['training_steps'],
                        mean_return=d['mean_return'],mean_achievements=d['mean_achievements'],
                        crafter_score=d['crafter_score'],mean_length=d['mean_length'],**d['achievement_success_pct']))
        if rows:
            with (study/'per_seed_results.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        lines=['# Spatial PPO versus MLP PPO: 100M','',
            'Three fresh seed-matched pairs; ordinary rewards, identical trainer/hyperparameters, no replay or planning. Frozen policies evaluated on 256 common standard-reset episodes (seed 17050). Results are provisional until all pairs complete.','',
            '| Budget | Condition | Seeds evaluated | Return | Achievements | Diamond (%) | Crafter score |',
            '|---|---|---:|---:|---:|---:|---:|']
        for label,_ in EVALUATIONS:
            for condition in ('baseline','spatial'):
                group=[r for r in rows if r['budget']==label and r['condition']==condition]
                if not group:continue
                lines.append(f"| {label} | {condition} | {len(group)}/3 | {np.mean([r['mean_return'] for r in group]):.3f} | {np.mean([r['mean_achievements'] for r in group]):.3f} | {np.mean([r['collect_diamond'] for r in group]):.2f} | {np.mean([r['crafter_score'] for r in group]):.2f} |")
        lines+=['','All 22 achievement rates are in the per-seed CSV. Training-seed sample size is three per condition; evaluation episodes are not independent training replications. Historical PPO baselines are retained as references, not pooled with these matched pairs.']
        (study/'results.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=STUDY);p.add_argument('--worker',action='store_true')
    p.add_argument('--condition',choices=('baseline','spatial'),default='spatial')
    p.add_argument('--seed',type=int,default=50);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--summarize-only',action='store_true');args=p.parse_args();args.study=args.study.resolve()
    if args.worker:worker(args);return
    if args.summarize_only:summarize(args.study);return
    args.study.mkdir(parents=True,exist_ok=True)
    if (args.study/'launch_manifest.json').exists():raise FileExistsError('Study already launched')
    for gpu in range(6):
        free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        if free<14000:raise RuntimeError(f'GPU{gpu} busy; no running job will be interrupted')
    children=[];manifest=[]
    for condition,offset in [('baseline',0),('spatial',3)]:
        for index,seed in enumerate((50,51,52)):
            gpu=offset+index
            command=[PYTHON,'-u',str(ROOT/'run_spatial_ppo_100m.py'),'--worker','--study',str(args.study),
                '--condition',condition,'--seed',str(seed),'--gpu',str(gpu)]
            log=(args.study/f'{condition}_seed{seed}.log').open('w')
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((condition,seed,proc,log))
            manifest.append(dict(condition=condition,seed=seed,gpu=gpu,pid=proc.pid,command=command))
            print(f'Started {condition} seed{seed} on GPU{gpu}',flush=True)
    write_json(args.study/'launch_manifest.json',dict(time=time.time(),jobs=manifest,
        requested_steps_per_run=100000000,actual_steps_per_run=STEPS,total_training_steps=6*STEPS))
    pending=children.copy();failed=[]
    while pending:
        for child in pending.copy():
            condition,seed,proc,log=child;code=proc.poll()
            if code is None:continue
            log.close();pending.remove(child)
            if code:failed.append(dict(condition=condition,seed=seed,exit_code=code))
            summarize(args.study)
        write_json(args.study/'progress.json',dict(state='running' if pending else ('failed' if failed else 'complete'),
            pending=[dict(condition=c,seed=s,pid=p.pid) for c,s,p,_ in pending],failed=failed,time=time.time()))
        if pending:time.sleep(30)
    if failed:raise RuntimeError(failed)


if __name__=='__main__':main()

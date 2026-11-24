"""Bounded 1M PPO screen followed by a data-dependent confirmation/refinement.

At most twelve fresh 1,048,576-step training runs. No simulator search, resets
to saved states, reward changes, pretrained teachers, or achievement labels.
"""
import argparse,csv,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
STUDY=ROOT/'concept_mapping/runs/ppo_sample_efficiency_leaps_1m'
CONDITIONS={'baseline':('mlp',0.),'spatial':('conditioned',0.),
    'self_imitation':('mlp',.1),'combined':('conditioned',.1)}


def write_json(path,data):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def worker(args):
    out=args.study/args.phase/args.condition/f'seed{args.seed}'
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_FLAGS='--xla_gpu_autotune_level=0',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        WANDB_MODE='online',WANDB_INIT_TIMEOUT='300')
    command=[PYTHON,'-u',str(ROOT/'train_ppo_leaps.py'),'--out',str(out/'training'),
        '--architecture',args.architecture,'--seed',str(args.seed),'--sil-coef',str(args.sil_coef),
        '--gae-lambda',str(args.gae_lambda),'--run-name',f'ppo-leaps-{args.phase}-{args.condition}-seed{args.seed}',
        '--wandb']
    write_json(out/'experiment_metadata.json',dict(command=command,seed=args.seed,gpu=args.gpu,
        evaluation_seed=args.eval_seed,gameplay_budget=1048576,extra_simulator_calls=0,
        fresh_weights=True,condition=args.condition,phase=args.phase))
    def status(**kw):write_json(out/'progress.json',dict(seed=args.seed,condition=args.condition,time=time.time(),**kw))
    status(state='training')
    with (out/'training.log').open('w') as f:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        status(state='training',pid=proc.pid);code=proc.wait()
    if code:status(state='failed',exit_code=code);raise RuntimeError('Training failed; see training.log')
    train_progress=json.loads((out/'training/progress.json').read_text())
    assert train_progress['state']=='training_complete' and train_progress['env_step']==1048576
    budgets=(1048576,) if args.phase=='screen' else (1048576,262144,524288)
    for step in budgets:
        status(state='evaluating',checkpoint=step)
        episodes=128 if step==1048576 else 64
        destination=out/'evaluation' if step==1048576 else out/f'evaluation_{step}'
        with (out/f'evaluation_{step}.log').open('w') as f:
            code=subprocess.call([PYTHON,'-u',str(ROOT/'evaluate_ppo_leaps.py'),
                '--checkpoint',str(out/'training/checkpoints'/str(step)),
                '--out',str(destination),'--episodes',str(episodes),'--seed',str(args.eval_seed)],
                cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if code:status(state='evaluation_failed',checkpoint=step,exit_code=code);raise RuntimeError('See evaluation log')
    status(state='complete',training_steps=1048576,training_seconds=train_progress['seconds'])


def summaries(study,phase):
    rows=[]
    for p in sorted((study/phase).glob('*/seed*/evaluation/summary.json')):
        d=json.loads(p.read_text());condition=p.parents[2].name
        rows.append(dict(phase=phase,condition=condition,seed=d['training_seed'],
            mean_return=d['mean_return'],mean_achievements=d['mean_achievements'],
            crafter_score=d['crafter_score'],mean_length=d['mean_length'],
            **d['achievement_success_pct']))
    if rows:
        with (study/f'{phase}_per_seed.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows


def report(study):
    lines=['# PPO sample-efficiency leaps: provisional results','',
        'Each fresh agent receives 1,048,576 ordinary gameplay transitions. No extra planning transitions, reward changes, semantic achievement supervision, or pretrained data. Frozen evaluations use standard resets and no replay/search. All 22 rates are in the per-seed CSVs.','',
        '| Phase | Condition | Seeds evaluated | Return | Achievements | Crafter score |',
        '|---|---|---:|---:|---:|---:|']
    aggregate={}
    for phase in ('screen','confirmation','refinement'):
        rows=summaries(study,phase)
        for condition in sorted({r['condition'] for r in rows}):
            group=[r for r in rows if r['condition']==condition]
            means={k:float(np.mean([r[k] for r in group])) for k in ('mean_return','mean_achievements','crafter_score')}
            aggregate[f'{phase}/{condition}']=dict(seeds=[r['seed'] for r in group],**means)
            lines.append(f"| {phase} | {condition} | {len(group)} | {means['mean_return']:.3f} | {means['mean_achievements']:.3f} | {means['crafter_score']:.2f} |")
    lines+=['','Two seeds per phase are a quick screen, not a precise estimate of generalization. Selection uses screen seeds 50/51; confirmation/refinement uses new training seeds 52/53 and new evaluation worlds. Screening and confirmation results are reported separately.','',
        'The historical 10M PPO reference is a target benchmark, not a matched control: it used a different environment-batch size and learning-rate horizon. Meeting its return alone does not establish a general 10x sample-efficiency claim.']
    (study/'results.md').write_text('\n'.join(lines)+'\n');write_json(study/'aggregate.json',aggregate)
    return aggregate


def run_phase(study,phase,jobs):
    children=[]
    for gpu,condition,seed,architecture,coef,trace in jobs:
        command=[PYTHON,'-u',str(ROOT/'run_ppo_leaps_study.py'),'--worker','--study',str(study),
            '--phase',phase,'--condition',condition,'--seed',str(seed),'--gpu',str(gpu),
            '--architecture',architecture,'--sil-coef',str(coef),'--gae-lambda',str(trace),
            '--eval-seed',str(15050 if phase=='screen' else 16050)]
        logpath=study/f'{phase}_{condition}_seed{seed}.log';log=logpath.open('w')
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        children.append((condition,seed,proc,log));print(f'Started {phase}/{condition}/seed{seed} GPU{gpu}',flush=True)
    failed=[];pending=children.copy()
    while pending:
        for child in pending.copy():
            condition,seed,proc,log=child;code=proc.poll()
            if code is None:continue
            log.close();pending.remove(child)
            if code:failed.append(dict(condition=condition,seed=seed,exit_code=code))
            report(study)
        write_json(study/'progress.json',dict(state='running' if pending else ('failed' if failed else 'phase_complete'),phase=phase,
            pending=[dict(condition=c,seed=s,pid=p.pid) for c,s,p,_ in pending],failed=failed,time=time.time()))
        if pending:time.sleep(15)
    if failed:raise RuntimeError(failed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=STUDY);p.add_argument('--worker',action='store_true')
    p.add_argument('--phase',default='screen');p.add_argument('--condition',default='baseline')
    p.add_argument('--seed',type=int,default=50);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--architecture',default='mlp');p.add_argument('--sil-coef',type=float,default=0.)
    p.add_argument('--gae-lambda',type=float,default=.8);p.add_argument('--eval-seed',type=int,default=15050)
    p.add_argument('--report-only',action='store_true');args=p.parse_args();args.study=args.study.resolve()
    if args.worker:worker(args);return
    if args.report_only:report(args.study);return
    if (args.study/'protocol.json').exists():raise FileExistsError('Existing study; use report-only to inspect')
    args.study.mkdir(parents=True,exist_ok=True)
    for gpu in range(8):
        free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        if free<14000:raise RuntimeError(f'GPU{gpu} busy; no running jobs will be stopped')
    protocol=dict(conditions=CONDITIONS,screen_seeds=[50,51],heldout_seeds=[52,53],
        training_steps_per_agent=1048576,max_agents=12,max_training_steps=12582912,
        primary='frozen mean episode return; all achievement rates and Crafter score secondary',
        seed_selection='screen50/51; new52/53 and evaluation seed16050 for confirmation/refinement',
        second_round_rule='If best variant mean gain >=0.25 and both paired seed gains >0, confirm it unchanged. Otherwise test spatial PPO with GAE lambda0.95 against the same baseline.',
        extra_simulator_calls=0,oracle_access=False,training_achievement_labels=False,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ('models/conditioned_symbolic.py','ppo_leap_utils.py','train_ppo_leaps.py','evaluate_ppo_leaps.py','run_ppo_leaps_study.py')})
    write_json(args.study/'protocol.json',protocol)
    references=[]
    for seed in (50,51,52):
        p=ROOT/f'concept_mapping/runs/ppo_memory_comparison_10m/mlp_seed{seed}/evaluation_10m/summary.json'
        if p.exists():references.append(json.loads(p.read_text()))
    write_json(args.study/'ten_million_reference.json',dict(seeds=[50,51,52][:len(references)],
        mean_return=float(np.mean([r['mean_return'] for r in references])),
        mean_achievements=float(np.mean([r['mean_achievements'] for r in references])),
        source='existing PPO MLP10M; benchmark reference, not a matched-control curve'))
    jobs=[]
    for condition,(architecture,coef) in CONDITIONS.items():
        for seed in (50,51):jobs.append((len(jobs),condition,seed,architecture,coef,.8))
    run_phase(args.study,'screen',jobs)
    rows=summaries(args.study,'screen')
    base={r['seed']:r['mean_return'] for r in rows if r['condition']=='baseline'}
    candidates=[]
    for condition in ('spatial','self_imitation','combined'):
        group=[r for r in rows if r['condition']==condition]
        gains=[r['mean_return']-base[r['seed']] for r in group]
        candidates.append((float(np.mean(gains)),condition,gains))
    gain,winner,gains=max(candidates)
    if gain>=.25 and min(gains)>0:
        phase='confirmation';architecture,coef=CONDITIONS[winner];trace=.8;candidate=winner
    else:
        phase='refinement';architecture,coef,trace='conditioned',0.,.95;candidate='spatial_long_credit'
    write_json(args.study/'second_round_decision.json',dict(phase=phase,screen_winner=winner,
        winner_paired_gains=gains,winner_mean_gain=gain,candidate=candidate,
        architecture=architecture,sil_coef=coef,gae_lambda=trace,
        note='Configuration selected only from screen data; held-out seed/world results not yet observed'))
    jobs=[]
    for condition,arch,c,tr in [('baseline','mlp',0.,.8),(candidate,architecture,coef,trace)]:
        for seed in (52,53):jobs.append((len(jobs),condition,seed,arch,c,tr))
    run_phase(args.study,phase,jobs);report(args.study)
    evaluation_calls=sum(json.loads(p.read_text())['evaluation_executed_transitions']
        for p in args.study.glob('**/evaluation*/summary.json'))
    write_json(args.study/'progress.json',dict(state='complete',max_training_steps=12582912,
        actual_training_steps=12582912,evaluation_executed_transitions=evaluation_calls,time=time.time()))


if __name__=='__main__':main()

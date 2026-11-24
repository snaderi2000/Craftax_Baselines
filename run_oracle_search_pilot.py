"""Persistent one-seed simulator-search training and frozen evaluation worker."""
import argparse,csv,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=50);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--study',type=Path,default=ROOT/'concept_mapping/runs/oracle_simulator_search_1m')
    args=p.parse_args();out=args.study.resolve()/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=False)
    free=int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:raise RuntimeError('GPU occupied; no other jobs will be stopped')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='online',WANDB_INIT_TIMEOUT='300')
    command=[PYTHON,'-u',str(ROOT/'train_oracle_search.py'),'--out',str(out/'training'),
        '--seed',str(args.seed),'--steps','1048576','--wandb']
    metadata=dict(seed=args.seed,gpu=args.gpu,command=command,gameplay_steps=1048576,
        search_steps=33554432,total_simulator_steps=34603008,ordinary_rewards=True,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ('models/oracle_search.py','train_oracle_search.py','evaluate_oracle_search.py')})
    (out/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def status(**kw):(out/'progress.json').write_text(json.dumps(dict(seed=args.seed,gpu=args.gpu,time=time.time(),**kw),indent=2)+'\n')
    status(state='training')
    with (out/'training.log').open('w') as f:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        status(state='training',pid=proc.pid);code=proc.wait()
    if code:status(state='training_failed',exit_code=code);raise RuntimeError('See training.log')
    checkpoint=out/'training/checkpoints/1048576'
    for mode in ('search','policy'):
        status(state='evaluating',mode=mode)
        with (out/f'evaluation_1m_{mode}.log').open('w') as f:
            code=subprocess.call([PYTHON,'-u',str(ROOT/'evaluate_oracle_search.py'),
                '--checkpoint',str(checkpoint),'--out',str(out/f'evaluation_1m_{mode}'),'--mode',mode],
                cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        if code:status(state='evaluation_failed',mode=mode,exit_code=code);raise RuntimeError('See evaluation log')
    rows=[]
    for label,path in [('PPO MLP',ROOT/f'concept_mapping/runs/ppo_memory_comparison_10m/mlp_seed{args.seed}/evaluation_1m/summary.json'),
        ('Learned-model EZ debug v2',ROOT/f'concept_mapping/runs/symbolic_ez_debug_v2_1m/seed{args.seed}/evaluation_1m_search/summary.json'),
        ('Exact-simulator search',out/'evaluation_1m_search/summary.json'),
        ('Distilled policy, no search',out/'evaluation_1m_policy/summary.json')]:
        if not path.exists():continue
        d=json.loads(path.read_text())
        rows.append(dict(agent=label,episodes=d['episodes'],mean_return=d['mean_return'],
            mean_achievements=d['mean_achievements'],mean_length=d['mean_length'],**d['achievement_success_pct']))
    with (args.study/'evaluation_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 1M simulator-search pilot: seed-matched frozen evaluations','',
        '128 standard-reset episodes per agent, evaluation seed 15050. Simulator search is privileged and spends extra simulator transitions; this is a diagnostic, not an equal-compute benchmark.','',
        '| Agent | Return | Achievements | Diamond (%) |','|---|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['agent']} | {r['mean_return']:.3f} | {r['mean_achievements']:.3f} | {r['collect_diamond']:.2f} |")
    lines+=['','All 22 achievement rates are in `evaluation_results.csv`. Search and gameplay costs are recorded separately in training and evaluation summaries.']
    (args.study/'results.md').write_text('\n'.join(lines)+'\n')
    status(state='complete',checkpoint=str(checkpoint))


if __name__=='__main__':main()

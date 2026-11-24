"""Matched baseline and stone-outcome PPO runs for a configurable seed/budget."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'concept_mapping/runs/stone_outcome_aux_20m_seed50'
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--total_timesteps',type=int,default=20000000)
    parser.add_argument('--gpu',type=int,default=7)
    parser.add_argument('--seed',type=int,default=50)
    parser.add_argument('--evaluation_seed',type=int,default=2050)
    parser.add_argument('--out',default=None)
    args=parser.parse_args()
    if args.total_timesteps<8192:raise ValueError('Budget must contain at least one PPO update')
    budget_name=f'{args.total_timesteps//1000000}m'
    OUT=Path(args.out).resolve() if args.out else ROOT/f'concept_mapping/runs/stone_outcome_aux_{budget_name}_seed{args.seed}'
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['baseline','stone_aux']:
        directory=OUT/name
        directory.mkdir(exist_ok=True)
        if (directory/'command.json').exists():raise RuntimeError(f'Refusing overwrite: {directory}')
    original=json.loads((ROOT/'concept_mapping/runs/diamond_density_curriculum_200m/replications_seed50/baseline/command.json').read_text())['command']
    common=original[:original.index('--run_name')]
    common[common.index('--total_timesteps')+1]=str(args.total_timesteps)
    common[common.index('--seed')+1]=str(args.seed)
    ix=common.index('--save_policy_milestones')+1
    common[ix]=','.join(s for s in common[ix].split(',') if int(s)<args.total_timesteps)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
             OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_INIT_TIMEOUT='300')
    completed=[]
    def progress(**status):
        (OUT/'progress.json').write_text(json.dumps(dict(completed=completed,**status),indent=2)+'\n')
    for name,coef in [('baseline','0'),('stone_aux','.1')]:
        free=int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<14000:
            progress(state='failed',error=f'GPU {args.gpu} no longer has sufficient free memory')
            raise RuntimeError(f'GPU {args.gpu} is occupied')
        directory=OUT/name
        command=common+['--run_name',f'stone-outcome-{name}-{budget_name}-seed{args.seed}','--stone_probe',
                        '--stone_aux_coef',coef,'--stone_aux_batch','256','--stone_replay_capacity','1024']
        (directory/'command.json').write_text(json.dumps(dict(command=command,gpu=args.gpu),indent=2)+'\n')
        with (directory/'training.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            progress(state='training',condition=name,gpu=args.gpu,pid=process.pid)
            print(f'Launched {name} GPU {args.gpu} PID {process.pid}',flush=True)
            code=process.wait()
        if code:
            progress(state='failed',condition=name,error=f'training exit {code}')
            raise RuntimeError('Training failed')
        run=re.findall(r'Run data is saved locally in (\S+)',(directory/'training.log').read_text())[-1]
        (directory/'run_manifest.json').write_text(json.dumps(dict(condition=name,local_run=run,seed=args.seed,
            evaluation_seed=args.evaluation_seed,
            environment_budget=args.total_timesteps,gpu=args.gpu,auxiliary_coefficient=float(coef),
            wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-",1)[-1]}'),indent=2)+'\n')
        progress(state='evaluating',condition=name,gpu=args.gpu)
        with (directory/'evaluation.log').open('w') as log:
            result=subprocess.run([PYTHON,'-u','evaluate_stone_outcome.py','--run',run,
                '--out',str(directory/'evaluation'),'--episodes','256','--seed',str(args.evaluation_seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            progress(state='failed',condition=name,error='evaluation failed')
            raise RuntimeError('Evaluation failed')
        completed.append(name)
    results={n:json.loads((OUT/n/'evaluation/summary.json').read_text()) for n in completed}
    (OUT/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    progress(state='complete')
    print(f'Both {budget_name} runs and fresh normal-environment evaluations complete.',flush=True)


if __name__=='__main__':main()

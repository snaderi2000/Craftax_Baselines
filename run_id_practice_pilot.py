"""One of eight matched 50M baseline/practice jobs with frozen evaluations."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--condition',choices=['baseline','practice'],required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = ROOT/f'concept_mapping/runs/id_practice_50m/seed{args.seed}_{args.condition}'
    if out.exists():
        raise RuntimeError(f'Refusing overwrite {out}')
    source = json.loads((ROOT/'concept_mapping/runs/diamond_density_curriculum_200m/replications_seed50/baseline/command.json').read_text())['command']
    command = source[:source.index('--run_name')]
    command[command.index('--seed')+1] = str(args.seed)
    command[command.index('--total_timesteps')+1] = '50000000'
    ix = command.index('--save_policy_milestones')+1
    milestones = sorted(set([int(s) for s in command[ix].split(',') if int(s)<50000000]+[5000000,20000000]))
    command[ix] = ','.join(map(str,milestones))
    command += ['--run_name',f'id-practice-{args.condition}-50m-seed{args.seed}',
                '--no-use_optimistic_resets','--id_practice','--id_practice_envs',
                '32' if args.condition=='practice' else '0','--id_practice_capacity','64','--id_practice_horizon','64']
    free = int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:
        raise RuntimeError('Assigned GPU occupied; no other jobs will be interrupted')
    out.mkdir(parents=True)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_INIT_TIMEOUT='300')
    def progress(**status):
        (out/'progress.json').write_text(json.dumps(dict(seed=args.seed,condition=args.condition,gpu=args.gpu,**status),indent=2)+'\n')
    (out/'command.json').write_text(json.dumps(dict(command=command,gpu=args.gpu),indent=2)+'\n')
    with (out/'training.log').open('w') as log:
        process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        progress(state='training',pid=process.pid)
        print(f'Launched {args.condition} seed {args.seed}, GPU {args.gpu}, PID {process.pid}',flush=True)
        code = process.wait()
    if code:
        progress(state='failed',error=f'training exit {code}')
        raise RuntimeError('Training failed')
    run = re.findall(r'Run data is saved locally in (\S+)',(out/'training.log').read_text())[-1]
    policy_dir = Path(run)/'files/policies'
    checkpoints = sorted(int(p.name) for p in policy_dir.iterdir() if p.name.isdigit())
    assert checkpoints[-1] == (50000000//8192)*8192
    manifest = dict(local_run=run,seed=args.seed,condition=args.condition,gpu=args.gpu,
        requested_training_steps=50000000,actual_training_steps=checkpoints[-1],evaluation_seed=2050,
        practice_envs=32 if args.condition=='practice' else 0,normal_envs=96 if args.condition=='practice' else 128,
        archive_capacity_per_id=64,practice_horizon=64,
        wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-",1)[-1]}')
    if args.condition=='practice':
        archive_path = Path(run)/'files/practice_archive'
        manifest['archive_directory'] = str(archive_path)
        manifest['archive_summary'] = json.loads((archive_path/'archive_summary.json').read_text())
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for budget in (5,20,50):
        checkpoint = min(s for s in checkpoints if s>=budget*1000000) if budget<50 else checkpoints[-1]
        progress(state='evaluating',budget_millions=budget,checkpoint=checkpoint,local_run=run)
        with (out/f'evaluation_{budget}m.log').open('w') as log:
            result = subprocess.run([PYTHON,'-u','evaluate_cppo_pilot.py','--run',run,
                '--out',str(out/f'normal_evaluation_{budget}m'),'--episodes','512','--seed','2050',
                '--checkpoint',str(checkpoint)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            progress(state='failed',error=f'{budget}M evaluation failed',local_run=run)
            raise RuntimeError('Evaluation failed')
    progress(state='complete',local_run=run)
    subprocess.run([PYTHON,'compare_id_practice_pilot.py'],check=True)
    print('Training and all three normal-environment evaluations complete.',flush=True)


if __name__ == '__main__':
    main()

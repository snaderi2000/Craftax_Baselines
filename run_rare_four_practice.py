"""Fresh rare-four practice runs: existing 50M variants or three-seed cap30 200M."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--full_episode',action='store_true')
    p.add_argument('--max_reuse',type=int,choices=[10,30],default=10)
    p.add_argument('--backward_curriculum',action='store_true')
    p.add_argument('--total_timesteps',type=int,choices=[50000000,200000000],default=50000000)
    args = p.parse_args()
    long_run = args.total_timesteps==200000000
    if long_run and (args.max_reuse!=30 or args.full_episode or args.backward_curriculum or args.seed not in (50,51,52)):
        raise ValueError('200M experiment is short cap30 with seeds 50,51,52 only')
    if args.backward_curriculum and (args.full_episode or args.max_reuse!=10):
        raise ValueError('Backward pilot keeps cap 10 and short episodes; do not combine interventions')
    if args.full_episode and args.max_reuse!=10:
        raise ValueError('Full-episode control keeps cap 10')
    os.chdir(ROOT)
    control = ROOT/f'concept_mapping/runs/id_practice_50m/seed{args.seed}_baseline'
    manifest = json.loads((control/'run_manifest.json').read_text())
    assert manifest['seed']==args.seed and manifest['actual_training_steps']==49995776
    for budget in (5,20,50):
        summary = json.loads((control/f'normal_evaluation_{budget}m/evaluation.json').read_text())
        assert summary['episodes']==512 and summary['evaluation_seed']==2050
    command = json.loads((control/'command.json').read_text())['command']
    assert command[command.index('--num_envs')+1]=='128'
    assert command[command.index('--total_timesteps')+1]=='50000000'
    command[command.index('--total_timesteps')+1]=str(args.total_timesteps)
    command[command.index('--id_practice_envs')+1]='13'
    command[command.index('--run_name')+1]=f'rare-four-practice10-50m-seed{args.seed}'
    command += ['--id_practice_rare_four','--id_practice_rarity_window','512',
                '--id_practice_rarity_refresh','128','--id_practice_rarity_warmup','512',
                '--id_practice_max_reuse',str(args.max_reuse)]
    variant = 'full_episode_practice_50m' if args.full_episode else 'rare_four_practice_50m'
    if args.full_episode:
        command += ['--id_practice_full_episode']
        command[command.index('--run_name')+1]=f'full-episode-rare-four-practice10-50m-seed{args.seed}'
    if args.max_reuse==30:
        variant=f'rare_four_reuse30_{args.total_timesteps//1000000}m'
        command[command.index('--run_name')+1]=f'rare-four-reuse30-practice10-{args.total_timesteps//1000000}m-seed{args.seed}'
    if long_run:
        index=command.index('--save_policy_milestones')+1
        milestones={int(s) for s in command[index].split(',')}
        milestones.update((50000000,75000000,100000000,150000000))
        command[index]=','.join(str(s) for s in sorted(milestones))
    if args.backward_curriculum:
        variant='backward_practice_50m'
        command+=['--id_practice_backward']
        command[command.index('--run_name')+1]=f'backward-rare-four-practice10-50m-seed{args.seed}'
    out = ROOT/f'concept_mapping/runs/{variant}/seed{args.seed}_practice'
    if out.exists():
        raise RuntimeError(f'Refusing overwrite {out}')
    free = int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:
        raise RuntimeError('GPU occupied; no existing process will be interrupted')
    out.mkdir(parents=True)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_INIT_TIMEOUT='300')
    def progress(**status):
        (out/'progress.json').write_text(json.dumps(dict(seed=args.seed,gpu=args.gpu,**status),indent=2)+'\n')
    (out/'command.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,
        hyperparameter_reference_directory=str(control),hyperparameter_reference_run=manifest['local_run'],
        baseline_directory=str(control),baseline_run=manifest['local_run']),indent=2)+'\n')
    budgets=(5,20,33,50,100,150,200) if long_run else (5,20,50)
    (out/'plan.json').write_text(json.dumps(dict(seed=args.seed,gpu=args.gpu,
        requested_steps=args.total_timesteps,fresh_training=True,standard_resets=True,
        max_snapshot_reuse=args.max_reuse,practice_offsets=[1,5,20],practice_horizon=64,
        success_bootstrap=False,evaluation_budgets_millions=budgets,evaluation_episodes=512,
        evaluation_seed=2050,historical_baseline_caveat='200M baselines use optimistic resets, not matched standard-reset controls.' if long_run else None),indent=2)+'\n')
    with (out/'training.log').open('w') as log:
        process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        progress(state='training',pid=process.pid)
        print(f'Launched rare-four seed {args.seed}, GPU {args.gpu}, PID {process.pid}',flush=True)
        code = process.wait()
    if code:
        progress(state='failed',error=f'training exit {code}')
        raise RuntimeError('Training failed')
    run = re.findall(r'Run data is saved locally in (\S+)',(out/'training.log').read_text())[-1]
    policy_dir = Path(run)/'files/policies'
    steps = sorted(int(p.name) for p in policy_dir.iterdir() if p.name.isdigit())
    assert steps[-1]==args.total_timesteps//8192*8192
    archive_path = Path(run)/'files/practice_archive'
    saved = dict(local_run=run,seed=args.seed,gpu=args.gpu,requested_training_steps=args.total_timesteps,
        actual_training_steps=steps[-1],evaluation_seed=2050,practice_envs=13,normal_envs=115,
        maximum_practice_fraction=13/128,normal_window=512,refresh_normal_episodes=128,
        minimum_normal_episodes=512,max_snapshot_reuse=args.max_reuse,baseline_run=manifest['local_run'],
        full_episode=args.full_episode,practice_horizon=None if args.full_episode else 64,
        backward_curriculum=args.backward_curriculum,
        offsets=[1,5,20,50] if args.backward_curriculum else [1,5,20],
        per_level_horizons=[64,64,64,128] if args.backward_curriculum else None,
        archive_directory=str(archive_path),archive_summary=json.loads((archive_path/'archive_summary.json').read_text()),
        wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-",1)[-1]}')
    (out/'run_manifest.json').write_text(json.dumps(saved,indent=2)+'\n')
    saved.update(fresh_training=True,learning_rate_schedule_budget=args.total_timesteps,
        standard_resets=True,success_bootstrap=False,evaluation_budgets_millions=budgets)
    (out/'run_manifest.json').write_text(json.dumps(saved,indent=2)+'\n')
    for budget in budgets:
        checkpoint = min(s for s in steps if s>=budget*1000000) if budget*1000000<args.total_timesteps else steps[-1]
        progress(state='evaluating',budget_millions=budget,local_run=run)
        with (out/f'evaluation_{budget}m.log').open('w') as log:
            result = subprocess.run([PYTHON,'-u','evaluate_cppo_pilot.py','--run',run,
                '--out',str(out/f'normal_evaluation_{budget}m'),'--episodes','512','--seed','2050',
                '--checkpoint',str(checkpoint)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            progress(state='failed',error=f'{budget}M evaluation failed',local_run=run)
            raise RuntimeError('Evaluation failed')
    if long_run:
        historical=ROOT/'concept_mapping/runs/diamond_density_curriculum_200m'
        historical=historical/'replications_seed50/baseline' if args.seed==50 else historical/f'replications_seed51_52/seed{args.seed}_baseline'
        source=json.loads((historical/'run_manifest.json').read_text())
        baseline_run=source['local_run']
        baseline_steps=sorted(int(p.name) for p in (Path(baseline_run)/'files/policies').iterdir() if p.name.isdigit())
        historical_out=out.parent/f'seed{args.seed}_historical_baseline'
        historical_out.mkdir(exist_ok=False)
        (historical_out/'run_manifest.json').write_text(json.dumps(dict(**source,
            use_optimistic_resets=True,matched_control=False,
            caveat='Same 200M budget and core PPO settings, but optimistic resets vs standard resets in cap30.'),indent=2)+'\n')
        for budget in (50,100,200):
            checkpoint=min(s for s in baseline_steps if s>=budget*1000000) if budget<200 else baseline_steps[-1]
            progress(state='evaluating_historical_baseline',budget_millions=budget,local_run=run)
            with (out/f'historical_baseline_evaluation_{budget}m.log').open('w') as log:
                result=subprocess.run([PYTHON,'-u','evaluate_cppo_pilot.py','--run',baseline_run,
                    '--out',str(historical_out/f'normal_evaluation_{budget}m'),'--episodes','512',
                    '--seed','2050','--checkpoint',str(checkpoint)],env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                progress(state='failed',error=f'Historical {budget}M evaluation failed',local_run=run)
                raise RuntimeError('Historical baseline evaluation failed')
    progress(state='complete',local_run=run)
    if long_run:
        subprocess.run([PYTHON,'compare_rare_four_200m.py'],check=True)
    elif args.max_reuse==30 or args.backward_curriculum:
        subprocess.run([PYTHON,'compare_practice_extensions.py','--variant',
                        'reuse30' if args.max_reuse==30 else 'backward'],check=True)
    else:
        subprocess.run([PYTHON,'compare_rare_four_practice.py']+(['--full_episode'] if args.full_episode else []),check=True)


if __name__=='__main__':
    main()

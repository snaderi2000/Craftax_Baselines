"""Matched two-seed 5M continuation: unchanged cap30 vs success bootstrap.

Restore each seed's own 49,995,776-step cap30 actor/critic and archive, including
remaining reuse quotas and rarity history. Fresh live episodes/RNG, reset Adam
identically in both arms, LR 1e-4 annealed over 610 updates. No simulator/reward
edits. Final total checkpoint 54,992,896 = 4,997,120 new interactions.
512 frozen normal episodes at +1M/+5M, seed 2050. This is a short recovery test,
not a fresh-training test or proof that target bootstrapping fixes forgetting.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
EXPERIMENT=ROOT/'concept_mapping/runs/practice_success_bootstrap_cap30_5m'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=[50,52],required=True)
    p.add_argument('--condition',choices=['control','bootstrap'],required=True)
    p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args();os.chdir(ROOT)
    source=ROOT/f'concept_mapping/runs/rare_four_reuse30_50m/seed{args.seed}_practice'
    parent=json.loads((source/'run_manifest.json').read_text())
    assert parent['seed']==args.seed and parent['max_snapshot_reuse']==30
    start=parent['actual_training_steps'];assert start==49995776
    command=json.loads((source/'command.json').read_text())['command']
    def setarg(name,value):command[command.index(name)+1]=str(value)
    setarg('--total_timesteps',start+5000000)
    setarg('--lr','0.0001')
    setarg('--run_name',f'cap30-success-{args.condition}-continue5m-seed{args.seed}')
    setarg('--save_policy_milestones',','.join(str(start+n) for n in (100000,500000,1000000,2200000,3300000)))
    command+=['--resume_from',str(Path(parent['local_run'])/'files/policies'/str(start)),
              '--resume_reset_optimizer','--id_practice_archive_from',parent['archive_directory']]
    if args.condition=='bootstrap':command+=['--id_practice_success_bootstrap']
    out=EXPERIMENT/f'seed{args.seed}_{args.condition}'
    if out.exists():raise RuntimeError(f'Refusing overwrite {out}')
    free=int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:raise RuntimeError('GPU occupied; existing jobs will not be interrupted')
    out.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
             OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_INIT_TIMEOUT='300')
    def progress(**status):
        (out/'progress.json').write_text(json.dumps(dict(seed=args.seed,condition=args.condition,gpu=args.gpu,**status),indent=2)+'\n')
    manifest=dict(seed=args.seed,condition=args.condition,gpu=args.gpu,parent_directory=str(source),
        parent_run=parent['local_run'],parent_checkpoint=start,new_requested_steps=5000000,
        new_actual_steps=4997120,final_checkpoint=54992896,optimizer_restarted=True,
        learning_rate=.0001,learning_rate_schedule='linear over continuation updates',
        restored_archive=parent['archive_directory'],reuse_quotas_preserved=True,
        rarity_history_preserved=True,live_episodes_and_phase_counters_restarted=True,
        training_rng_seed=args.seed,evaluation_seed=2050,practice_envs=13,num_envs=128,
        success_bootstrap=args.condition=='bootstrap',timeout_bootstrap=True,
        true_terminal_bootstrap=False,raw_environment_rewards_unchanged=True,
        limitation='Two-seed short continuation; bootstraps existing mixed-data critic, not a separate normal-only critic.')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'command.json').write_text(json.dumps(dict(command=command),indent=2)+'\n')
    with (out/'training.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        progress(state='training',pid=process.pid);print(f'{args.condition}, seed {args.seed}, GPU {args.gpu}, PID {process.pid}',flush=True)
        code=process.wait()
    if code:
        progress(state='failed',error=f'training exit {code}');raise RuntimeError('Training failed')
    run=re.findall(r'Run data is saved locally in (\S+)',(out/'training.log').read_text())[-1]
    steps=sorted(int(p.name) for p in (Path(run)/'files/policies').iterdir() if p.name.isdigit())
    assert steps[-1]==manifest['final_checkpoint']
    archive=Path(run)/'files/practice_archive'
    manifest.update(local_run=run,archive_directory=str(archive),
        archive_summary=json.loads((archive/'archive_summary.json').read_text()),
        wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-",1)[-1]}')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for budget in (1,5):
        checkpoint=min(s for s in steps if s>=start+1000000) if budget==1 else steps[-1]
        progress(state='evaluating',new_budget_millions=budget,local_run=run)
        with (out/f'evaluation_{budget}m.log').open('w') as log:
            result=subprocess.run([PYTHON,'-u','evaluate_cppo_pilot.py','--run',run,
                '--out',str(out/f'normal_evaluation_plus{budget}m'),'--episodes','512',
                '--seed','2050','--checkpoint',str(checkpoint)],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            progress(state='failed',error=f'+{budget}M evaluation failed');raise RuntimeError('Evaluation failed')
    progress(state='complete',local_run=run)
    subprocess.run([PYTHON,'compare_practice_success_bootstrap.py'],check=True)


if __name__=='__main__':main()

"""Launch five matched 100M pilots on GPUs 0--4 and compare after completion."""
import json
import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'concept_mapping/runs/cppo_100m_pilot'
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
CONDITIONS=['cppo','baseline','bc_ppo','frozen_critic','shuffled_cppo']


class ExistingProcess:
    """Observe an already-launched child after restarting the supervisor."""
    def __init__(self,pid,log):self.pid,self.log=pid,log
    def poll(self):
        try:
            command=Path(f'/proc/{self.pid}/cmdline').read_bytes()
            if b'cppo-pilot-' in command:return None
        except FileNotFoundError:pass
        return 0 if 'saved runner state to' in self.log.read_text() else 1


def available_memory(gpu):
    data=subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(data.strip())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recover',action='store_true')
    parser.add_argument('--retired_supervisor_pid',type=int,default=None)
    args=parser.parse_args()
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    children={};logs={};waiting=set(CONDITIONS);completed=set();retries={c:0 for c in CONDITIONS}
    memory_required={c:3500 for c in CONDITIONS}
    environment=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_INIT_TIMEOUT='300')
    common=[PYTHON,'-u','ppo.py','--env_name','Craftax-Classic-Symbolic-v1',
        '--num_envs','128','--total_timesteps','1e8','--num_steps','64',
        '--update_epochs','4','--num_minibatches','8','--gamma','.99','--gae_lambda','.8',
        '--clip_eps','.2','--vf_coef','.5','--ent_coef','.01','--lr','.0002',
        '--max_grad_norm','1.0','--anneal_lr','--seed','50','--save_policy',
        '--save_policy_milestones','65536,100000,150000,220000,330000,500000,750000,1000000,1500000,2200000,3300000,5000000,7500000,10000000,15000000,22000000,33000000,50000000,75000000']
    memory=str(OUT/'memory/culture.npz')
    for c in CONDITIONS:
        directory=OUT/c;directory.mkdir(exist_ok=True)
        if (directory/'run_manifest.json').exists() and not args.recover:
            raise RuntimeError(f'Existing pilot {c}; refusing to overwrite it')
    if args.recover:
        progress=json.loads((OUT/'progress.json').read_text())
        for c,pid in progress.get('pids',{}).items():
            directory=OUT/c
            if (directory/'evaluation.json').exists():
                completed.add(c);waiting.discard(c);continue
            process=ExistingProcess(pid,directory/'training.log')
            code=process.poll()
            if code in [None,0]:
                children[c]=process;waiting.discard(c)
            else:
                backup=directory/f'training_failed_{int(time.time())}.log'
                (directory/'training.log').rename(backup)
                print(f'Preserved failed {c} log at {backup}; scheduled retry',flush=True)
    while len(completed)<len(CONDITIONS):
        for gpu,c in enumerate(CONDITIONS):
            directory=OUT/c
            if c in waiting and available_memory(gpu)>=memory_required[c]:
                command=common+['--run_name',f'cppo-pilot-{c}-100m-seed50']
                if c=='frozen_critic':
                    command+=['--component_init_from','wandb/run-20260630_214658-m0mw4end/files/policies/9999941632','--train_component','actor']
                elif c=='bc_ppo':
                    command+=['--culture_path',memory,'--bc_steps','1000']
                elif c in ['cppo','shuffled_cppo']:
                    command+=['--culture_path',memory,'--culture_coef','.1','--culture_query_batch','512']
                    if c=='shuffled_cppo':command+=['--culture_shuffle_actions']
                env=dict(environment,CUDA_VISIBLE_DEVICES=str(gpu))
                if gpu==4:env['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
                logs[c]=(directory/'training.log').open('w')
                children[c]=subprocess.Popen(command,env=env,stdout=logs[c],stderr=subprocess.STDOUT,start_new_session=True)
                (directory/'command.json').write_text(json.dumps(dict(command=command,gpu=gpu),indent=2))
                waiting.remove(c);print(f'Launched {c} on GPU {gpu}, PID {children[c].pid}',flush=True)
            if c in children and c not in completed:
                code=children[c].poll()
                if code is None:continue
                if code!=0:
                    if c in logs:logs[c].close()
                    backup=directory/f'training_failed_{int(time.time())}.log'
                    (directory/'training.log').rename(backup)
                    retries[c]+=1
                    # Wait for a mostly empty GPU before retrying a failure.
                    # Keep monitoring other conditions instead of exiting.
                    memory_required[c]=12000
                    waiting.add(c);del children[c]
                    print(f'{c} exited {code}; preserved {backup}; queued retry until GPU {gpu} has 12 GB free',flush=True)
                    continue
                if c in logs:logs[c].close()
                log=(directory/'training.log').read_text()
                matches=re.findall(r'Run data is saved locally in (\S+)',log)
                if not matches:raise RuntimeError(f'Cannot identify W&B directory: {c}')
                run=matches[-1];rid=run.rsplit('-',1)[-1]
                manifest=dict(condition=c,local_run=run,wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{rid}',
                              gpu=gpu,seed=50,environment_budget=100000000)
                (directory/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                env=dict(environment,CUDA_VISIBLE_DEVICES=str(gpu))
                env['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
                with (directory/'evaluation.log').open('w') as f:
                    subprocess.run([PYTHON,'-u','evaluate_cppo_pilot.py','--run',run,'--out',str(directory)],
                                   env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                completed.add(c);print(f'Completed and evaluated {c}',flush=True)
        (OUT/'progress.json').write_text(json.dumps(dict(waiting=sorted(waiting),completed=sorted(completed),
            pids={c:p.pid for c,p in children.items()},retries=retries,updated_unix_time=time.time()),indent=2)+'\n')
        if len(completed)<len(CONDITIONS):time.sleep(20)
    with (OUT/'comparison.log').open('w') as f:
        subprocess.run([PYTHON,'-u','compare_cppo_pilot.py','--out',str(OUT)],
                       env=environment,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('All pilots completed; comparison.csv, learning_curves.pdf, evaluation_comparison.pdf, results.md are ready.',flush=True)
    if args.retired_supervisor_pid:
        retired=Path(f'/proc/{args.retired_supervisor_pid}/cmdline')
        if retired.exists() and b'run_cppo_100m_pilot.py' in retired.read_bytes():
            os.kill(args.retired_supervisor_pid,signal.SIGCONT)
            os.kill(args.retired_supervisor_pid,signal.SIGTERM)


if __name__=='__main__':main()

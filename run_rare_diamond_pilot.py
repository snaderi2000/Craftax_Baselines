"""Two fresh 100M rare-event runs; reuse the already completed PPO baseline."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'concept_mapping/runs/cppo_rare_diamond_100m_pilot'
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    os.chdir(ROOT)
    memory = OUT/'memory/culture.npz'
    if not memory.exists():
        raise RuntimeError('Build rare_event_culture.py memory first')
    names = ['rare_diamond', 'shuffled_rare_diamond']
    for name in names:
        directory = OUT/name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/'command.json').exists():
            raise RuntimeError(f'Refusing to overwrite {directory}')
    common = [PYTHON, '-u', 'ppo.py', '--env_name', 'Craftax-Classic-Symbolic-v1',
              '--num_envs', '128', '--total_timesteps', '1e8', '--num_steps', '64',
              '--update_epochs', '4', '--num_minibatches', '8', '--gamma', '.99',
              '--gae_lambda', '.8', '--clip_eps', '.2', '--vf_coef', '.5',
              '--ent_coef', '.01', '--lr', '.0002', '--max_grad_norm', '1.0',
              '--anneal_lr', '--seed', '50', '--save_policy',
              '--save_policy_milestones',
              '65536,100000,150000,220000,330000,500000,750000,1000000,1500000,2200000,3300000,5000000,7500000,10000000,15000000,22000000,33000000,50000000,75000000',
              '--culture_path', str(memory), '--culture_diamond_oracle',
              '--culture_coef', '1.0', '--culture_query_batch', '512',
              '--culture_threshold', '.9']
    environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       WANDB_INIT_TIMEOUT='300')
    jobs = {}
    for gpu, name in enumerate(names):
        env = dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))
        command = common + ['--run_name', f'cppo-rare-diamond-{name}-100m-seed50']
        if name.startswith('shuffled'):
            command.append('--culture_shuffle_actions')
        directory = OUT/name
        (directory/'command.json').write_text(json.dumps(dict(command=command, gpu=gpu), indent=2))
        with (directory/'training.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        jobs[name] = (process, env)
        print(f'Launched {name} GPU {gpu} PID {process.pid}', flush=True)
    completed, failures = [], {}
    while len(completed) + len(failures) < len(names):
        for name, (process, env) in jobs.items():
            if name in completed or name in failures:
                continue
            code = process.poll()
            if code is None:
                continue
            if code:
                failures[name] = f'training exit {code}'
                continue
            directory = OUT/name
            matches = re.findall(r'Run data is saved locally in (\S+)', (directory/'training.log').read_text())
            if not matches:
                failures[name] = 'Could not identify saved W&B run'
                continue
            run = matches[-1]
            manifest = dict(condition=name, local_run=run,
                            wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-", 1)[-1]}',
                            seed=50, environment_budget=100000000, memory=str(memory))
            (directory/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            with (directory/'evaluation.log').open('w') as log:
                result = subprocess.run([PYTHON, '-u', 'evaluate_diamond_opportunities.py',
                                         '--run', run, '--out', str(directory/'diamond_opportunities'),
                                         '--episodes', '512', '--seed', '2050'],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                failures[name] = f'evaluation exit {result.returncode}'
            else:
                completed.append(name)
                print(f'Completed and evaluated {name}', flush=True)
        (OUT/'progress.json').write_text(json.dumps(dict(completed=completed, failures=failures,
             pids={n:p.pid for n,(p,_) in jobs.items()}, updated_unix_time=time.time()), indent=2)+'\n')
        if len(completed) + len(failures) < len(names):
            time.sleep(20)
    if failures:
        raise RuntimeError(str(failures))
    summaries = {}
    for name in names:
        summaries[name] = json.loads((OUT/name/'diamond_opportunities/summary.json').read_text())
    original = ROOT/'concept_mapping/runs/cppo_100m_pilot/diamond_opportunities'
    for name in ['baseline', 'cppo']:
        summaries[name] = json.loads((original/name/'summary.json').read_text())
    (OUT/'comparison.json').write_text(json.dumps(summaries, indent=2)+'\n')
    print('Complete: comparison.json contains the matched diamond-opportunity results.', flush=True)


if __name__ == '__main__':
    main()

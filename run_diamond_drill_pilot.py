"""One fresh 200M mixed normal/drill PPO pilot with normal-only evaluation."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'concept_mapping/runs/diamond_skill_drill_200m_seed50'
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'command.json').exists():
        raise RuntimeError('Refusing to overwrite existing drill run')
    source = ROOT/'concept_mapping/runs/diamond_density_curriculum_200m/replications_seed50/baseline/command.json'
    original = json.loads(source.read_text())['command']
    command = original[:original.index('--run_name')] + [
        '--run_name','diamond-skill-drill-200m-seed50','--diamond_drills',
        '--diamond_drill_envs','32','--diamond_drill_horizon','64']
    gpu = 6
    free = int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'],text=True).strip())
    if free < 14000:
        raise RuntimeError(f'GPU {gpu} does not have enough free memory ({free} MiB)')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_INIT_TIMEOUT='300')
    (OUT/'command.json').write_text(json.dumps(dict(command=command,gpu=gpu),indent=2)+'\n')
    def progress(**status):
        (OUT/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
    with (OUT/'training.log').open('w') as log:
        process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        progress(state='training',gpu=gpu,pid=process.pid)
        print(f'Launched drill pilot GPU {gpu} PID {process.pid}',flush=True)
        code = process.wait()
    if code:
        progress(state='failed',error=f'training exit {code}')
        raise RuntimeError('Training failed')
    run = re.findall(r'Run data is saved locally in (\S+)',(OUT/'training.log').read_text())[-1]
    (OUT/'run_manifest.json').write_text(json.dumps(dict(local_run=run,seed=50,gpu=gpu,
        environment_budget=200000000,normal_envs=96,drill_envs=32,drill_horizon=64,
        wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-",1)[-1]}'),indent=2)+'\n')
    progress(state='evaluating',gpu=gpu,local_run=run)
    for budget,checkpoint in [(200,None),(100,100007936)]:
        evaluation = OUT/f'normal_evaluation_{budget}m'
        cmd = [PYTHON,'-u','evaluate_diamond_opportunities.py','--run',run,'--out',str(evaluation),
               '--episodes','512','--seed','2050']
        if checkpoint is not None:
            cmd += ['--checkpoint',str(checkpoint)]
        with (OUT/f'evaluation_{budget}m.log').open('w') as log:
            result = subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            progress(state='failed',error=f'{budget}M normal evaluation failed')
            raise RuntimeError('Evaluation failed')
        comparison = {'drill':json.loads((evaluation/'summary.json').read_text())}
        folder = 'normal_evaluation' if budget == 200 else 'normal_evaluation_100m'
        references = ROOT/'concept_mapping/runs/diamond_density_curriculum_200m/replications_seed50'
        for name in ['baseline','curriculum']:
            comparison[name] = json.loads((references/name/folder/'summary.json').read_text())
        (OUT/f'comparison_{budget}m.json').write_text(json.dumps(comparison,indent=2)+'\n')
    progress(state='complete',gpu=gpu,local_run=run)
    print('Completed training and normal-only evaluations.',flush=True)


if __name__ == '__main__':
    main()

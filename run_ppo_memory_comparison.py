"""One GPU worker for fresh MLP/GRU/Transformer PPO 10M comparisons."""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
STUDY=ROOT/'concept_mapping/runs/ppo_memory_comparison_10m'
BUDGET=10000000


def command_for(architecture,seed):
    trainer='ppo.py' if architecture=='mlp' else 'ppo_rnn_og.py'
    command=[PYTHON,'-u',str(ROOT/trainer),'--env_name','Craftax-Classic-Symbolic-v1',
        '--num_envs','1024','--total_timesteps',str(BUDGET),'--num_steps','64',
        '--update_epochs','4','--num_minibatches','8','--gamma','.99','--gae_lambda','.8',
        '--clip_eps','.2','--vf_coef','.5','--ent_coef','.01','--lr','.0002',
        '--max_grad_norm','1.','--anneal_lr','--seed',str(seed),'--layer_size','512',
        '--activation','tanh','--num_repeats','1','--use_optimistic_resets',
        '--optimistic_reset_ratio','16','--death_penalty','0','--save_policy',
        '--save_policy_milestones','65536,1000000,2200000,5000000,7500000',
        '--use_wandb','--wandb_project','Craftax_Baselines',
        '--run_name',f'ppo-memory-{architecture}-10m-seed{seed}']
    if architecture!='mlp':command+=['--diamond_reward','1','--memory_type',architecture,
        '--transformer_dim','256','--transformer_layers','2','--transformer_heads','4','--transformer_context','64']
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--jobs',nargs='+',required=True,help='Sequential architecture:seed jobs')
    args=parser.parse_args()
    for spec in args.jobs:
        architecture,seed=spec.split(':');seed=int(seed)
        if architecture not in ('mlp','gru','transformer'):raise ValueError(spec)
        out=STUDY/f'{architecture}_seed{seed}'
        if out.exists():raise FileExistsError(out)
        free=int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
            '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<14000:raise RuntimeError(f'GPU {args.gpu} occupied: {free} MiB free')
        out.mkdir(parents=True)
        command=command_for(architecture,seed)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),
            XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',WANDB_MODE='online',WANDB_INIT_TIMEOUT='300')
        metadata=dict(architecture=architecture,seed=seed,gpu=args.gpu,command=command,
            requested_steps=BUDGET,actual_steps=BUDGET//65536*65536,fresh_training=True,
            standard_rewards=True,death_penalty=0,diamond_reward=1,
            optimistic_resets=True,optimistic_reset_ratio=16,rollout_steps=64,num_envs=1024,
            transformer=dict(layers=2,width=256,heads=4,context_including_current=64,
                feedforward_width=1024,activation='gelu',dropout=0,pre_layer_norm=True,
                history='raw observations, recomputed under current parameters; episode-masked',
                training='exact sliding windows, rematerialized; temporal sequences preserved') if architecture=='transformer' else None,
            architecture_caveat='Existing MLP uses independent tanh branches; GRU uses shared ReLU encoder; Transformer replaces the GRU/encoder width and adds standard attention/FFN blocks. Architecture comparison, not parameter-matched memory isolation.',
            evaluation=dict(seed=15050,episodes=128,standard_resets=True,checkpoints='1M,5M,final10M'),
            source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                for p in ('ppo.py','ppo_rnn_og.py','models/actor_critic_transformer.py','evaluate_ppo_memory.py')})
        (out/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        def status(**kw):
            (out/'progress.json').write_text(json.dumps(dict(architecture=architecture,seed=seed,
                gpu=args.gpu,time=time.time(),**kw),indent=2)+'\n')
        with (out/'training.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            status(state='training',pid=process.pid)
            code=process.wait()
        matches=re.findall(r'Run data is saved locally in (\S+)',(out/'training.log').read_text())
        run=Path(matches[-1]) if matches else None
        if code:
            status(state='failed',exit_code=code,local_run=str(run));raise RuntimeError(f'{spec} training failed')
        if run is None:raise RuntimeError('No W&B run found')
        if not run.is_absolute():run=ROOT/run
        checkpoints=sorted(int(p.name) for p in (run/'files/policies').iterdir() if p.name.isdigit())
        assert checkpoints[-1]==metadata['actual_steps']
        (out/'run_manifest.json').write_text(json.dumps(dict(**metadata,local_run=str(run),
            checkpoints=checkpoints,wandb_url='https://wandb.ai/synaderi-uc-davis/Craftax_Baselines/runs/'+run.name.rsplit('-',1)[-1]),indent=2)+'\n')
        for budget in (1,5,10):
            step=min(s for s in checkpoints if s>=budget*1000000) if budget<10 else checkpoints[-1]
            status(state='evaluating',local_run=str(run),budget_millions=budget,checkpoint=step)
            with (out/f'evaluation_{budget}m.log').open('w') as log:
                code=subprocess.call([PYTHON,'-u',str(ROOT/'evaluate_ppo_memory.py'),'--run',str(run),
                    '--out',str(out/f'evaluation_{budget}m'),'--architecture',architecture,'--checkpoint',str(step)],
                    cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
            if code:
                status(state='evaluation_failed',budget_millions=budget,local_run=str(run));raise RuntimeError('Evaluation failed')
        status(state='complete',local_run=str(run))
        subprocess.run([PYTHON,str(ROOT/'summarize_ppo_memory_comparison.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()

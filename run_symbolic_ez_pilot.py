"""Persistent worker: fresh symbolic EZ-style training, then frozen evaluations."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
STUDY=ROOT/'concept_mapping/runs/symbolic_ez_10m'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--study',type=Path,default=STUDY)
    p.add_argument('--real-sequence-training',action='store_true')
    p.add_argument('--early-eval-gpu',type=int,help='Evaluate the 1M checkpoint while training continues; shared GPU evaluations are serialized')
    p.add_argument('--steps',type=int,choices=(1048576,9961472),default=9961472)
    p.add_argument('--absolute-search-q',action='store_true')
    p.add_argument('--updates-per-step',type=int,default=1)
    p.add_argument('--reanalyse-interval',type=int,default=16)
    p.add_argument('--reanalyse-fraction',type=float,default=.25)
    p.add_argument('--lr-decay-steps',type=int)
    p.add_argument('--run-name')
    args=p.parse_args();study=args.study.resolve();out=study/f'seed{args.seed}'
    if out.exists():raise FileExistsError(out)
    free=int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<14000:raise RuntimeError('GPU occupied; existing jobs will not be interrupted')
    command=[PYTHON,'-u',str(ROOT/'train_symbolic_ez.py'),'--out',str(out/'training'),
        '--seed',str(args.seed),'--steps',str(args.steps),'--wandb',
        '--updates-per-step',str(args.updates_per_step),'--reanalyse-interval',str(args.reanalyse_interval),
        '--reanalyse-fraction',str(args.reanalyse_fraction)]
    if args.real_sequence_training:command.append('--real-sequence-training')
    if args.absolute_search_q:command.append('--absolute-search-q')
    if args.lr_decay_steps is not None:command+=['--lr-decay-steps',str(args.lr_decay_steps)]
    if args.run_name:command+=['--run-name',args.run_name]
    out.mkdir(parents=True)
    metadata=dict(seed=args.seed,gpu=args.gpu,command=command,requested_interactions=1000000 if args.steps==1048576 else 10000000,
        real_sequence_training=args.real_sequence_training,early_eval_gpu=args.early_eval_gpu,
        actual_interactions=args.steps,baseline='concept_mapping/runs/ppo_memory_comparison_10m/mlp_seed'+str(args.seed),
        baseline_matched_fields=['seed','standard rewards','optimistic reset ratio16','1024 environments',f'{args.steps} interactions','frozen evaluation seed15050 and128episodes'],
        model_fields='Only symbolic observations, chosen actions, actual rewards, and done flags; no full simulator states or achievement labels.',
        upstream='EfficientZeroV2@12b9e775df1f8afb4a1dc854cf722ea84fbfb61d',
        search='vendored mctx0.0.5 Gumbel MuZero;32 simulations,8 root candidates,depth5;learned model only',
        differences_from_ezv2='See SYMBOLIC_EZ_NOTICE.md: symbolic decoder, history GRU, immediate reward, fractional root reanalysis; pilot is not a full reproduction.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('models/symbolic_ez.py','symbolic_ez_replay.py','train_symbolic_ez.py','evaluate_symbolic_ez.py')})
    (out/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',WANDB_MODE='online',WANDB_INIT_TIMEOUT='300')
    def status(**kw):(out/'progress.json').write_text(json.dumps(dict(seed=args.seed,gpu=args.gpu,time=time.time(),**kw),indent=2)+'\n')
    def evaluate(step,budget,mode,eval_environment):
        destination=out/f'evaluation_{budget}m_{mode}'
        if (destination/'summary.json').exists():return 0
        with (out/f'evaluation_{budget}m_{mode}.log').open('w') as log:
            return subprocess.call([PYTHON,'-u',str(ROOT/'evaluate_symbolic_ez.py'),
                '--checkpoint',str(out/'training/checkpoints'/str(step)),
                '--out',str(destination),'--mode',mode],
                cwd=ROOT,env=eval_environment,stdout=log,stderr=subprocess.STDOUT)
    with (out/'training.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        status(state='training',pid=process.pid)
        early_attempted=False
        while True:
            try:
                code=process.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                checkpoint=out/'training/checkpoints/1048576/config.json'
                if args.early_eval_gpu is not None and not early_attempted and checkpoint.exists():
                    early_attempted=True
                    with Path(f'/tmp/symbolic_ez_evaluation_gpu{args.early_eval_gpu}.lock').open('w') as lock:
                        fcntl.flock(lock,fcntl.LOCK_EX)
                        eval_environment=dict(environment,CUDA_VISIBLE_DEVICES=str(args.early_eval_gpu))
                        early_code=evaluate(1048576,1,'search',eval_environment)
                    (out/'early_evaluation.json').write_text(json.dumps(dict(
                        checkpoint=1048576,gpu=args.early_eval_gpu,exit_code=early_code,time=time.time()),indent=2)+'\n')
                    subprocess.run([PYTHON,str(ROOT/'summarize_symbolic_ez_pilot.py'),'--study',str(study)],cwd=ROOT,check=True)
    if code:
        status(state='failed',exit_code=code);raise RuntimeError('Training failed; inspect training.log')
    checkpoints=sorted(int(p.name) for p in (out/'training/checkpoints').iterdir() if p.name.isdigit())
    assert checkpoints[-1]==args.steps
    for budget in ((1,) if args.steps==1048576 else (1,5,10)):
        step=min(s for s in checkpoints if s>=budget*1000000) if budget<10 else checkpoints[-1]
        modes=('search','policy') if step==args.steps else ('search',)
        for mode in modes:
            status(state='evaluating',checkpoint=step,mode=mode)
            code=evaluate(step,budget,mode,environment)
            if code:
                status(state='evaluation_failed',checkpoint=step,mode=mode);raise RuntimeError('Evaluation failed')
    status(state='complete',final_checkpoint=checkpoints[-1])
    subprocess.run([PYTHON,str(ROOT/'summarize_symbolic_ez_pilot.py'),'--study',str(study)],cwd=ROOT,check=True)


if __name__=='__main__':main()

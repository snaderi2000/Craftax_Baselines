"""Compare warmed one-/multi-GPU PPO updates from one preset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from ppo_experiment_config import Experiment
from run_reward_shaped_architectures import ROOT, PYTHON, environment, write_json


CORE_SOURCES = ('ppo_rnn_og.py', 'ppo_data_parallel.py', 'ppo.py', 'wrappers.py',
                'ppo_experiment_config.py', 'run_reward_shaped_architectures.py',
                'symbolic_map_memory.py', 'models/conditioned_symbolic.py',
                'models/actor_critic.py', 'models/actor_critic_map_mlp.py', 'policy_temperature.py',
                'training_time_limit.py')


def fingerprints():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in CORE_SOURCES}


def worker(args, spec):
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    from ppo_data_parallel import compile_train
    from run_reward_shaped_architectures import inference_config

    count=args.worker_devices
    cfg=inference_config(spec.metadata(spec.architectures[0],ROOT,PYTHON))
    start=int(Path(spec.resume_from).name) if spec.resume_from else 0
    cfg.update(DATA_PARALLEL_DEVICES=count,START_TIMESTEPS=start,
        TOTAL_TIMESTEPS=start+args.updates*spec.batch_steps,
        RESUME_FROM=str(Path(spec.resume_from).resolve().parent) if spec.resume_from else None,
        RESUME_STEP=start or None,DEBUG=False,USE_WANDB=False,SAVE_POLICY=False,
        SAVE_POLICY_MILESTONES=(),NUM_REPEATS=1,TRAINING_DEADLINE=0.,
        TRAIN_ICM=False,TRAIN_COMPONENT='both',COMPONENT_INIT_FROM=None)
    before=fingerprints()
    print(json.dumps(dict(phase='compiling',devices=count,global_batch_steps=spec.batch_steps)),flush=True)
    train=compile_train(cfg)
    key=jax.random.PRNGKey(spec.seed)
    began=time.perf_counter();result=train(key);jax.block_until_ready(result)
    print(json.dumps(dict(phase='warmup_complete',seconds=time.perf_counter()-began)),flush=True)
    durations=[]
    for repeat in range(args.repeats):
        began=time.perf_counter();result=train(jax.random.fold_in(key,repeat+1));jax.block_until_ready(result)
        elapsed=time.perf_counter()-began
        assert int(np.asarray(result['runner_state'][-1]))==args.updates
        durations.append(elapsed)
        print(json.dumps(dict(phase='measured',devices=count,repeat=repeat,seconds=elapsed,
            sps=args.updates*spec.batch_steps/elapsed)),flush=True)
    state=result['runner_state'][0];obs=result['runner_state'][2]
    if not spec.resume_from:
        assert int(state.step)==args.updates*spec.ppo.update_epochs*spec.ppo.num_minibatches
    assert all(np.isfinite(np.asarray(x)).all()
               for x in jax.tree_util.tree_leaves((state.params,state.opt_state)))
    assert all(x.sharding.is_fully_replicated for x in jax.tree_util.tree_leaves(state.params))
    assert obs.shape[0]==spec.ppo.num_envs
    if count>1:
        assert len(obs.addressable_shards)==count
        assert not obs.sharding.is_fully_replicated
        assert all(s.data.shape[0]==spec.ppo.num_envs//count for s in obs.addressable_shards)
    assert before==fingerprints(),'Trainer source changed during measurement'
    result=dict(devices=count,visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
        global_num_envs=spec.ppo.num_envs,local_num_envs=spec.ppo.num_envs//count,
        num_steps=spec.ppo.num_steps,update_epochs=spec.ppo.update_epochs,
        global_minibatches=spec.ppo.num_minibatches,global_batch_steps=spec.batch_steps,
        checkpoint=spec.resume_from,updates_per_measurement=args.updates,
        measured_seconds=durations,median_sps=args.updates*spec.batch_steps/statistics.median(durations),
        parameters=sum(x.size for x in jax.tree_util.tree_leaves(state.params)),
        optimizer_steps=int(state.step),
        finite_parameters_and_optimizer=True,replicated_parameters=True,
        observation_sharding=str(obs.sharding),jax_version=jax.__version__,jaxlib_version=jaxlib.__version__,
        source_sha256=before,
        caveat='Short warmed full-update benchmark; compilation, checkpoint I/O, and W&B logging excluded. No learning-quality claim.')
    write_json(args.out/f'devices_{count}.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',default='5,6');p.add_argument('--device-counts',default='1,2')
    p.add_argument('--updates',type=int,default=4);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--worker-devices',type=int,help=argparse.SUPPRESS)
    args=p.parse_args();spec=Experiment.load(args.config)
    if len(spec.architectures)!=1 or spec.architectures[0].name not in ('gru','spatial_gru','spatial_map_gru','map_mlp'):
        raise ValueError('Benchmark requires one GRU or feedforward map_mlp architecture')
    if args.updates<1 or args.repeats<1:raise ValueError('Updates and repeats must be positive')
    if args.worker_devices is not None:
        worker(args,spec);return
    gpus=tuple(map(int,args.gpus.split(',')));counts=tuple(map(int,args.device_counts.split(',')))
    if len(set(gpus))!=len(gpus) or any(n not in (1,2,4) or n>len(gpus) for n in counts):
        raise ValueError('Provide unique GPUs and supported device counts fitting that list')
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    from cap_reward_shaped_runs import try_evaluation_lock
    from run_ppo_study import require_free
    locks=[]
    try:
        for gpu in sorted(gpus):
            lock=try_evaluation_lock(gpu)
            if lock is None:raise RuntimeError(f'GPU {gpu} is reserved')
            locks.append(lock)
        require_free(gpus);before=fingerprints();results=[]
        write_json(args.out/'manifest.json',dict(state='running',started_at=time.time(),config=spec.to_dict(),gpus=gpus,counts=counts))
        for count in counts:
            command=[PYTHON,'-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
                '--out',str(args.out),'--worker-devices',str(count),'--updates',str(args.updates),'--repeats',str(args.repeats)]
            with (args.out/f'devices_{count}.log').open('w') as log:
                process=subprocess.run(command,cwd=ROOT,env=environment(gpus[:count]),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            if process.returncode:raise RuntimeError(f'{count}-GPU benchmark failed; inspect devices_{count}.log')
            results.append(json.loads((args.out/f'devices_{count}.json').read_text()))
            assert before==fingerprints(),'Source changed between benchmark cases'
        reference=next((r['median_sps'] for r in results if r['devices']==1),None)
        if reference:
            for r in results:r['speedup']=r['median_sps']/reference
        write_json(args.out/'comparison.json',dict(state='complete',finished_at=time.time(),results=results))
        print(json.dumps(results,indent=2),flush=True)
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':main()

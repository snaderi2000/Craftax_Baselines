"""Evaluate a completed pilot using a fresh common evaluation RNG seed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from orbax.checkpoint import CheckpointManager,CheckpointManagerOptions,PyTreeCheckpointer
from craftax.craftax_env import make_craftax_env_from_name
from models.actor_critic import ActorCritic
from wrappers import LogWrapper,AutoResetEnvWrapper,BatchEnvWrapper


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--seed',type=int,default=1050)
    p.add_argument('--checkpoint',type=int,default=None)
    args=p.parse_args()
    run=Path(args.run)
    manager=CheckpointManager(str(run/'files/policies'),PyTreeCheckpointer(),CheckpointManagerOptions(create=False,max_to_keep=1))
    step=args.checkpoint if args.checkpoint is not None else max(manager.all_steps())
    if step not in manager.all_steps():
        raise ValueError(f'Checkpoint {step} unavailable')
    params=manager.restore(step)['params']
    env=BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',False))),32)
    ep=env.default_params;net=ActorCritic(17,512)
    key=jax.random.PRNGKey(args.seed); key,rkey=jax.random.split(key)
    obs,state=env.reset(rkey,ep)
    def transition(carry,_):
        key,obs,state=carry
        key,ak,ek=jax.random.split(key,3)
        pi,_=net.apply(params,obs)
        obs,state,reward,done,info=env.step(ek,state,pi.sample(seed=ak),ep)
        return (key,obs,state),(done,info)
    block=jax.jit(lambda carry:jax.lax.scan(transition,carry,None,length=64))
    carry=(key,obs,state);rows=[];eval_steps=0
    quotas=np.full(32,args.episodes//32,np.int32)
    quotas[:args.episodes%32]+=1
    completed=np.zeros(32,np.int32)
    while len(rows)<args.episodes:
        carry,(done,info)=block(carry)
        done=np.asarray(done);info=jax.tree_util.tree_map(np.asarray,info)
        for t,e in np.argwhere(done):
            if completed[e]>=quotas[e]:continue
            row={k:float(v[t,e]) for k,v in info.items() if np.asarray(v).ndim==2
                 and (k.startswith('Achievements') or k.startswith('achievements') or
                      k in ['returned_episode_returns','returned_episode_lengths'])}
            rows.append(row)
            completed[e]+=1
        eval_steps+=64*32
        if eval_steps>args.episodes*100000:
            raise RuntimeError('Evaluation exceeded safety limit')
    rows=rows[:args.episodes]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'evaluation_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    summary.update(episodes=len(rows),checkpoint=step,evaluation_seed=args.seed)
    summary['median_total_return']=float(np.median([r['returned_episode_returns'] for r in rows]))
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

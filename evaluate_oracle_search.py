# SPDX-License-Identifier: GPL-3.0-only
"""Frozen full-episode evaluation with exact search or its distilled policy."""
import argparse,csv,json,time
from pathlib import Path
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement,BlockType
from models.oracle_search import OraclePolicyValue,oracle_search


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=('search','policy'),default='search')
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--seed',type=int,default=15050)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint).resolve();cfg=json.loads((checkpoint/'config.json').read_text())
    weights=ocp.PyTreeCheckpointer().restore(str(checkpoint))['params']
    model=OraclePolicyValue(cfg['width']);env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    batch=args.batch
    assert args.episodes%batch==0,'Use a batch dividing the episode count so transition counts are exact'
    @jax.jit
    def evaluate(ids):
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed),i))(ids)
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        obs,state=jax.vmap(env.reset,in_axes=(0,None))(split[:,0],env.default_params)
        carry=(jnp.asarray(0),obs,state,jnp.zeros((batch,cfg['width'])),split[:,1],
            jnp.zeros(batch,jnp.int32),jnp.zeros(batch),jnp.ones(batch,bool),jnp.zeros(batch,bool),
            jnp.zeros(batch),jnp.zeros(batch,jnp.int32),jnp.zeros((batch,len(Achievement)),bool),
            jnp.zeros(batch,bool))
        def body(c):
            t,obs,state,h,keys,pa,pr,resets,finished,returns,lengths,ach,lava=c
            split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
            h=model.apply(weights,h,obs,pa,pr,resets,method=model.represent)
            if args.mode=='search':
                sk=jax.random.fold_in(jax.random.PRNGKey(args.seed+90000),ids[0]+t)
                actions,_,_=oracle_search(model,weights,env,state,h,sk,cfg['simulations'],cfg['max_depth'],cfg['candidates'])
            else:
                logits,_=model.apply(weights,h,method=model.prediction)
                actions=jax.vmap(lambda p,k:jax.random.categorical(k,p))(logits,split[:,1])
            no,ns,r,d,_=jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],state,actions,env.default_params)
            active=~finished
            tile=ns.map[jnp.arange(batch),ns.player_position[:,0],ns.player_position[:,1]]
            return (t+1,no,ns,h,split[:,0],actions,r,jnp.zeros(batch,bool),finished|d,
                returns+active*r,lengths+active,jnp.where(active[:,None],ns.achievements,ach),
                jnp.where(active,tile==BlockType.LAVA.value,lava))
        carry=jax.lax.while_loop(lambda c:(c[0]<env.default_params.max_timesteps)&~c[8].all(),body,carry)
        return carry[0],carry[8:]
    start=time.time();rows=[];executed_steps=0
    for offset in range(0,args.episodes,batch):
        ticks,(finished,returns,lengths,ach,lava)=jax.device_get(evaluate(jnp.arange(offset,offset+batch)))
        # Compiled batched evaluation also calls the simulator for finished slots.
        executed_steps+=int(ticks)*batch
        for i in range(batch):rows.append(dict(episode=offset+i,total_reward=float(returns[i]),length=int(lengths[i]),
            terminated=bool(finished[i]),lava=bool(lava[i]),**{a.name.lower():int(ach[i,a.value]) for a in Achievement}))
        print(f'Evaluated {len(rows)}/{args.episodes}',flush=True)
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    active=sum(r['length'] for r in rows)
    summary=dict(checkpoint=str(checkpoint),mode=args.mode,episodes=len(rows),evaluation_seed=args.seed,
        standard_resets=True,privileged_simulator_search=args.mode=='search',
        mean_return=float(np.mean([r['total_reward'] for r in rows])),
        mean_length=float(np.mean([r['length'] for r in rows])),
        mean_achievements=float(np.mean([sum(r[a.name.lower()] for a in Achievement) for r in rows])),
        achievement_success_pct={a.name.lower():float(np.mean([r[a.name.lower()] for r in rows])*100) for a in Achievement},
        active_episode_transitions=active,executed_environment_transitions=executed_steps,
        executed_search_transitions=executed_steps*cfg['simulations'] if args.mode=='search' else 0,
        seconds=time.time()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

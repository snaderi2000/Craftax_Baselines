"""Frozen ordinary-environment evaluation of the PPO sample-efficiency screen."""
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
from models.conditioned_symbolic import make_network


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--seed',type=int,default=15050);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint).resolve();cfg=json.loads((checkpoint/'config.json').read_text())
    params=ocp.PyTreeCheckpointer().restore(str(checkpoint))['params'];net=make_network(cfg['architecture'])
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False);batch=args.batch
    assert args.episodes%batch==0
    @jax.jit
    def evaluate(ids):
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed),i))(ids)
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        obs,state=jax.vmap(env.reset,in_axes=(0,None))(split[:,0],env.default_params)
        initial=(jnp.asarray(0),obs,state,split[:,1],jnp.zeros(batch,bool),jnp.zeros(batch),
            jnp.zeros(batch,jnp.int32),jnp.zeros((batch,len(Achievement)),bool),jnp.zeros(batch,bool))
        def step(c):
            t,obs,state,keys,finished,returns,lengths,ach,lava=c
            split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
            pi,_=net.apply(params,obs)
            action=jax.vmap(lambda p,k:jax.random.categorical(k,p))(pi.logits,split[:,1])
            no,ns,r,d,_=jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],state,action,env.default_params)
            active=~finished
            tile=ns.map[jnp.arange(batch),ns.player_position[:,0],ns.player_position[:,1]]
            return (t+1,no,ns,split[:,0],finished|d,returns+active*r,lengths+active,
                jnp.where(active[:,None],ns.achievements,ach),jnp.where(active,tile==BlockType.LAVA.value,lava))
        return jax.lax.while_loop(lambda c:(c[0]<env.default_params.max_timesteps)&~c[4].all(),step,initial)
    rows=[];executed=0;start=time.time()
    for offset in range(0,args.episodes,batch):
        t,_,_,_,finished,returns,lengths,ach,lava=jax.device_get(evaluate(jnp.arange(offset,offset+batch)))
        executed+=int(t)*batch
        for i in range(batch):rows.append(dict(episode=offset+i,total_reward=float(returns[i]),length=int(lengths[i]),
            terminated=bool(finished[i]),lava=bool(lava[i]),**{a.name.lower():int(ach[i,a.value]) for a in Achievement}))
        print(f'Evaluated {len(rows)}/{args.episodes}',flush=True)
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rates={a.name.lower():float(np.mean([r[a.name.lower()] for r in rows])) for a in Achievement}
    result=dict(checkpoint=str(checkpoint),architecture=cfg['architecture'],sil_coef=cfg['sil_coef'],
        training_seed=cfg['seed'],training_steps=int(checkpoint.name),planned_training_steps=cfg['steps'],
        episodes=len(rows),evaluation_seed=args.seed,
        standard_resets=True,search=False,mean_return=float(np.mean([r['total_reward'] for r in rows])),
        mean_length=float(np.mean([r['length'] for r in rows])),
        mean_achievements=float(np.mean([sum(r[a.name.lower()] for a in Achievement) for r in rows])),
        achievement_success_pct={k:v*100 for k,v in rates.items()},
        crafter_score=float((np.exp(np.mean(np.log1p(100*np.array(list(rates.values()))))))-1),
        evaluation_executed_transitions=executed,seconds=time.time()-start)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

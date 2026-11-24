# SPDX-License-Identifier: GPL-3.0-only
"""Matched frozen-policy evaluation, with/without learned-model search."""
import argparse,csv,json,time
from pathlib import Path
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from models.symbolic_ez import SymbolicEZ,search,scalar_decode,MAP_DIM
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement,BlockType


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--seed',type=int,default=15050)
    p.add_argument('--mode',choices=['search','policy'],default='search')
    p.add_argument('--batch',type=int,default=32)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    checkpoint=Path(args.checkpoint);cfg=json.loads((checkpoint/'config.json').read_text())
    weights=ocp.PyTreeCheckpointer().restore(str(checkpoint.resolve()))['params']
    model=SymbolicEZ(cfg['width']);env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    ep=env.default_params;batch=args.batch
    @jax.jit
    def evaluate(ids):
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed),i))(ids)
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        obs,state=jax.vmap(env.reset,in_axes=(0,None))(split[:,0],ep)
        initial=(jnp.asarray(0),obs,state,jnp.zeros((batch,cfg['width'])),split[:,1],
            jnp.zeros(batch,jnp.int32),jnp.zeros(batch),jnp.ones(batch,bool),jnp.zeros(batch,bool),
            jnp.zeros(batch),jnp.zeros(batch,jnp.int32),jnp.zeros((batch,len(Achievement)),bool),jnp.zeros(batch,bool),jnp.zeros(16))
        def cond(c):return (c[0]<ep.max_timesteps)&~c[8].all()
        def body(c):
            t,obs,state,h,keys,pa,pr,resets,finished,returns,lengths,ach,lava,diagnostics=c
            split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
            h=model.apply(weights,h,obs,pa,pr,resets,method=model.represent)
            if args.mode=='search':
                key=jax.random.fold_in(jax.random.PRNGKey(args.seed+90000),ids[0]+t)
                actions,_,_=search(model,weights,h,key,cfg['simulations'],absolute_q=cfg.get('absolute_search_q',False))
            else:
                pi,_=model.apply(weights,h,method=model.prediction)
                actions=jax.vmap(lambda logits,k:jax.random.categorical(k,logits))(pi,split[:,1])
            predicted,r,cont=model.apply(weights,h,actions,method=model.dynamics)
            decoded=model.apply(weights,predicted,method=model.observation)
            next_obs,state,reward,done,_=jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],state,actions,ep)
            active=~finished
            returns+=active*reward;lengths+=active
            ach=jnp.where(active[:,None],state.achievements,ach)
            tile=state.map[jnp.arange(batch),state.player_position[:,0],state.player_position[:,1]]
            lava=jnp.where(active,tile==BlockType.LAVA.value,lava)
            counts=jnp.rint(obs[:,MAP_DIM:MAP_DIM+16]*10)
            actual=jnp.rint(next_obs[:,MAP_DIM:MAP_DIM+16]*10).clip(0,9)
            predicted_counts=decoded[:,MAP_DIM:MAP_DIM+160].reshape(batch,16,10).argmax(-1)
            changed=(actual!=counts)&active[:,None]
            rw=scalar_decode(r);reward_event=(jnp.abs(reward)>.05)&active
            terminal=done&active;positive=(reward>.05)&active;negative=(reward<-.05)&active
            diagnostics+=jnp.array([active.sum(),(jnp.abs(rw-reward)*active).sum(),reward_event.sum(),
                ((jnp.sign(rw)==jnp.sign(reward))&reward_event).sum(),changed.sum(),
                ((predicted_counts==actual)&changed).sum(),
                ((jax.nn.sigmoid(cont)>.5)==~done).astype(jnp.float32).dot(active),
                (jnp.square(rw-reward)*active).sum(),terminal.sum(),
                ((1-jax.nn.sigmoid(cont))*terminal).sum(),
                ((jax.nn.sigmoid(cont)<.5)&terminal).sum(),
                (predicted_counts[:,12]*terminal).sum(),positive.sum(),(rw*positive).sum(),
                negative.sum(),(rw*negative).sum()])
            return t+1,next_obs,state,h,split[:,0],actions,reward,jnp.zeros(batch,bool),finished|done,returns,lengths,ach,lava,diagnostics
        final=jax.lax.while_loop(cond,body,initial)
        return final[8:]
    rows=[];diagnostics=np.zeros(16);start=time.time()
    for offset in range(0,args.episodes,batch):
        done,returns,lengths,ach,lava,d=jax.device_get(evaluate(jnp.arange(offset,offset+batch)))
        diagnostics+=d
        for i in range(min(batch,args.episodes-offset)):
            rows.append(dict(episode=offset+i,total_reward=float(returns[i]),length=int(lengths[i]),
                terminated=bool(done[i]),lava=bool(lava[i]),**{a.name.lower():int(ach[i,a.value]) for a in Achievement}))
        print(f'Evaluated {len(rows)}/{args.episodes} episodes',flush=True)
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    d=diagnostics
    summary=dict(checkpoint=str(checkpoint),mode=args.mode,episodes=len(rows),evaluation_seed=args.seed,
        standard_resets=True,mean_return=float(np.mean([r['total_reward'] for r in rows])),
        mean_length=float(np.mean([r['length'] for r in rows])),
        mean_achievements=float(np.mean([sum(r[a.name.lower()] for a in Achievement) for r in rows])),
        achievement_success_pct={a.name.lower():float(np.mean([r[a.name.lower()] for r in rows])*100) for a in Achievement},
        model_diagnostics=dict(transitions=int(d[0]),reward_mae=float(d[1]/max(1,d[0])),
            reward_event_transitions=int(d[2]),reward_sign_accuracy_on_events=float(d[3]/max(1,d[2])),
            changed_inventory_intrinsic_fields=int(d[4]),accuracy_on_changed_fields=float(d[5]/max(1,d[4])),
            continuation_accuracy=float(d[6]/max(1,d[0])),reward_mse=float(d[7]/max(1,d[0])),
            terminal_transitions=int(d[8]),
            mean_termination_probability_on_terminal=float(d[9]/d[8]) if d[8] else None,
            termination_recall_at_half=float(d[10]/d[8]) if d[8] else None,
            mean_predicted_health_on_terminal=float(d[11]/d[8]) if d[8] else None,
            positive_reward_events=int(d[12]),
            mean_predicted_reward_on_positive=float(d[13]/d[12]) if d[12] else None,
            negative_reward_events=int(d[14]),
            mean_predicted_reward_on_negative=float(d[15]/d[14]) if d[14] else None),
        seconds=time.time()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

"""Paired frozen-policy continuations from the first out-of-view remembered resource opportunity."""
import argparse
import csv
from collections import deque
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_reward_shaped_ppo import jax, jnp, ocp, make_craftax_env_from_name
import numpy as np
from craftax.craftax_classic.constants import Achievement
from models.actor_critic_map_mlp import map_mlp_network
from symbolic_map_memory import with_map_memory
from fixed_world_seeds import read_world_seeds
from cap_reward_shaped_runs import try_evaluation_lock

MOVES = ((0,-1),(0,1),(-1,0),(1,0))


def route(terrain,pos,codes):
    """Choose a reachable target by shortest known walk to an adjacent square."""
    walk=np.isin(terrain,[3,7,8,14])
    origin=tuple(pos)
    q=deque([origin]);seen={origin};target=None
    while q and target is None:
        r,c=q.popleft()
        for dr,dc in MOVES:
            p=(r+dr,c+dc)
            if not (0<=p[0]<64 and 0<=p[1]<64):continue
            if terrain[p] in codes:
                target=p;break
            if walk[p] and p not in seen:
                seen.add(p);q.append(p)
    commands=np.zeros((64,64),np.int8)
    if target is None:return commands,(-100,-100)
    r,c=target
    edges=[(r+dr,c+dc) for dr,dc in MOVES if 0<=r+dr<64 and 0<=c+dc<64 and walk[r+dr,c+dc]]
    q=deque(edges);seen=set(edges)
    while q:
        r,c=q.popleft()
        for i,(dr,dc) in enumerate(MOVES):
            p=(r-dr,c-dc)
            if 0<=p[0]<64 and 0<=p[1]<64 and walk[p] and p not in seen:
                commands[p]=i+1;seen.add(p);q.append(p)
    return commands,target


def run(args):
    if str(args.gpu)!=os.getenv('CUDA_VISIBLE_DEVICES'):raise ValueError('GPU mismatch')
    lock=try_evaluation_lock(args.gpu)
    if lock is None:raise RuntimeError('Probe GPU reserved')
    cfg=json.loads((args.teacher/'inference_config.json').read_text())
    checkpoint=Path(json.loads((args.teacher/'progress.json').read_text())['final_checkpoint'])
    net=map_mlp_network(cfg)
    env=with_map_memory(make_craftax_env_from_name(cfg['ENV_NAME'],False),cfg)
    weights=ocp.PyTreeCheckpointer().restore(str(checkpoint/'default' if (checkpoint/'default').exists() else checkpoint))['params']
    worlds=np.asarray(read_world_seeds(args.worlds),np.uint32)
    args.out.mkdir(exist_ok=False,parents=True)
    def tree_where(mask,new,old):
        return jax.tree_util.tree_map(lambda a,b:jnp.where(mask.reshape((len(mask),)+(1,)*(a.ndim-1)),a,b),new,old)
    def keys_for(ids,seed):
        return jax.vmap(lambda idx:jax.random.fold_in(jax.random.PRNGKey(seed),idx))(ids)
    def advance(s,k,a):
        parts=jax.vmap(lambda x:jax.random.split(x,2))(k)
        obs,n,_,done,_=jax.vmap(env.step,in_axes=(0,0,0,None))(parts[:,0],s,a,env.default_params)
        return obs,n,parts[:,1],done
    def act(obs,k):
        pi,_=net.apply(weights,obs)
        return jax.vmap(lambda p,x:jax.random.categorical(jax.random.fold_in(x,71),p))(pi.logits,k)
    rows=[]; counts={}
    for kind in ('diamond','plant'):
        codes=(11,) if kind=='diamond' else (16,17)
        achievement=(Achievement.COLLECT_DIAMOND if kind=='diamond' else Achievement.EAT_PLANT).value
        @jax.jit
        def screen(world,ids):
            k=keys_for(ids,983051)
            obs,s=jax.vmap(env.reset,in_axes=(0,None))(jax.vmap(jax.random.PRNGKey)(world),env.default_params)
            # Keep a lane at its first eligible pre-action state.
            def advance_screen(c):
                t,o,s,k,stop,found=c
                visible=o[:,:1323].reshape(len(ids),7,9,21)
                if kind=='diamond':
                    known=(s.terrain==11).any((1,2)); local=visible[:,:,:,10].any((1,2)); ready=s.env_state.inventory.iron_pickaxe>0
                else:
                    known=((s.terrain==16)|(s.terrain==17)).any((1,2)); local=visible[:,:,:,15:17].any((1,2,3))
                    ready=s.env_state.achievements[:,Achievement.PLACE_PLANT.value]
                hit=~stop & known & ~local & ready & ~s.env_state.achievements[:,achievement]
                no,ns,nk,done=advance(s,k,act(o,k))
                keep=stop|hit
                return (t+1,tree_where(keep,o,no),tree_where(keep,s,ns),tree_where(keep,k,nk),
                        keep|done|ns.env_state.achievements[:,achievement],found|hit)
            return jax.lax.while_loop(lambda c:(c[0]<3072)&~c[4].all(),advance_screen,
                        (0,obs,s,k,jnp.zeros(len(ids),bool),jnp.zeros(len(ids),bool)))
        @jax.jit
        def branch(o,s,k,commands,target,guided):
            def step(c,t):
                o,s,k,done,near,reward_step,dead,steered=c
                pos=s.position;close=(abs(pos-target).sum(-1)==1)
                action=act(o,k)
                move=commands[jnp.arange(pos.shape[0]),pos[:,0],pos[:,1]]
                use=guided&(near<0)&~close&(t<128)&(move>0)&~done
                no,ns,nk,d=advance(s,k,jnp.where(use,move,action))
                close_after=(abs(ns.position-target).sum(-1)==1)
                near=jnp.where(~done & (near<0)&(close|close_after),t+jnp.where(close,0,1),near)
                reached=ns.env_state.achievements[:,achievement]
                reward_step=jnp.where(~done & reached & (reward_step<0),t+1,reward_step)
                death=(ns.env_state.player_health<=0)|(ns.env_state.map[jnp.arange(pos.shape[0]),ns.position[:,0],ns.position[:,1]]==14)
                return (no,ns,nk,done|d,near,reward_step,dead|(~done&d&death),steered+use),None
            b=len(k); blanks=jnp.full(b,-1,jnp.int32)
            output,_=jax.lax.scan(step,(o,s,k,jnp.zeros(b,bool),blanks,blanks,jnp.zeros(b,bool),jnp.zeros(b,jnp.int32)),jnp.arange(256))
            return output[4:]
        counts[kind]=dict(screened=0,eligible=0,routable=0)
        for offset in range(0,len(worlds),64):
            ids=np.arange(offset,min(offset+64,len(worlds)))
            _,o,s,_,_,hit=screen(jnp.asarray(worlds[ids]),jnp.asarray(ids))
            hit=np.asarray(hit); terrain,pos=jax.device_get((s.terrain,s.position))
            routes=[route(m,p,codes) if h else (np.zeros((64,64),np.int8),(-100,-100)) for m,p,h in zip(terrain,pos,hit)]
            commands=jnp.asarray(np.stack([x[0] for x in routes]))
            target=np.asarray([x[1] for x in routes]); valid=target[:,0]>=0
            counts[kind]['screened']+=len(ids);counts[kind]['eligible']+=int(hit.sum());counts[kind]['routable']+=int(valid.sum())
            if valid.any():
                for rep in range(4):
                    k=keys_for(ids,983101+rep)
                    for guided in (False,True):
                        near,collected,dead,steered=jax.device_get(branch(o,s,k,commands,jnp.asarray(target),guided))
                        for i in np.flatnonzero(valid):
                            rows.append(dict(resource=kind,world_seed=int(worlds[ids[i]]),attempt=rep,guided=guided,
                              snapshot_step=int(s.elapsed[i]),target_row=int(target[i,0]),target_col=int(target[i,1]),
                              adjacent_step=int(near[i]),achievement_step=int(collected[i]),died=bool(dead[i]),guided_actions=int(steered[i])))
            print(kind,counts[kind],flush=True)
            with (args.out/'pairs.csv').open('w') as f:
                if rows:
                    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            (args.out/'screening.json').write_text(json.dumps(counts,indent=2)+'\n')
    (args.out/'complete.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),counts=counts,rows=len(rows)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher','worlds','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    run(p.parse_args())

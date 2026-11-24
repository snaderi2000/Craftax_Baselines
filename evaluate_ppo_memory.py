"""Seed-matched standard-environment evaluation for MLP, GRU, and Transformer PPO."""
import argparse,csv,json,time
from pathlib import Path
from concept_mapping.render_ppo_episodes import _load_wandb_config
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from models.actor_critic import ActorCritic
from models.actor_critic_transformer import ActorCriticTransformer
from ppo_rnn_og import ActorCriticRNN,ScannedRNN
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement,BlockType


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--architecture',choices=['mlp','gru','transformer'],required=True)
    p.add_argument('--checkpoint',type=int,required=True)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--seed',type=int,default=15050)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg=_load_wandb_config(args.run)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    ep=env.default_params
    params=ocp.PyTreeCheckpointer().restore(str(Path(args.run).resolve()/'files/policies'/str(args.checkpoint)/'default'))['params']
    architecture=args.architecture
    net=(ActorCritic(17,cfg['LAYER_SIZE']) if architecture=='mlp' else
         ActorCriticRNN(17,cfg) if architecture=='gru' else ActorCriticTransformer(17,cfg))
    batch=32
    def memory():
        if architecture=='mlp':return jnp.zeros((batch,1))
        if architecture=='gru':return ScannedRNN.initialize_carry(batch,cfg['LAYER_SIZE'])
        return ActorCriticTransformer.initialize_carry(batch,cfg['TRANSFORMER_CONTEXT'],1345)
    @jax.jit
    def evaluate(ids):
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed),i))(ids)
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        obs,state=jax.vmap(env.reset,in_axes=(0,None))(split[:,0],ep)
        # Slots are whole independent episodes, with no reset distribution bias.
        initial=(jnp.asarray(0),obs,state,memory(),split[:,1],jnp.zeros(batch,bool),
                 jnp.zeros(batch),jnp.zeros(batch,jnp.int32),jnp.zeros((batch,len(Achievement)),bool),jnp.zeros(batch,bool))
        def cond(c):return (c[0]<ep.max_timesteps)&~c[5].all()
        def body(c):
            t,obs,state,h,keys,finished,returns,lengths,ach,lava=c
            split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
            if architecture=='mlp':pi,_=net.apply(params,obs)
            else:
                h,dist,_=net.apply(params,h,(obs[None],jnp.zeros((1,batch),bool)))
                pi=type(dist)(logits=dist.logits[0])
            actions=jax.vmap(lambda logits,k:jax.random.categorical(k,logits))(pi.logits,split[:,1])
            obs,state,reward,done,_=jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],state,actions,ep)
            active=~finished
            returns=returns+active*reward;lengths=lengths+active
            ach=jnp.where(active[:,None],state.achievements,ach)
            tile=state.map[jnp.arange(batch),state.player_position[:,0],state.player_position[:,1]]
            lava=jnp.where(active,tile==BlockType.LAVA.value,lava)
            return t+1,obs,state,h,split[:,0],finished|done,returns,lengths,ach,lava
        final=jax.lax.while_loop(cond,body,initial)
        return final[5:]
    start=time.time();rows=[]
    for offset in range(0,args.episodes,batch):
        result=jax.device_get(evaluate(jnp.arange(offset,offset+batch)))
        done,returns,lengths,ach,lava=result
        for i in range(min(batch,args.episodes-offset)):
            rows.append(dict(episode=offset+i,total_reward=float(returns[i]),length=int(lengths[i]),
                terminated=bool(done[i]),lava=bool(lava[i]),
                **{a.name.lower():int(ach[i,a.value]) for a in Achievement}))
        print(f'Evaluated {len(rows)}/{args.episodes} episodes',flush=True)
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(architecture=architecture,run=args.run,checkpoint=args.checkpoint,
        episodes=len(rows),evaluation_seed=args.seed,standard_resets=True,
        max_steps=ep.max_timesteps,mean_return=float(np.mean([r['total_reward'] for r in rows])),
        mean_length=float(np.mean([r['length'] for r in rows])),
        achievement_success_pct={a.name.lower():float(np.mean([r[a.name.lower()] for r in rows])*100) for a in Achievement},
        mean_achievements=float(np.mean([sum(r[a.name.lower()] for a in Achievement) for r in rows])),
        seconds=time.time()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

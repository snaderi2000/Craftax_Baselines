# SPDX-License-Identifier: GPL-3.0-only
"""Evaluate a frozen PPO checkpoint with exact-simulator MCTS; no training."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from concept_mapping.render_ppo_episodes import _load_wandb_config
from models.actor_critic import ActorCritic
from models.ppo_oracle_search import search
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement,BlockType


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--checkpoint',type=int,default=1048576)
    p.add_argument('--episodes',type=int,default=128);p.add_argument('--seed',type=int,default=15050)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--simulations',type=int,default=32)
    p.add_argument('--max-depth',type=int,default=5);p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--gumbel-scale',type=float,default=1.)
    args=p.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assert args.episodes%args.batch==0
    config=_load_wandb_config(args.run)
    network=ActorCritic(17,config['LAYER_SIZE'],config.get('ACTIVATION','tanh'))
    checkpoint=Path(args.run).resolve()/'files/policies'/str(args.checkpoint)/'default'
    params=ocp.PyTreeCheckpointer().restore(str(checkpoint))['params']
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False);batch=args.batch
    metric_names=('root_actions_visited','max_depth_reached','hit_depth_limit','mean_node_depth',
        'target_entropy','differs_from_greedy_ppo','root_visited_q_range')
    metadata=dict(vars(args),run=str(Path(args.run).resolve()),out=str(out),training=False,
        frozen_ppo_policy_and_value=True,privileged_full_simulator_search=True,ordinary_rewards=True,
        stochastic_search='one sampled next state per expanded edge, not chance-node search',
        q_rescale=False,source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ('models/ppo_oracle_search.py','models/actor_critic.py','evaluate_ppo_mcts.py')})
    (out/'experiment_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    @jax.jit
    def evaluate(ids):
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(args.seed),i))(ids)
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        obs,state=jax.vmap(env.reset,in_axes=(0,None))(split[:,0],env.default_params)
        carry=(jnp.asarray(0),obs,state,split[:,1],jnp.zeros(batch,bool),jnp.zeros(batch),
            jnp.zeros(batch,jnp.int32),jnp.zeros((batch,len(Achievement)),bool),jnp.zeros(batch,bool),
            jnp.zeros((len(metric_names),batch)),jnp.zeros(batch,jnp.int32))
        def body(c):
            t,obs,state,keys,finished,returns,lengths,ach,lava,stats,action_counts=c
            split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
            sk=jax.random.fold_in(jax.random.PRNGKey(args.seed+90000),ids[0]+t)
            actions,_,diagnostics=search(network,params,env,state,obs,sk,
                args.simulations,args.max_depth,args.candidates,args.gumbel_scale)
            no,ns,r,d,_=jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],state,actions,env.default_params)
            active=~finished
            tile=ns.map[jnp.arange(batch),ns.player_position[:,0],ns.player_position[:,1]]
            stats+=jnp.stack([diagnostics[k] for k in metric_names])*active[None]
            return (t+1,no,ns,split[:,0],finished|d,returns+active*r,lengths+active,
                jnp.where(active[:,None],ns.achievements,ach),jnp.where(active,tile==BlockType.LAVA.value,lava),stats,
                action_counts+active*(actions==5))
        return jax.lax.while_loop(lambda c:(c[0]<env.default_params.max_timesteps)&~c[4].all(),body,carry)
    start=time.time();rows=[];executed=0;totals=np.zeros(len(metric_names))
    for offset in range(0,args.episodes,batch):
        result=jax.device_get(evaluate(jnp.arange(offset,offset+batch)))
        ticks,_,_,_,finished,returns,lengths,ach,lava,stats,do_count=result
        executed+=int(ticks)*batch;totals+=stats.sum(-1)
        for i in range(batch):
            row=dict(episode=offset+i,total_reward=float(returns[i]),length=int(lengths[i]),
                terminated=bool(finished[i]),lava=bool(lava[i]),do_actions=int(do_count[i]),
                **{a.name.lower():int(ach[i,a.value]) for a in Achievement},
                **{k:float(stats[j,i]/max(1,lengths[i])) for j,k in enumerate(metric_names)})
            rows.append(row)
        progress=dict(state='evaluating',episodes=len(rows),mean_return=float(np.mean([r['total_reward'] for r in rows])),
            elapsed_seconds=time.time()-start)
        (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
    with (out/'episodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    active=sum(r['length'] for r in rows)
    summary=dict(run=args.run,checkpoint=args.checkpoint,episodes=args.episodes,evaluation_seed=args.seed,
        standard_resets=True,simulations=args.simulations,max_depth=args.max_depth,candidates=args.candidates,
        gumbel_scale=args.gumbel_scale,mean_return=float(np.mean([r['total_reward'] for r in rows])),
        mean_length=float(np.mean([r['length'] for r in rows])),
        mean_achievements=float(np.mean([sum(r[a.name.lower()] for a in Achievement) for r in rows])),
        achievement_success_pct={a.name.lower():float(np.mean([r[a.name.lower()] for r in rows])*100) for a in Achievement},
        search_diagnostics={k:float(totals[i]/max(1,active)) for i,k in enumerate(metric_names)},
        active_episode_transitions=active,executed_environment_transitions=executed,
        executed_search_transitions=executed*args.simulations,seconds=time.time()-start)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'progress.json').write_text(json.dumps(dict(state='complete',**summary),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

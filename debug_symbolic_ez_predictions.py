"""Frozen-checkpoint audit of consequential symbolic world-model predictions."""
import json
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


def main():
    checkpoint=Path('concept_mapping/runs/symbolic_ez_10m/seed50/training/checkpoints/1048576')
    cfg=json.loads((checkpoint/'config.json').read_text())
    params=ocp.PyTreeCheckpointer().restore(str(checkpoint.resolve()))['params']
    model=SymbolicEZ(cfg['width']);env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    ep=env.default_params;batch=32
    keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(15050),i))(jnp.arange(batch))
    split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
    obs,state=jax.jit(jax.vmap(env.reset,in_axes=(0,None)))(split[:,0],ep)
    keys=split[:,1];hidden=jnp.zeros((batch,cfg['width']))
    pa=jnp.zeros(batch,int);pr=jnp.zeros(batch);resets=jnp.ones(batch,bool)
    finished=np.zeros(batch,bool);records=[]
    @jax.jit
    def tick(obs,state,h,pa,pr,resets,keys,t):
        h=model.apply(params,h,obs,pa,pr,resets,method=model.represent)
        actions,policy,sv=search(model,params,h,jax.random.fold_in(jax.random.PRNGKey(105050),t),32)
        next_h,r,c=model.apply(params,h,actions,method=model.dynamics)
        decoded=model.apply(params,next_h,method=model.observation)
        counts=decoded[:,MAP_DIM:MAP_DIM+160].reshape(batch,16,10).argmax(-1)
        _,root_v=model.apply(params,h,method=model.prediction)
        all_h=jnp.repeat(h,17,axis=0);all_a=jnp.tile(jnp.arange(17),batch)
        _,all_r,_=model.apply(params,all_h,all_a,method=model.dynamics)
        split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
        next_obs,state,rewards,dones,_=jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],state,actions,ep)
        return (next_obs,state,h,actions,rewards,jnp.zeros(batch,bool),split[:,0],dones,
                scalar_decode(r),jax.nn.sigmoid(c),counts,scalar_decode(root_v),sv,scalar_decode(all_r).reshape(batch,17))
    for t in range(10000):
        current=np.rint(np.asarray(obs)[:,MAP_DIM:MAP_DIM+16]*10).astype(int)
        result=tick(obs,state,hidden,pa,pr,resets,keys,jnp.asarray(t))
        obs,state,hidden,pa,pr,resets,keys,done,rhat,cont,counts,rootv,sv,allr=result
        next_counts=np.rint(np.asarray(obs)[:,MAP_DIM:MAP_DIM+16]*10).astype(int)
        for i in np.flatnonzero(~finished):
            records.append(dict(episode=int(i),step=t,action=int(pa[i]),reward=float(pr[i]),
                reward_prediction=float(rhat[i]),termination_probability=float(1-cont[i]),
                terminal=bool(done[i]),root_value=float(rootv[i]),search_value=float(sv[i]),
                current_counts=current[i].tolist(),next_counts=next_counts[i].tolist(),
                predicted_counts=np.asarray(counts[i]).tolist(),predicted_rewards_all_actions=np.asarray(allr[i]).tolist()))
        finished|=np.asarray(done)
        if finished.all():break
    terminal=[r for r in records if r['terminal']]
    positive=[r for r in records if r['reward']>.05]
    negative=[r for r in records if r['reward']<-.05]
    zeros=[r for r in records if abs(r['reward'])<=.05]
    result=dict(episodes=32,transitions=len(records),
        terminal_probability_mean=float(np.mean([r['termination_probability'] for r in terminal])),
        terminal_recall_at_half=float(np.mean([r['termination_probability']>.5 for r in terminal])),
        terminal_health_prediction_mean=float(np.mean([r['predicted_counts'][12] for r in terminal])),
        reward_positive_events=len(positive),reward_negative_events=len(negative),
        mean_predicted_reward_on_positive=float(np.mean([r['reward_prediction'] for r in positive])),
        mean_predicted_reward_on_negative=float(np.mean([r['reward_prediction'] for r in negative])),
        mean_predicted_reward_on_zero=float(np.mean([r['reward_prediction'] for r in zeros])),
        value_at_terminal_step=float(np.mean([r['root_value'] for r in terminal])),
        search_value_at_terminal_step=float(np.mean([r['search_value'] for r in terminal])))
    out=Path('concept_mapping/runs/symbolic_ez_10m/debug_predictions_1m');out.mkdir(exist_ok=True)
    (out/'transitions.json').write_text(json.dumps(records)+'\n')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

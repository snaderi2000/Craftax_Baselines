# SPDX-License-Identifier: GPL-3.0-only
# Symbolic JAX pilot adapted from EfficientZeroV2 (EVAR Lab); see SYMBOLIC_EZ_NOTICE.md.
"""Fresh learned-model Gumbel-search agent; no simulator calls inside search."""
import argparse,csv,json,os,time
from pathlib import Path
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from craftax.craftax_env import make_craftax_env_from_name
from wrappers import OptimisticResetVecEnvWrapper,GymnaxWrapper
from models.symbolic_ez import SymbolicEZ,search,training_loss,encode_sequence,scalar_decode
from symbolic_ez_replay import SequenceReplay


class ObservationLabelWrapper(GymnaxWrapper):
    def step(self,key,state,action,params):
        obs,state,reward,done,info=self._env.step(key,state,action,params)
        return obs,state,reward,done,dict(info,next_observation=obs)


def make_components(width=256,simulations=32,absolute_q=False):
    model=SymbolicEZ(width)
    def actor(params,h,obs,pa,pr,reset,key):
        h=model.apply(params,h,obs,pa,pr,reset,method=model.represent)
        a,p,v=search(model,params,h,key,simulations,absolute_q=absolute_q)
        return h,a,p,v
    return model,jax.jit(actor)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=50)
    p.add_argument('--steps',type=int,default=9961472);p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--width',type=int,default=256)
    p.add_argument('--simulations',type=int,default=32);p.add_argument('--capacity-time',type=int,default=512)
    p.add_argument('--burnin',type=int,default=64);p.add_argument('--warmup-time',type=int,default=128)
    p.add_argument('--unroll',type=int,default=5);p.add_argument('--td',type=int,default=5)
    p.add_argument('--wandb',action='store_true');p.add_argument('--verify-interval',type=int,default=128)
    p.add_argument('--reanalyse-interval',type=int,default=16)
    p.add_argument('--real-sequence-training',action='store_true',help='Also supervise real posterior GRU sequences; optional debugging fix, default preserves original pilot')
    p.add_argument('--absolute-search-q',action='store_true',help='Keep search value differences in reward units instead of normalizing tiny ranges to one')
    p.add_argument('--updates-per-step',type=int,default=1,help='Replay gradient updates per batched environment step')
    p.add_argument('--reanalyse-fraction',type=float,default=.25)
    p.add_argument('--lr-decay-steps',type=int,help='Interaction horizon for annealing, default the run budget')
    p.add_argument('--run-name',help='Explicit W&B name for diagnostic variants')
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    assert args.num_envs%16==0
    assert args.warmup_time>args.burnin+args.unroll+args.td
    assert args.capacity_time>args.burnin+args.unroll+args.td+1
    assert args.steps%args.num_envs==0 and args.updates_per_step>0
    assert args.reanalyse_interval>0 and 0<args.reanalyse_fraction<=1
    config=dict(vars(args),gamma=.99,lr=.0002,grad_clip=1.,target_ema=.01,
        replay_prioritized_fraction=.8,priority_alpha=.6,importance_beta=.4,
        reset='optimistic ratio16',standard_rewards=True,death_penalty=0,diamond_reward=1,
        observation_dim=1345,actions=17,loss_weights=dict(policy=1,value=.5,reward=1,
            termination=1,observation=1,consistency=2),
        implementation='Symbolic EZ-style pilot; deterministic latent dynamics, immediate reward head, fractional root-only value reanalysis; not a full EZ-V2 reproduction.')
    model,actor=make_components(args.width,args.simulations,args.absolute_search_q)
    key=jax.random.PRNGKey(args.seed);key,initkey,resetkey=jax.random.split(key,3)
    dummy=jnp.zeros((1,1345));h=jnp.zeros((1,args.width));a=jnp.zeros(1,jnp.int32)
    weights=model.init(initkey,h,dummy,a,jnp.zeros(1),jnp.ones(1,bool))
    config['parameters']=sum(int(v.size) for v in jax.tree_util.tree_leaves(weights))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.wandb:
        import wandb
        wb=wandb.init(project='Craftax_Baselines',entity='synaderi-uc-davis',
            name=args.run_name or f'symbolic-ez-style-{round(args.steps/1000000)}m{"-real-sequence" if args.real_sequence_training else ""}-seed{args.seed}',config=config,dir=str(out.resolve()))
        (out/'wandb.json').write_text(json.dumps(dict(url=wb.url,id=wb.id),indent=2)+'\n')
    decay_steps=args.lr_decay_steps if args.lr_decay_steps is not None else args.steps
    total_updates=max(1,(decay_steps//args.num_envs-args.warmup_time)*args.updates_per_step)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(optax.linear_schedule(.0002,0.,total_updates),eps=1e-5))
    state=TrainState.create(apply_fn=model.apply,params=weights,tx=optimizer);target=weights
    @jax.jit
    def update(state,target,batch):
        loss_fn=lambda params:training_loss(model,params,target,batch,args.burnin,args.unroll,args.td,args.real_sequence_training)
        (loss,(metrics,priorities,root)),grads=jax.value_and_grad(loss_fn,has_aux=True)(state.params)
        norm=optax.global_norm(grads)
        state=state.apply_gradients(grads=grads)
        target=jax.tree_util.tree_map(lambda old,new:.99*old+.01*new,target,state.params)
        return state,target,dict(metrics,loss=loss,gradient_norm=norm),priorities,root
    @jax.jit
    def reanalyse(target,batch,key):
        z=encode_sequence(model,target,batch['initial_h'],batch)[args.burnin]
        _,policy,values=search(model,target,z,key,args.simulations,absolute_q=args.absolute_search_q)
        return policy,values
    base=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    env=OptimisticResetVecEnvWrapper(ObservationLabelWrapper(base),args.num_envs,16)
    reset=jax.jit(env.reset);env_step=jax.jit(env.step)
    obs,env_state=reset(resetkey,base.default_params)
    hidden=jnp.zeros((args.num_envs,args.width));pa=jnp.zeros(args.num_envs,jnp.int32)
    pr=jnp.zeros(args.num_envs);resets=jnp.ones(args.num_envs,bool)
    replay=SequenceReplay(args.num_envs,args.capacity_time,args.width,args.burnin,args.unroll,args.td,args.seed)
    returns=np.zeros(args.num_envs);lengths=np.zeros(args.num_envs,int)
    completed=[];history=[];milestones=[65536,1048576,2228224,5046272,7536640,args.steps]
    saved=set();start=time.time();last_metrics={};updates=0;search_reanalyses=0
    with (out/'training_episodes.csv').open('w') as episodes_file:
        episode_writer=None
        for time_index in range(args.steps//args.num_envs):
            key,ak,ek,rk=jax.random.split(key,4)
            previous_hidden=hidden
            hidden,actions,policies,values=actor(state.params,hidden,obs,pa,pr,resets,ak)
            if time_index<args.warmup_time:
                actions=jax.random.randint(rk,(args.num_envs,),0,17)
                policies=jnp.full((args.num_envs,17),1/17)
            next_obs,env_state,rewards,dones,info=env_step(ek,env_state,actions,base.default_params)
            rewards_np,dones_np=np.asarray(rewards),np.asarray(dones)
            replay.append(np.asarray(obs),np.asarray(info['next_observation']),hidden=np.asarray(previous_hidden),
                prev_action=np.asarray(pa),prev_reward=np.asarray(pr),reset=np.asarray(resets),
                action=np.asarray(actions),reward=rewards_np,done=dones_np,
                policy=np.asarray(policies),search_value=np.asarray(values))
            returns+=rewards_np;lengths+=1
            done_ids=np.flatnonzero(dones_np)
            for i in done_ids:
                row=dict(env_step=(time_index+1)*args.num_envs,return_=float(returns[i]),length=int(lengths[i]),
                    **{k:float(np.asarray(v)[i]) for k,v in info.items() if k.startswith('Achievements/')})
                if episode_writer is None:
                    episode_writer=csv.DictWriter(episodes_file,fieldnames=list(row));episode_writer.writeheader()
                episode_writer.writerow(row);completed.append(row)
            returns[done_ids]=0;lengths[done_ids]=0
            obs=next_obs;pa=actions;pr=rewards;resets=dones
            for _ in range(args.updates_per_step if time_index>=args.warmup_time else 0):
                batch,slots=replay.sample(args.batch)
                batch['root_reanalysed']=np.zeros(args.batch,bool)
                batch['root_search_value']=np.zeros(args.batch,np.float32)
                if updates%args.reanalyse_interval==0:
                    key,rk=jax.random.split(key)
                    refreshed_policy,refreshed_value=reanalyse(target,batch,rk)
                    n=max(1,int(args.batch*args.reanalyse_fraction))
                    refreshed_policy=np.asarray(refreshed_policy);refreshed_value=np.asarray(refreshed_value)
                    replay.reanalyse(slots[:n],refreshed_policy[:n],refreshed_value[:n])
                    batch['policy'][args.burnin,:n]=refreshed_policy[:n]
                    batch['root_reanalysed'][:n]=True;batch['root_search_value'][:n]=refreshed_value[:n]
                    search_reanalyses+=n
                # Labels and burn-in arrays only: no simulator state reaches this function.
                batch={k:jnp.asarray(v) for k,v in batch.items() if k!='search_value'}
                state,target,metrics,priorities,_=update(state,target,batch)
                priorities=np.asarray(priorities)
                if not np.isfinite(priorities).all():raise FloatingPointError('Nonfinite priorities')
                replay.update_priorities(slots,priorities);updates+=1
                if updates%args.verify_interval==0 or updates==1:
                    last_metrics={k:float(v) for k,v in metrics.items()}
                    if not all(np.isfinite(v) for v in last_metrics.values()):raise FloatingPointError(last_metrics)
                    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(state.params)):
                        raise FloatingPointError('Nonfinite weights')
            steps=(time_index+1)*args.num_envs
            if time_index%32==0 or steps==args.steps:
                recent=completed[-256:]
                log=dict(env_step=steps,updates=updates,elapsed_seconds=time.time()-start,
                    sps=steps/max(1,time.time()-start),episodes=len(completed),reanalysed_roots=search_reanalyses,
                    **{'model/'+k:v for k,v in last_metrics.items()})
                if recent:
                    log.update(episode_return=float(np.mean([r['return_'] for r in recent])),
                        episode_length=float(np.mean([r['length'] for r in recent])),
                        **{k:float(np.mean([r[k] for r in recent])) for k in recent[0] if k.startswith('Achievements/')})
                history.append(log);print(json.dumps(log),flush=True)
                (out/'progress.json').write_text(json.dumps(dict(state='training',**log),indent=2)+'\n')
                if args.wandb:wandb.log(log,step=steps)
                episodes_file.flush()
            due=[m for m in milestones if m<=steps and m not in saved]
            if due:
                path=out/'checkpoints'/str(steps)
                ocp.PyTreeCheckpointer().save(str(path.resolve()),dict(params=state.params,target=target,
                    optimizer=state.opt_state,gradient_step=state.step))
                (path/'config.json').write_text(json.dumps(config,indent=2)+'\n')
                saved.update(due);print(f'Saved checkpoint {steps}',flush=True)
        (out/'training_history.json').write_text(json.dumps(history)+'\n')
        final=dict(state='training_complete',env_step=args.steps,updates=updates,
            seconds=time.time()-start,parameters=config['parameters'],checkpoint=str(path.resolve()))
        (out/'progress.json').write_text(json.dumps(final,indent=2)+'\n')
        print(json.dumps(final),flush=True)
    if args.wandb:wandb.finish()


if __name__=='__main__':main()

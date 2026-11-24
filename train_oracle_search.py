# SPDX-License-Identifier: GPL-3.0-only
"""Fresh policy/value training using actual-simulator search as the teacher."""
import argparse,csv,hashlib,json,time
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
from models.oracle_search import OraclePolicyValue,oracle_search,lambda_returns
from models.symbolic_ez import scalar_decode,scalar_target,cross_entropy,encode_sequence
from wrappers import OptimisticResetVecEnvWrapper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=1048576)
    p.add_argument('--seed',type=int,default=50);p.add_argument('--num-envs',type=int,default=1024)
    p.add_argument('--rollout',type=int,default=64);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--minibatches',type=int,default=8);p.add_argument('--width',type=int,default=256)
    p.add_argument('--simulations',type=int,default=32);p.add_argument('--max-depth',type=int,default=5)
    p.add_argument('--candidates',type=int,default=8);p.add_argument('--lr-decay-steps',type=int,default=9961472)
    p.add_argument('--wandb',action='store_true')
    args=p.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assert args.steps%(args.num_envs*args.rollout)==0
    assert args.num_envs%16==0 and args.num_envs%args.minibatches==0
    model=OraclePolicyValue(args.width)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    wrapped=OptimisticResetVecEnvWrapper(env,args.num_envs,16)
    key=jax.random.PRNGKey(args.seed);key,ik,rk=jax.random.split(key,3)
    weights=model.init(ik,jnp.zeros((1,args.width)),jnp.zeros((1,1345)),
        jnp.zeros(1,jnp.int32),jnp.zeros(1),jnp.ones(1,bool))
    config=dict(vars(args),out=str(out),implementation='Exact-simulator Gumbel MCTS with recurrent policy/value distillation; not PPO or full EfficientZero V2',
        observation_dim=1345,actions=17,gamma=.99,gae_lambda=.8,lr=.0002,grad_clip=1.,value_loss_weight=.5,
        ordinary_rewards=True,death_penalty=0,diamond_reward=1,reset='optimistic ratio16',
        learned_world_model=False,search_privileged_full_simulator_state=True,
        search_rescale_q=False,warmup_interactions=0,
        policy_inputs='symbolic observation, previous action/reward, recurrent carry; no global simulator state',
        value_targets='lambda returns from actual gameplay rewards; terminal transitions never bootstrap',
        planned_gameplay_transitions=args.steps,planned_search_transitions=args.steps*args.simulations,
        parameters=sum(int(x.size) for x in jax.tree_util.tree_leaves(weights)),
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ('models/oracle_search.py','models/symbolic_ez.py','train_oracle_search.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.wandb:
        import wandb
        wb=wandb.init(project='Craftax_Baselines',entity='synaderi-uc-davis',
            name=f'oracle-simulator-search-1m-seed{args.seed}',config=config,dir=str(out))
        (out/'wandb.json').write_text(json.dumps(dict(id=wb.id,url=wb.url),indent=2)+'\n')
    decay_updates=max(1,args.lr_decay_steps//(args.num_envs*args.rollout))*args.epochs*args.minibatches
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(
        optax.linear_schedule(.0002,0.,decay_updates),eps=1e-5))
    train=TrainState.create(apply_fn=model.apply,params=weights,tx=optimizer)
    obs,state=jax.jit(wrapped.reset)(rk,env.default_params)
    carry=(obs,state,jnp.zeros((args.num_envs,args.width)),jnp.zeros(args.num_envs,jnp.int32),
        jnp.zeros(args.num_envs),jnp.ones(args.num_envs,bool),key)

    @jax.jit
    def rollout(params,carry):
        def step(carry,_):
            obs,state,h,pa,pr,resets,key=carry
            key,sk,ek=jax.random.split(key,3)
            h=model.apply(params,h,obs,pa,pr,resets,method=model.represent)
            action,policy,search_value=oracle_search(model,params,env,state,h,sk,
                args.simulations,args.max_depth,args.candidates)
            _,v=model.apply(params,h,method=model.prediction)
            next_obs,next_state,reward,done,info=wrapped.step(ek,state,action,env.default_params)
            metrics={k:v for k,v in info.items() if k.startswith('Achievements/')}
            record=dict(obs=obs,prev_action=pa,prev_reward=pr,reset=resets,
                policy=policy,action=action,reward=reward,done=done,value=scalar_decode(v),search_value=search_value)
            return (next_obs,next_state,h,action,reward,done,key),(record,metrics)
        carry,(data,metrics)=jax.lax.scan(step,carry,None,length=args.rollout)
        obs,_,h,pa,pr,resets,_=carry
        h=model.apply(params,h,obs,pa,pr,resets,method=model.represent)
        _,v=model.apply(params,h,method=model.prediction)
        returns=lambda_returns(data['reward'],data['done'],data['value'],scalar_decode(v))
        return carry,data,metrics,returns

    @jax.jit
    def learn(train,data,returns,initial_h,key):
        def loss(params,batch,targets,h):
            hidden=encode_sequence(model,params,h,batch)
            pi,value=model.apply(params,hidden,method=model.prediction)
            policy_loss=cross_entropy(pi,batch['policy']).mean()
            value_loss=cross_entropy(value,scalar_target(targets)).mean()
            return policy_loss+.5*value_loss,dict(policy_loss=policy_loss,value_loss=value_loss)
        def epoch(carry,_):
            train,key=carry;key,pk=jax.random.split(key)
            indices=jax.random.permutation(pk,args.num_envs).reshape(args.minibatches,-1)
            def update(train,ids):
                b={k:v[:,ids] for k,v in data.items() if k in ('obs','prev_action','prev_reward','reset','policy')}
                (value,metrics),grads=jax.value_and_grad(loss,has_aux=True)(train.params,b,returns[:,ids],initial_h[ids])
                norm=optax.global_norm(grads)
                return train.apply_gradients(grads=grads),dict(metrics,loss=value,gradient_norm=norm)
            train,metrics=jax.lax.scan(update,train,indices)
            return (train,key),metrics
        (train,key),metrics=jax.lax.scan(epoch,(train,key),None,length=args.epochs)
        return train,key,jax.tree_util.tree_map(jnp.mean,metrics)

    running_return=np.zeros(args.num_envs);running_length=np.zeros(args.num_envs,int)
    episodes=[];history=[];start=time.time();saved=set()
    milestones=(65536,262144,524288,1048576,args.steps)
    print(json.dumps(dict(state='compiling',parameters=config['parameters'],gameplay_budget=args.steps,
        search_budget=config['planned_search_transitions'])),flush=True)
    with (out/'training_episodes.csv').open('w') as f:
        writer=None
        for iteration in range(args.steps//(args.num_envs*args.rollout)):
            initial_h=carry[2]
            carry,data,achievement_metrics,targets=rollout(train.params,carry)
            key,uk=jax.random.split(carry[-1])
            train,key,metrics=learn(train,data,targets,initial_h,uk)
            carry=(*carry[:-1],key)
            reward,done=jax.device_get((data['reward'],data['done']))
            achievement_metrics=jax.device_get(achievement_metrics)
            for t in range(args.rollout):
                running_return+=reward[t];running_length+=1
                for i in np.flatnonzero(done[t]):
                    row=dict(env_step=(iteration*args.rollout+t+1)*args.num_envs,
                        return_=float(running_return[i]),length=int(running_length[i]),
                        **{k:float(v[t,i]) for k,v in achievement_metrics.items()})
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);episodes.append(row)
                running_return[done[t]]=0;running_length[done[t]]=0
            f.flush();step=(iteration+1)*args.num_envs*args.rollout
            metrics={k:float(v) for k,v in metrics.items()}
            if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError(metrics)
            if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(train.params)):
                raise FloatingPointError('Nonfinite weights')
            recent=episodes[-256:]
            log=dict(env_step=step,gameplay_transitions=step,search_transitions=step*args.simulations,
                total_simulator_transitions=step*(1+args.simulations),
                elapsed_seconds=time.time()-start,updates=int(train.step),episodes=len(episodes),
                search_target_entropy=float((-(data['policy']*jnp.log(data['policy']+1e-30)).sum(-1)).mean()),
                **{'train/'+k:v for k,v in metrics.items()})
            if recent:log.update(episode_return=float(np.mean([r['return_'] for r in recent])),
                episode_length=float(np.mean([r['length'] for r in recent])),
                **{k:float(np.mean([r[k] for r in recent])) for k in recent[0] if k.startswith('Achievements/')})
            history.append(log);print(json.dumps(log),flush=True)
            (out/'progress.json').write_text(json.dumps(dict(state='training',**log),indent=2)+'\n')
            if args.wandb:wandb.log(log,step=step)
            due=[s for s in milestones if s<=step and s not in saved]
            if due:
                path=out/'checkpoints'/str(step)
                ocp.PyTreeCheckpointer().save(str(path),dict(params=train.params,optimizer=train.opt_state,gradient_step=train.step))
                (path/'config.json').write_text(json.dumps(config,indent=2)+'\n');saved.update(due)
        final=dict(state='training_complete',env_step=args.steps,updates=int(train.step),
            seconds=time.time()-start,search_transitions=args.steps*args.simulations,
            total_simulator_transitions=args.steps*(args.simulations+1),checkpoint=str(path))
        (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
        (out/'progress.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
    if args.wandb:wandb.finish()


if __name__=='__main__':main()

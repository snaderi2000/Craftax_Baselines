"""Bounded PPO sample-efficiency experiments using ordinary gameplay only."""
import argparse,csv,hashlib,json,os,time
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
from wrappers import OptimisticResetVecEnvWrapper
from models.conditioned_symbolic import make_network
from models.oracle_search import lambda_returns
from ppo_leap_utils import EpisodeReplay


def apply_gradients(state,grads,lr):
    delta,opt=state.tx.update(grads,state.opt_state,state.params)
    delta=jax.tree_util.tree_map(lambda d:lr*d,delta)
    return state.replace(step=state.step+1,opt_state=opt,params=optax.apply_updates(state.params,delta))


def ppo_loss(network,params,b,clip=.2):
    pi,value=network.apply(params,b['obs'])
    ratio=jnp.exp(pi.log_prob(b['action'])-b['log_prob'])
    advantage=(b['advantage']-b['advantage'].mean())/(b['advantage'].std()+1e-8)
    actor=-jnp.minimum(ratio*advantage,jnp.clip(ratio,1-clip,1+clip)*advantage).mean()
    clipped=b['value']+jnp.clip(value-b['value'],-clip,clip)
    critic=.5*jnp.maximum((value-b['target'])**2,(clipped-b['target'])**2).mean()
    entropy=pi.entropy().mean()
    kl=((ratio-1)-(pi.log_prob(b['action'])-b['log_prob'])).mean()
    return actor+.5*critic-.01*entropy,dict(actor=actor,critic=critic,entropy=entropy,approx_kl=kl)


def self_imitation_loss(network,params,anchor_params,b,coef=.1):
    pi,value=network.apply(params,b['obs'])
    anchor,_=network.apply(anchor_params,b['obs'])
    advantage=jax.nn.relu(b['returns']-value)
    actor=-jax.lax.stop_gradient(jnp.minimum(advantage,5.))*pi.log_prob(b['action'])
    critic=.5*advantage**2
    anchor_logits=jax.lax.stop_gradient(anchor.logits)
    anchor_probs=jax.nn.softmax(anchor_logits)
    kl=(anchor_probs*(anchor_logits-pi.logits)).sum(-1)
    total=(b['weights']*(coef*(actor+.5*critic)+kl)).mean()
    return total,(dict(imitation_actor=actor.mean(),imitation_critic=critic.mean(),
        imitation_kl=kl.mean(),positive_fraction=(advantage>0).mean()),advantage)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=50)
    p.add_argument('--architecture',choices=('mlp','conditioned'),default='mlp')
    p.add_argument('--steps',type=int,default=1048576);p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--rollout',type=int,default=64);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--minibatches',type=int,default=8);p.add_argument('--sil-coef',type=float,default=0.)
    p.add_argument('--sil-updates',type=int,default=16);p.add_argument('--sil-batch',type=int,default=512)
    p.add_argument('--lr-horizon',type=int,default=1048576);p.add_argument('--run-name',required=True)
    p.add_argument('--gae-lambda',type=float,default=.8)
    p.add_argument('--checkpoint-milestones',help='Comma-separated requested interaction counts; saved at the next completed rollout, plus the final checkpoint')
    p.add_argument('--wandb',action='store_true');args=p.parse_args()
    assert args.steps%(args.num_envs*args.rollout)==0 and args.num_envs%16==0
    if args.lr_horizon<args.steps:
        raise ValueError('Set --lr-horizon to cover the full training budget; refusing to silently freeze the learning rate early')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    net=make_network(args.architecture)
    key=jax.random.PRNGKey(args.seed);key,ik,rk=jax.random.split(key,3)
    params=net.init(ik,jnp.zeros((1,1345)))
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.scale_by_adam(eps=1e-5),optax.scale(-1.))
    train=TrainState.create(apply_fn=net.apply,params=params,tx=tx).replace(step=jnp.asarray(0,jnp.int32))
    config=dict(vars(args),out=str(out),gamma=.99,gae_lambda=args.gae_lambda,clip=.2,vf_coef=.5,entropy_coef=.01,
        lr=.0002,grad_clip=1.,reset='optimistic ratio16',standard_rewards=True,extra_simulator_calls=0,
        achievement_labels_used_for_learning=False,pretrained_weights=False,
        self_imitation=dict(coef=args.sil_coef,kl_anchor=1.,policy_advantage_clip=5.,
            replay_capacity=131072,priority_alpha=.6,importance_beta=.4,uniform_fraction=.2,
            targets='full discounted returns from completed factual episodes; terminal reward included'),
        parameters=sum(int(x.size) for x in jax.tree_util.tree_leaves(params)),
        runtime_flags=dict(XLA_FLAGS=os.environ.get('XLA_FLAGS',''),preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE','')),
        sources={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ('train_ppo_leaps.py','models/conditioned_symbolic.py','ppo_leap_utils.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.wandb:
        import wandb
        wb=wandb.init(project='Craftax_Baselines',entity='synaderi-uc-davis',name=args.run_name,config=config,dir=str(out))
        (out/'wandb.json').write_text(json.dumps(dict(id=wb.id,url=wb.url),indent=2)+'\n')
    base=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    env=OptimisticResetVecEnvWrapper(base,args.num_envs,16)
    obs,state=jax.jit(env.reset)(rk,base.default_params);carry=(obs,state,key)
    @jax.jit
    def rollout(params,carry):
        def body(carry,_):
            obs,state,key=carry;key,ak,ek=jax.random.split(key,3)
            pi,value=net.apply(params,obs);action=pi.sample(seed=ak)
            no,ns,r,d,info=env.step(ek,state,action,base.default_params)
            data=dict(obs=obs,action=action,value=value,log_prob=pi.log_prob(action),reward=r,done=d)
            return (no,ns,key),(data,{k:v for k,v in info.items() if k.startswith('Achievements/')})
        carry,(data,metrics)=jax.lax.scan(body,carry,None,length=args.rollout)
        _,last_value=net.apply(params,carry[0])
        targets=lambda_returns(data['reward'],data['done'],data['value'],last_value,gae_lambda=args.gae_lambda)
        data=dict(data,target=targets,advantage=targets-data['value'])
        return carry,data,metrics
    @jax.jit
    def learn(train,data,key,lr):
        flat=jax.tree_util.tree_map(lambda x:x.reshape((-1,)+x.shape[2:]),data)
        size=args.num_envs*args.rollout
        def epoch(carry,_):
            train,key=carry;key,sk=jax.random.split(key)
            ids=jax.random.permutation(sk,size).reshape(args.minibatches,-1)
            def update(train,ix):
                b=jax.tree_util.tree_map(lambda x:x[ix],flat)
                (loss,metrics),grads=jax.value_and_grad(lambda p:ppo_loss(net,p,b),has_aux=True)(train.params)
                return apply_gradients(train,grads,lr),dict(metrics,loss=loss,gradient_norm=optax.global_norm(grads))
            train,metrics=jax.lax.scan(update,train,ids)
            return (train,key),metrics
        (train,key),metrics=jax.lax.scan(epoch,(train,key),None,length=args.epochs)
        return train,key,jax.tree_util.tree_map(jnp.mean,metrics)
    @jax.jit
    def imitate(train,batches,lr):
        anchor=train.params
        def update(train,b):
            (loss,(metrics,advantages)),grads=jax.value_and_grad(
                lambda p:self_imitation_loss(net,p,anchor,b,args.sil_coef),has_aux=True)(train.params)
            return apply_gradients(train,grads,lr),dict(metrics,imitation_loss=loss),advantages
        def scan_step(train,b):
            train,metrics,advantages=update(train,b)
            return train,(metrics,advantages)
        train,(metrics,advantages)=jax.lax.scan(scan_step,train,batches)
        return train,jax.tree_util.tree_map(jnp.mean,metrics),advantages
    replay=EpisodeReplay(args.num_envs,seed=args.seed) if args.sil_coef else None
    ep_return=np.zeros(args.num_envs);ep_length=np.zeros(args.num_envs,int);episodes=[];history=[]
    start=time.time();saved=set()
    requested=([int(s) for s in args.checkpoint_milestones.split(',')] if args.checkpoint_milestones
               else [65536,262144,524288,1048576])
    if any(s<=0 for s in requested):raise ValueError('Checkpoint milestones must be positive')
    milestones=sorted(set([s for s in requested if s<=args.steps]+[args.steps]))
    print(json.dumps(dict(state='compiling',parameters=config['parameters'],configuration=args.run_name)),flush=True)
    with (out/'training_episodes.csv').open('w') as f:
        writer=None
        for iteration in range(args.steps//(args.num_envs*args.rollout)):
            carry,data,achievement_metrics=rollout(train.params,carry)
            step=(iteration+1)*args.num_envs*args.rollout
            lr=.0002*max(0.,1-(step-args.num_envs*args.rollout)/args.lr_horizon)
            key,uk=jax.random.split(carry[-1]);train,key,metrics=learn(train,data,uk,jnp.asarray(lr,jnp.float32))
            carry=(*carry[:-1],key)
            obs,action,reward,done,ach=jax.device_get((data['obs'],data['action'],data['reward'],data['done'],achievement_metrics))
            if replay is not None:
                replay.add_rollout(obs,action,reward,done)
                if replay.size:
                    b,ids=replay.sample(args.sil_updates*args.sil_batch)
                    b={k:jnp.asarray(v.reshape((args.sil_updates,args.sil_batch)+v.shape[1:])) for k,v in b.items()}
                    train,sil_metrics,advantages=imitate(train,b,jnp.asarray(lr,jnp.float32))
                    replay.update(ids,np.asarray(advantages).reshape(-1));metrics=dict(metrics,**sil_metrics)
            for t in range(args.rollout):
                ep_return+=reward[t];ep_length+=1
                for i in np.flatnonzero(done[t]):
                    row=dict(env_step=(iteration*args.rollout+t+1)*args.num_envs,return_=float(ep_return[i]),length=int(ep_length[i]),
                        **{k:float(v[t,i]) for k,v in ach.items()})
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);episodes.append(row)
                ep_return[done[t]]=0;ep_length[done[t]]=0
            f.flush();metrics={k:float(v) for k,v in metrics.items()}
            if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError(metrics)
            recent=episodes[-256:]
            log=dict(env_step=step,updates=int(train.step),elapsed_seconds=time.time()-start,
                episodes=len(episodes),lr=lr,replay_size=replay.size if replay else 0,
                **{'train/'+k:v for k,v in metrics.items()})
            if recent:log.update(episode_return=float(np.mean([r['return_'] for r in recent])),
                mean_length=float(np.mean([r['length'] for r in recent])),
                **{k:float(np.mean([r[k] for r in recent])) for k in recent[0] if k.startswith('Achievements/')})
            history.append(log);print(json.dumps(log),flush=True)
            (out/'progress.json').write_text(json.dumps(dict(state='training',**log),indent=2)+'\n')
            if args.wandb:wandb.log(log,step=step)
            due=[s for s in milestones if s<=step and s not in saved]
            if due:
                path=out/'checkpoints'/str(step)
                ocp.PyTreeCheckpointer().save(str(path),dict(params=train.params,opt_state=train.opt_state,gradient_step=train.step))
                (path/'config.json').write_text(json.dumps(config,indent=2)+'\n');saved.update(due)
        final=dict(state='training_complete',env_step=args.steps,updates=int(train.step),
            seconds=time.time()-start,checkpoint=str(path),parameters=config['parameters'],extra_simulator_calls=0)
        (out/'training_history.json').write_text(json.dumps(history,indent=2)+'\n')
        (out/'progress.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)
    if args.wandb:wandb.finish()


if __name__=='__main__':main()

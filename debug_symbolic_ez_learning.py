"""Exact-simulator search controls and fixed-real-batch model fitting diagnostics.

Simulator states are used only in the explicit oracle tests and data collection.
The fitted model receives observations, factual recurrent history, actions, and
recorded next-step labels; these diagnostics do not train a policy.
"""
import argparse,functools,json,time
from pathlib import Path
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from models.symbolic_ez import SymbolicEZ,mctx,scalar_target,scalar_decode,cross_entropy,observation_loss,MAP_DIM
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Action,BlockType
from train_symbolic_ez import ObservationLabelWrapper
from wrappers import OptimisticResetVecEnvWrapper


def oracle_checks(env):
    params=env.default_params
    _,base=env.reset(jax.random.PRNGKey(765),params)
    p=base.player_position
    clean=base.replace(map=jnp.full_like(base.map,BlockType.GRASS.value),
        zombies=base.zombies.replace(mask=jnp.zeros_like(base.zombies.mask)),
        cows=base.cows.replace(mask=jnp.zeros_like(base.cows.mask)),
        skeletons=base.skeletons.replace(mask=jnp.zeros_like(base.skeletons.mask)),
        arrows=base.arrows.replace(mask=jnp.zeros_like(base.arrows.mask)),
        player_direction=jnp.asarray(Action.RIGHT.value),achievements=jnp.zeros_like(base.achievements))
    results={}
    for label,tile,drink in [('wood',BlockType.TREE.value,9),('water',BlockType.WATER.value,0),('lava',BlockType.LAVA.value,9)]:
        state=clean.replace(map=clean.map.at[p[0],p[1]+1].set(tile),player_drink=jnp.asarray(drink))
        state=jax.tree_util.tree_map(lambda x:x[None],state)
        def recurrent(_,key,actions,embedding):
            obs,ns,r,d,_=jax.vmap(env.step,in_axes=(0,0,0,None))(
                jax.random.split(jax.random.PRNGKey(234),len(actions)),embedding,actions,params)
            # For the lava diagnostic only, use the explicitly stated -2 death cost.
            if label=='lava':r=r-2*d
            return mctx.RecurrentFnOutput(reward=r,discount=.99*(1-d),
                prior_logits=jnp.zeros((len(actions),17)),value=jnp.zeros(len(actions))),ns
        @jax.jit
        def run(state):
            result=mctx.gumbel_muzero_policy({},jax.random.PRNGKey(456),
                mctx.RootFnOutput(prior_logits=jnp.zeros((1,17)),value=jnp.zeros(1),embedding=state),
                recurrent,num_simulations=64,max_depth=1,max_num_considered_actions=17,gumbel_scale=0.)
            return result.action,result.action_weights,jax.vmap(lambda tree:tree.qvalues(0))(result.search_tree)
        action,weights,q=run(state)
        selected=int(action[0]);values=np.asarray(q[0])
        if label in ('wood','water'):assert selected==Action.DO.value,(label,selected,values)
        else:assert selected!=Action.RIGHT.value and values[Action.RIGHT.value]<-1.9
        results[label]=dict(action=selected,qvalues=values.tolist(),target=np.asarray(weights[0]).tolist())
    return results


def noise_checks():
    # A toy dynamics model with action differences of only 0.000001 reward.
    def recurrent(_,key,actions,embedding):
        r=jnp.where(actions==5,1e-6,0.)
        return mctx.RecurrentFnOutput(reward=r,discount=jnp.zeros(len(actions)),
            prior_logits=jnp.zeros((len(actions),17)),value=jnp.zeros(len(actions))),embedding
    results={}
    for name,rescale in [('original_rescaled',True),('absolute_q',False)]:
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value,rescale_values=rescale)
        result=jax.jit(lambda:mctx.gumbel_muzero_policy({},jax.random.PRNGKey(42),
            mctx.RootFnOutput(prior_logits=jnp.zeros((1,17)),value=jnp.zeros(1),embedding=jnp.zeros((1,1))),
            recurrent,num_simulations=32,max_num_considered_actions=17,max_depth=1,gumbel_scale=0.,qtransform=qtransform))()
        weights=np.asarray(result.action_weights[0])
        results[name]=dict(max_policy_weight=float(weights.max()),entropy=float(-(weights*np.log(weights+1e-30)).sum()),target=weights.tolist())
    return results


def collect(model,weights,env,batch=128,steps=256):
    wrapped=OptimisticResetVecEnvWrapper(ObservationLabelWrapper(env),batch,16)
    reset=jax.jit(wrapped.reset);step=jax.jit(wrapped.step)
    obs,state=reset(jax.random.PRNGKey(875),env.default_params)
    h=jnp.zeros((batch,model.width));pa=jnp.zeros(batch,jnp.int32);pr=jnp.zeros(batch);resets=jnp.ones(batch,bool)
    @jax.jit
    def encode(h,obs,pa,pr,resets):
        z=model.apply(weights,h,obs,pa,pr,resets,method=model.represent)
        pi,_=model.apply(weights,z,method=model.prediction)
        return z,pi
    records={k:[] for k in ('obs','next_obs','initial_h','prev_action','prev_reward','reset','action','reward','done','env')}
    key=jax.random.PRNGKey(832)
    for t in range(steps):
        key,ak,rk,ek=jax.random.split(key,4)
        nh,pi=encode(h,obs,pa,pr,resets)
        policy_action=jax.random.categorical(ak,pi)
        random_action=jax.random.randint(rk,(batch,),0,17)
        action=jnp.where(jnp.arange(batch)%2==0,random_action,policy_action)
        no,ns,reward,done,info=step(ek,state,action,env.default_params)
        row=dict(obs=obs,next_obs=info['next_observation'],initial_h=h,prev_action=pa,
            prev_reward=pr,reset=resets,action=action,reward=reward,done=done,env=jnp.arange(batch))
        for k,v in row.items():records[k].append(np.asarray(v))
        obs,state,h,pa,pr,resets=no,ns,nh,action,reward,done
    return {k:np.concatenate(v) for k,v in records.items()}


def fit_check(model,weights,data,updates):
    rng=np.random.default_rng(25)
    train=data['env']<96
    selections=[]
    for mask in ((data['reward']>.05),(data['reward']<-.05),data['done'],(abs(data['reward'])<=.05)):
        ids=np.flatnonzero(mask&train)
        if len(ids):selections.extend(rng.choice(ids,min(64,len(ids)),replace=False))
    ids=np.unique(selections)
    b={k:jnp.asarray(v[ids]) for k,v in data.items() if k!='env'}
    def predictions(p,b):
        h=model.apply(p,b['initial_h'],b['obs'],b['prev_action'],b['prev_reward'],b['reset'],method=model.represent)
        nh,r,c=model.apply(p,h,b['action'],method=model.dynamics)
        decoded=model.apply(p,nh,method=model.observation)
        return r,c,decoded
    def loss(p):
        r,c,decoded=predictions(p,b)
        return (cross_entropy(r,scalar_target(b['reward']))+
            optax.sigmoid_binary_cross_entropy(c,1-b['done'].astype(jnp.float32))+
            observation_loss(decoded,b['next_obs'])).mean()
    optimizer=optax.chain(optax.clip_by_global_norm(10.),optax.adam(.001))
    opt=optimizer.init(weights)
    @jax.jit
    def update(p,opt):
        value,g=jax.value_and_grad(loss)(p)
        delta,opt=optimizer.update(g,opt,p)
        return optax.apply_updates(p,delta),opt,value
    predict=jax.jit(predictions)
    def metrics(p):
        r,c,_=predict(p,b);r=np.asarray(scalar_decode(r));death=np.asarray(1-jax.nn.sigmoid(c))
        actual=np.asarray(b['reward']);done=np.asarray(b['done'])
        return dict(loss=float(loss(p)),mean_prediction_positive=float(r[actual>.05].mean()),
            mean_true_positive=float(actual[actual>.05].mean()),
            mean_prediction_negative=float(r[actual<-.05].mean()),
            mean_true_negative=float(actual[actual<-.05].mean()),
            death_recall=float((death[done]>.5).mean()),death_false_positive=float((death[~done]>.5).mean()))
    first=metrics(weights)
    for _ in range(updates):weights,opt,lossval=update(weights,opt)
    last=metrics(weights)
    return dict(training_examples=len(ids),positive_examples=int((np.asarray(b['reward'])>.05).sum()),
        terminal_examples=int(np.asarray(b['done']).sum()),updates=updates,before=first,after=last,
        note='Fits a fixed balanced batch, not a held-out performance result.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fit-updates',type=int,default=400)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    start=time.time();env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    result=dict(oracle_search=oracle_checks(env),tiny_reward_search=noise_checks())
    (out/'search_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    checkpoint=Path(args.checkpoint).resolve();config=json.loads((checkpoint/'config.json').read_text())
    weights=ocp.PyTreeCheckpointer().restore(str(checkpoint))['params'];model=SymbolicEZ(config['width'])
    data=collect(model,weights,env)
    np.savez_compressed(out/'real_transitions.npz',**data)
    result['data']=dict(transitions=len(data['reward']),positive=int((data['reward']>.05).sum()),negative=int((data['reward']<-.05).sum()),terminals=int(data['done'].sum()))
    print(json.dumps(result['data']),flush=True)
    result['fixed_batch_fit']=fit_check(model,weights,data,args.fit_updates)
    result['seconds']=time.time()-start
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['fixed_batch_fit'],indent=2),flush=True)


if __name__=='__main__':main()

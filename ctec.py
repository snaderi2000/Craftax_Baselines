"""Temporal contrastive exploration for recurrent PPO.

Independent implementation of C-TeC (Mohamed et al., ICLR 2026),
https://arxiv.org/abs/2603.02008 . InfoNCE learns state-action/future
representations; the reward is positive L2 distance, NOT InfoNCE loss.
Our mixed-reward adaptation bounds its mean scale relative to task reward,
warms up the encoder, and fades curiosity late in training. No privileged
state, achievement labels, or cross-episode positive pairs are used.
"""
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

DEFAULTS=dict(ctec_coef=0.,ctec_width=1024,ctec_dim=64,ctec_batch=512,
    ctec_updates=2,ctec_lr=3e-4,ctec_gamma=.3,ctec_warmup_steps=1000000,
    ctec_ramp_steps=9000000,ctec_fade_start=.8,ctec_clip=5.)


class Encoder(nn.Module):
    width:int
    dim:int
    @nn.compact
    def __call__(self,x):
        for _ in range(3):x=nn.relu(nn.Dense(self.width)(x))
        return nn.Dense(self.dim)(x)


class TemporalModel(nn.Module):
    width:int=1024
    dim:int=64
    @nn.compact
    def __call__(self,obs,action,future):
        return (Encoder(self.width,self.dim,name='current')(jnp.concatenate((obs,jax.nn.one_hot(action,17)),axis=-1)),
                Encoder(self.width,self.dim,name='future')(future))


@struct.dataclass
class CuriosityState:
    learner:object
    distance_mean:object
    distance_sq:object
    task_abs_mean:object
    updates:object


class ExplorationTrainState(TrainState):
    curiosity:object=None


def initialize(config,key):
    model=TemporalModel(config.get('CTEC_WIDTH',1024),config.get('CTEC_DIM',64))
    params=model.init(key,jnp.zeros((1,1345)),jnp.zeros(1,jnp.int32),jnp.zeros((1,1345)))
    learner=TrainState.create(apply_fn=model.apply,params=params,
        tx=optax.chain(optax.clip_by_global_norm(1.),optax.adam(config.get('CTEC_LR',3e-4),eps=1e-5)))
    return CuriosityState(learner,jnp.array(0.),jnp.array(1.),jnp.array(0.),jnp.array(0,jnp.int32))


def sample_future_indices(key,resets,gamma):
    """Sample strictly later observations in the SAME episode, truncated geometric.

    resets[t] means obs[t] starts a new episode, as in recurrent PPO. Exclude
    terminal transitions whose successor is already autoreset, and the final
    rollout observation (no recorded future). No arbitrary terminal fallback.
    """
    t,n=resets.shape
    indices=jnp.broadcast_to(jnp.arange(t)[:,None],(t,n))
    # Earliest subsequent reset is an exclusive bound.
    next_reset=jnp.where(resets,indices,t)
    next_reset=jax.lax.associative_scan(jnp.minimum,next_reset,axis=0,reverse=True)
    end=jnp.concatenate((next_reset[1:],jnp.full((1,n),t)),axis=0)
    count=jnp.maximum(end-indices-1,0)
    u=jax.random.uniform(key,(t,n),minval=1e-6,maxval=1.-1e-6)
    offset=1+jnp.floor(jnp.log1p(-u*(1.-gamma**count))/jnp.log(gamma)).astype(jnp.int32)
    future=jnp.minimum(indices+jnp.minimum(offset,jnp.maximum(count,1)),t-1)
    return future,count>0


def distance(x,y):return jnp.sqrt(jnp.sum(jnp.square(x-y),axis=-1)+1e-6)


def contrastive_loss(params,apply,obs,actions,future,valid):
    x,y=apply(params,obs,actions,future)
    # Pairwise L2 using norms/matmul avoids a B x B x D allocation.
    d2=jnp.maximum(jnp.sum(x*x,-1)[:,None]+jnp.sum(y*y,-1)[None,:]-2*x@y.T,0.)
    logits=-jnp.sqrt(d2+1e-6)
    logits=jnp.where(valid[None,:],logits,-1e9)
    losses=jax.nn.logsumexp(logits,axis=-1)-jnp.diag(logits)
    loss=jnp.sum(jnp.where(valid,losses,0.))/jnp.maximum(valid.sum(),1)
    return loss,dict(positive_distance=jnp.sum(jnp.where(valid,distance(x,y),0.))/jnp.maximum(valid.sum(),1),
        accuracy=jnp.sum(valid*(jnp.argmax(logits,axis=-1)==jnp.arange(obs.shape[0])))/jnp.maximum(valid.sum(),1))


def coefficient(config,update):
    steps=(update.astype(jnp.float32)+1)*config['NUM_ENVS']*config['NUM_STEPS']+float(config.get('START_TIMESTEPS',0))
    ramp=jnp.clip((steps-config.get('CTEC_WARMUP_STEPS',1000000))/max(config.get('CTEC_RAMP_STEPS',9000000),1),0.,1.)
    horizon=float(config.get('LR_HORIZON_TIMESTEPS') or config['TOTAL_TIMESTEPS'])
    fade=config.get('CTEC_FADE_START',.8)
    decay=jnp.clip((1.-steps/horizon)/(1.-fade),0.,1.)
    return config.get('CTEC_COEF',0.)*ramp*decay


def update_curiosity(state,obs,actions,resets,task_rewards,key,update,config):
    """Frozen pre-update bonus, then separate representation optimization.

    All reductions are global under GSPMD. Sampled contrastive batches are
    replicated (small) for a single global negative pool, never local losses.
    """
    from ppo_data_parallel import replicate
    raw=jax.lax.stop_gradient(obs[...,:1345])
    t,n,_=raw.shape
    reward_key,train_key=jax.random.split(key)
    future_ids,valid=sample_future_indices(reward_key,resets,config.get('CTEC_GAMMA',.3))
    flat=raw.reshape((-1,1345));act=actions.reshape(-1)
    # Encode each observation once; gather future embeddings afterwards.
    current,future=state.learner.apply_fn(state.learner.params,flat,act,flat)
    future=future.reshape(t,n,-1)[future_ids,jnp.arange(n)[None,:]]
    reward=jnp.where(valid,distance(current.reshape(t,n,-1),future),0.)
    count=jnp.maximum(valid.sum(),1)
    mean=reward.sum()/count;sq=jnp.square(reward).sum()/count
    weight=jnp.where(state.updates==0,1.,.01)
    dm=state.distance_mean+weight*(mean-state.distance_mean)
    ds=state.distance_sq+weight*(sq-state.distance_sq)
    tm=state.task_abs_mean+weight*(jnp.mean(jnp.abs(task_rewards))-state.task_abs_mean)
    norm=jnp.clip(reward/jnp.sqrt(jnp.maximum(ds,1e-6)),0.,config.get('CTEC_CLIP',5.))
    # Bound expected curiosity budget relative to actual task rewards. The
    # reward remains nonnegative; task rewards/penalties are never rescaled.
    norm=jnp.where(valid,norm,0.)
    scale=coefficient(config,update)*tm/jnp.maximum(jnp.mean(norm),1e-6)
    bonus=jax.lax.stop_gradient(scale*norm)
    def learn(learner,k):
        k1,k2=jax.random.split(k)
        f,v=sample_future_indices(k1,resets,config.get('CTEC_GAMMA',.3))
        sample=jax.random.randint(k2,(config.get('CTEC_BATCH',512),),0,t*n)
        ti,ei=sample//n,sample%n
        batch=replicate((raw[ti,ei],actions[ti,ei],raw[f[ti,ei],ei],v[ti,ei]),config)
        (loss,metrics),grad=jax.value_and_grad(contrastive_loss,has_aux=True)(learner.params,learner.apply_fn,*batch)
        grad=replicate(grad,config)
        learner=replicate(learner.apply_gradients(grads=grad),config)
        return learner,dict(loss=loss,grad_norm=optax.global_norm(grad),**metrics)
    learner,metrics=jax.lax.scan(learn,state.learner,jax.random.split(train_key,config.get('CTEC_UPDATES',2)))
    updated=CuriosityState(learner,dm,ds,tm,state.updates+1)
    logs={'training/ctec_'+k:jnp.mean(v) for k,v in metrics.items()}
    logs.update({'training/ctec_bonus':jnp.mean(bonus),'training/ctec_raw':mean,
        'training/ctec_coef':coefficient(config,update),'training/ctec_valid_fraction':jnp.mean(valid),
        'training/task_abs_reward':jnp.mean(jnp.abs(task_rewards)),
        'training/ctec_scale':scale})
    return replicate(updated,config),bonus,logs

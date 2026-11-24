# SPDX-License-Identifier: GPL-3.0-only
"""Symbolic policy/value learning with exact-simulator Gumbel MCTS.

Observation/history encoder and categorical value support follow symbolic_ez.
There is no learned dynamics, reward, or termination model in this agent.
"""
import functools
import flax.linen as nn
import jax
import jax.numpy as jnp
from models.symbolic_ez import SymbolicEZ,SUPPORT,scalar_decode,mctx


class OraclePolicyValue(SymbolicEZ):
    def setup(self):
        self.obs_layer=nn.Dense(self.width)
        self.gru=nn.GRUCell(self.width)
        self.pred_layer=nn.Dense(self.width)
        self.policy=nn.Dense(self.actions,kernel_init=nn.initializers.normal(.01))
        self.value=nn.Dense(len(SUPPORT),kernel_init=nn.initializers.zeros)

    def __call__(self,h,obs,previous_action,previous_reward,reset):
        return self.prediction(self.represent(h,obs,previous_action,previous_reward,reset))


def make_oracle_recurrent(model,env):
    """Functional simulator branches; terminal branches are absorbing.

Search has its own RNG stream and never updates the live environment carry.
One stochastic sample is used per expanded edge, not an expectation over all
possible environment outcomes (this is not stochastic/chance-node MCTS).
"""
    def recurrent(params,key,actions,embedding):
        state,h,finished=embedding['state'],embedding['hidden'],embedding['terminal']
        obs,next_state,reward,done,_=jax.vmap(env.step,in_axes=(0,0,0,None))(
            jax.random.split(key,len(actions)),state,actions,env.default_params)
        next_state=jax.tree_util.tree_map(lambda old,new:jnp.where(
            finished.reshape((len(actions),)+(1,)*(old.ndim-1)),old,new),state,next_state)
        reward=jnp.where(finished,0.,reward)
        terminal=finished|done
        next_h=model.apply(params,h,obs,actions,reward,jnp.zeros(len(actions),bool),method=model.represent)
        next_h=jnp.where(finished[:,None],h,next_h)
        logits,value=model.apply(params,next_h,method=model.prediction)
        result=mctx.RecurrentFnOutput(reward=reward,discount=.99*(1-terminal),
            prior_logits=jnp.where(terminal[:,None],0.,logits),
            value=jnp.where(terminal,0.,scalar_decode(value)))
        return result,dict(state=next_state,hidden=next_h,terminal=terminal)
    return recurrent


def oracle_search(model,params,env,state,h,key,simulations=32,max_depth=5,
                  candidates=8,gumbel_scale=1.):
    logits,value=model.apply(params,h,method=model.prediction)
    result=mctx.gumbel_muzero_policy(params,key,
        mctx.RootFnOutput(prior_logits=logits,value=scalar_decode(value),embedding=dict(
            state=state,hidden=h,terminal=jnp.zeros(h.shape[0],bool))),
        make_oracle_recurrent(model,env),num_simulations=simulations,
        max_num_considered_actions=min(candidates,simulations),max_depth=max_depth,
        gumbel_scale=gumbel_scale,
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value,rescale_values=False))
    return result.action,result.action_weights,result.search_tree.summary().value


def lambda_returns(rewards,dones,values,last_value,gamma=.99,gae_lambda=.8):
    """Real-trajectory lambda returns; terminal rewards included, no reset bootstrap."""
    def body(carry,x):
        next_value,next_advantage=carry
        reward,done,value=x
        discount=gamma*(1-done)
        advantage=reward+discount*next_value-value+discount*gae_lambda*next_advantage
        return (value,advantage),advantage+value
    return jax.lax.scan(body,(last_value,jnp.zeros_like(last_value)),
        (rewards,dones,values),reverse=True)[1]

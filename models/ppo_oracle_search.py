# SPDX-License-Identifier: GPL-3.0-only
"""Frozen feedforward PPO policy/value guidance for actual-simulator MCTS."""
import functools
import jax
import jax.numpy as jnp
from models.symbolic_ez import mctx


def make_recurrent(network,env):
    def recurrent(params,key,actions,embedding):
        state,finished=embedding['state'],embedding['terminal']
        obs,next_state,reward,done,_=jax.vmap(env.step,in_axes=(0,0,0,None))(
            jax.random.split(key,len(actions)),state,actions,env.default_params)
        next_state=jax.tree_util.tree_map(lambda old,new:jnp.where(
            finished.reshape((len(actions),)+(1,)*(old.ndim-1)),old,new),state,next_state)
        terminal=finished|done
        policy,value=network.apply(params,obs)
        return mctx.RecurrentFnOutput(reward=jnp.where(finished,0.,reward),
            discount=.99*(1-terminal),prior_logits=jnp.where(terminal[:,None],0.,policy.logits),
            value=jnp.where(terminal,0.,value)),dict(state=next_state,terminal=terminal)
    return recurrent


def search(network,params,env,state,obs,key,simulations=32,max_depth=5,candidates=8,gumbel_scale=1.):
    policy,value=network.apply(params,obs)
    result=mctx.gumbel_muzero_policy(params,key,
        mctx.RootFnOutput(prior_logits=policy.logits,value=value,
            embedding=dict(state=state,terminal=jnp.zeros(obs.shape[0],bool))),
        make_recurrent(network,env),num_simulations=simulations,max_depth=max_depth,
        max_num_considered_actions=min(simulations,candidates,17),gumbel_scale=gumbel_scale,
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value,rescale_values=False))
    tree=result.search_tree
    depths=jnp.zeros_like(tree.parents)
    def body(i,depths):
        parent=tree.parents[:,i]
        d=depths[jnp.arange(obs.shape[0]),jnp.maximum(parent,0)]+1
        return depths.at[:,i].set(jnp.where(parent>=0,d,0))
    depths=jax.lax.fori_loop(1,simulations+1,body,depths)
    visited=tree.children_visits[:,0]>0
    summary=tree.summary();q=summary.qvalues
    stats=dict(root_actions_visited=visited.sum(-1).astype(jnp.float32),
        max_depth_reached=depths.max(-1).astype(jnp.float32),
        hit_depth_limit=(depths.max(-1)>=max_depth).astype(jnp.float32),
        mean_node_depth=depths.sum(-1)/jnp.maximum((tree.parents[:,1:]>=0).sum(-1),1),
        target_entropy=-(result.action_weights*jnp.log(result.action_weights+1e-30)).sum(-1),
        differs_from_greedy_ppo=(result.action!=jnp.argmax(policy.logits,-1)).astype(jnp.float32),
        root_visited_q_range=jnp.max(jnp.where(visited,q,-jnp.inf),-1)-jnp.min(jnp.where(visited,q,jnp.inf),-1))
    return result.action,result.action_weights,stats

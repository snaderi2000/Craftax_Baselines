"""Frozen PPO scalar-value and simulator-branch integration checks."""
import unittest
from types import SimpleNamespace
import jax
if not hasattr(jax,'tree'):
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
from models.ppo_oracle_search import make_recurrent,search


class Network:
    def apply(self,params,obs):
        return SimpleNamespace(logits=jnp.zeros((len(obs),17))),jnp.full(len(obs),3.)


class Environment:
    default_params=None
    def step(self,key,state,action,params):
        return jnp.zeros(2),state+1,(action==2).astype(jnp.float32),action==3,{}


class PPOOracleTests(unittest.TestCase):
    def test_scalar_critic_values_and_terminal_bootstrap(self):
        fn=make_recurrent(Network(),Environment())
        root=dict(state=jnp.zeros(2,jnp.int32),terminal=jnp.zeros(2,bool))
        out,child=fn({},jax.random.PRNGKey(1),jnp.array([2,3]),root)
        np.testing.assert_array_equal(out.reward,[1.,0.])
        np.testing.assert_array_equal(out.value,[3.,0.])
        np.testing.assert_allclose(out.discount,[.99,0.])
        out,next_child=fn({},jax.random.PRNGKey(2),jnp.array([2,2]),child)
        np.testing.assert_array_equal(out.reward,[1.,0.])
        np.testing.assert_array_equal(next_child['state'],[2,1])

    def test_search_improvement_coverage_depth_and_state_isolation(self):
        state=jnp.zeros(2,jnp.int32);obs=jnp.zeros((2,2))
        action,weights,stats=jax.jit(lambda s:search(Network(),{},Environment(),s,obs,
            jax.random.PRNGKey(3),64,1,17,0.))(state)
        np.testing.assert_array_equal(action,[2,2])
        np.testing.assert_allclose(weights.sum(-1),1.,atol=1e-6)
        np.testing.assert_array_equal(stats['root_actions_visited'],17.)
        np.testing.assert_array_equal(stats['max_depth_reached'],1.)
        np.testing.assert_array_equal(state,0)
        for value in stats.values():self.assertTrue(np.isfinite(value).all())


if __name__=='__main__':unittest.main()

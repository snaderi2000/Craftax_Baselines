import unittest
import numpy as np
import jax
import jax.numpy as jnp
from models.actor_critic import ActorCritic
from stone_outcome import initialize_head,empty_replay,append_replay,sample_replay,outcome_loss


class StoneTests(unittest.TestCase):
    def test_balanced_and_empty_replay(self):
        buffer = empty_replay(4,10)
        batch = sample_replay(buffer,jax.random.PRNGKey(50),8)
        self.assertEqual(float(batch[-1].sum()),0.)
        buffer = append_replay(buffer,jnp.ones((6,10)),jnp.full(6,5),jnp.array([1,1,1,1,0,0],bool),jnp.array([0,1,1,0,0,0],bool))
        np.testing.assert_array_equal(buffer['counts'],[2,2])
        batch = sample_replay(buffer,jax.random.PRNGKey(50),8)
        self.assertEqual(float(batch[2].sum()),4.)
        self.assertEqual(float(batch[-1].sum()),8.)

    def test_aux_gradients_and_unchanged_policy_initialization(self):
        net = ActorCritic(17,16)
        obs = jnp.ones((2,10))
        original = net.init(jax.random.PRNGKey(50),obs)
        params = initialize_head(original,jax.random.PRNGKey(51),16)
        pi,value = net.apply(params,obs)
        original_pi,original_value = net.apply(original,obs)
        np.testing.assert_array_equal(pi.logits_parameter(),original_pi.logits_parameter())
        np.testing.assert_array_equal(value,original_value)
        batch = (obs,jnp.array([5,5]),jnp.array([0.,1.]),jnp.ones(2))
        grads = jax.grad(lambda p:outcome_loss(p,batch))(params)['params']
        self.assertTrue(any(np.any(np.asarray(v)!=0) for v in jax.tree_util.tree_leaves(grads['Dense_0'])))
        for i in range(3,8):
            self.assertTrue(all(np.all(np.asarray(v)==0) for v in jax.tree_util.tree_leaves(grads[f'Dense_{i}'])))


if __name__ == '__main__':
    unittest.main()

"""Small deterministic checks for retrieval and selective imitation."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from culture import CultureMemory, build_policy_target, encode
from models.actor_critic import ActorCritic


class CultureTests(unittest.TestCase):
    def setUp(self):
        self.memory=CultureMemory(dict(embeddings=np.eye(3,dtype=np.float32),
            actions=np.array([[1],[2],[3]]),outcomes=np.array([1.,.5,0.],np.float32),
            returns=np.array([2.,1.,-1.],np.float32)),{})

    def test_matching_and_gate(self):
        sim,ix=self.memory.retrieve(jnp.array([[1.,0.,0.],[-1.,0.,0.]]),2)
        q,gate,_=build_policy_target(self.memory,sim,ix,threshold=.9)
        self.assertAlmostEqual(float(q[0,1]),1.)
        self.assertAlmostEqual(float(gate[0]),1.)
        self.assertEqual(float(gate[1]),0.)
        self.assertTrue(np.array_equal(np.asarray(q[1]),np.zeros(17)))

    def test_no_cultural_gradient_to_critic(self):
        net=ActorCritic(17,16)
        obs=jnp.ones((2,10))
        params=net.init(jax.random.PRNGKey(50),obs)
        def loss(p):
            pi,_=net.apply(p,obs)
            return -pi.log_prob(jnp.array([1,2])).mean()
        grads=jax.grad(loss)(params)['params']
        self.assertTrue(all(np.all(np.asarray(x)==0) for i in range(4,8)
                            for x in jax.tree_util.tree_leaves(grads[f'Dense_{i}'])))
        z=encode(params,obs)
        np.testing.assert_allclose(np.linalg.norm(z,axis=-1),1,atol=1e-6)


if __name__=='__main__':unittest.main()

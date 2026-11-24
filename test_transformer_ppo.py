"""Temporal correctness checks for the exact-window PPO Transformer."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from models.actor_critic_transformer import ActorCriticTransformer


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=dict(LAYER_SIZE=16,TRANSFORMER_DIM=16,TRANSFORMER_HEADS=4,
                     TRANSFORMER_LAYERS=2,TRANSFORMER_CONTEXT=8)
        cls.net=ActorCriticTransformer(17,cls.cfg)
        cls.obs=jax.random.normal(jax.random.PRNGKey(1),(12,2,10))
        cls.resets=jnp.zeros((12,2),bool).at[4,0].set(True).at[9,1].set(True)
        cls.carry=cls.net.initialize_carry(2,8,10)
        cls.params=cls.net.init(jax.random.PRNGKey(2),cls.carry,(cls.obs,cls.resets))

    def test_online_equals_sequence_and_chunked_training(self):
        carry=self.carry;probs=[];values=[]
        for t in range(12):
            carry,p,v=self.net.apply(self.params,carry,(self.obs[t:t+1],self.resets[t:t+1]))
            probs.append(p.probs[0]);values.append(v[0])
        final,p,v=self.net.apply(self.params,self.carry,(self.obs,self.resets))
        np.testing.assert_allclose(p.probs,np.stack(probs),atol=2e-6)
        np.testing.assert_allclose(v,np.stack(values),atol=2e-5)
        c,_,_=self.net.apply(self.params,self.carry,(self.obs[:6],self.resets[:6]))
        _,cp,cv=self.net.apply(self.params,c,(self.obs[6:],self.resets[6:]))
        np.testing.assert_allclose(cp.probs,p.probs[6:],atol=2e-6)
        np.testing.assert_allclose(cv,v[6:],atol=2e-5)
        np.testing.assert_array_equal(final[1],carry[1])

    def test_future_and_pre_reset_history_cannot_leak(self):
        _,p,v=self.net.apply(self.params,self.carry,(self.obs,self.resets))
        future=self.obs.at[7:].set(100.)
        _,fp,fv=self.net.apply(self.params,self.carry,(future,self.resets))
        np.testing.assert_allclose(fp.probs[:7],p.probs[:7],atol=2e-6)
        np.testing.assert_allclose(fv[:7],v[:7],atol=2e-5)
        changed=self.obs.at[:4,0].set(-100.)
        _,rp,rv=self.net.apply(self.params,self.carry,(changed,self.resets))
        np.testing.assert_allclose(rp.probs[4:,0],p.probs[4:,0],atol=2e-6)
        np.testing.assert_allclose(rv[4:,0],v[4:,0],atol=2e-5)

    def test_exact_context_and_finite_gradients(self):
        resets=jnp.zeros_like(self.resets)
        _,p,v=self.net.apply(self.params,self.carry,(self.obs,resets))
        changed=self.obs.at[:4].set(100.) # outside final token's eight-step context
        _,q,w=self.net.apply(self.params,self.carry,(changed,resets))
        np.testing.assert_allclose(q.probs[-1],p.probs[-1],atol=2e-6)
        np.testing.assert_allclose(w[-1],v[-1],atol=2e-5)
        def loss(params):
            _,pi,value=self.net.apply(params,self.carry,(self.obs,self.resets))
            return -pi.log_prob(jnp.zeros((12,2),int)).mean()+jnp.square(value).mean()
        gradients=jax.grad(loss)(self.params)
        assert all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(gradients))
        assert any(np.any(a!=0) for a in jax.tree_util.tree_leaves(gradients['params']['temporal']))


if __name__=='__main__':unittest.main()

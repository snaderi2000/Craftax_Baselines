"""History ablation must affect rollout/replay identically and preserve weights."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from ppo_rnn_og import ActorCriticRNN,ScannedRNN


class GRUMemoryControlTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'LAYER_SIZE':16,'SYMBOLIC_ENCODER':'mlp'}
        self.normal=ActorCriticRNN(3,self.cfg)
        self.reset=ActorCriticRNN(3,dict(self.cfg,RESET_GRU_EVERY_STEP=True))
        self.obs=jax.random.normal(jax.random.PRNGKey(8),(7,3,11))
        self.dones=jnp.zeros((7,3),bool).at[0].set(True).at[4,1].set(True)
        self.hidden=ScannedRNN.initialize_carry(3,16)
        self.params=self.normal.init(jax.random.PRNGKey(50),self.hidden,(self.obs,self.dones))

    def test_identical_initial_parameters_and_default_behavior(self):
        other=self.reset.init(jax.random.PRNGKey(50),self.hidden,(self.obs,self.dones))
        self.assertEqual(jax.tree_util.tree_structure(self.params),jax.tree_util.tree_structure(other))
        for a,b in zip(jax.tree_util.tree_leaves(self.params),jax.tree_util.tree_leaves(other)):
            np.testing.assert_array_equal(a,b)
        explicit=ActorCriticRNN(3,dict(self.cfg,RESET_GRU_EVERY_STEP=False,GRU_CONCAT_OBSERVATION=False))
        a=self.normal.apply(self.params,self.hidden,(self.obs,self.dones))
        b=explicit.apply(self.params,self.hidden,(self.obs,self.dones))
        np.testing.assert_array_equal(a[1].logits,b[1].logits)

    def test_reset_outputs_depend_only_on_current_observation(self):
        _,pi,value=self.reset.apply(self.params,self.hidden,(self.obs,self.dones))
        changed=self.obs.at[:-1].set(-self.obs[:-1]*3)
        _,changed_pi,changed_value=self.reset.apply(self.params,jnp.ones_like(self.hidden),(changed,self.dones))
        np.testing.assert_array_equal(pi.logits[-1],changed_pi.logits[-1])
        np.testing.assert_array_equal(value[-1],changed_value[-1])
        _,normal_pi,_=self.normal.apply(self.params,self.hidden,(self.obs,self.dones))
        _,different_pi,_=self.normal.apply(self.params,self.hidden,(changed,self.dones))
        self.assertGreater(float(jnp.max(jnp.abs(normal_pi.logits[-1]-different_pi.logits[-1]))),1e-6)

    def test_streaming_collection_matches_sequence_replay_and_ratio_one(self):
        concat=ActorCriticRNN(3,dict(self.cfg,GRU_CONCAT_OBSERVATION=True))
        for net in (self.normal,self.reset,concat):
            params=net.init(jax.random.PRNGKey(50),self.hidden,(self.obs,self.dones))
            hidden=self.hidden;logits=[];values=[]
            for t in range(len(self.obs)):
                hidden,pi,v=net.apply(params,hidden,(self.obs[t:t+1],self.dones[t:t+1]))
                logits.append(pi.logits[0]);values.append(v[0])
            replay_hidden,pi,v=net.apply(params,self.hidden,(self.obs,self.dones))
            np.testing.assert_allclose(pi.logits,jnp.stack(logits),atol=1e-6)
            np.testing.assert_allclose(v,jnp.stack(values),atol=1e-6)
            np.testing.assert_allclose(hidden,replay_hidden,atol=1e-6)
            actions=jnp.zeros(self.dones.shape,jnp.int32)
            from distrax import Categorical
            ratio=jnp.exp(pi.log_prob(actions)-Categorical(logits=jnp.stack(logits)).log_prob(actions))
            np.testing.assert_allclose(ratio,1.,atol=1e-6)

    def test_concat_preserves_encoder_gru_initialization_and_hidden_dynamics(self):
        concat=ActorCriticRNN(3,dict(self.cfg,GRU_CONCAT_OBSERVATION=True))
        params=concat.init(jax.random.PRNGKey(50),self.hidden,(self.obs,self.dones))
        for name in ('Dense_0','ScannedRNN_0'):
            for a,b in zip(jax.tree_util.tree_leaves(params['params'][name]),jax.tree_util.tree_leaves(self.params['params'][name])):
                np.testing.assert_array_equal(a,b)
        normal_hidden,_,_=self.normal.apply(self.params,self.hidden,(self.obs,self.dones))
        concat_hidden,pi,value=concat.apply(params,self.hidden,(self.obs,self.dones))
        np.testing.assert_array_equal(normal_hidden,concat_hidden)
        self.assertEqual(pi.logits.shape,(7,3,3));self.assertEqual(value.shape,(7,3))
        size=lambda tree:sum(a.size for a in jax.tree_util.tree_leaves(tree))
        self.assertEqual(size(params)-size(self.params),2*16*16)

    def test_observation_bypass_works_when_recurrent_path_is_zero(self):
        from copy import deepcopy
        concat=ActorCriticRNN(3,dict(self.cfg,GRU_CONCAT_OBSERVATION=True))
        params=concat.init(jax.random.PRNGKey(50),self.hidden,(self.obs,self.dones))
        params=deepcopy(params)
        params['params']['ScannedRNN_0']=jax.tree_util.tree_map(jnp.zeros_like,params['params']['ScannedRNN_0'])
        grad=jax.grad(lambda obs:concat.apply(params,self.hidden,(obs,self.dones))[2][-1].sum())(self.obs)
        self.assertTrue(bool(jnp.isfinite(grad).all()))
        self.assertGreater(float(jnp.linalg.norm(grad[-1])),1e-6)
        np.testing.assert_array_equal(grad[:-1],0.)

    def test_memory_gradient_present_only_when_enabled(self):
        for net,enabled in ((self.normal,True),(self.reset,False)):
            grad=jax.grad(lambda obs:net.apply(self.params,self.hidden,(obs,self.dones))[2][-1].sum())(self.obs)
            self.assertTrue(bool(jnp.isfinite(grad).all()))
            if enabled:self.assertGreater(float(jnp.linalg.norm(grad[0])),1e-8)
            else:np.testing.assert_array_equal(grad[:-1],0.)


if __name__=='__main__':unittest.main()

"""PPO's sampling, likelihood ratio and entropy refer to the same policy."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from policy_temperature import tempered_policy


class TrainingTemperatureTests(unittest.TestCase):
    def test_default_keeps_distribution_and_samples_exactly(self):
        pi=distrax.Categorical(logits=jnp.array([[0.,1.,2.],[-1.,.5,.25]]))
        actual=tempered_policy(pi,1.)
        self.assertIs(pi,actual)
        key=jax.random.PRNGKey(71)
        np.testing.assert_array_equal(pi.sample(seed=key),actual.sample(seed=key))

    def test_rollout_and_update_likelihoods_agree_with_temperature(self):
        from models.actor_critic import ActorCritic
        net=ActorCritic(3,16)
        obs=jax.random.normal(jax.random.PRNGKey(72),(32,12))
        params=net.init(jax.random.PRNGKey(73),obs)
        raw,_=net.apply(params,obs)
        behavior=tempered_policy(raw,.25)
        actions=behavior.sample(seed=jax.random.PRNGKey(74))
        old_logp=behavior.log_prob(actions)
        def objective(p):
            raw,_=net.apply(p,obs)
            pi=tempered_policy(raw,.25)
            ratio=jnp.exp(pi.log_prob(actions)-old_logp)
            return ratio.mean()+.01*pi.entropy().mean(),ratio
        (_,ratio),grads=jax.value_and_grad(objective,has_aux=True)(params)
        np.testing.assert_allclose(ratio,jnp.ones(32),atol=1e-6)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(grads)))
        expected=jax.nn.softmax(raw.logits_parameter()/.25)
        np.testing.assert_allclose(behavior.probs,expected,rtol=2e-6,atol=2e-6)
        expected_entropy=-(expected*jnp.log(expected)).sum(-1)
        np.testing.assert_allclose(behavior.entropy(),expected_entropy,rtol=2e-6,atol=2e-6)

    def test_invalid_training_temperature_rejected(self):
        pi=distrax.Categorical(logits=jnp.array([0.,1.]))
        for t in (0.,-1.,float('inf'),float('nan')):
            with self.assertRaises(ValueError):tempered_policy(pi,t)


if __name__=='__main__':unittest.main()

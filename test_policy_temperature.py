"""Check unchanged baseline sampling and reduced-randomness action selection."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from evaluate_reward_shaped_ppo import select_policy_actions


class PolicyTemperatureTests(unittest.TestCase):
    def test_default_is_bitwise_identical_to_previous_sampler(self):
        keys=jax.random.split(jax.random.PRNGKey(87),128)
        logits=jax.random.normal(jax.random.PRNGKey(88),(128,17))
        expected=jax.vmap(lambda p,k:jax.random.categorical(k,p))(logits,keys)
        np.testing.assert_array_equal(select_policy_actions(logits,keys,1.),expected)

    def test_greedy_is_key_independent_and_low_temperature_concentrates(self):
        n=8192;logits=jnp.broadcast_to(jnp.array([0.,1.,2.]),(n,3))
        keys=jax.random.split(jax.random.PRNGKey(89),n)
        np.testing.assert_array_equal(select_policy_actions(logits,keys,0.),jnp.full(n,2))
        ordinary=select_policy_actions(logits,keys,1.)
        colder=select_policy_actions(logits,keys,.25)
        self.assertGreater(float((colder==2).mean()),float((ordinary==2).mean())+.2)
        for invalid in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):select_policy_actions(logits,keys,invalid)


if __name__=='__main__':unittest.main()

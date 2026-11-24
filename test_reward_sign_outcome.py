"""CPU tests: factual sign labels, replay balancing and actor-only gradients."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from models.actor_critic import ActorCritic
from reward_sign_outcome import (labels_from_rewards, initialize_head,
    empty_replay, append_replay, sample_replay, outcome_loss)


class RewardSignTests(unittest.TestCase):
    def test_labels_and_replay(self):
        rewards = jnp.array([-1., 0., .1, -.1, 0., 1.])
        np.testing.assert_array_equal(labels_from_rewards(rewards), [0,1,2,0,1,2])
        obs = jnp.arange(24.).reshape(6,4)
        b = append_replay(empty_replay(4,4), obs, jnp.arange(6), rewards)
        np.testing.assert_array_equal(b['counts'], [2,2,2])
        batch = sample_replay(b, jax.random.PRNGKey(1), 24)
        np.testing.assert_array_equal(np.bincount(np.asarray(batch[2])), [8,8,8])
        for row, action, label in zip(*batch[:3]):
            np.testing.assert_array_equal(row, obs[int(action)])
            self.assertEqual(int(label), int(labels_from_rewards(rewards)[int(action)]))
        one = append_replay(empty_replay(2,4), obs[:1], jnp.array([0]), jnp.array([0.]))
        np.testing.assert_array_equal(sample_replay(one,jax.random.PRNGKey(2),8)[2], np.ones(8))
        np.testing.assert_array_equal(sample_replay(empty_replay(2,4),jax.random.PRNGKey(2),8)[3], np.zeros(8))

    def test_gradients(self):
        key = jax.random.PRNGKey(0)
        obs = jnp.ones((6,4))
        net = ActorCritic(17,32)
        original = net.init(key,obs)
        params = initialize_head(original, key,32)
        before_pi, before_v = net.apply(original,obs)
        after_pi, after_v = net.apply(params,obs)
        np.testing.assert_array_equal(before_pi.logits_parameter(),after_pi.logits_parameter())
        np.testing.assert_array_equal(before_v,after_v)
        batch = (obs,jnp.arange(6),jnp.array([0,1,2,0,1,2]),jnp.ones(6))
        grads = jax.grad(outcome_loss)(params,batch)['params']
        for i in range(3,8):
            for g in jax.tree_util.tree_leaves(grads[f'Dense_{i}']):
                np.testing.assert_array_equal(g, np.zeros_like(g))
        self.assertGreater(float(jnp.linalg.norm(grads['Dense_0']['kernel'])),0)
        self.assertGreater(float(jnp.linalg.norm(grads['RewardSign']['kernel'])),0)
        empty = sample_replay(empty_replay(2,4),key,6)
        self.assertEqual(float(outcome_loss(params,empty)),0)


if __name__ == '__main__':
    unittest.main()

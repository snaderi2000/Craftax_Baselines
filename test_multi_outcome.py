import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.envs.craftax_state import Inventory
from craftax.craftax_classic.constants import Achievement
from models.actor_critic import ActorCritic
from multi_outcome import (OUTCOMES, N_OUTCOMES, MultiOutcomeWrapper, transition_labels,
    initialize_head, empty_replay, append_replay, sample_replay, outcome_loss)


class MultiOutcomeTests(unittest.TestCase):
    def test_labels_and_passthrough(self):
        before = SimpleNamespace(achievements=jnp.zeros(22,bool),inventory=Inventory())
        after = SimpleNamespace(achievements=before.achievements.at[Achievement.COLLECT_STONE.value].set(True),
                                inventory=Inventory(stone=1))
        target = transition_labels(before,after)
        self.assertEqual(N_OUTCOMES,34)
        self.assertEqual(int(target.sum()),2)
        self.assertTrue(target[OUTCOMES.index('achievement_collect_stone')])
        self.assertTrue(target[OUTCOMES.index('inventory_increase_stone')])
        repeated = SimpleNamespace(achievements=after.achievements,inventory=Inventory(stone=2))
        self.assertEqual(int(transition_labels(after,repeated).sum()),1)
        crafting = SimpleNamespace(achievements=after.achievements.at[Achievement.MAKE_STONE_PICKAXE.value].set(True),
                                  inventory=Inventory(stone=0,stone_pickaxe=1))
        target = transition_labels(after,crafting)
        self.assertEqual(int(target.sum()),2)
        self.assertTrue(target[OUTCOMES.index('inventory_increase_stone_pickaxe')])
        self.assertFalse(target[OUTCOMES.index('inventory_increase_stone')])
        class FakeEnv:
            def step(self, key, state, action, params):
                return jnp.zeros(4), after, jnp.asarray(-.1), jnp.asarray(True), {}
        obs, state, reward, done, info = MultiOutcomeWrapper(FakeEnv()).step(None,before,5,None)
        self.assertIs(state,after)
        self.assertAlmostEqual(float(reward),-.1,places=6)
        self.assertTrue(done)
        self.assertTrue(info['outcome/achievement_collect_stone'])

    def test_balanced_replay(self):
        obs = jnp.arange(24.).reshape(6,4)
        labels = jnp.tile(jnp.array([0,1,0,1,0,1],bool)[:,None],(1,N_OUTCOMES))
        b = append_replay(empty_replay(4,4),obs,jnp.arange(6),labels)
        np.testing.assert_array_equal(b['counts'],np.full(2*N_OUTCOMES,3))
        batch = sample_replay(b,jax.random.PRNGKey(1),2*N_OUTCOMES)
        counts = np.zeros((N_OUTCOMES,2),int)
        for row, action, output, label, mask in zip(*batch):
            np.testing.assert_array_equal(row,obs[int(action)])
            self.assertEqual(int(label),int(labels[int(action),int(output)]))
            self.assertEqual(float(mask),1)
            counts[int(output),int(label)]+=1
        np.testing.assert_array_equal(counts,np.ones_like(counts))
        none = append_replay(empty_replay(4,4),obs,jnp.arange(6),jnp.zeros_like(labels))
        np.testing.assert_array_equal(sample_replay(none,jax.random.PRNGKey(2),256)[3],np.zeros(256))
        np.testing.assert_array_equal(sample_replay(empty_replay(4,4),jax.random.PRNGKey(2),256)[4],np.zeros(256))

    def test_initialization_and_gradients(self):
        key = jax.random.PRNGKey(0)
        obs = jnp.ones((6,4))
        net = ActorCritic(17,32)
        original = net.init(key,obs)
        params = initialize_head(original,key,32)
        pi,v = net.apply(params,obs)
        p0,v0 = net.apply(original,obs)
        np.testing.assert_array_equal(pi.logits_parameter(),p0.logits_parameter())
        np.testing.assert_array_equal(v,v0)
        batch = (obs,jnp.arange(6),jnp.arange(6),jnp.array([0.,1.,0.,1.,0.,1.]),jnp.ones(6))
        grads = jax.grad(outcome_loss)(params,batch)['params']
        self.assertGreater(float(jnp.linalg.norm(grads['Dense_0']['kernel'])),0)
        for i in range(3,8):
            for g in jax.tree_util.tree_leaves(grads[f'Dense_{i}']):
                np.testing.assert_array_equal(g,np.zeros_like(g))
        self.assertEqual(float(outcome_loss(params,sample_replay(empty_replay(4,4),key,6))),0)


if __name__ == '__main__':
    unittest.main()

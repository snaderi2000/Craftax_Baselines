import unittest
import jax
import jax.numpy as jnp
import numpy as np
from cluster_free_practice import ClusterFreePractice, sampling_probabilities
from test_id_practice import FakeEnv


class ClusterFreeTests(unittest.TestCase):
    def setup_bank(self, horizon=30):
        env = ClusterFreePractice(FakeEnv(), 4, 'uniform', 1, 16, horizon, 0, 2)
        _, state = env.reset(jax.random.PRNGKey(0))
        for t in range(30):
            _, state, _, _, _ = env.step(jax.random.PRNGKey(t+1), state, jnp.zeros(4, jnp.int32))
        return env, state

    def test_snapshot_exact_offset_and_no_achievement_grouping(self):
        env, state = self.setup_bank()
        count = int(state.bank_total)
        self.assertEqual(count, 4)
        np.testing.assert_array_equal(state.bank.timestep[:count], np.full(count, 20))
        self.assertFalse(np.asarray(state.bank.achievements[:count, 0]).any())
        self.assertEqual(int(state.entry[-1]) >= 0, True)
        for t in range(6):
            _, state, _, done, _ = env.step(jax.random.PRNGKey(t+100), state, jnp.zeros(4, jnp.int32))
            if t == 4:
                self.assertFalse(bool(done[-1]))  # A reward does not end practice.
        self.assertEqual(int(state.bank_total), count)  # Practice events never enter bank.

    def test_bootstrap_target_and_true_terminal(self):
        env, state = self.setup_bank(horizon=2)
        for t in range(2):
            _, state, _, _, info = env.step(jax.random.PRNGKey(t+100), state, jnp.zeros(4, jnp.int32))
            state, errors = env.record_prediction(state, info, jnp.full(4, .2), jnp.full(4, 2.), .99)
        self.assertTrue(bool(info['cf/truncated'][-1]))
        self.assertAlmostEqual(float(errors[-1]), 2*.99**2-.2, places=5)
        entry = int(info['cf/entry'][-1])
        self.assertEqual(int(state.bank_info['updates'][entry]), 1)
        raw = state.raw.replace(timestep=state.raw.timestep.at[-1].set(29))
        state = state.replace(raw=raw, elapsed=jnp.zeros(4, jnp.int32))
        _, state, _, _, info = env.step(jax.random.PRNGKey(201), state, jnp.zeros(4, jnp.int32))
        state, errors = env.record_prediction(state, info, jnp.full(4, .2), jnp.full(4, 100.), .99)
        self.assertFalse(bool(info['cf/truncated'][-1]))
        self.assertAlmostEqual(float(errors[-1]), .2, places=5)

    def test_health_only_reward_is_not_admitted(self):
        env, state = self.setup_bank()
        before = state.candidate_count
        successor = state.raw.replace(player_health=state.raw.player_health+1)
        updated = env._record(state, successor, jnp.full(4, .1), jnp.zeros(4, bool), jax.random.PRNGKey(9))
        np.testing.assert_array_equal(updated.candidate_count, before)

    def test_priority_mixture_and_reuse_limit(self):
        usable = jnp.asarray([True, True, False])
        p = np.asarray(sampling_probabilities(jnp.asarray([.1, 2., 99.]), usable, True, jax.random.PRNGKey(0)))
        self.assertGreater(p[1], p[0]); self.assertEqual(p[2], 0.)
        np.testing.assert_allclose(p.sum(), 1.)
        env, state = self.setup_bank()
        for t in range(20):
            starts, entries, _, state = env._starts(jax.random.PRNGKey(100+t), state.raw, state,
                                                   jnp.asarray([False, False, False, True]))
        self.assertLessEqual(int(state.bank_info['reuse'].max()), 2)

    def test_identical_normal_warmup_across_conditions(self):
        environments = [ClusterFreePractice(FakeEnv(), 4, mode, 1, 16, 2, 1000, 2)
                        for mode in ['baseline', 'uniform', 'prioritized']]
        states = [env.reset(jax.random.PRNGKey(0))[1] for env in environments]
        for t in range(35):
            states = [env.step(jax.random.PRNGKey(t+1), state, jnp.zeros(4, jnp.int32))[1]
                      for env, state in zip(environments, states)]
        for state in states[1:]:
            for a, b in zip(jax.tree_util.tree_leaves(states[0]), jax.tree_util.tree_leaves(state)):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()

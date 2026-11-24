import unittest
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from wrappers import LogWrapper, AutoResetEnvWrapper, BatchEnvWrapper
from diamond_density_curriculum import enrich_map, DiamondDensityCurriculum


class DensityTests(unittest.TestCase):
    def test_exact_count_and_only_stone_changes(self):
        grid = jnp.full((16, 16), 4, dtype=jnp.int32).at[0, 0].set(10)
        grid = grid.at[0, 1].set(9).at[0, 2].set(8).at[0, 3].set(2)
        result = np.asarray(enrich_map(jax.random.PRNGKey(50), grid, 10))
        self.assertEqual(int((result == 10).sum()), 10)
        original = np.asarray(grid)
        changed = result != original
        self.assertTrue(np.all(original[changed] == 4))
        self.assertTrue(np.all(result[changed] == 10))
        np.testing.assert_array_equal(enrich_map(jax.random.PRNGKey(50), grid, 1), grid)

    def test_schedule_and_pass_through_rewards(self):
        base = BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(
            make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False))), 2)
        params = base.default_params.replace(max_timesteps=1)
        env = DiamondDensityCurriculum(base, start=4, probability=1., multiplier=10)
        key = jax.random.PRNGKey(50)
        obs, state = env.reset(key, params)
        self.assertFalse(np.asarray(state.dense).any())
        action = jnp.zeros(2, dtype=jnp.int32)
        for interactions in [2, 4]:
            key = jax.random.fold_in(key, interactions)
            normal_obs, normal_state, normal_reward, normal_done, _ = base.step(key, state.inner, action, params)
            obs, state, reward, done, _ = env.step(key, state, action, params)
            np.testing.assert_array_equal(reward, normal_reward)
            np.testing.assert_array_equal(done, normal_done)
            self.assertEqual(int(state.interactions), interactions)
            if interactions == 2:
                np.testing.assert_array_equal(obs, normal_obs)
                np.testing.assert_array_equal(state.inner.env_state.map, normal_state.env_state.map)
                self.assertFalse(np.asarray(state.dense).any())
            else:
                self.assertTrue(np.asarray(state.dense).all())
                for old, new in zip(np.asarray(normal_state.env_state.map), np.asarray(state.inner.env_state.map)):
                    changed = old != new
                    self.assertTrue(np.all(old[changed] == 4))
                    self.assertTrue(np.all(new[changed] == 10))
                    self.assertEqual(int((new == 10).sum()), min(10*int((old == 10).sum()), int((old == 4).sum())+int((old == 10).sum())))
                np.testing.assert_array_equal(state.inner.env_state.achievements, normal_state.env_state.achievements)


if __name__ == '__main__':
    unittest.main()

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from diamond_skill_drill import DiamondDrill, drill_position


class DrillTests(unittest.TestCase):
    def test_reachability_and_hidden_diamond(self):
        grid = jnp.full((16,16), 2, dtype=jnp.int32).at[8,8].set(10)
        valid, position, target = drill_position(jax.random.PRNGKey(50), grid)
        self.assertTrue(bool(valid))
        self.assertTrue(np.all(np.abs(np.asarray(position-target)) <= [3,4]))
        sealed = jnp.full((16,16), 4, dtype=jnp.int32).at[8,8].set(10)
        self.assertFalse(bool(drill_position(jax.random.PRNGKey(50), sealed)[0]))

    def test_fresh_resets_and_timeout_reward(self):
        base = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
        env = DiamondDrill(base, horizon=1)
        params = base.default_params
        reset = jax.jit(lambda key: env.reset(key, params))
        maps = []
        for seed in [50,51,52]:
            key = jax.random.PRNGKey(seed)
            obs, state = reset(key)
            s = state.env_state
            maps.append(np.asarray(s.map))
            self.assertTrue(np.asarray(obs[:1323]).reshape(7,9,21)[:,:,10].any())
            self.assertIn(int(s.map[s.player_position[0],s.player_position[1]]), [2,7,13])
            for name in ['wood_pickaxe','stone_pickaxe','iron_pickaxe']:
                self.assertEqual(int(getattr(s.inventory,name)),1)
            self.assertEqual(int(s.inventory.diamond),0)
            self.assertFalse(bool(s.achievements[19]))
            for name in ['player_health','player_food','player_drink','player_energy']:
                self.assertEqual(int(getattr(s,name)),9)
            expected_obs, expected_state, expected_reward, _, _ = base.step(key,s,0,params)
            actual_obs, actual_state, reward, done, info = env.step(key,state,0,params)
            self.assertTrue(bool(done))
            self.assertTrue(bool(info['drill/truncated']))
            self.assertFalse(bool(info['drill/success']))
            self.assertEqual(float(reward),float(expected_reward))
            np.testing.assert_array_equal(actual_obs,expected_obs)
            np.testing.assert_array_equal(info['drill/bootstrap_obs'],expected_obs)
            np.testing.assert_array_equal(actual_state.env_state.map,expected_state.map)
        self.assertFalse(np.array_equal(maps[0],maps[1]))
        self.assertFalse(np.array_equal(maps[1],maps[2]))


if __name__ == '__main__':
    unittest.main()

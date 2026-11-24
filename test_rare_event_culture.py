"""Oracle gating and pass-through tests without rolling out an agent."""
import unittest
from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
from rare_event_culture import DiamondOracleWrapper


class Passthrough:
    def step(self, rng, state, action, params):
        return jnp.zeros(2), state, jnp.asarray(0.75), jnp.asarray(False), {}


class OracleTests(unittest.TestCase):
    def state(self, diamond=True, pickaxe=1, collected=False):
        grid = jnp.zeros((12, 12), dtype=jnp.int32)
        if diamond:
            grid = grid.at[5, 6].set(10)
        return SimpleNamespace(map=grid, player_position=jnp.array([5, 5]),
                               inventory=SimpleNamespace(iron_pickaxe=pickaxe),
                               achievements=jnp.zeros(22, dtype=bool).at[19].set(collected))

    def test_gate_and_unchanged_transition(self):
        wrapper = DiamondOracleWrapper(Passthrough())
        for diamond, pickaxe, collected, expected in [(True, 1, False, True),
                                                      (False, 1, False, False),
                                                      (True, 0, False, False),
                                                      (True, 1, True, False)]:
            state = self.state(diamond, pickaxe, collected)
            obs, next_state, reward, done, info = wrapper.step(None, state, 0)
            self.assertEqual(bool(info['diamond_oracle']), expected)
            self.assertIs(next_state, state)
            self.assertEqual(float(reward), 0.75)
            self.assertFalse(bool(done))
            np.testing.assert_array_equal(obs, np.zeros(2))


if __name__ == '__main__':
    unittest.main()

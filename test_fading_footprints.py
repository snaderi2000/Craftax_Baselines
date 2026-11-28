"""Timing and orientation contract for the 914/1,024 SRU V3 trail."""
import unittest

import numpy as np
from craftax.craftax_classic.constants import Action, BlockType
from craftax.craftax_classic.renderer import render_craftax_symbolic

import test_symbolic_motion_inputs as motion_tests
from symbolic_motion_inputs import with_motion_inputs
from wrappers import GymnaxWrapper


class FootprintTest(unittest.TestCase):
    def test_move_block_revisit_reset(self):
        fixture = motion_tests.SymbolicMotionInputTests
        fixture.setUpClass()
        # Spawn (32,32), grass to the east and north, stone two tiles east.
        raw = fixture.raw.replace(map=fixture.raw.map.at[32, 34].set(BlockType.STONE.value))

        class FixedWorld(GymnaxWrapper):
            def reset(self, key, params):
                return render_craftax_symbolic(raw), raw

        decay = 2 ** (-1 / 100)
        # Each entry is a full observation's nonzero trail cells (row, column).
        sequence = [
            (None,         {(3, 4): 1}),
            (Action.RIGHT, {(3, 3): decay,    (3, 4): 1}),
            (Action.RIGHT, {(3, 3): decay**2, (3, 4): 1}),  # blocked by stone
            (Action.LEFT,  {(3, 4): 1,        (3, 5): decay}),  # revisit spawn
            (Action.UP,    {(3, 4): 1, (4, 4): decay, (4, 5): decay**2}),
            (None,         {(3, 4): 1}),  # new episode at the same spawn
        ]
        for mode in ('trail', 'control'):
            env = with_motion_inputs(FixedWorld(fixture.env), dict(
                MOTION_INPUTS=True, POSITION_TIME_INPUTS=True, SRU_EPISODE_MEMORY=mode))
            for action, cells in sequence:
                with self.subTest(mode=mode, action=action, cells=cells):
                    if action is None:
                        obs, state = env.reset(fixture.key, env.default_params)
                        self.assertEqual(int((state.visited >= 0).sum()), 1)
                    else:
                        obs, state, _, done, _ = env.step(
                            fixture.key, state, action.value, env.default_params)
                        self.assertFalse(bool(done))
                    self.assertEqual(obs.shape, (1462,))
                    np.testing.assert_array_equal(obs[1385:1399], 0)
                    expected = np.zeros((7, 9), dtype=np.float32)
                    if mode == 'trail':
                        for position, value in cells.items():
                            expected[position] = value
                    np.testing.assert_allclose(obs[1399:].reshape(7, 9), expected,
                                               rtol=1e-6, atol=0)

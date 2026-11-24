import unittest

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from failure_seed_replay import FailureSeedAutoReset, MAX_WEIGHT, ReplayState, SLOTS, next_world
from symbolic_motion_inputs import ClassicMotionInputWrapper
from wrappers import ClassicLavaPenaltyWrapper, LogWrapper


class FailureSeedReplayTests(unittest.TestCase):
    def test_weighted_sampling_retirement_and_capacity(self):
        keys = jax.random.split(jax.random.PRNGKey(29), SLOTS)
        state = ReplayState(None, keys[0], jnp.int32(-1), keys, jnp.zeros(SLOTS, jnp.int32))
        fresh, choice = jax.random.split(jax.random.PRNGKey(12))
        state = next_world(state, False, fresh, choice)
        np.testing.assert_array_equal(state.failures, [1]+[0]*(SLOTS-1))
        state = state.replace(slot=jnp.int32(0), world_key=keys[0])
        state = next_world(state, False, fresh, choice)
        self.assertEqual(int(state.failures[0]), 2)
        scores = state.failures.at[0].set(MAX_WEIGHT).at[1].set(1)
        state = state.replace(slot=jnp.int32(0), failures=scores)
        draws = jax.vmap(lambda k: next_world(state, False, fresh, k).slot)(
            jax.random.split(choice, 8192))
        counts = np.bincount(np.asarray(draws)+1, minlength=3) / len(draws)
        np.testing.assert_allclose(counts, [.5, .5*MAX_WEIGHT/(MAX_WEIGHT+1), .5/(MAX_WEIGHT+1)], atol=.025)
        retired = next_world(state, True, fresh, choice)
        self.assertEqual(int(retired.failures[0]), 0)
        self.assertNotEqual(int(retired.slot), 0)
        full = state.replace(slot=jnp.int32(-1), failures=jnp.ones(SLOTS, jnp.int32), world_key=fresh)
        skipped = next_world(full, False, fresh, choice)
        np.testing.assert_array_equal(skipped.failures, full.failures)
        np.testing.assert_array_equal(skipped.keys, full.keys)

    def test_actual_terrain_reward_and_terminal_before_reset(self):
        raw = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
        base = LogWrapper(ClassicMotionInputWrapper(ClassicLavaPenaltyWrapper(raw, 2.)))
        env = FailureSeedAutoReset(base)
        params = raw.default_params.replace(max_timesteps=2)
        key = jax.random.PRNGKey(791)
        initial, state = env.reset(key, params)
        rng = jax.random.PRNGKey(421)
        first, running, reward, done, _ = env.step(rng, state, 0, params)
        expected = base.step(jax.random.split(rng)[1], state.env_state, 0, params)
        np.testing.assert_array_equal(first, expected[0])
        self.assertEqual(float(reward), float(expected[2]))
        self.assertFalse(bool(done))
        self.assertFalse(np.asarray(running.failures).any())
        for seed in range(32):
            obs, restart, _, done, _ = env.step(jax.random.PRNGKey(seed), running, 0, params)
            self.assertTrue(bool(done))
            if int(restart.slot) >= 0:
                break
        else:
            self.fail('No replay sampled')
        np.testing.assert_array_equal(restart.world_key, key)
        np.testing.assert_array_equal(obs, initial)
        before = state.env_state
        for a, b in zip(jax.tree_util.tree_leaves(restart.env_state), jax.tree_util.tree_leaves(before)):
            np.testing.assert_array_equal(a, b)
        # Complete all 22 on the retry. Check the terminal state before autoreset.
        _, running, *_ = env.step(rng, restart, 0, params)
        logged = running.env_state
        motion = logged.env_state
        sim = motion.env_state
        motion = motion.replace(env_state=sim.replace(achievements=jnp.ones_like(sim.achievements)))
        running = running.replace(env_state=logged.replace(env_state=motion))
        _, cleared, _, done, info = env.step(jax.random.PRNGKey(8), running, 0, params)
        self.assertTrue(bool(done) and bool(info['curriculum/replayed_all22']))
        self.assertFalse(np.asarray(cleared.failures).any())
        self.assertEqual(int(cleared.slot), -1)

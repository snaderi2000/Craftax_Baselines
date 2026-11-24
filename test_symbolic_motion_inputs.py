"""Real transition alignment, collisions, resets and recurrent replay for motion inputs."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Action, BlockType
from craftax.craftax_classic.renderer import render_craftax_symbolic
from symbolic_motion_inputs import ClassicMotionInputWrapper, with_motion_inputs
from symbolic_map_memory import simulator_state
from wrappers import AutoResetEnvWrapper, BatchEnvWrapper, OptimisticResetVecEnvWrapper, ClassicLavaPenaltyWrapper


class SymbolicMotionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = jax.random.PRNGKey(912)
        cls.env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
        _, state = cls.env.reset(cls.key, cls.env.default_params)
        cls.raw = state.replace(map=jnp.full_like(state.map, BlockType.GRASS.value),
            mob_map=jnp.zeros_like(state.mob_map), player_position=jnp.array([32, 32]),
            player_direction=Action.RIGHT.value, player_health=9, player_food=9,
            player_drink=9, player_energy=9, is_sleeping=False,
            zombies=state.zombies.replace(mask=jnp.zeros_like(state.zombies.mask)),
            cows=state.cows.replace(mask=jnp.zeros_like(state.cows.mask)),
            skeletons=state.skeletons.replace(mask=jnp.zeros_like(state.skeletons.mask)),
            arrows=state.arrows.replace(mask=jnp.zeros_like(state.arrows.mask)))
        cls.motion = ClassicMotionInputWrapper(cls.env)

    def initialize(self, raw=None):
        raw = self.raw if raw is None else raw
        return self.motion.initialize(render_craftax_symbolic(raw), raw)

    def test_opt_in_and_reset_clear_suffix_without_changing_observation_or_world(self):
        self.assertIs(with_motion_inputs(self.env, {}), self.env)
        obs, state = self.motion.reset(self.key, self.env.default_params)
        expected, raw = self.env.reset(self.key, self.env.default_params)
        self.assertEqual(obs.shape, (1364,))
        self.assertEqual(self.motion.observation_space().shape, obs.shape)
        np.testing.assert_array_equal(obs[:1345], expected)
        np.testing.assert_array_equal(obs[1345:], 0.)
        for a, b in zip(jax.tree_util.tree_leaves(simulator_state(state)), jax.tree_util.tree_leaves(raw)):
            np.testing.assert_array_equal(a, b)

    def test_displacement_is_actual_xy_and_action_matches_the_preceding_transition(self):
        for action, delta in ((Action.LEFT, [-1, 0]), (Action.RIGHT, [1, 0]),
                              (Action.UP, [0, -1]), (Action.DOWN, [0, 1]), (Action.NOOP, [0, 0])):
            _, state = self.initialize()
            obs, state, r, done, info = self.motion.step(self.key, state, action.value, self.env.default_params)
            expected, raw, er, ed, ei = self.env.step(self.key, self.raw, action.value, self.env.default_params)
            np.testing.assert_array_equal(obs[:1345], expected)
            np.testing.assert_array_equal(obs[1345:1362], np.eye(17)[action.value])
            np.testing.assert_array_equal(obs[-2:], delta)
            self.assertEqual(float(r), float(er))
            self.assertEqual(bool(done), bool(ed))
            for a, b in zip(jax.tree_util.tree_leaves((state.env_state, info)), jax.tree_util.tree_leaves((raw, ei))):
                np.testing.assert_array_equal(a, b)

    def test_blocked_move_and_interaction_record_attempt_without_fictitious_motion(self):
        raw = self.raw.replace(map=self.raw.map.at[32, 33].set(BlockType.STONE.value))
        _, state = self.initialize(raw)
        obs, state, _, _, _ = self.motion.step(self.key, state, Action.RIGHT.value, self.env.default_params)
        np.testing.assert_array_equal(obs[-2:], [0, 0])
        self.assertEqual(int(obs[1345:1362].argmax()), Action.RIGHT.value)
        obs, _, _, _, _ = self.motion.step(self.key, state, Action.DO.value, self.env.default_params)
        np.testing.assert_array_equal(obs[-2:], [0, 0])
        self.assertEqual(int(obs[1345:1362].argmax()), Action.DO.value)

    def test_suffix_does_not_expose_absolute_position_or_unseen_terrain(self):
        first, _ = self.initialize()
        moved = self.raw.replace(player_position=jnp.array([20, 20]))
        other, _ = self.initialize(moved)
        np.testing.assert_array_equal(first, other)
        hidden = self.raw.replace(map=self.raw.map.at[10, 10].set(BlockType.DIAMOND.value))
        _, state = self.initialize(hidden)
        _, reference = self.initialize()
        a = self.motion.step(self.key, state, Action.LEFT.value, self.env.default_params)[0]
        b = self.motion.step(self.key, reference, Action.LEFT.value, self.env.default_params)[0]
        np.testing.assert_array_equal(a, b)

    def test_autoresets_zero_only_new_episode_inputs(self):
        _, terminal = self.initialize(self.raw.replace(timestep=9999))
        _, alive = self.initialize()
        states = jax.tree_util.tree_map(lambda a, b: jnp.stack((a, b)), terminal, alive)
        for env in (BatchEnvWrapper(AutoResetEnvWrapper(self.motion), 2),
                    OptimisticResetVecEnvWrapper(self.motion, 2, 1)):
            obs, state, _, done, _ = env.step(self.key, states,
                jnp.array([Action.LEFT.value, Action.RIGHT.value]), self.env.default_params)
            np.testing.assert_array_equal(done, [True, False])
            np.testing.assert_array_equal(obs[0, 1345:], 0.)
            np.testing.assert_array_equal(obs[1, -2:], [1, 0])
            np.testing.assert_array_equal(state.previous_position, state.env_state.player_position)
            # A new episode's next displacement must not subtract the old world's position.
            next_obs, _, _, _, _ = env.step(self.key, state,
                jnp.array([Action.NOOP.value]*2), self.env.default_params)
            np.testing.assert_array_equal(next_obs[:, -2:], 0.)

    def test_lava_reward_is_preserved_and_terminal_state_is_unwrapped(self):
        env = ClassicMotionInputWrapper(ClassicLavaPenaltyWrapper(self.env, 2.))
        raw = self.raw.replace(map=self.raw.map.at[32, 33].set(BlockType.LAVA.value))
        _, state = env.initialize(render_craftax_symbolic(raw), raw)
        obs, state, r, done, _ = env.step(self.key, state, Action.RIGHT.value, self.env.default_params)
        _, expected, er, ed, _ = env._env.step(self.key, raw, Action.RIGHT.value, self.env.default_params)
        self.assertTrue(bool(done))
        self.assertEqual(float(r), float(er))
        self.assertEqual(bool(done), bool(ed))
        np.testing.assert_array_equal(obs[-2:], [1, 0])
        np.testing.assert_array_equal(simulator_state(state).player_position, expected.player_position)

    def test_augmented_online_and_sequence_replay_match_and_motion_has_gradients(self):
        from ppo_rnn_og import ActorCriticRNN, ScannedRNN
        obs, state = self.initialize()
        sequence = [obs]
        for i, action in enumerate((Action.LEFT, Action.NOOP, Action.RIGHT, Action.DOWN)):
            obs, state, _, _, _ = self.motion.step(jax.random.fold_in(self.key, i), state, action.value, self.env.default_params)
            sequence.append(obs)
        sequence = jnp.stack(sequence)[:, None]
        reset = jnp.array([[True], [False], [False], [True], [False]])
        hidden = ScannedRNN.initialize_carry(1, 16)
        net = ActorCriticRNN(17, {'LAYER_SIZE':16, 'SYMBOLIC_ENCODER':'mlp', 'GRU_CONCAT_OBSERVATION':True})
        params = net.init(self.key, hidden, (sequence, reset))
        final, pi, values = net.apply(params, hidden, (sequence, reset))
        h = hidden
        online_logits, online_values = [], []
        for t in range(len(sequence)):
            h, p, v = net.apply(params, h, (sequence[t:t+1], reset[t:t+1]))
            online_logits.append(p.logits)
            online_values.append(v)
        np.testing.assert_allclose(pi.logits, jnp.concatenate(online_logits), atol=1e-6)
        np.testing.assert_allclose(values, jnp.concatenate(online_values), atol=1e-6)
        np.testing.assert_allclose(final, h, atol=1e-6)
        grads = jax.grad(lambda obs: net.apply(params, hidden, (obs, reset))[2].sum())(sequence)
        self.assertTrue(bool(jnp.isfinite(grads).all()))
        self.assertGreater(float(jnp.linalg.norm(grads[..., 1345:1362])), 0.)
        self.assertGreater(float(jnp.linalg.norm(grads[..., -2:])), 0.)


if __name__ == '__main__':
    unittest.main()

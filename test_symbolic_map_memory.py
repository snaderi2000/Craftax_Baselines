"""Observed-only terrain memory, odometry, resets, and temporal model checks."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Action, BlockType
from craftax.craftax_classic.renderer import render_craftax_symbolic
from symbolic_map_memory import ClassicMapMemoryWrapper, PLAYER_BIT, PLAYER_FEATURES, display_local_view, map_discovery_reward
from wrappers import AutoResetEnvWrapper, OptimisticResetVecEnvWrapper, BatchEnvWrapper


class SymbolicMapMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = jax.random.PRNGKey(912)
        cls.env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
        _, state = cls.env.reset(cls.key, cls.env.default_params)
        cls.state = state.replace(map=jnp.full_like(state.map, BlockType.GRASS.value),
            mob_map=jnp.zeros_like(state.mob_map), player_position=jnp.array([32, 32]),
            player_direction=Action.RIGHT.value, player_health=9, player_food=9,
            player_drink=9, player_energy=9, is_sleeping=False,
            zombies=state.zombies.replace(mask=jnp.zeros_like(state.zombies.mask)),
            cows=state.cows.replace(mask=jnp.zeros_like(state.cows.mask)),
            skeletons=state.skeletons.replace(mask=jnp.zeros_like(state.skeletons.mask)),
            arrows=state.arrows.replace(mask=jnp.zeros_like(state.arrows.mask)))
        cls.memory = ClassicMapMemoryWrapper(cls.env, 33)

    def initialize(self, state=None):
        state = self.state if state is None else state
        return self.memory.initialize(render_craftax_symbolic(state), state)

    def test_initial_view_contains_only_observed_terrain(self):
        obs, state = self.initialize()
        self.assertEqual(obs.shape, self.memory.observation_space().shape)
        self.assertEqual(int((state.terrain > 0).sum()), 63)
        self.assertEqual(int(state.terrain[0, 0]), 0)
        np.testing.assert_array_equal(obs[:1345], render_craftax_symbolic(self.state))
        # Changing unobserved simulator terrain cannot alter the memory input.
        changed = self.state.replace(map=self.state.map.at[10, 10].set(BlockType.DIAMOND.value))
        other, _ = self.initialize(changed)
        np.testing.assert_array_equal(obs, other)

    def test_new_cells_reward_only_first_observations_inside_the_board(self):
        tracked = ClassicMapMemoryWrapper(self.env, 33, track_coverage=True)
        for pos, actions, expected in [
                ((32, 32), (Action.NOOP, Action.RIGHT, Action.LEFT, Action.RIGHT, Action.RIGHT), (0, 7, 0, 0, 7)),
                ((0, 0), (Action.LEFT, Action.RIGHT, Action.DOWN), (0, 4, 6))]:
            raw = self.state.replace(player_position=jnp.array(pos))
            obs, state = tracked.initialize(render_craftax_symbolic(raw), raw)
            old_obs, old_state = self.initialize(raw)
            np.testing.assert_array_equal(obs, old_obs)
            for i, (action, count) in enumerate(zip(actions, expected)):
                key = jax.random.fold_in(self.key, i)
                before = int((state.terrain != 0).sum())
                obs, state, reward, done, info = tracked.step(key, state, action.value, self.env.default_params)
                old_obs, old_state, old_reward, old_done, _ = self.memory.step(key, old_state, action.value, self.env.default_params)
                self.assertEqual(int(info['map_new_cells']), count)
                self.assertEqual(int((state.terrain != 0).sum()) - before, count)
                self.assertAlmostEqual(float(map_discovery_reward(info['map_new_cells'], done, 0.001)), count * 0.001)
                np.testing.assert_array_equal(obs, old_obs)
                self.assertEqual(float(reward), float(old_reward))
                self.assertEqual(bool(done), bool(old_done))

    def test_map_coverage_terminal_step_and_autoreset(self):
        tracked = ClassicMapMemoryWrapper(self.env, 33, track_coverage=True)
        for raw in (self.state.replace(timestep=9999),
                    self.state.replace(map=self.state.map.at[32, 33].set(BlockType.LAVA.value))):
            _, state = tracked.initialize(render_craftax_symbolic(raw), raw)
            for wrapper in (BatchEnvWrapper(AutoResetEnvWrapper(tracked), 1), OptimisticResetVecEnvWrapper(tracked, 1, 1)):
                batched = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], state)
                _, reset, _, done, info = wrapper.step(self.key, batched, jnp.array([Action.RIGHT.value]), self.env.default_params)
                self.assertTrue(bool(done[0]))
                self.assertEqual(int(info['map_new_cells'][0]), 7)
                self.assertEqual(float(map_discovery_reward(info['map_new_cells'], done, 0.001)[0]), 0.)
                self.assertEqual(int(reset.elapsed[0]), 0)
                _, _, _, _, first = wrapper.step(self.key, reset, jnp.array([Action.NOOP.value]), self.env.default_params)
                self.assertEqual(int(first['map_new_cells'][0]), 0)

    def test_fixed_board_keeps_dated_terrain_and_moves_current_occupants(self):
        memory = ClassicMapMemoryWrapper(self.env, global_readout=True, single_map=True, board_only=True)
        raw = self.state.replace(map=self.state.map.at[32, 33].set(BlockType.WATER.value))
        live = render_craftax_symbolic(raw)
        # Two distinct visible entity channels can occupy the same cell.
        grid = live[:-PLAYER_FEATURES].reshape(7, 9, 21).at[3, 5, 17].set(1).at[3, 5, 20].set(1)
        obs, state = memory.initialize(live.at[:-PLAYER_FEATURES].set(grid.ravel()), raw)
        cells = obs[:-PLAYER_FEATURES].reshape(64, 64, 3)
        self.assertEqual(obs.shape, memory.observation_space().shape)
        np.testing.assert_array_equal(cells[32, 33], [BlockType.WATER.value+1, 0, 9])
        self.assertEqual(int(cells[32, 32, 2]), PLAYER_BIT)
        self.assertEqual(int((cells[..., 0] != 0).sum()), 63)
        np.testing.assert_array_equal(obs[-PLAYER_FEATURES:], live[-PLAYER_FEATURES:])
        np.testing.assert_array_equal(display_local_view(obs, raw, {'MAP_BOARD_ONLY': True}), live)

        moved = raw.replace(player_position=jnp.array([0, 0]), map=raw.map.at[32, 33].set(BlockType.DIAMOND.value))
        obs, _ = memory.observe(render_craftax_symbolic(moved),
                                state.replace(env_state=moved, position=moved.player_position, elapsed=10))
        cells = obs[:-PLAYER_FEATURES].reshape(64, 64, 3)
        np.testing.assert_allclose(cells[32, 33], [BlockType.WATER.value+1, np.log1p(10)/np.log(10001.), 0])
        self.assertEqual(int(cells[32, 32, 2]), 0)
        self.assertEqual(int(cells[0, 0, 2]), PLAYER_BIT)
        self.assertEqual(int((cells[..., 0] != 0).sum()), 63 + 4*5)
        self.assertEqual(int(cells[-1, -1, 0]), 0)

    def test_fixed_board_heads_receive_only_board_features(self):
        from models.actor_critic_map_mlp import map_mlp_network
        memory = ClassicMapMemoryWrapper(self.env, global_readout=True, single_map=True, board_only=True)
        obs, _ = memory.initialize(render_craftax_symbolic(self.state), self.state)
        net = map_mlp_network(dict(MAP_BOARD_ONLY=True, MAP_SINGLE_READOUT=True,
                                   MAP_GLOBAL_READOUT=True, LAYER_SIZE=32))
        weights = net.init(self.key, obs)
        self.assertEqual(weights['params']['world_positions']['kernel'].shape, (4096, 64))
        for head in ('Dense_0', 'Dense_4'):
            self.assertEqual(weights['params']['actor_critic'][head]['kernel'].shape[0], 18*64)
        def both_outputs(batch):
            policy, value = net.apply(weights, batch)
            return policy.logits, value
        # Player attributes can affect a decision only through the marked world cell.
        changed = obs.at[-PLAYER_FEATURES:].set(0)
        original_board = jnp.stack((obs, changed))
        marker_field = (32*64+32) * 3 + 2
        for outputs in both_outputs(original_board.at[:, marker_field].set(0)):
            np.testing.assert_array_equal(outputs[0], outputs[1])
        for outputs in both_outputs(original_board):
            self.assertFalse(np.array_equal(outputs[0], outputs[1]))
        with self.assertRaises(ValueError):
            net.apply(weights, jnp.zeros(1345))

    def test_actual_displacement_handles_collisions_and_preserves_rewards(self):
        for tile, delta in ((BlockType.STONE, [0, 0]), (BlockType.GRASS, [0, 1])):
            raw = self.state.replace(map=self.state.map.at[32, 33].set(tile.value))
            _, state = self.initialize(raw)
            actual = self.memory.step(self.key, state, Action.RIGHT.value, self.env.default_params)
            expected = self.env.step(self.key, raw, Action.RIGHT.value, self.env.default_params)
            np.testing.assert_array_equal(actual[1].position-state.position, delta)
            self.assertEqual(float(actual[2]), float(expected[2]))
            self.assertEqual(bool(actual[3]), bool(expected[3]))
            for a, b in zip(jax.tree_util.tree_leaves(actual[1].env_state),
                            jax.tree_util.tree_leaves(expected[1])):
                np.testing.assert_array_equal(a, b)

    def test_remembers_offscreen_terrain_and_refreshes_changed_tiles(self):
        raw = self.state.replace(map=self.state.map.at[32, 28].set(BlockType.WATER.value))
        _, state = self.initialize(raw)
        for step in range(8):
            obs, state, _, _, _ = self.memory.step(jax.random.fold_in(self.key, step),
                state, Action.RIGHT.value, self.env.default_params)
        self.assertEqual(int(state.terrain[32, 28]), BlockType.WATER.value+1)
        self.assertLess(int(state.last_seen[32, 28]), int(state.elapsed))
        terrain = obs[1345:1345+33**2].reshape((33, 33))
        self.assertEqual(int(terrain[16, 4]), BlockType.WATER.value+1)
        # The backing map also retains locations beyond the 33x33 readout.
        for step in range(12):
            _, state, _, _, _ = self.memory.step(jax.random.fold_in(self.key, step+20),
                state, Action.RIGHT.value, self.env.default_params)
        self.assertEqual(int(state.terrain[32, 28]), BlockType.WATER.value+1)
        # A hidden modification is not revealed until it is observed again.
        state = state.replace(env_state=state.env_state.replace(
            map=state.env_state.map.at[32, 28].set(BlockType.STONE.value)))
        obs, state = self.memory.observe(render_craftax_symbolic(state.env_state), state)
        self.assertEqual(int(state.terrain[32, 28]), BlockType.WATER.value+1)
        for step in range(20):
            obs, state, _, _, _ = self.memory.step(jax.random.fold_in(self.key, step+8),
                state, Action.LEFT.value, self.env.default_params)
        self.assertEqual(int(state.terrain[32, 28]), BlockType.STONE.value+1)
        self.assertEqual(int(state.last_seen[32, 28]), int(state.elapsed))

    def test_map_boundary_does_not_wrap_writes_or_reveal_unseen_tiles(self):
        raw = self.state.replace(player_position=jnp.array([0, 0]))
        _, state = self.initialize(raw)
        self.assertEqual(int((state.terrain > 0).sum()), 4*5)
        self.assertEqual(int(state.terrain[-1, -1]), 0)

    def test_global_readout_retains_the_age_of_terrain_outside_the_crop(self):
        memory = ClassicMapMemoryWrapper(self.env, 33, global_readout=True)
        raw = self.state.replace(map=self.state.map.at[32, 28].set(BlockType.WATER.value))
        obs, state = memory.initialize(render_craftax_symbolic(raw), raw)
        for step in range(20):
            obs, state, *_ = memory.step(jax.random.fold_in(self.key, step), state,
                                       Action.RIGHT.value, self.env.default_params)
        self.assertGreater(int(state.position[1])-28, 16)
        self.assertEqual(obs.shape, memory.observation_space().shape)
        crop, board, ages = obs[1345:2434], obs[3523:7619], obs[7619:-2]
        self.assertFalse(jnp.any(crop == BlockType.WATER.value+1))
        self.assertEqual(int(board.reshape(64, 64)[32, 28]), BlockType.WATER.value+1)
        self.assertAlmostEqual(float(ages.reshape(64, 64)[32, 28]),
                               np.log1p(20)/np.log(10001.), places=6)
        self.assertEqual(int(board[0]), 0)
        np.testing.assert_array_equal(jnp.rint(obs[-2:]*63), state.position)

    def test_standard_and_optimistic_autoresets_clear_only_finished_maps(self):
        _, terminal = self.initialize(self.state.replace(timestep=9999))
        terminal = terminal.replace(terrain=terminal.terrain.at[2, 2].set(9),
                                    last_seen=terminal.last_seen.at[2, 2].set(0))
        _, alive = self.initialize()
        batched = jax.tree_util.tree_map(lambda a, b: jnp.stack((a, b)), terminal, alive)
        for env in (BatchEnvWrapper(AutoResetEnvWrapper(self.memory), 2),
                    OptimisticResetVecEnvWrapper(self.memory, 2, 1)):
            obs, state, _, done, _ = env.step(self.key, batched,
                jnp.array([Action.NOOP.value]*2), self.env.default_params)
            np.testing.assert_array_equal(done, [True, False])
            np.testing.assert_array_equal(state.elapsed, [0, 1])
            self.assertLessEqual(int((state.terrain[0] > 0).sum()), 63)
            self.assertEqual(obs.shape, (2, 3523))

    def test_sequence_matches_online_gradients_reach_memory_encoder(self):
        from ppo_rnn_og import ActorCriticRNN, ScannedRNN
        obs, _ = self.initialize()
        sequence = jnp.broadcast_to(obs, (4, 2, obs.shape[0]))
        resets = jnp.array([[False, False], [False, True], [True, False], [False, False]])
        net = ActorCriticRNN(17, {'LAYER_SIZE':32, 'SYMBOLIC_ENCODER':'spatial_map', 'MAP_VIEW_SIZE':33})
        hidden = ScannedRNN.initialize_carry(2, 32)
        params = net.init(self.key, hidden, (sequence, resets))
        final, pi, values = net.apply(params, hidden, (sequence, resets))
        h = hidden; online = []
        for t in range(4):
            h, _, v = net.apply(params, h, (sequence[t:t+1], resets[t:t+1]))
            online.append(v)
        np.testing.assert_allclose(values, jnp.concatenate(online), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(final, h, atol=2e-5, rtol=2e-5)
        grads = jax.grad(lambda p: net.apply(p, hidden, (sequence, resets))[2].sum())(params)
        self.assertTrue(np.isfinite(np.asarray(values)).all())
        self.assertTrue(np.isfinite(np.asarray(pi.logits)).all())
        for name in ('memory_conv_0', 'memory_conv_1', 'memory_conv_2'):
            self.assertGreater(float(jnp.linalg.norm(grads['params'][name]['kernel'])), 0.)


if __name__ == '__main__':
    unittest.main()

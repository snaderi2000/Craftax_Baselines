"""A feedforward map reader uses memory and has no convolution or recurrence."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from models.actor_critic_map_mlp import MapMLPActorCritic,map_mlp_network


class MapMLPTests(unittest.TestCase):
    def test_board_adapter_uses_frozen_v1_hidden_interface(self):
        from experiments.distill_frozen_v1 import v1_targets
        from models.actor_critic import ActorCritic
        from symbolic_map_memory import BOARD_OBS_SIZE
        teacher = map_mlp_network({'LAYER_SIZE': 16})
        x = jax.random.uniform(jax.random.PRNGKey(16), (3, 3523))
        trained = teacher.init(jax.random.PRNGKey(17), x)
        pi, value, hidden = v1_targets(teacher, trained, x)
        cfg = dict(LAYER_SIZE=16, MAP_V1_ATTENTION_ADAPTER=True, MAP_SINGLE_READOUT=True,
                   MAP_GLOBAL_READOUT=True, MAP_BOARD_ONLY=True)
        student = map_mlp_network(cfg)
        obs = jnp.zeros((3, BOARD_OBS_SIZE)).at[:, 0].set(3.).at[:, 2].set(16.)
        params = student.init(jax.random.PRNGKey(18), obs)['params']
        frozen = {k: trained['params']['actor_critic'][k] for k in params['actor_critic']}
        copied_pi, copied_v = ActorCritic(17,16).apply({'params': frozen}, hidden, from_first_hidden=True)
        np.testing.assert_array_equal(copied_pi.probs, pi.probs)
        np.testing.assert_array_equal(copied_v, value)
        params['actor_critic'] = frozen
        # Observation ages at unobserved cells cannot influence the attention read.
        changed = obs.at[:, 301].set(1.)
        for before, after in zip(student.apply({'params':params}, obs, method=student.with_hidden)[1:],
                                 student.apply({'params':params}, changed, method=student.with_hidden)[1:]):
            np.testing.assert_array_equal(before, after)

    def test_widened_heads_retain_adam_moments_and_can_use_new_units(self):
        import optax
        from flax.training.train_state import TrainState
        from experiments.distill_single_map import extend_board_state
        from symbolic_map_memory import BOARD_OBS_SIZE
        cfg = dict(MAP_BOARD_ONLY=True, MAP_GLOBAL_READOUT=True, MAP_SINGLE_READOUT=True)
        obs = jnp.zeros((1, BOARD_OBS_SIZE))
        marker = (32*64+32)*3
        obs = obs.at[0, marker:marker+3].set(jnp.array([4., 0., 16.]))
        def create(width):
            net = map_mlp_network(dict(cfg, LAYER_SIZE=width))
            return TrainState.create(apply_fn=net.apply, params=net.init(jax.random.PRNGKey(width), obs),
                                     tx=optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-4)))
        original = create(8)
        original = original.apply_gradients(grads=jax.tree_util.tree_map(jnp.ones_like, original.params))
        widened = extend_board_state(original, create(12))
        p0, v0 = original.apply_fn(original.params, obs)
        p1, v1 = widened.apply_fn(widened.params, obs)
        np.testing.assert_allclose(p0.probs,p1.probs,atol=1e-6)
        np.testing.assert_allclose(v0,v1,atol=1e-6)
        self.assertEqual(int(widened.opt_state[1][0].count), int(original.step))
        for old_m, new_m in zip(original.opt_state[1][0][1:], widened.opt_state[1][0][1:]):
            kernel = lambda moments: moments['params']['actor_critic']['Dense_1']['kernel']
            np.testing.assert_array_equal(kernel(new_m)[:8,:8], kernel(old_m))
            np.testing.assert_array_equal(kernel(new_m)[8:,:], 0.)
            np.testing.assert_array_equal(kernel(new_m)[:,8:], 0.)
        def loss(params):
            pi, value = widened.apply_fn(params, obs)
            return pi.logits[0, 0] + value[0]
        grads = jax.grad(loss)(widened.params)['params']['actor_critic']
        for head in ('Dense_3','Dense_7'):
            self.assertTrue(np.any(np.asarray(grads[head]['kernel'][8:]) != 0))

    def test_player_attention_tracks_relative_cells_and_masks_unknowns(self):
        from flax.core import unfreeze
        from models.actor_critic_map_mlp import PlayerBoardAttention
        reader = PlayerBoardAttention(8)
        index = 12*64 + 9
        cells = jnp.zeros((4096, 4)).at[index].set(jnp.array([1., 0., 0., 1.]))
        cells = cells.at[index+3].set(jnp.array([0., 2., 1., 0.]))
        terrain = jnp.zeros(4096, int).at[jnp.array([index,index+3])].set(1)
        player = jnp.arange(4096) == index
        params = unfreeze(reader.init(jax.random.PRNGKey(21), cells, terrain, player))
        np.testing.assert_array_equal(reader.apply(params, cells, terrain, player), 0.)
        params['params']['output']['kernel'] = jax.random.normal(jax.random.PRNGKey(22), (7, 8))
        read = reader.apply(params, cells, terrain, player)
        np.testing.assert_array_equal(read, reader.apply(params, cells.at[0].set(100.), terrain, player))
        shift = 30*64 + 18
        translated = [jnp.roll(x, shift, axis=0) for x in (cells, terrain, player)]
        np.testing.assert_allclose(read, reader.apply(params, *translated), atol=1e-6)
        self.assertFalse(np.allclose(read, reader.apply(params, cells, terrain, jnp.roll(player, 3))))

    def test_fine_gate_keeps_quadrant_addresses(self):
        from flax.core import unfreeze
        from models.actor_critic_map_mlp import GlobalMapReader
        reader = GlobalMapReader(width=32, spatial_regions=True, fine_shortcut=True)
        grid = jnp.arange(4096, dtype=jnp.float32).reshape(64, 64)
        obs = jnp.zeros((1,8194)).at[0,4096:8192].set(grid.flatten())
        weights = unfreeze(reader.init(jax.random.PRNGKey(713), obs))
        params = weights['params']
        params['patches']['kernel'] = jnp.zeros_like(params['patches']['kernel']).at[0,0,4,0].set(1.)
        params['regions']['kernel'] = jnp.zeros_like(params['regions']['kernel'])
        np.testing.assert_array_equal(reader.apply(weights, obs), 0.)
        params['fine_gate'] = jnp.ones(32)
        result = np.asarray(reader.apply(weights, obs))[0]
        for quadrant, (row,col) in enumerate(((0,0),(0,4),(4,0),(4,4))):
            np.testing.assert_array_equal(result[:,:,quadrant*8], grid[row::8,col::8])

    def test_spatial_regions_keep_order_and_relative_coordinates(self):
        from flax.core import unfreeze
        from models.actor_critic_map_mlp import GlobalMapReader
        reader = GlobalMapReader(width=32, spatial_regions=True)
        memory = jnp.zeros((1, 8194)).at[0, -2:].set(jnp.array([10., 20.]) / 63)
        params = unfreeze(reader.init(jax.random.PRNGKey(210), memory))
        # Route the terrain age and relative coordinates of cell (3,5) in
        # each 8x8 region to channels 0..2, with a bias to keep ReLUs active.
        kernel = jnp.zeros_like(params['params']['patches']['kernel'])
        for i, feature in enumerate((4,6,7)):
            kernel = kernel.at[3,1,feature,i].set(1.)
        params['params']['patches'].update(kernel=kernel, bias=jnp.ones(8))
        kernel = jnp.zeros_like(params['params']['regions']['kernel'])
        for i in range(3):
            kernel = kernel.at[0,1,i,i].set(1.)
        params['params']['regions']['kernel'] = kernel
        row, col = 19, 37
        reading = memory.at[0,4096+row*64+col].set(.75)
        grid = reader.apply(params, reading)
        self.assertEqual(grid.shape, (1,8,8,32))
        np.testing.assert_allclose(grid[0,2,4,:3], [1.75, 1+(row-10)/63, 1+(col-20)/63], atol=1e-6)
        self.assertEqual(float(grid[0,2,5,0]), 1.)
        moved = reader.apply(params, reading.at[0,-1].add(1/63))
        np.testing.assert_allclose(moved[0,2,4,:3]-grid[0,2,4,:3], [0,0,-1/63], atol=1e-6)
        cfg = dict(LAYER_SIZE=32, MAP_GLOBAL_READOUT=True, MAP_SINGLE_READOUT=True, MAP_SPATIAL_REGIONS=True)
        network = map_mlp_network(cfg)
        weights = network.init(jax.random.PRNGKey(211), jnp.zeros((1,9539)))['params']
        for head in ('Dense_0', 'Dense_4'):
            self.assertEqual(weights['actor_critic'][head]['kernel'].shape, (3393,32))
        self.assertEqual(set(weights), {'actor_critic', 'global_reader'})
        for incompatible in ('MAP_INTERNAL_CROP', 'MAP_PATCH_MLP', 'MAP_FULL_TERRAIN'):
            with self.assertRaises(ValueError):
                map_mlp_network(dict(cfg, **{incompatible: True}))

    def test_observed_world_input_reproduces_crop_reader_at_edges(self):
        from symbolic_map_memory import ClassicMapMemoryWrapper, MapMemoryState
        positions = jnp.array([(0, 0), (0, 63), (63, 0), (63, 63), (32, 32), (25, -1)])
        terrain = (jnp.arange(4096).reshape(64, 64) % 18).astype(jnp.uint8)
        last = jnp.where(terrain != 0, terrain.astype(jnp.int32), -1)
        live = jnp.zeros(1345)
        full = ClassicMapMemoryWrapper(None, global_readout=True)
        single = ClassicMapMemoryWrapper(None, global_readout=True, single_map=True)
        def emit(pos):
            state = MapMemoryState(None, terrain, last, pos, pos, jnp.array(120))
            return full.observe(live, state)[0], single.observe(live, state)[0]
        original, compact = jax.vmap(emit)(positions)
        net = MapMLPActorCritic(width=32, global_readout=True)
        weights = net.init(jax.random.PRNGKey(22), original)
        p, v = net.apply(weights, original)
        other_p, other_v = net.apply(weights, compact, method=net.from_observed_world)
        np.testing.assert_array_equal(other_p.logits, p.logits)
        np.testing.assert_array_equal(other_v, v)
        one_map = map_mlp_network(dict(LAYER_SIZE=32, MAP_GLOBAL_READOUT=True,
                                       MAP_SINGLE_READOUT=True, MAP_INTERNAL_CROP=True))
        derived_p, derived_v = one_map.apply(weights, compact)
        np.testing.assert_array_equal(derived_p.logits, p.logits)
        np.testing.assert_array_equal(derived_v, v)

    def test_dense_patch_reader_preserves_world_and_within_patch_coordinates(self):
        from flax.core import unfreeze
        from models.actor_critic_map_mlp import GlobalMapReader
        reader = GlobalMapReader(width=32, features_only=True, patch_mlp=True)
        memory = jnp.zeros((1, 8194))
        params = unfreeze(reader.init(jax.random.PRNGKey(42), memory))
        weights = params['params']
        # Read the age of cell (3, 5) within the patch at (2, 4).
        patch = 2*8+4
        weights['patch_weights'] = jnp.zeros_like(weights['patch_weights']).at[patch, (3*8+5)*6+4, 0].set(1.)
        weights['features']['kernel'] = jnp.zeros_like(weights['features']['kernel']).at[patch*4, 0].set(1.)
        for row, col, expected in ((19,37,0.75), (19,38,0.), (27,37,0.)):
            obs = memory.at[0,4096+row*64+col].set(0.75)
            self.assertEqual(float(reader.apply(params, obs)[0,0]), expected)

    def test_single_map_transfer_on_initial_observation(self):
        from craftax.craftax_env import make_craftax_env_from_name
        from symbolic_map_memory import with_map_memory
        from models.actor_critic_map_mlp import transfer_map_mlp_params
        raw = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', False)
        cfg = dict(SYMBOLIC_ENCODER='map_mlp', MAP_GLOBAL_READOUT=True, LAYER_SIZE=32)
        compact_cfg = dict(cfg, MAP_SINGLE_READOUT=True)
        key = jax.random.PRNGKey(17)
        teacher_obs, _ = with_map_memory(raw, cfg).reset(key, raw.default_params)
        env = with_map_memory(raw, compact_cfg)
        obs, memory = env.reset(key, raw.default_params)
        self.assertEqual(obs.shape, (9539,))
        self.assertEqual(env.observation_space().shape, obs.shape)
        np.testing.assert_array_equal(obs[:1345], teacher_obs[:1345])
        np.testing.assert_array_equal(obs[1345:], teacher_obs[3523:])
        seen = np.asarray(memory.last_seen >= 0).ravel()
        self.assertEqual(seen.sum(), 63)
        np.testing.assert_array_equal(obs[1345:5441] > 0, seen)
        np.testing.assert_array_equal(obs[5441:-2], ~seen)
        teacher, student = map_mlp_network(cfg), map_mlp_network(compact_cfg)
        original = teacher.init(key, teacher_obs[None])
        regional = map_mlp_network(dict(compact_cfg, MAP_SPATIAL_REGIONS=True))
        migrated = transfer_map_mlp_params(regional.init(key, obs[None]), original)
        # At reset, the live window contains all known terrain. A teacher with
        # its global residual still at zero can be reproduced by the live path.
        reference_pi, reference_v = teacher.apply(original, teacher_obs[None])
        regional_pi, regional_v = regional.apply(migrated, obs[None])
        np.testing.assert_allclose(regional_pi.logits, reference_pi.logits, atol=2e-5)
        np.testing.assert_allclose(regional_v, reference_v, atol=2e-5)
        # Exercise a trained (nonzero) global readout as well as the live branch.
        from flax.core import unfreeze
        original = unfreeze(original)
        residual = original['params']['global_reader']['residual']['kernel']
        original['params']['global_reader']['residual']['kernel'] = jax.random.normal(key, residual.shape) * 0.02
        transferred = transfer_map_mlp_params(student.init(key, obs[None]), original)
        pi, value = student.apply(transferred, obs[None])
        original_pi, original_value = teacher.apply(original, teacher_obs[None])
        # At this test width the shared live weights fit in rank 128 exactly.
        np.testing.assert_allclose(pi.logits, original_pi.logits, atol=2e-5)
        np.testing.assert_allclose(value, original_value, atol=2e-5)

    def test_memory_changes_outputs_and_receives_gradients(self):
        cells=33**2
        obs=jnp.concatenate((jnp.zeros(1345),(jnp.arange(cells)%18).astype(jnp.float32),
                             jnp.linspace(0,1,cells)))[None]
        net=MapMLPActorCritic(width=32)
        params=net.init(jax.random.PRNGKey(812),obs)
        pi,value=net.apply(params,obs)
        changed=obs.at[:,1345:1345+cells].set(0)
        other_pi,other_value=net.apply(params,changed)
        self.assertGreater(float(jnp.max(jnp.abs(pi.logits-other_pi.logits))),1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(value-other_value))),1e-6)
        grads=jax.grad(lambda p:net.apply(p,obs)[1].sum())(params)
        embedding_grad=grads['params']['terrain_embedding']['embedding']
        self.assertGreater(float(jnp.linalg.norm(embedding_grad)),0.)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(grads)))
        self.assertEqual(set(params['params']),{'terrain_embedding','actor_critic'})
        # All learned arrays are vectors/matrices; there are no spatial kernels.
        self.assertTrue(all(x.ndim<=2 for x in jax.tree_util.tree_leaves(params)))
        self.assertEqual(pi.logits.shape,(1,17));self.assertEqual(value.shape,(1,))

    def test_default_size_and_configured_view(self):
        net=map_mlp_network({'LAYER_SIZE':512,'MAP_VIEW_SIZE':33,'MAP_EMBEDDING_SIZE':4})
        params=jax.eval_shape(net.init,jax.random.PRNGKey(813),jax.ShapeDtypeStruct((1,3523),jnp.float32))
        self.assertEqual(sum(x.size for x in jax.tree_util.tree_leaves(params)),8013914)
        small=map_mlp_network({'LAYER_SIZE':16,'MAP_VIEW_SIZE':17,'MAP_EMBEDDING_SIZE':8})
        obs=jnp.zeros((2,1345+2*17**2))
        params=small.init(jax.random.PRNGKey(814),obs)
        self.assertEqual(small.apply(params,obs)[0].logits.shape,(2,17))
        with self.assertRaises(ValueError):small.init(jax.random.PRNGKey(815),jnp.zeros((1,1345)))


if __name__=='__main__':unittest.main()

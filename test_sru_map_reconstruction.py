import unittest

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from models.sru_map_reconstruction import TargetState, TargetWrapper, query_labels, init_decoder, reconstruction_loss
from ppo_rnn_og import ActorCriticRNN


@struct.dataclass
class World:
    map: object
    player_position: object


class ReconstructionTests(unittest.TestCase):
    def test_alignment_age_unknown_and_edges(self):
        terrain = jnp.full((64, 64), 2).at[0, 0].set(10)
        world = World(terrain, jnp.array([0, 0]))
        state = TargetState(world, jnp.zeros((70, 72), jnp.int32), jnp.full((4, 2), -1), jnp.int32(0))
        # Upper-left of the board is at the center of the padded local observation.
        patch = jnp.pad(terrain, ((3, 3), (4, 4)), constant_values=1)[:7, :9]
        obs = jnp.concatenate((jax.nn.one_hot(patch, 21).ravel(), jnp.zeros(41)))
        obj = TargetWrapper(None)
        state = obj.record(obs, state)
        coords = jnp.array([[0, 0], [20, 20], [-1, 0], [63, 63]])
        self.assertEqual(int(query_labels(state, coords)[0, -1]), 0)  # currently visible
        # Diamond has vanished in simulator state since it was observed.
        state = state.replace(tick=jnp.int32(150), env_state=World(terrain.at[0, 0].set(2), jnp.array([12, 12])))
        result = np.asarray(query_labels(state, coords))
        np.testing.assert_array_equal(result[0], [-12, -12, 10, 150, 1])
        np.testing.assert_array_equal(result[1], [8, 8, 2, -1, 1])
        self.assertEqual(result[2, -1], 0)
        self.assertEqual(result[3, 3], -1)
        np.testing.assert_array_equal(state.landmarks[3], [0, 0])

    def test_map_loss_alone_trains_recurrence_and_preserves_policy_interface(self):
        cfg = dict(LAYER_SIZE=16, SRU_GRU=True, GRU_CONCAT_OBSERVATION=True)
        network = ActorCriticRNN(17, cfg)
        key = jax.random.PRNGKey(41)
        x = (jax.random.normal(key, (16, 3, 1364)), jnp.zeros((16, 3), bool))
        h = jnp.zeros((3, 16))
        params = network.init(key, h, x)
        base = network.apply(params, h, x)
        extended = network.apply(params, h, x, return_memory=True)
        np.testing.assert_array_equal(base[1].logits, extended[1].logits)
        np.testing.assert_array_equal(base[2], extended[2])
        decoder = init_decoder(16, key)
        queries = jnp.broadcast_to(jnp.array([7, -2, 10, 100, 1]), (2, 3, 4, 5))
        def objective(p):
            sequence = network.apply(p, h, x, return_memory=True)[3]
            return reconstruction_loss(decoder.params, sequence[::8], queries)[0]
        grads = jax.grad(objective)(params)['params']
        self.assertGreater(np.linalg.norm(grads['ScannedRNN_0']['spatial_transform']['kernel']), 0.)
        np.testing.assert_array_equal(grads['Dense_3']['kernel'], 0.)
        loss, stats = reconstruction_loss(decoder.params, extended[3][::8], queries.at[..., 4].set(0))
        self.assertEqual(float(loss), 0.)
        self.assertEqual(int(stats[-1]), 0)

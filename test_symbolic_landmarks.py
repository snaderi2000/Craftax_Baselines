import unittest
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
from symbolic_landmarks import LandmarkState, landmark_features


class LandmarkTests(unittest.TestCase):
    def test_landmark_frames_ages_and_masking(self):
        grid=jnp.zeros((64,64),jnp.uint8).at[8,9].set(11).at[8,18].set(11).at[7,8].set(16)
        s=LandmarkState(SimpleNamespace(terrain=grid,last_seen=jnp.full((64,64),20),
                           position=jnp.array([8,8]),elapsed=jnp.int32(20)),
                        jnp.zeros((64,64),bool),jnp.array([7,8]),jnp.int32(10))
        locations=np.asarray(landmark_features(s,'locations')).reshape(6,5)
        np.testing.assert_allclose(locations[0],[1,0,1/63,1/126,0])
        np.testing.assert_array_equal(locations[[1,3,4,5]],0)
        events=np.asarray(landmark_features(s,'events')).reshape(6,5)
        np.testing.assert_allclose(events[4],[1,-1/63,0,1/126,np.log1p(10)/np.log(10001.)],rtol=1e-6)
        np.testing.assert_array_equal(events[:4],locations[:4])
        np.testing.assert_array_equal(landmark_features(s,'masked'),0)
        # A step counter alone cannot preserve an invalidated planting site.
        s.env_state.terrain=s.env_state.terrain.at[7,8].set(3)
        np.testing.assert_array_equal(landmark_features(s,'events').reshape(6,5)[4],0)

    def test_frontier_prioritizes_physically_unvisited_sector(self):
        grid=jnp.zeros((64,64),jnp.uint8).at[8,8].set(3).at[8,16].set(3)
        s=LandmarkState(SimpleNamespace(terrain=grid,last_seen=jnp.zeros((64,64)),
                          position=jnp.array([8,8]),elapsed=jnp.int32(0)),
                        jnp.zeros((64,64),bool).at[8,8].set(True),jnp.zeros(2,jnp.int32),jnp.int32(-1))
        np.testing.assert_allclose(landmark_features(s,'events').reshape(6,5)[5],[1,0,8/63,8/126,0])

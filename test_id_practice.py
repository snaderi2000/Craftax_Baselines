import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from id_practice import IDPractice,choose


@struct.dataclass
class FakeState:
    timestep: object
    achievements: object
    player_health: object
    map: object


class FakeEnv:
    def reset(self,key,params=None):
        raw = FakeState(jnp.asarray(0),jnp.zeros(22,bool),jnp.asarray(9),
                        jax.random.randint(key,(2,2),0,10))
        return self.get_obs(raw),raw
    def get_obs(self,state):
        return jnp.asarray([state.timestep,state.achievements.sum(),state.player_health],jnp.float32)
    def step(self,key,state,action,params=None):
        time = state.timestep+1
        flag = state.achievements.at[0].set(state.achievements[0] | (time==25))
        raw = state.replace(timestep=time,achievements=flag)
        reward = (flag & ~state.achievements).sum().astype(jnp.float32)
        return self.get_obs(raw),raw,reward,time>=30,{}


class PracticeTests(unittest.TestCase):
    def setup_archive(self,horizon=64):
        env = IDPractice(FakeEnv(),4,1,16,horizon)
        _,state = env.reset(jax.random.PRNGKey(0))
        for t in range(30):
            _,state,_,_,_ = env.step(jax.random.PRNGKey(t+1),state,jnp.zeros(4,jnp.int32))
        return env,state
    def test_offsets_exact_restore_and_no_practice_capture(self):
        env,state = self.setup_archive()
        self.assertEqual(int(state.archive_counts[0]),12)
        for k in range(12):
            lag = int(state.archive_lags[0,k])
            self.assertIn(lag,(1,5,20))
            self.assertEqual(int(state.archive.timestep[0,k]),25-lag)
            self.assertFalse(state.archive.achievements[0,k,0])
        self.assertEqual(int(state.target[-1]),0)
        self.assertFalse(state.raw.achievements[-1,0])
        snapshots = [tuple(np.asarray(x[0,k]).tobytes() for x in jax.tree_util.tree_leaves(state.archive)) for k in range(12)]
        current = tuple(np.asarray(x[-1]).tobytes() for x in jax.tree_util.tree_leaves(state.raw))
        self.assertIn(current,snapshots)
        total = int(state.archive_totals.sum())
        for t in range(19):
            _,state,_,_,_ = env.step(jax.random.PRNGKey(t+100),state,jnp.zeros(4,jnp.int32))
        self.assertEqual(int(state.archive_totals.sum()),total)
    def test_timeout_and_success_pre_reset_observation(self):
        env,state = self.setup_archive(horizon=2)
        index = int(np.flatnonzero(np.asarray(state.archive_lags[0])==20)[0])
        candidate = jax.tree_util.tree_map(lambda x:x[0,index],state.archive)
        raw = jax.tree_util.tree_map(lambda x,y:x.at[-1].set(y),state.raw,candidate)
        state = state.replace(raw=raw,elapsed=jnp.zeros(4,jnp.int32))
        _,state,_,_,_ = env.step(jax.random.PRNGKey(101),state,jnp.zeros(4,jnp.int32))
        _,state,reward,done,info = env.step(jax.random.PRNGKey(102),state,jnp.zeros(4,jnp.int32))
        self.assertTrue(done[-1]); self.assertTrue(info['practice/truncated'][-1])
        self.assertFalse(info['returned_episode'][-1]); self.assertEqual(float(reward[-1]),0.)
        self.assertEqual(float(info['practice/bootstrap_obs'][-1,0]),float(candidate.timestep+2))
        raw = state.raw.replace(timestep=state.raw.timestep.at[-1].set(24),
            achievements=state.raw.achievements.at[-1].set(jnp.zeros(22,bool)))
        state = state.replace(raw=raw,elapsed=jnp.zeros(4,jnp.int32))
        _,_,reward,done,info = env.step(jax.random.PRNGKey(103),state,jnp.zeros(4,jnp.int32))
        self.assertTrue(done[-1]); self.assertTrue(info['practice/success'][-1])
        self.assertFalse(info['practice/truncated'][-1]); self.assertEqual(float(reward[-1]),1.)
    def test_baseline_and_empty_fallback(self):
        env = IDPractice(FakeEnv(),4,0,16)
        _,state = env.reset(jax.random.PRNGKey(0))
        for t in range(30):
            _,state,_,done,info = env.step(jax.random.PRNGKey(t+1),state,jnp.zeros(4,jnp.int32))
        self.assertTrue(done.all()); self.assertTrue(info['returned_episode'].all())
        self.assertTrue((state.target==-1).all()); self.assertEqual(int(state.archive_counts.sum()),0)
        practice = IDPractice(FakeEnv(),4,1,16)
        _,fresh = practice.reset(jax.random.PRNGKey(0))
        raw,targets = practice._starts(jax.random.PRNGKey(1),fresh.raw,fresh)
        self.assertTrue((targets==-1).all())
        for a,b in zip(jax.tree_util.tree_leaves(raw),jax.tree_util.tree_leaves(fresh.raw)):
            np.testing.assert_array_equal(a,b)


if __name__ == '__main__':
    unittest.main()

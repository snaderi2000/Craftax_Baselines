import unittest
import jax
import jax.numpy as jnp
import numpy as np
from rare_four_practice import RareFourPractice
from test_id_practice import FakeEnv


class RareFourTests(unittest.TestCase):
    def test_normal_only_window_and_periodic_selection(self):
        env = RareFourPractice(FakeEnv(),4,1,16,64,window=4,refresh=2,min_episodes=4)
        _,state = env.reset(jax.random.PRNGKey(0))
        state = state.replace(archive_counts=state.archive_counts.at[:8].set(1))
        flags = jnp.zeros((4,22),bool).at[:,0].set(True).at[:3,1].set(True).at[:2,2].set(True).at[:1,3].set(True)
        raw = state.raw.replace(achievements=flags)
        same = env._update_rarity(state,raw,jnp.zeros(4,bool),jax.random.PRNGKey(1))
        self.assertEqual(int(same.rarity_total),0)
        state = env._update_rarity(state,raw,jnp.ones(4,bool),jax.random.PRNGKey(2))
        np.testing.assert_array_equal(np.flatnonzero(state.rarity_selected),[4,5,6,7])
        np.testing.assert_array_equal(state.rarity_sum[:4],[4,3,2,1])
        old = state.rarity_selected
        # One additional normal episode does not trigger refresh.
        state = env._update_rarity(state,raw.replace(achievements=jnp.ones((4,22),bool)),
                                  jnp.array([1,0,0,0],bool),jax.random.PRNGKey(3))
        np.testing.assert_array_equal(state.rarity_selected,old)
        self.assertEqual(int(state.rarity_total),5)
        # Ring removed the oldest episode before adding the new one.
        self.assertEqual(int(state.rarity_sum[0]),4)
        self.assertEqual(int(state.rarity_sum[4]),1)

    def test_simultaneous_resets_obey_cap_and_fallback(self):
        env = RareFourPractice(FakeEnv(),16,13,16,max_reuse=10)
        _,state = env.reset(jax.random.PRNGKey(0))
        archive = jax.tree_util.tree_map(lambda a,x:a.at[0,0].set(x[0]),state.archive,state.raw)
        state = state.replace(archive=archive,archive_counts=state.archive_counts.at[0].set(1),
                              rarity_selected=state.rarity_selected.at[0].set(True))
        _,targets,state = env._reset_choices(jax.random.PRNGKey(1),state.raw,state,jnp.ones(16,bool))
        self.assertEqual(int((targets>=0).sum()),10)
        self.assertEqual(int(state.reuse_counts.max()),10)
        self.assertEqual(int(state.rarity_starts[0]),10)
        self.assertEqual(int(state.rarity_unique_starts[0]),1)
        starts,targets,state = env._reset_choices(jax.random.PRNGKey(2),state.raw,state,jnp.ones(16,bool))
        self.assertTrue((targets==-1).all())
        for a,b in zip(jax.tree_util.tree_leaves(starts),jax.tree_util.tree_leaves(state.raw)):
            np.testing.assert_array_equal(a,b)
        # Proposals for live, non-reset slots never consume snapshot quota.
        state = state.replace(reuse_counts=jnp.zeros_like(state.reuse_counts))
        _,_,state = env._reset_choices(jax.random.PRNGKey(3),state.raw,state,jnp.zeros(16,bool))
        self.assertEqual(int(state.reuse_counts.sum()),0)

    def test_equal_task_sampling(self):
        env = RareFourPractice(FakeEnv(),257,256,16,max_reuse=1000)
        _,state = env.reset(jax.random.PRNGKey(0))
        state = state.replace(archive_counts=state.archive_counts.at[:4].set(jnp.array([1,2,3,4])),
                              rarity_selected=state.rarity_selected.at[:4].set(True))
        _,targets,state = env._reset_choices(jax.random.PRNGKey(20),state.raw,state,jnp.ones(257,bool))
        counts = np.bincount(np.asarray(targets[1:]),minlength=4)
        self.assertTrue(np.all(counts>30)); self.assertTrue(np.all(counts<110))
        self.assertEqual(int(counts.sum()),256)

    def test_practice_terminal_excluded_and_overwrite_renews_quota(self):
        env = RareFourPractice(FakeEnv(),4,1,16)
        _,state = env.reset(jax.random.PRNGKey(0))
        state = state.replace(target=jnp.zeros(4,jnp.int32),
            raw=state.raw.replace(timestep=jnp.full(4,29)))
        _,state,_,_,_ = env.step(jax.random.PRNGKey(1),state,jnp.zeros(4,jnp.int32))
        self.assertEqual(int(state.rarity_total),0)
        state = state.replace(target=jnp.full(4,-1,jnp.int32),cursor=jnp.asarray(0),
            history_count=jnp.zeros(4,jnp.int32),episode_serial=jnp.ones(4,jnp.int32),
            reuse_counts=jnp.full((22,16),10),archive_counts=jnp.full(22,16),archive_totals=jnp.full(22,16),
            archive_event_steps=jnp.full((22,16),25),
            raw=state.raw.replace(timestep=jnp.full(4,24),achievements=jnp.zeros((4,22),bool)))
        _,next_raw,_,_,_ = env.step_fn(jax.random.split(jax.random.PRNGKey(2),4),state.raw,jnp.zeros(4,jnp.int32),None)
        updated = env._record(state,next_raw)
        self.assertEqual(int((updated.reuse_counts[0]==0).sum()),4)


if __name__ == '__main__':
    unittest.main()

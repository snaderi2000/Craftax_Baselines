import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from id_practice import save_archive
from practice_continuation import bootstrap_reward,read_archive,seed_archive
from rare_four_practice import RareFourPractice
from test_id_practice import FakeEnv


class BootstrapTests(unittest.TestCase):
    def test_masks_real_death_timeout_success_and_ongoing(self):
        reward=jnp.array([1.,1.,0.,0.])
        value=jnp.array([2.,100.,-2.,5.])
        info={'practice/truncated':jnp.array([0,0,1,0],bool),
              'practice/success':jnp.array([1,1,0,0],bool),
              'practice/ended':jnp.array([1,1,1,0],bool),
              'practice/environment_terminal':jnp.array([0,1,0,0],bool)}
        corrected,mask,_=bootstrap_reward(reward,value,info,.99,True)
        np.testing.assert_allclose(corrected,[2.98,1.,-1.98,0.],rtol=1e-6)
        np.testing.assert_array_equal(mask,[1,0,1,0])
        control,_,_=bootstrap_reward(reward,value,info,.99,False)
        np.testing.assert_allclose(control,[1.,1.,-1.98,0.],rtol=1e-6)
        np.testing.assert_array_equal(reward,[1.,1.,0.,0.])
        # Critic bootstrap target is detached, not optimized toward its own target.
        grad=jax.grad(lambda v:bootstrap_reward(reward,v,info,.99,True)[0].sum())(value)
        np.testing.assert_array_equal(grad,np.zeros(4))

    def test_post_action_value_not_reset_state_and_raw_reward_unchanged(self):
        env=RareFourPractice(FakeEnv(),4,1,16,window=4,refresh=2,min_episodes=4)
        _,state=env.reset(jax.random.PRNGKey(0))
        state=state.replace(target=state.target.at[-1].set(0),
                            raw=state.raw.replace(timestep=state.raw.timestep.at[-1].set(24)))
        obs,_,reward,done,info=env.step(jax.random.PRNGKey(1),state,jnp.zeros(4,jnp.int32))
        self.assertTrue(done[-1]);self.assertEqual(float(reward[-1]),1.)
        self.assertEqual(float(obs[-1,0]),0.)
        self.assertEqual(float(info['practice/bootstrap_obs'][-1,0]),25.)
        corrected,_,_=bootstrap_reward(reward,info['practice/bootstrap_obs'][:,0],info,.99,True)
        self.assertAlmostEqual(float(corrected[-1]),25.75,places=5)

    def test_own_bank_restore_preserves_reuse_and_avoids_provenance_collisions(self):
        env=RareFourPractice(FakeEnv(),4,1,16,window=4,refresh=2,min_episodes=4,max_reuse=30)
        _,state=env.reset(jax.random.PRNGKey(0))
        for t in range(30):_,state,_,_,_=env.step(jax.random.PRNGKey(t+1),state,jnp.zeros(4,jnp.int32))
        state=state.replace(reuse_counts=state.reuse_counts.at[0,0].set(7))
        with tempfile.TemporaryDirectory() as directory:
            save_archive(state,directory)
            payload=read_archive(directory,16,4,30)
            _,fresh=env.reset(jax.random.PRNGKey(4))
            seeded=jax.jit(lambda s:seed_archive(s,payload))(fresh)
            np.testing.assert_array_equal(seeded.archive_counts,state.archive_counts)
            np.testing.assert_array_equal(seeded.reuse_counts,state.reuse_counts)
            np.testing.assert_array_equal(seeded.rarity_sum,state.rarity_sum)
            for a,b in zip(jax.tree_util.tree_leaves(seeded.archive),jax.tree_util.tree_leaves(state.archive)):
                np.testing.assert_array_equal(a,b)
            self.assertTrue((seeded.episode_serial>state.archive_episodes.max()).all())
            self.assertTrue((seeded.target==-1).all())
            self.assertEqual(int(seeded.total_steps),0)


if __name__=='__main__':unittest.main()

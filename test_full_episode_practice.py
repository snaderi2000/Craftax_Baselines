import unittest
import jax
import jax.numpy as jnp
from rare_four_practice import RareFourPractice
from test_id_practice import FakeEnv


class LongEnv(FakeEnv):
    def step(self,key,state,action,params=None):
        obs,raw,reward,_,info = super().step(key,state,action,params)
        return obs,raw,reward,raw.timestep>=100,info


class FullEpisodeTests(unittest.TestCase):
    def test_no_goal_or_64_step_end_and_count_once(self):
        env = RareFourPractice(LongEnv(),4,1,16,64,full_episode=True)
        _,state = env.reset(jax.random.PRNGKey(0))
        state = state.replace(target=state.target.at[-1].set(0))
        total_reward=0.
        earned=0
        for t in range(100):
            _,state,reward,done,info = env.step(jax.random.PRNGKey(t+1),state,jnp.zeros(4,jnp.int32))
            total_reward += float(reward[-1])
            earned += int(info['practice/target_earned_now'][-1])
            self.assertFalse(info['practice/truncated'].any())
            if t<99:
                self.assertFalse(done[-1])
                self.assertFalse(info['practice/success'][-1])
            else:
                self.assertTrue(done[-1]);self.assertTrue(info['practice/success'][-1])
                self.assertFalse(info['returned_episode'][-1])
        self.assertEqual(total_reward,1.)
        self.assertEqual(earned,1)
        self.assertEqual(int(state.successes_by_id[0]),1)
        self.assertEqual(int(state.attempts_by_id[0]),1)
        self.assertEqual(int(state.timeouts_by_id.sum()),0)
        self.assertEqual(int(state.rarity_total),3) # Only normal slots entered tally.


if __name__=='__main__':
    unittest.main()

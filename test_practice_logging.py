import unittest
from unittest.mock import patch
import numpy as np
import jax
import jax.numpy as jnp

from logz import batch_logging
from rare_four_practice import RareFourPractice
from test_id_practice import FakeEnv


class PracticeLoggingTests(unittest.TestCase):
    def test_logging_retains_practice_and_rarity_scalars(self):
        info={'returned_episode_returns':1.,'returned_episode_lengths':10.,
              'practice/steps/wake_up':123.,'rare_four/starts/wake_up':7.,'backward/level/wake_up':2.}
        logged=batch_logging.create_log_dict(info,{})
        for k in info:
            if '/' in k:self.assertEqual(logged[k],info[k])
        batch_logging.batch_logs.clear();batch_logging.log_times.clear()
        config=dict(NUM_REPEATS=1,DEBUG=False,NUM_STEPS=64,NUM_ENVS=128)
        with patch.object(batch_logging.wandb,'log') as log:
            batch_logging.batch_log(0,logged,config)
            self.assertEqual(float(log.call_args.args[0]['practice/steps/wake_up']),123.)

    def test_per_id_steps_match_total_practice_steps(self):
        env=RareFourPractice(FakeEnv(),4,1,16,window=4,refresh=2,min_episodes=4)
        _,state=env.reset(jax.random.PRNGKey(0))
        state=state.replace(target=state.target.at[-1].set(0))
        for t in range(10):_,state,_,_,_=env.step(jax.random.PRNGKey(t+1),state,jnp.zeros(4,jnp.int32))
        self.assertEqual(int(state.practice_steps_by_id[0]),10)
        self.assertEqual(int(state.practice_steps_by_id.sum()),int(state.practice_steps))
        np.testing.assert_array_equal(state.practice_steps_by_id[1:],np.zeros(21))


if __name__=='__main__':unittest.main()

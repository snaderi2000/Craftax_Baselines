import unittest
import jax
import jax.numpy as jnp
import numpy as np
from backward_practice import BackwardPractice
from test_id_practice import FakeEnv


class EventEnv(FakeEnv):
    def step(self,key,state,action,params=None):
        time=state.timestep+1
        flags=state.achievements.at[0].set(state.achievements[0] | (time==80))
        raw=state.replace(timestep=time,achievements=flags)
        return self.get_obs(raw),raw,(flags & ~state.achievements).sum().astype(jnp.float32),time>=100,{}


class BackwardTests(unittest.TestCase):
    def setup_bank(self):
        env=BackwardPractice(EventEnv(),4,1,64,64)
        _,state=env.reset(jax.random.PRNGKey(0))
        def advance(t,state):
            return env.step(jax.random.fold_in(jax.random.PRNGKey(1),t),state,jnp.zeros(4,jnp.int32))[1]
        state=jax.lax.fori_loop(0,80,advance,state)
        return env,state
    def test_separate_banks_offsets_and_eligibility(self):
        env,state=self.setup_bank()
        np.testing.assert_array_equal(state.backward_bank_counts[0],[4,4,4,4])
        for level,lag in enumerate((1,5,20,50)):
            indices=level*16+np.arange(4)
            np.testing.assert_array_equal(state.archive_lags[0,indices],np.full(4,lag))
            np.testing.assert_array_equal(state.archive.timestep[0,indices],np.full(4,80-lag))
            self.assertFalse(state.archive.achievements[0,indices,0].any())
        self.assertEqual(int(env._all_usable(state)[0].sum()),16)
        self.assertEqual(int(env._usable(state)[0].sum()),4)
        np.testing.assert_array_equal(state.archive_lags[0,np.asarray(env._usable(state)[0])],np.ones(4))
    def test_mastery_minimum_and_live_episode_level(self):
        env,state=self.setup_bank()
        state=state.replace(target=state.target.at[-1].set(0),practice_level=state.practice_level.at[-1].set(0))
        ended=jnp.array([0,0,0,1],bool)
        success=ended
        update=jax.jit(env._update_curriculum)
        for _ in range(127):
            state=update(state,ended,success)
        self.assertEqual(int(state.backward_level[0]),0)
        state=update(state,ended,success)
        self.assertEqual(int(state.backward_level[0]),1)
        self.assertEqual(int(state.practice_level[-1]),0) # Existing episode retains its starting level.
        state=state.replace(rarity_selected=state.rarity_selected.at[0].set(True))
        starts,targets,state=env._reset_choices(jax.random.PRNGKey(4),state.raw,state,ended)
        self.assertEqual(int(targets[-1]),0); self.assertEqual(int(state.practice_level[-1]),1)
        self.assertEqual(int(starts.timestep[-1]),75)
        self.assertEqual(int(env._time_limit(state)[-1]),64)
        state=state.replace(practice_level=state.practice_level.at[-1].set(3))
        self.assertEqual(int(env._time_limit(state)[-1]),128)
    def test_no_promotion_without_success_or_next_bank(self):
        env,state=self.setup_bank()
        state=state.replace(target=state.target.at[-1].set(0),practice_level=state.practice_level.at[-1].set(0))
        ended=jnp.array([0,0,0,1],bool)
        update=jax.jit(env._update_curriculum)
        for _ in range(128):state=update(state,ended,jnp.zeros(4,bool))
        self.assertEqual(int(state.backward_level[0]),0)
        state=state.replace(backward_attempt_counts=jnp.zeros((22,4),jnp.int32),backward_last_check=jnp.zeros((22,4),jnp.int32),
            backward_outcomes=jnp.zeros((22,4,128),jnp.int32),backward_bank_counts=state.backward_bank_counts.at[0,1].set(0))
        for _ in range(128):state=update(state,ended,ended)
        self.assertEqual(int(state.backward_level[0]),0)


if __name__=='__main__':unittest.main()

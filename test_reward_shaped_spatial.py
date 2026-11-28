"""Reward semantics and temporal correctness for the shaped-reward study."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Action, Achievement, BlockType
from wrappers import ClassicAchievementRewardWrapper, ClassicDeathPenaltyWrapper, ClassicLavaPenaltyWrapper
from ppo_rnn_og import ActorCriticRNN, ScannedRNN


class RewardShapedSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
        cls.key=jax.random.PRNGKey(7550)
        _,state=cls.env.reset(cls.key,cls.env.default_params)
        cls.state=state.replace(map=jnp.full_like(state.map,BlockType.GRASS.value),
            mob_map=jnp.zeros_like(state.mob_map),player_position=jnp.array([32,32]),
            player_direction=Action.RIGHT.value,player_food=5,player_health=9,
            player_drink=9,player_energy=9,is_sleeping=False,
            achievements=jnp.zeros_like(state.achievements),
            zombies=state.zombies.replace(mask=jnp.zeros_like(state.zombies.mask)),
            cows=state.cows.replace(mask=jnp.zeros_like(state.cows.mask)),
            skeletons=state.skeletons.replace(mask=jnp.zeros_like(state.skeletons.mask)),
            arrows=state.arrows.replace(mask=jnp.zeros_like(state.arrows.mask)))
        cls.shaped=ClassicAchievementRewardWrapper(ClassicDeathPenaltyWrapper(cls.env,.5),5.,5.)

    def test_completion_is_a_true_terminal_with_the_original_final_reward(self):
        from wrappers import ClassicCompletionWrapper
        for bonus in (0., 5.):
            env = ClassicCompletionWrapper(self.env, bonus)
            flags = jnp.ones_like(self.state.achievements).at[Achievement.EAT_PLANT.value].set(False)
            state = self.state.replace(achievements=flags, map=self.state.map.at[32,33].set(BlockType.RIPE_PLANT.value))
            for action, won in [(Action.NOOP, False), (Action.DO, True)]:
                raw = self.env.step(self.key, state, action.value, self.env.default_params)
                actual = env.step(self.key, state, action.value, self.env.default_params)
                self.assertEqual(bool(actual[3]), won)
                self.assertEqual(bool(actual[4]['all22_win']), won)
                self.assertAlmostEqual(float(actual[2]-raw[2]), bonus if won else 0.)
                if won:
                    self.assertEqual(float(actual[4]['discount']), 0.)
                    self.assertGreater(float(actual[4]['score']), 99.)
                    after = env.step(self.key, actual[1], Action.NOOP.value, self.env.default_params)
                    self.assertFalse(bool(after[4]['all22_win']))

    def test_actual_diamond_and_plant_receive_five_once(self):
        for tile,event in [(BlockType.DIAMOND,Achievement.COLLECT_DIAMOND),(BlockType.RIPE_PLANT,Achievement.EAT_PLANT)]:
            with self.subTest(event=event.name):
                state=self.state.replace(map=self.state.map.at[32,33].set(tile.value),
                    inventory=self.state.inventory.replace(iron_pickaxe=1))
                _,base,r,done,_=self.env.step(self.key,state,Action.DO.value,self.env.default_params)
                _,actual,shaped,actual_done,_=self.shaped.step(self.key,state,Action.DO.value,self.env.default_params)
                self.assertTrue(bool(base.achievements[event.value]))
                self.assertFalse(bool(done));self.assertEqual(bool(done),bool(actual_done))
                self.assertAlmostEqual(float(shaped-r),4.,places=5)
                for a,b in zip(jax.tree_util.tree_leaves(base),jax.tree_util.tree_leaves(actual)):
                    np.testing.assert_array_equal(a,b)
                # A later encounter with the same resource has no new bonus.
                repeated=state.replace(achievements=base.achievements)
                raw=self.env.step(self.key,repeated,Action.DO.value,self.env.default_params)[2]
                reward=self.shaped.step(self.key,repeated,Action.DO.value,self.env.default_params)[2]
                self.assertAlmostEqual(float(reward),float(raw),places=5)

    def test_real_lava_death_cost_and_timeout_exemption(self):
        for lava in [True,False]:
            state=self.state.replace(map=self.state.map.at[32,33].set(BlockType.LAVA.value if lava else BlockType.GRASS.value),
                timestep=0 if lava else 9999)
            raw=self.env.step(self.key,state,Action.RIGHT.value,self.env.default_params)
            shaped=self.shaped.step(self.key,state,Action.RIGHT.value,self.env.default_params)
            self.assertTrue(bool(raw[3]));self.assertTrue(bool(shaped[3]))
            self.assertAlmostEqual(float(shaped[2]-raw[2]),-.5 if lava else 0.,places=5)

    def test_lava_only_penalty_preserves_ordinary_death_timeout_and_default_achievements(self):
        wrapper=ClassicLavaPenaltyWrapper(self.env,2.)
        lava=self.state.replace(map=self.state.map.at[32,33].set(BlockType.LAVA.value))
        starvation=self.state.replace(player_health=1,player_food=0,player_recover=-15.)
        timeout=self.state.replace(timestep=9999)
        for label,state,action,cost in [('lava',lava,Action.RIGHT.value,2.),
                ('starvation',starvation,Action.NOOP.value,0.),('timeout',timeout,Action.NOOP.value,0.)]:
            with self.subTest(case=label):
                raw=self.env.step(self.key,state,action,self.env.default_params)
                actual=wrapper.step(self.key,state,action,self.env.default_params)
                self.assertTrue(bool(raw[3]));self.assertTrue(bool(actual[3]))
                self.assertAlmostEqual(float(actual[2]-raw[2]),-cost,places=5)
                for a,b in zip(jax.tree_util.tree_leaves(raw[1]),jax.tree_util.tree_leaves(actual[1])):
                    if np.issubdtype(np.asarray(a).dtype, np.floating):
                        # Wrapped and bare GPU kernels can round state updates differently.
                        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7, equal_nan=False)
                    else:
                        np.testing.assert_array_equal(a, b)
        for tile,event in [(BlockType.DIAMOND,Achievement.COLLECT_DIAMOND),(BlockType.RIPE_PLANT,Achievement.EAT_PLANT)]:
            state=self.state.replace(map=self.state.map.at[32,33].set(tile.value),
                inventory=self.state.inventory.replace(iron_pickaxe=1))
            raw=self.env.step(self.key,state,Action.DO.value,self.env.default_params)
            actual=wrapper.step(self.key,state,Action.DO.value,self.env.default_params)
            self.assertTrue(bool(actual[1].achievements[event.value]))
            self.assertAlmostEqual(float(actual[2]),float(raw[2]),places=5)
            self.assertAlmostEqual(float(actual[2]),1.,places=5)

    def test_spatial_gru_sequence_matches_online_and_resets(self):
        net=ActorCriticRNN(17,{'LAYER_SIZE':32,'SYMBOLIC_ENCODER':'spatial'})
        obs=jax.random.normal(self.key,(4,2,1345))
        resets=jnp.array([[False,False],[False,True],[True,False],[False,False]])
        hidden=ScannedRNN.initialize_carry(2,32)
        params=net.init(self.key,hidden,(obs,resets))
        h,pi,v=net.apply(params,hidden,(obs,resets))
        hh=hidden;values=[];logits=[]
        for t in range(4):
            hh,p,value=net.apply(params,hh,(obs[t:t+1],resets[t:t+1]))
            values.append(value);logits.append(p.logits)
        np.testing.assert_allclose(h,hh,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(v,jnp.concatenate(values),atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(pi.logits,jnp.concatenate(logits),atol=2e-5,rtol=2e-5)
        reset=jnp.ones((1,2),bool)
        zero=net.apply(params,hidden,(obs[:1],reset))
        altered=net.apply(params,jnp.ones_like(hidden),(obs[:1],reset))
        np.testing.assert_allclose(zero[0],altered[0],atol=1e-6)
        np.testing.assert_allclose(zero[2],altered[2],atol=1e-6)


if __name__=='__main__':unittest.main()

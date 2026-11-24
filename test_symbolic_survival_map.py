"""Observed-only navigation, action feedback, reset and matched-control checks."""
import unittest
import jax,jax.numpy as jnp,numpy as np
from symbolic_survival_map import water_route_features,ClassicSurvivalMapWrapper,FEATURE_SIZE
from symbolic_map_memory import ClassicMapMemoryWrapper
from craftax.craftax_classic.renderer import render_craftax_symbolic
from craftax.craftax_classic.constants import Action,BlockType


class SurvivalMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_symbolic_map_memory import SymbolicMapMemoryTests as Base
        Base.setUpClass();cls.raw=Base.state;cls.env=Base.env;cls.key=Base.key
        cls.map=ClassicMapMemoryWrapper(cls.env,33)
        cls.enabled=ClassicSurvivalMapWrapper(cls.map,'enabled')
        cls.masked=ClassicSurvivalMapWrapper(cls.map,'masked')

    def initialize(self,raw,wrapper=None):
        obs,m=self.map.initialize(render_craftax_symbolic(raw),raw)
        return (wrapper or self.enabled).initialize(obs,m)

    def test_known_water_beyond_crop_routes_and_unknown_wall_lava_block(self):
        terrain=jnp.zeros((64,64),jnp.uint8).at[32,5:33].set(3).at[32,4].set(4)
        local=jnp.zeros(1345).at[1339].set(1.)
        route=jax.jit(water_route_features)
        v=route(terrain,jnp.array([32,32]),local)
        self.assertAlmostEqual(float(v[0]),27/128)
        self.assertAlmostEqual(float(v[1]),26/128)
        self.assertEqual(float(v[5]),1.)
        for blocking in (0,5,15):
            v=route(terrain.at[32,18].set(blocking),jnp.array([32,32]),local)
            self.assertEqual(float(v[5]),0.)
        adjacent=route(terrain,jnp.array([32,5]),local)
        self.assertEqual(float(adjacent[0]),0.)
        self.assertEqual(float(adjacent[10]),1.)
        self.assertEqual(float(adjacent[14]),1.)

    def test_no_unseen_map_leak_and_masked_control_identical_dynamics(self):
        raw=self.raw.replace(map=self.raw.map.at[5,5].set(BlockType.WATER.value))
        o,s=self.initialize(raw)
        o2,_=self.initialize(self.raw)
        np.testing.assert_array_equal(o,o2)
        masked,ms=self.initialize(raw,self.masked)
        np.testing.assert_array_equal(masked[:-FEATURE_SIZE],o[:-FEATURE_SIZE])
        np.testing.assert_array_equal(masked[-FEATURE_SIZE:],0.)
        result=self.enabled.step(self.key,s,Action.RIGHT.value,self.env.default_params)
        control=self.masked.step(self.key,ms,Action.RIGHT.value,self.env.default_params)
        np.testing.assert_array_equal(result[0][:-FEATURE_SIZE],control[0][:-FEATURE_SIZE])
        for a,b in zip(jax.tree_util.tree_leaves(result[1:]),jax.tree_util.tree_leaves(control[1:])):
            np.testing.assert_array_equal(a,b)

    def test_actual_failed_movement_depletion_duration_and_sleep(self):
        raw=self.raw.replace(player_energy=0,map=self.raw.map.at[32,33].set(BlockType.STONE.value))
        _,s=self.initialize(raw)
        o,s,_,_,_=self.enabled.step(self.key,s,Action.RIGHT.value,self.env.default_params)
        self.assertEqual(int(s.depleted_steps[2]),2)
        self.assertEqual(int(s.blocked_streak),1)
        np.testing.assert_array_equal(o[-4:-2],0.)
        self.assertEqual(float(o[-2]),1.)
        _,s,_,_,_=self.enabled.step(self.key,s,Action.SLEEP.value,self.env.default_params)
        self.assertTrue(bool(s.last_sleeping))
        o,s,_,_,_=self.enabled.step(self.key,s,Action.RIGHT.value,self.env.default_params)
        self.assertEqual(float(o[-2]),0.)
        self.assertEqual(int(s.blocked_streak),0)

    def test_autoreset_clears_counters_and_control_initialization_matches(self):
        from wrappers import AutoResetEnvWrapper
        raw=self.raw.replace(player_health=1,player_energy=0,player_recover=-15.)
        _,s=self.initialize(raw)
        wrapped=AutoResetEnvWrapper(self.enabled)
        o,s,_,done,_=wrapped.step(self.key,s,Action.NOOP.value,self.env.default_params)
        self.assertTrue(bool(done));np.testing.assert_array_equal(s.depleted_steps,0)
        self.assertEqual(int(s.blocked_streak),0);np.testing.assert_array_equal(o[-24:],0.)
        from ppo_rnn_og import ActorCriticRNN
        cfg=dict(LAYER_SIZE=16,SYMBOLIC_ENCODER='map_mlp',MAP_VIEW_SIZE=9,GRU_CONCAT_OBSERVATION=True)
        obs=jnp.zeros((3,2,1345+162+FEATURE_SIZE));h=jnp.zeros((2,16));reset=jnp.zeros((3,2),bool)
        nets=[ActorCriticRNN(17,dict(cfg,MAP_SURVIVAL_FEATURES=m)) for m in ('masked','enabled')]
        params=[n.init(self.key,h,(obs,reset)) for n in nets]
        for a,b in zip(jax.tree_util.tree_leaves(params[0]),jax.tree_util.tree_leaves(params[1])):np.testing.assert_array_equal(a,b)
        def loss(o):return nets[1].apply(params[1],h,(o,reset))[2].sum()
        gradient=jax.grad(loss)(obs.at[...,0].set(1.))
        self.assertGreater(float(jnp.linalg.norm(gradient[...,-FEATURE_SIZE:])),0.)


if __name__=='__main__':unittest.main()

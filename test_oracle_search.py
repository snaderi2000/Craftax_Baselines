"""Behavioral checks for exact-simulator search, absorption, and real targets."""
import unittest
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
from models.oracle_search import OraclePolicyValue,oracle_search,make_oracle_recurrent,lambda_returns
from models.symbolic_ez import scalar_target
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import BlockType,Action


class ToyPolicyValue:
    width=1
    def prediction(self):pass
    def represent(self):pass
    def apply(self,params,*args,method):
        if method==self.represent:return args[0]+1
        return jnp.zeros((len(args[0]),17)),jnp.log(jnp.maximum(scalar_target(jnp.zeros(len(args[0]))),1e-30))


class OracleTests(unittest.TestCase):
    def test_terminal_reward_and_reset_rewards_do_not_mix(self):
        rewards=jnp.array([[1.,2.],[100.,3.]])
        dones=jnp.array([[1.,0.],[0.,1.]])
        values=jnp.zeros((2,2));last=jnp.ones(2)*999
        targets=lambda_returns(rewards,dones,values,last)
        np.testing.assert_allclose(targets[0],[1.,2.+.99*.8*3.],rtol=1e-5)
        np.testing.assert_allclose(targets[1,1],3.)

    def test_search_collects_real_wood_without_mutating_root(self):
        env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
        _,s=env.reset(jax.random.PRNGKey(91),env.default_params)
        p=s.player_position
        s=s.replace(map=jnp.full_like(s.map,BlockType.GRASS.value).at[p[0],p[1]+1].set(BlockType.TREE.value),
            zombies=s.zombies.replace(mask=jnp.zeros_like(s.zombies.mask)),
            cows=s.cows.replace(mask=jnp.zeros_like(s.cows.mask)),
            skeletons=s.skeletons.replace(mask=jnp.zeros_like(s.skeletons.mask)),
            arrows=s.arrows.replace(mask=jnp.zeros_like(s.arrows.mask)),
            player_direction=jnp.asarray(Action.RIGHT.value),achievements=jnp.zeros_like(s.achievements))
        batch=jax.tree_util.tree_map(lambda x:x[None],s)
        before=[np.array(x) for x in jax.tree_util.tree_leaves(batch)]
        action,policy,value=jax.jit(lambda state:oracle_search(ToyPolicyValue(),{},env,state,
            jnp.zeros((1,1)),jax.random.PRNGKey(52),64,1,17,0.))(batch)
        self.assertEqual(int(action[0]),Action.DO.value)
        self.assertGreater(float(policy[0,Action.DO.value]),.9)
        np.testing.assert_allclose(policy.sum(-1),1.,atol=1e-6)
        self.assertTrue(np.isfinite(value).all())
        for old,new in zip(before,jax.tree_util.tree_leaves(batch)):np.testing.assert_array_equal(old,new)

    def test_terminal_search_branch_is_absorbing(self):
        class Env:
            default_params=None
            def step(self,key,state,action,params):
                # Deliberately rewards illegal steps after death to expose leaks.
                reward=jnp.where(state>0,100.,-.1)
                return jnp.zeros(1345),state+1,reward,jnp.asarray(True),{}
        recurrent=make_oracle_recurrent(ToyPolicyValue(),Env())
        embedding=dict(state=jnp.zeros(2,jnp.int32),hidden=jnp.zeros((2,1)),terminal=jnp.zeros(2,bool))
        out,child=recurrent({},jax.random.PRNGKey(1),jnp.zeros(2,jnp.int32),embedding)
        np.testing.assert_allclose(out.reward,-.1);np.testing.assert_array_equal(out.discount,0.)
        np.testing.assert_array_equal(out.value,0.)
        out,grandchild=recurrent({},jax.random.PRNGKey(2),jnp.zeros(2,jnp.int32),child)
        np.testing.assert_array_equal(out.reward,0.);np.testing.assert_array_equal(out.discount,0.)
        np.testing.assert_array_equal(grandchild['state'],child['state'])
        np.testing.assert_array_equal(grandchild['hidden'],child['hidden'])

    def test_recurrent_encoder_resets_and_has_no_dynamics_parameters(self):
        model=OraclePolicyValue(16);obs=jnp.ones((2,1345));h=jnp.zeros((2,16));a=jnp.zeros(2,jnp.int32)
        weights=model.init(jax.random.PRNGKey(1),h,obs,a,jnp.zeros(2),jnp.ones(2,bool))
        clean=model.apply(weights,h,obs,a,jnp.zeros(2),jnp.ones(2,bool))
        dirty=model.apply(weights,h+7,obs,a+5,jnp.ones(2),jnp.ones(2,bool))
        for x,y in zip(clean,dirty):np.testing.assert_allclose(x,y,atol=1e-6)
        self.assertEqual(set(weights['params']),{'obs_layer','gru','pred_layer','policy','value'})


if __name__=='__main__':unittest.main()

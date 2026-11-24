"""Behavioral checks for the symbolic model/search pilot before long training."""
import unittest
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import optax
from models.symbolic_ez import (SymbolicEZ,search,training_loss,scalar_target,scalar_decode,
    bootstrap_returns,BLOCKS,CELLS,MAP_DIM)
from symbolic_ez_replay import SequenceReplay,pack_observation,unpack_observation,SumTree


def example_observation(n):
    terrain=np.eye(BLOCKS,dtype=np.float32)[np.zeros((n,CELLS),int)]
    mobs=np.zeros((n,CELLS,4),np.float32)
    stats=np.zeros((n,22),np.float32);stats[:,:16]=np.float16(.9);stats[:,16]=1.;stats[:,20]=1.
    return np.concatenate([np.concatenate([terrain,mobs],-1).reshape(n,MAP_DIM),stats],-1)


class SymbolicEZTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=SymbolicEZ(16)
        cls.obs=jnp.asarray(example_observation(2));cls.h=jnp.zeros((2,16));cls.actions=jnp.zeros(2,int)
        cls.params=cls.model.init(jax.random.PRNGKey(0),cls.h,cls.obs,cls.actions,jnp.zeros(2),jnp.zeros(2,bool))

    def test_scalar_targets_and_terminal_bootstrap(self):
        values=jnp.array([-2.,-.1,0.,1.,5.,22.])
        targets=scalar_target(values)
        np.testing.assert_allclose(targets.sum(-1),1.,atol=1e-6)
        # Decode the expected transformed support (not logits) directly.
        from models.symbolic_ez import SUPPORT,symexp
        np.testing.assert_allclose(symexp((targets*SUPPORT).sum(-1)),values,atol=2e-4)
        result=bootstrap_returns(jnp.array([[1.,1.],[100.,2.]]),jnp.array([[1.,0.],[0.,0.]]),jnp.array([1000.,3.]))
        np.testing.assert_allclose(result,[1.,1.+.99*2+.99**2*3],atol=1e-5)

    def test_reset_masks_memory_action_reward(self):
        apply=lambda h,a,r:self.model.apply(self.params,h,self.obs,a,r,jnp.ones(2,bool),method=self.model.represent)
        np.testing.assert_allclose(apply(self.h,self.actions,jnp.zeros(2)),apply(jnp.ones_like(self.h),jnp.ones(2,int)*5,jnp.ones(2)*9),atol=1e-6)

    def test_terminal_observation_labels_survive_auto_reset(self):
        from train_symbolic_ez import ObservationLabelWrapper
        from wrappers import OptimisticResetVecEnvWrapper
        class Fake:
            def reset(self,key,params):return jnp.ones(3),jnp.asarray(0)
            def step(self,key,state,action,params):
                return -jnp.ones(3),state+1,jnp.asarray(-.1),jnp.asarray(True),{}
        env=OptimisticResetVecEnvWrapper(ObservationLabelWrapper(Fake()),16,16)
        obs,state=env.reset(jax.random.PRNGKey(1),None)
        next_obs,_,reward,done,info=env.step(jax.random.PRNGKey(2),state,jnp.zeros(16,int),None)
        np.testing.assert_array_equal(next_obs,np.ones((16,3)))
        np.testing.assert_array_equal(info['next_observation'],-np.ones((16,3)))
        assert np.asarray(done).all()

    def test_search_is_normalized_and_prefers_known_terminal_reward(self):
        class Toy:
            def prediction(self):pass
            def dynamics(self):pass
            def apply(self,p,*args,method):
                if method==self.prediction:
                    h=args[0];logits=jnp.zeros((len(h),17)).at[:,3].set(8.)
                    return logits,jnp.log(jnp.maximum(scalar_target(jnp.zeros(len(h))),1e-30))
                h,action=args
                reward=(action==3).astype(jnp.float32)
                return h,jnp.log(jnp.maximum(scalar_target(reward),1e-30)),jnp.full(len(h),-30.)
        toy=Toy()
        a,pi,v=search(toy,{},self.h,jax.random.PRNGKey(5),32,0.)
        np.testing.assert_array_equal(a,[3,3]);np.testing.assert_allclose(pi.sum(-1),1.,atol=1e-5)
        # Root value averages explored outcomes, including inferior actions.
        assert np.isfinite(v).all() and (np.asarray(v)>0.).all() and (np.asarray(v)<=1.001).all()
        np.testing.assert_array_equal(np.asarray(pi).argmax(-1),[3,3])

    def test_replay_roundtrip_and_ring_sequence_integrity(self):
        obs=example_observation(2)
        np.testing.assert_array_equal(unpack_observation(*pack_observation(obs)),obs)
        replay=SequenceReplay(2,16,16,4,2,2,seed=2)
        for t in range(40):
            replay.append(obs,obs,action=np.full(2,t%17),prev_action=np.zeros(2),prev_reward=np.zeros(2),
                reward=np.full(2,t),done=np.zeros(2),reset=np.zeros(2,bool),
                policy=np.full((2,17),1/17),hidden=np.zeros((2,16)),search_value=np.zeros(2))
        batch,slots=replay.sample(8)
        assert batch['obs'].shape==(9,8,1345)
        np.testing.assert_array_equal(np.diff(batch['reward'],axis=0),np.ones((8,8)))
        assert np.isfinite(batch['weights']).all()
        replay.update_priorities(slots,np.arange(1,9))
        np.testing.assert_allclose(replay.tree.tree[1],replay.tree.tree[replay.tree.size:].sum())

    def test_search_does_not_amplify_tiny_q_differences_when_disabled(self):
        class TinyReward:
            def prediction(self):pass
            def dynamics(self):pass
            def apply(self,p,*args,method):
                if method==self.prediction:
                    return jnp.zeros((len(args[0]),17)),jnp.log(jnp.maximum(scalar_target(jnp.zeros(len(args[0]))),1e-30))
                h,action=args
                reward=(action==3).astype(jnp.float32)*1e-6
                return h,jnp.log(jnp.maximum(scalar_target(reward),1e-30)),jnp.full(len(h),-30.)
        _,weights,_=search(TinyReward(),{},self.h,jax.random.PRNGKey(12),32,0.,absolute_q=True)
        np.testing.assert_allclose(np.asarray(weights),np.full((2,17),1/17),atol=2e-5)

    def test_masked_unroll_and_trainable_finite_losses(self):
        length=7;batch={
            'obs':jnp.broadcast_to(self.obs,(length,2,1345)),
            'next_obs':jnp.broadcast_to(self.obs,(length,2,1345)),
            'action':jnp.zeros((length,2),int),'prev_action':jnp.zeros((length,2),int),
            'prev_reward':jnp.zeros((length,2)),'reset':jnp.zeros((length,2),bool),
            'reward':jnp.zeros((length,2)).at[2].set(1.),
            'done':jnp.zeros((length,2)).at[2].set(1.),
            'policy':jnp.ones((length,2,17))/17,'initial_h':self.h,'weights':jnp.ones(2),
            'root_reanalysed':jnp.zeros(2,bool),'root_search_value':jnp.zeros(2)}
        loss=lambda p,b:training_loss(self.model,p,self.params,b,2,2,2)[0]
        first=float(loss(self.params,batch));altered=dict(batch)
        altered['reward']=batch['reward'].at[3:].set(99.)
        altered['obs']=batch['obs'].at[3:].set(10.)
        np.testing.assert_allclose(loss(self.params,altered),first,atol=2e-5)
        optimizer=optax.adam(.001);params=self.params;state=optimizer.init(params)
        @jax.jit
        def update(params,state):
            value,gradient=jax.value_and_grad(loss)(params,batch)
            delta,state=optimizer.update(gradient,state,params)
            return optax.apply_updates(params,delta),state,value,optax.global_norm(gradient)
        for _ in range(12):params,state,value,norm=update(params,state)
        assert np.isfinite(value) and np.isfinite(norm)
        assert float(loss(params,batch))<first

    def test_real_sequence_fix_trains_gru_beyond_first_observation(self):
        length=7
        batch=dict(obs=jnp.broadcast_to(self.obs,(length,2,1345)),
            next_obs=jnp.broadcast_to(self.obs,(length,2,1345)),action=jnp.zeros((length,2),int),
            prev_action=jnp.zeros((length,2),int),prev_reward=jnp.zeros((length,2)),
            reset=jnp.zeros((length,2),bool),reward=jnp.ones((length,2)),done=jnp.zeros((length,2)),
            policy=jax.nn.one_hot(jnp.zeros((length,2),int),17),initial_h=self.h,
            weights=jnp.ones(2),root_reanalysed=jnp.zeros(2,bool),root_search_value=jnp.zeros(2))
        def objective(obs,fix):
            return training_loss(self.model,self.params,self.params,dict(batch,obs=obs),2,2,2,fix)[0]
        original=jax.grad(lambda obs:objective(obs,False))(batch['obs'])
        fixed=jax.grad(lambda obs:objective(obs,True))(batch['obs'])
        np.testing.assert_array_equal(original[3:5],np.zeros_like(original[3:5]))
        assert float(jnp.linalg.norm(fixed[3:5]))>1e-7
        # Burn-in stays detached; labels after the unroll must not leak gradients.
        np.testing.assert_array_equal(fixed[:2],np.zeros_like(fixed[:2]))
        np.testing.assert_array_equal(fixed[5:],np.zeros_like(fixed[5:]))


if __name__=='__main__':unittest.main()

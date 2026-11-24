"""Behavioral checks for spatial memory, recurrent PPO replay and learned recall."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.actor_critic_neural_map import NeuralMapCell, initialize_carry
from ppo_rnn_og import ActorCriticRNN, initialize_recurrent_carry


class NeuralMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=dict(LAYER_SIZE=32,NEURAL_MAP_CHANNELS=4,SYMBOLIC_ENCODER='map_mlp',
            MAP_VIEW_SIZE=9,MAP_EMBEDDING_SIZE=2,GRU_CONCAT_OBSERVATION=True)
        cls.net=ActorCriticRNN(17,cls.cfg)
        cls.obs=jax.random.normal(jax.random.PRNGKey(10),(5,2,1345+162+2))*.1
        cls.obs=cls.obs.at[...,1345:1345+81].set(1).at[...,-2:].set(30)
        cls.done=jnp.zeros((5,2),bool).at[3,0].set(True)
        cls.carry=initialize_recurrent_carry(2,cls.cfg)
        cls.params=cls.net.init(jax.random.PRNGKey(3),cls.carry,(cls.obs,cls.done))

    def test_streamed_replay_and_episode_reset(self):
        c,pi,v=self.net.apply(self.params,self.carry,(self.obs,self.done))
        h=self.carry;logits=[];values=[]
        step=jax.jit(self.net.apply)
        for i in range(5):
            h,p,val=step(self.params,h,(self.obs[i:i+1],self.done[i:i+1]))
            logits.append(p.logits);values.append(val)
        np.testing.assert_allclose(jnp.concatenate(logits),pi.logits,atol=2e-6)
        np.testing.assert_allclose(jnp.concatenate(values),v,atol=2e-6)
        for a,b in zip(c,h):np.testing.assert_allclose(a,b,atol=2e-6)
        contaminated=(jnp.ones_like(h[0])*100,jnp.ones_like(h[1])*100,jnp.ones_like(h[2]))
        reset=jnp.ones((1,2),bool)
        a=self.net.apply(self.params,contaminated,(self.obs[:1],reset))
        b=self.net.apply(self.params,self.carry,(self.obs[:1],reset))
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(x,y)

    def test_border_write_and_map_ablation(self):
        obs=self.obs[:1].at[...,-2:].set(0)
        c,_,_=self.net.apply(self.params,self.carry,(obs,self.done[:1]))
        self.assertEqual(int(c[2][0].sum()),20)  # rows 0..3, cols 0..4
        np.testing.assert_array_equal(c[1][:,-1],0)
        ablated=ActorCriticRNN(17,dict(self.cfg,NEURAL_MAP_RESET_EVERY_STEP=True))
        same=ablated.init(jax.random.PRNGKey(3),self.carry,(self.obs,self.done))
        for a,b in zip(jax.tree_util.tree_leaves(same),jax.tree_util.tree_leaves(self.params)):
            np.testing.assert_array_equal(a,b)
        contaminated=(c[0],jnp.ones_like(c[1])*10,jnp.ones_like(c[2]))
        a=ablated.apply(self.params,contaminated,(obs,self.done[:1]))
        b=ablated.apply(self.params,(c[0],self.carry[1],self.carry[2]),(obs,self.done[:1]))
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(x,y)

    def test_gradients_through_earlier_writes_without_gru_history(self):
        cfg=dict(self.cfg,RESET_GRU_EVERY_STEP=True)
        net=ActorCriticRNN(17,cfg)
        obs=self.obs[:3,:1].at[1:,:,-2:].set(50)  # late read must attend to old region
        done=jnp.zeros((3,1),bool)
        carry=initialize_carry(1,cfg)
        def loss(params,inputs):return net.apply(params,carry,(inputs,done))[2][-1].sum()
        dp,dx=jax.jit(jax.grad(loss,argnums=(0,1)))(self.params,obs)
        self.assertGreater(float(jnp.linalg.norm(dx[0,:,:1323])),1e-9)
        memory=dp['params']['NeuralMapActorCritic_0']['memory']
        for name in ('write_candidate','write_gate','attention_query'):
            self.assertGreater(float(jnp.linalg.norm(memory[name]['kernel'])),1e-10)
        def reset_loss(inputs):return net.apply(self.params,carry,(inputs,done.at[1].set(True)))[2][-1].sum()
        np.testing.assert_array_equal(jax.grad(reset_loss)(obs)[0],0)

    def test_two_device_ppo_gradient_and_optimizer_parity(self):
        if jax.local_device_count()<2:self.skipTest('Two devices required')
        from jax.sharding import Mesh
        from flax.training.train_state import TrainState
        from ppo_data_parallel import AXIS, replicate, shard_batch
        from ppo_rnn_og import Transition, recurrent_ppo_loss
        cfg=dict(self.cfg,DATA_PARALLEL_DEVICES=2,CLIP_EPS=.2,VF_COEF=.5,ENT_COEF=.01)
        obs=self.obs[:3];done=self.done[:3]
        _,pi,value=self.net.apply(self.params,self.carry,(obs,done))
        actions=jnp.arange(6).reshape(3,2)%17
        offsets=jnp.linspace(-.4,.4,6).reshape(3,2)
        trajectory=Transition(done,actions,value+offsets,jnp.zeros_like(value),
            pi.log_prob(actions)-offsets,obs,{})
        advantages=jnp.array([[9.,-5.],[10.,-8.],[11.,-7.]])
        targets=value+offsets
        tx=optax.chain(optax.clip_by_global_norm(.1),optax.adam(.001,eps=1e-5))
        state=TrainState.create(apply_fn=self.net.apply,params=self.params,tx=tx)
        def update(state,h,traj,adv,target,parallel):
            config=dict(cfg,_DATA_PARALLEL_ACTIVE=parallel)
            state=replicate(state,config);h=shard_batch(h,0,config)
            traj,adv,target=shard_batch((traj,adv,target),1,config)
            loss,grads=jax.value_and_grad(lambda p:recurrent_ppo_loss(
                p,self.net,h,traj,adv,target,cfg)[0])(state.params)
            grads=replicate(grads,config)
            return loss,grads,replicate(state.apply_gradients(grads=grads),config)
        inputs=(state,self.carry,trajectory,advantages,targets)
        reference=jax.jit(lambda *x:update(*x,False),device=jax.devices()[0])(*inputs)
        with Mesh(np.asarray(jax.devices()[:2]),(AXIS,)):
            parallel=jax.jit(lambda *x:update(*x,True))(*inputs)
            jax.block_until_ready(parallel)
        for a,b in zip(jax.tree_util.tree_leaves(reference),jax.tree_util.tree_leaves(parallel)):
            np.testing.assert_allclose(a,b,rtol=2e-4,atol=4e-6)

    def test_wrapper_uses_only_existing_observation_and_odometry(self):
        from craftax.craftax_env import make_craftax_env_from_name
        from symbolic_map_memory import with_map_memory
        base=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
        plain=with_map_memory(base,dict(self.cfg,NEURAL_MAP_CHANNELS=0))
        neural=with_map_memory(base,self.cfg)
        key=jax.random.PRNGKey(9)
        obs,state=neural.reset(key,base.default_params)
        original,original_state=plain.reset(key,base.default_params)
        np.testing.assert_array_equal(obs[:-2],original)
        np.testing.assert_array_equal(obs[-2:],original_state.position)
        self.assertEqual(neural.observation_space().shape,obs.shape)
        # Changing unseen raw simulator tiles cannot alter the augmentation.
        from symbolic_neural_map import NeuralMapPositionWrapper
        changed=state.replace(env_state=state.env_state.replace(map=jnp.zeros_like(state.env_state.map)))
        np.testing.assert_array_equal(NeuralMapPositionWrapper.augment(original,changed),obs)

    def test_learns_delayed_spatial_recall_with_erased_gru(self):
        # Train a cue/readout at separated locations; erase GRU before every
        # step, and read only after the cue is no longer visible. The two lanes
        # have identical final observations and opposite earlier cues.
        cfg=dict(LAYER_SIZE=16,NEURAL_MAP_CHANNELS=4)
        cell=NeuralMapCell(16,4,reset_gru=True)
        carry=initialize_carry(2,cfg)
        embed=jnp.zeros((2,16));visible=jnp.zeros((2,63,21))
        cue=visible.at[0,:,0].set(-1).at[1,:,0].set(1)
        inputs=(embed,cue,jnp.full((2,2),15),jnp.zeros(2,bool))
        params=cell.init(jax.random.PRNGKey(77),carry,inputs)
        # Head consumes the learned map read, never the GRU.
        params={'cell':params,'head':jax.random.normal(jax.random.PRNGKey(78),(128,))*.01}
        opt=optax.adam(.01);state=opt.init(params)
        def logits(p,erase=False):
            c,_=cell.apply(p['cell'],carry,inputs)
            c,_=cell.apply(p['cell'],c,(embed,visible,jnp.full((2,2),45),jnp.zeros(2,bool)))
            if erase:c=(c[0],carry[1],carry[2])
            _,read=cell.apply(p['cell'],c,(embed,visible,jnp.full((2,2),45),jnp.zeros(2,bool)))
            return read[:,-128:]@p['head']
        def loss(p):return optax.sigmoid_binary_cross_entropy(logits(p),jnp.array([0.,1.])).mean()
        @jax.jit
        def update(p,s):
            value,grads=jax.value_and_grad(loss)(p)
            updates,s=opt.update(grads,s,p)
            return optax.apply_updates(p,updates),s,value
        for _ in range(120):params,state,value=update(params,state)
        self.assertLess(float(value),.08)
        np.testing.assert_array_equal(logits(params)>0,[False,True])
        erased=logits(params,True)
        np.testing.assert_allclose(erased[0],erased[1],atol=1e-6)


if __name__=='__main__':unittest.main()

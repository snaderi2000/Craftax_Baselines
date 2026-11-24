"""Checks for real-return replay, self-imitation gradients, and spatial learning."""
import unittest
from types import SimpleNamespace
import jax
if not hasattr(jax,'tree'):
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from models.conditioned_symbolic import ConditionedSymbolicActorCritic
from ppo_leap_utils import EpisodeReplay,discounted_episode_returns
from train_ppo_leaps import self_imitation_loss


class Toy:
    def apply(self,p,obs):
        return distrax.Categorical(logits=jnp.broadcast_to(p['actor'],(len(obs),2))),jnp.full(len(obs),p['value'])


class LeapTests(unittest.TestCase):
    def test_terminal_return_included_and_unfinished_episode_not_replayed(self):
        np.testing.assert_allclose(discounted_episode_returns([1.,0.,2.]),[1+2*.99**2,2*.99,2.],rtol=1e-6)
        replay=EpisodeReplay(1,capacity=4)
        obs=np.zeros((2,1,1345),np.float32);actions=np.zeros((2,1),int)
        replay.add_rollout(obs,actions,np.array([[1.],[0.]]),np.zeros((2,1),bool))
        self.assertEqual(replay.size,0)
        replay.add_rollout(obs[:1],actions[:1],np.array([[2.]]),np.ones((1,1),bool))
        self.assertEqual(replay.size,3)
        np.testing.assert_allclose(replay.returns[:3],[1+2*.99**2,2*.99,2.],rtol=1e-6)
        replay.add_rollout(obs,actions,np.array([[3.],[4.]]),np.ones((2,1),bool))
        self.assertEqual(replay.size,4)
        b,ids=replay.sample(1000)
        self.assertTrue(np.isfinite(b['weights']).all());self.assertTrue((ids<4).all())
        replay.update(ids,np.arange(1000)%5)
        self.assertTrue(np.isfinite(replay.tree.tree).all())

    def test_only_better_observed_returns_drive_imitation(self):
        p=dict(actor=jnp.zeros(2),value=jnp.asarray(.2))
        b=dict(obs=jnp.zeros((4,2)),action=jnp.ones(4,jnp.int32),returns=jnp.ones(4),weights=jnp.ones(4))
        grad=jax.grad(lambda p:self_imitation_loss(Toy(),p,p,b)[0])(p)
        self.assertLess(float(grad['actor'][1]),0.)
        self.assertLess(float(grad['value']),0.)
        worse=dict(b,returns=jnp.zeros(4))
        loss,_=self_imitation_loss(Toy(),p,p,worse)
        self.assertAlmostEqual(float(loss),0.,places=6)
        # The separate imitation objective anchors drift relative to the PPO snapshot.
        changed=dict(p,actor=jnp.array([2.,-2.]))
        anchored,_=self_imitation_loss(Toy(),changed,p,worse)
        self.assertGreater(float(anchored),0.)

    def test_conditioned_network_learns_two_factor_interaction(self):
        model=ConditionedSymbolicActorCritic(channels=8,width=16,blocks=1)
        obs=jnp.zeros((4,1345)).at[2:,500].set(1.).at[jnp.array([1,3]),1323].set(.9)
        target=jnp.array([0.,0.,0.,1.])
        params=model.init(jax.random.PRNGKey(1),obs)
        opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0002));state=opt.init(params)
        def loss(p):
            pi,value=model.apply(p,obs)
            return ((value-target)**2).mean()-.05*pi.log_prob(target.astype(jnp.int32)).mean()
        @jax.jit
        def update(p,state):
            value,g=jax.value_and_grad(loss)(p);delta,state=opt.update(g,state,p)
            return optax.apply_updates(p,delta),state,value
        initial=float(loss(params))
        for _ in range(500):params,state,value=update(params,state)
        pi,predicted=model.apply(params,obs)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertLess(float(loss(params)),initial*.5)
        self.assertLess(float(((predicted-target)**2).mean()),.03)


if __name__=='__main__':unittest.main()

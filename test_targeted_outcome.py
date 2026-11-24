import unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Action,BlockType,Achievement
from craftax.craftax_classic.envs.craftax_state import Inventory
from models.actor_critic import ActorCritic
from targeted_outcome import (TASKS,N_TASKS,TargetedOutcomeWrapper,attempt_labels,
    initialize_head,empty_replay,append_replay,sample_replay,outcome_loss)


def state(tile=BlockType.STONE,inventory=None,flags=None,sleeping=False):
    mobs = SimpleNamespace(position=jnp.zeros((1,2),jnp.int32),mask=jnp.zeros(1,bool))
    return SimpleNamespace(map=jnp.full((3,3),tile.value),player_position=jnp.array([1,1]),
        player_direction=2,inventory=inventory or Inventory(),
        achievements=jnp.zeros(22,bool) if flags is None else flags,
        is_sleeping=sleeping,cows=mobs,zombies=mobs,skeletons=mobs)


class TargetedTests(unittest.TestCase):
    def test_relevance_repeats_and_events(self):
        self.assertEqual(N_TASKS,22)
        stone = TASKS.index('collect_stone')
        before = state(flags=jnp.zeros(22,bool).at[stone].set(True),inventory=Inventory(stone=1))
        after = state(flags=before.achievements,inventory=Inventory(stone=2))
        eligible,labels = attempt_labels(before,after,Action.DO.value)
        self.assertTrue(eligible[stone]); self.assertTrue(labels[stone])
        self.assertFalse(eligible[TASKS.index('make_stone_pickaxe')])
        self.assertFalse(attempt_labels(before,after,Action.LEFT.value)[0].any())
        failed = state(flags=before.achievements,inventory=Inventory(stone=1))
        self.assertFalse(attempt_labels(before,failed,Action.DO.value)[1][stone])
        craft = TASKS.index('make_stone_pickaxe')
        eligible,labels = attempt_labels(state(),state(inventory=Inventory(stone_pickaxe=1)),Action.MAKE_STONE_PICKAXE.value)
        self.assertTrue(eligible[craft]); self.assertTrue(labels[craft])
        water = TASKS.index('collect_drink')
        flags = jnp.zeros(22,bool).at[water].set(True)
        self.assertFalse(attempt_labels(state(BlockType.WATER,flags=flags),state(BlockType.WATER,flags=flags),Action.DO.value)[0][water])
        self.assertTrue(attempt_labels(state(BlockType.WATER),state(BlockType.WATER,flags=flags),Action.DO.value)[1][water])
        for name in TASKS:
            self.assertEqual(len(attempt_labels(state(),state(),Action.DO.value)[0]),22)

    def test_balance_eligible_only(self):
        obs = jnp.arange(24.).reshape(6,4)
        eligible = jnp.zeros((6,22),bool).at[:,9].set(True)
        labels = jnp.zeros((6,22),bool).at[1::2,9].set(True)
        b = append_replay(empty_replay(4,4),obs,jnp.arange(6),labels,eligible)
        self.assertEqual(int(b['counts'].sum()),6)
        batch = sample_replay(b,jax.random.PRNGKey(0),256)
        self.assertEqual(int(batch[-1].sum()),256)
        active = np.asarray(batch[-1])>0
        self.assertTrue(np.all(np.asarray(batch[2])[active]==9))
        self.assertEqual(int(np.asarray(batch[3])[active].sum()),128)
        self.assertEqual(len(batch[0]),22*256)
        for row,action,label in zip(batch[0][active],batch[1][active],batch[3][active]):
            np.testing.assert_array_equal(row,obs[int(action)])
            self.assertEqual(int(label),int(labels[int(action),9]))

    def test_gradients_initialization_and_passthrough(self):
        key = jax.random.PRNGKey(0)
        obs = jnp.ones((6,4)); net = ActorCritic(17,32)
        original = net.init(key,obs); params = initialize_head(original,key,32)
        pi,v = net.apply(params,obs); p0,v0 = net.apply(original,obs)
        np.testing.assert_array_equal(pi.logits_parameter(),p0.logits_parameter())
        np.testing.assert_array_equal(v,v0)
        batch = (obs,jnp.arange(6),jnp.full(6,9),jnp.array([0.,1.,0.,1.,0.,1.]),jnp.ones(6))
        grads = jax.grad(outcome_loss)(params,batch)['params']
        self.assertGreater(float(jnp.linalg.norm(grads['Dense_0']['kernel'])),0)
        for i in range(3,8):
            for g in jax.tree_util.tree_leaves(grads[f'Dense_{i}']):
                np.testing.assert_array_equal(g,np.zeros_like(g))
        self.assertEqual(float(outcome_loss(params,sample_replay(empty_replay(4,4),key,8))),0)
        after = state(inventory=Inventory(stone=1))
        class FakeEnv:
            def step(self,key,state,action,params):
                return obs,after,jnp.asarray(1.),jnp.asarray(True),{}
        _,returned,reward,done,info = TargetedOutcomeWrapper(FakeEnv()).step(None,state(),5,None)
        self.assertIs(returned,after); self.assertEqual(float(reward),1.); self.assertTrue(done)
        self.assertTrue(info['targeted/success/collect_stone'])


if __name__ == '__main__':
    unittest.main()

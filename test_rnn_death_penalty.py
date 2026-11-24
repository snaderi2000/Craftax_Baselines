import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from wrappers import ClassicDeathPenaltyWrapper, ClassicDiamondRewardWrapper
from ppo_rnn_og import ActorCriticRNN, ScannedRNN


@struct.dataclass
class State:
    map: object
    player_position: object
    player_health: object
    done: object
    achievements: object = None


class FakeEnv:
    def step(self, key, state, action, params=None):
        return jnp.zeros(1), state, jnp.asarray(.1), state.done, {}


class FakeAchievementEnv:
    def step(self, key, state, action, params=None):
        from craftax.craftax_classic.constants import Achievement
        flags = state.achievements
        index = jnp.where(action == 1, Achievement.COLLECT_DIAMOND.value, Achievement.COLLECT_WOOD.value)
        flags = flags.at[index].set(flags[index] | (action != 0))
        health = jnp.maximum(state.player_health-1, 0)
        successor = state.replace(achievements=flags, player_health=health)
        reward = (flags & ~state.achievements).sum() + .1*(health-state.player_health)
        return jnp.zeros(1), successor, reward, state.done, {}


class RNNDeathPenaltyTests(unittest.TestCase):
    def test_diamond_reward_is_five_total_once_and_preserves_other_rewards(self):
        from craftax.craftax_classic.constants import BlockType
        state = State(jnp.full((2,2),BlockType.GRASS.value),jnp.array([0,0]),
                      jnp.asarray(9),jnp.asarray(False),jnp.zeros(22,bool))
        wrapper = ClassicDiamondRewardWrapper(FakeAchievementEnv(), 5.)
        key = jax.random.PRNGKey(0)
        _, successor, first, _, _ = wrapper.step(key,state,1)
        self.assertAlmostEqual(float(first), 4.9, places=5)  # +5 diamond, -.1 health
        _, _, repeat, _, _ = wrapper.step(key,successor,1)
        self.assertAlmostEqual(float(repeat), -.1, places=5)  # No repeat bonus
        _, _, other, _, _ = wrapper.step(key,state,2)
        self.assertAlmostEqual(float(other), .9, places=5)  # Wood still +1
        _, _, ordinary, _, _ = wrapper.step(key,state,0)
        self.assertAlmostEqual(float(ordinary), -.1, places=5)

    def test_diamond_and_death_adjustments_compose(self):
        from craftax.craftax_classic.constants import BlockType
        state = State(jnp.full((2,2),BlockType.GRASS.value),jnp.array([0,0]),
                      jnp.asarray(1),jnp.asarray(True),jnp.zeros(22,bool))
        wrapper = ClassicDiamondRewardWrapper(ClassicDeathPenaltyWrapper(FakeAchievementEnv(), .5), 5.)
        _, _, reward, done, _ = wrapper.step(jax.random.PRNGKey(0),state,1)
        self.assertTrue(bool(done))
        self.assertAlmostEqual(float(reward), 4.4, places=5)  # +5 -.1 -.5

    def test_default_diamond_reward_matches_original(self):
        from craftax.craftax_classic.constants import BlockType
        state = State(jnp.full((2,2),BlockType.GRASS.value),jnp.array([0,0]),
                      jnp.asarray(9),jnp.asarray(False),jnp.zeros(22,bool))
        base = ClassicDeathPenaltyWrapper(FakeAchievementEnv(), .5)
        key = jax.random.PRNGKey(0)
        expected = base.step(key,state,1)
        actual = ClassicDiamondRewardWrapper(base, 1.).step(key,state,1)
        np.testing.assert_allclose(expected[2],actual[2])
        np.testing.assert_array_equal(expected[1].achievements,actual[1].achievements)

    def test_death_cost_and_timeout_exemption(self):
        from craftax.craftax_classic.constants import BlockType
        for cost in (2., .15, .5):
            wrapper = ClassicDeathPenaltyWrapper(FakeEnv(), cost)
            for lava, health, terminal, expected in [(True,9,True,.1-cost), (False,0,True,.1-cost),
                                                      (False,9,True,.1), (False,9,False,.1)]:
                with self.subTest(cost=cost, lava=lava, health=health, terminal=terminal):
                    tile = BlockType.LAVA.value if lava else BlockType.GRASS.value
                    state = State(jnp.full((2,2),tile),jnp.array([0,0]),jnp.asarray(health),jnp.asarray(terminal))
                    _, _, reward, _, _ = wrapper.step(jax.random.PRNGKey(0),state,0)
                    self.assertAlmostEqual(float(reward),expected,places=5)

    def test_rnn_resets_hidden_state_on_episode_boundary(self):
        net = ActorCriticRNN(17, {'LAYER_SIZE':16})
        obs = jnp.zeros((1,2,1345))
        reset = jnp.ones((1,2),bool)
        hidden = ScannedRNN.initialize_carry(2,16)
        weights = net.init(jax.random.PRNGKey(0),hidden,(obs,reset))
        a, _, av = net.apply(weights,hidden,(obs,reset))
        b, _, bv = net.apply(weights,jnp.ones_like(hidden),(obs,reset))
        np.testing.assert_allclose(a,b)
        np.testing.assert_allclose(av,bv)


if __name__ == '__main__':
    unittest.main()

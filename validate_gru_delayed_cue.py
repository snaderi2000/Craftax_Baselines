"""Positive control using the actual recurrent PPO trainer on a delayed-cue task.

Only the environment factory is substituted. Rollout collection, episode reset,
GAE, sequence shuffling, PPO clipping and Optax updates use ppo_rnn_og.make_train.
This toy task is a pipeline diagnostic, not evidence of Craftax performance.
"""
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces
import ppo_rnn_og as recurrent


@struct.dataclass
class CueState:
    cue:object
    t:object


class DelayedCue:
    default_params=None
    delay=7

    def action_space(self,params=None):return spaces.Discrete(2)
    def observation_space(self,params=None):return spaces.Box(-1.,1.,(3,))
    def observe(self,state):
        return jnp.array([jnp.where(state.t==0,2.*state.cue-1.,0.),
                          (state.t==self.delay).astype(jnp.float32),state.t/self.delay],jnp.float32)
    def reset(self,key,params=None):
        state=CueState(jax.random.randint(key,(),0,2),jnp.asarray(0,jnp.int32))
        return self.observe(state),state
    def step(self,key,state,action,params=None):
        done=state.t==self.delay
        reward=(done&(action==state.cue)).astype(jnp.float32)
        successor=state.replace(t=state.t+1)
        return self.observe(successor),successor,reward,done,{}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    outcomes=[];initial_params=None
    for reset in (False,True):
        cfg=dict(ENV_NAME='DelayedCue-v0',NUM_ENVS=256,NUM_STEPS=32,TOTAL_TIMESTEPS=2097152,
            NUM_MINIBATCHES=4,UPDATE_EPOCHS=4,GAMMA=.99,GAE_LAMBDA=.8,CLIP_EPS=.2,
            VF_COEF=.5,ENT_COEF=.01,LR=.001,MAX_GRAD_NORM=1.,ANNEAL_LR=False,
            LAYER_SIZE=32,SYMBOLIC_ENCODER='mlp',MEMORY_TYPE='gru',RESET_GRU_EVERY_STEP=reset,
            DEATH_PENALTY=0.,LAVA_PENALTY=0.,DIAMOND_REWARD=1.,PLANT_REWARD=1.,
            USE_OPTIMISTIC_RESETS=False,DEBUG=False,USE_WANDB=False,SAVE_POLICY=False,SEED=50)
        began=time.time()
        with patch.object(recurrent,'make_craftax_env_from_name',return_value=DelayedCue()):
            train=jax.jit(recurrent.make_train(cfg))
            result=train(jax.random.PRNGKey(50))
            jax.block_until_ready(result)
        state=result['runner_state'][0]
        assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves((state.params,state.opt_state)))
        # All decision observations are identical across balanced cue classes.
        # Independent network application uses no saved training hidden state.
        cues=jnp.arange(2048)%2
        obs=jnp.zeros((8,2048,3),jnp.float32).at[0,:,0].set(2.*cues-1.)
        obs=obs.at[7,:,1].set(1.).at[:,:,2].set(jnp.arange(8)[:,None]/7.)
        dones=jnp.zeros((8,2048),bool).at[0].set(True)
        network=recurrent.ActorCriticRNN(2,cfg);hidden=recurrent.ScannedRNN.initialize_carry(2048,32)
        weights=network.init(jax.random.PRNGKey(0),hidden,(obs[:1],dones[:1]))
        if initial_params is None:initial_params=weights
        else:
            for a,b in zip(jax.tree_util.tree_leaves(weights),jax.tree_util.tree_leaves(initial_params)):
                np.testing.assert_array_equal(a,b)
        _,pi,_=jax.jit(network.apply)(state.params,hidden,(obs,dones))
        probability=float(jnp.exp(pi.log_prob(jnp.broadcast_to(cues,(8,2048))))[-1].mean())
        accuracy=float((jnp.argmax(pi.logits[-1],axis=-1)==cues).mean())
        report=dict(reset_gru_every_step=reset,steps=cfg['TOTAL_TIMESTEPS'],delay_steps=7,
            expected_success_probability=probability,greedy_success_rate=accuracy,
            optimizer_updates=int(state.step),seconds=time.time()-began,parameters_and_optimizer_finite=True)
        outcomes.append(report);print(json.dumps(report),flush=True)
        del result,train
    passed=outcomes[0]['expected_success_probability']>.90 and abs(outcomes[1]['expected_success_probability']-.5)<.01
    summary=dict(passed=passed,outcomes=outcomes,identical_initial_parameters=True,
        pipeline='Actual ppo_rnn_og.make_train; environment factory replaced only',
        caveat='Small memory positive control; not a Craftax benchmark. Identical decision observations guarantee chance for the no-history control.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not passed:raise RuntimeError('Delayed-cue learning check failed; inspect before launching Craftax runs')


if __name__=='__main__':main()

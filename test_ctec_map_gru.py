"""Behavioral checks for map recurrence and episode-safe temporal exploration."""
import unittest
import tempfile
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ctec import (sample_future_indices,initialize,update_curiosity,contrastive_loss,
                  ExplorationTrainState,coefficient)
from ppo_rnn_og import ActorCriticRNN


class TemporalExplorationTests(unittest.TestCase):
    def test_two_device_curiosity_optimizer_and_reward_parity(self):
        import os,subprocess,sys
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=2',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--parallel-worker'],
            env=env,text=True,capture_output=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_future_pairs_never_cross_reset_or_point_to_present(self):
        resets=jnp.array([[1,1],[0,0],[1,0],[0,1],[1,0],[0,0]],bool)
        episodes=np.cumsum(np.asarray(resets),axis=0)
        for gamma in (.01,.3,.99):
            f,v=jax.jit(sample_future_indices)(jax.random.PRNGKey(8),resets,gamma)
            for t in range(6):
                for n in range(2):
                    if v[t,n]:
                        self.assertGreater(int(f[t,n]),t)
                        self.assertEqual(episodes[t,n],episodes[int(f[t,n]),n])
                    else:self.assertTrue(t==5 or bool(resets[t+1,n]))
        _,valid=sample_future_indices(jax.random.PRNGKey(1),jnp.ones((8,3),bool),.3)
        self.assertFalse(np.any(valid))

    def test_contrastive_learning_predicts_real_future_pairs(self):
        cfg=dict(CTEC_WIDTH=32,CTEC_DIM=8,CTEC_LR=.002)
        state=initialize(cfg,jax.random.PRNGKey(2)).learner
        obs=jnp.eye(16,1345);actions=jnp.arange(16)%17;future=jnp.roll(obs,17,axis=-1)
        valid=jnp.ones(16,bool)
        def loss(p):return contrastive_loss(p,state.apply_fn,obs,actions,future,valid)[0]
        @jax.jit
        def step(s):return s.apply_gradients(grads=jax.grad(loss)(s.params))
        first=float(loss(state.params))
        for _ in range(100):state=step(state)
        self.assertLess(float(loss(state.params)),first*.7)
        self.assertTrue(np.isfinite(float(loss(state.params))))

    def test_bonus_warmup_scale_terminal_mask_and_auxiliary_checkpoint(self):
        import orbax.checkpoint as ocp
        cfg=dict(CTEC_WIDTH=16,CTEC_DIM=8,CTEC_BATCH=16,CTEC_UPDATES=2,
            CTEC_COEF=.1,CTEC_WARMUP_STEPS=64,CTEC_RAMP_STEPS=64,
            NUM_ENVS=4,NUM_STEPS=8,TOTAL_TIMESTEPS=10000000000)
        initial=initialize(cfg,jax.random.PRNGKey(3))
        obs=jax.random.normal(jax.random.PRNGKey(4),(8,4,1345))
        action=jnp.zeros((8,4),jnp.int32);resets=jnp.zeros((8,4),bool).at[4,0].set(True)
        task=jnp.zeros((8,4)).at[2,0].set(-2).at[1,2].set(1)
        run=jax.jit(lambda s,u:update_curiosity(s,obs,action,resets,task,jax.random.PRNGKey(5),u,cfg))
        state,bonus,logs=run(initial,jnp.array(0))
        np.testing.assert_array_equal(bonus,0.)
        state,bonus,logs=run(state,jnp.array(10))
        self.assertGreater(float(bonus.mean()),0)
        self.assertLessEqual(float(bonus.mean()),.1*float(jnp.abs(task).mean())+1e-6)
        self.assertEqual(float(bonus[3,0]),0.)
        np.testing.assert_array_equal(bonus[-1],0.)
        self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state)))
        # Full PPO + contrastive optimizers and running moments survive reload.
        policy=ExplorationTrainState.create(apply_fn=lambda p,x:x,params={'w':jnp.ones(2)},tx=optax.adam(.001),curiosity=state)
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'32');check=ocp.PyTreeCheckpointer();check.save(path,policy)
            loaded=check.restore(path,item=policy)
            resumed=run(loaded.curiosity,jnp.array(11));control=run(state,jnp.array(11))
            for a,b in zip(jax.tree_util.tree_leaves(resumed),jax.tree_util.tree_leaves(control)):
                np.testing.assert_array_equal(a,b)
        _,end_bonus,_=run(state,jnp.array(400000000))
        np.testing.assert_array_equal(end_bonus,0.)
        # Large absolute horizons and continuation offsets must never enter
        # JAX as int32 literals (10B overflows even when the other operand is float).
        continuation=dict(cfg,START_TIMESTEPS=10000000000,LR_HORIZON_TIMESTEPS=12000000000)
        value=jax.jit(lambda u:coefficient(continuation,u))(jnp.array(0))
        self.assertGreater(float(value),0.)
        self.assertLess(float(value),.1)

    def test_map_gru_streaming_reset_and_temporal_gradient(self):
        cfg=dict(LAYER_SIZE=16,SYMBOLIC_ENCODER='map_mlp',MAP_VIEW_SIZE=9,
                 MAP_EMBEDDING_SIZE=4,GRU_CONCAT_OBSERVATION=True)
        net=ActorCriticRNN(17,cfg);n=2;t=5
        obs=jax.random.normal(jax.random.PRNGKey(6),(t,n,1345+162))
        obs=obs.at[...,1345:1426].set(jax.random.randint(jax.random.PRNGKey(7),(t,n,81),0,18))
        resets=jnp.zeros((t,n),bool).at[3,0].set(True);h=jnp.zeros((n,16))
        params=net.init(jax.random.PRNGKey(8),h,(obs,resets))
        whole,pi,values=net.apply(params,h,(obs,resets));stream=[]
        for i in range(t):
            h,p,v=net.apply(params,h,(obs[i:i+1],resets[i:i+1]));stream.append(p.logits)
        np.testing.assert_allclose(jnp.concatenate(stream),pi.logits,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(whole,h,rtol=1e-5,atol=1e-6)
        different=net.apply(params,jnp.ones_like(h)*10,(obs,resets))[1].logits
        np.testing.assert_allclose(different[3:,0],pi.logits[3:,0],rtol=1e-5,atol=1e-6)
        def temporal_loss(x):return net.apply(params,jnp.zeros_like(h),(x,resets))[2][-1,1]
        grad=jax.grad(temporal_loss)(obs)
        self.assertGreater(float(jnp.linalg.norm(grad[0,1,:1345])),1e-8)
        # Curiosity configuration must not change policy initialization.
        other=ActorCriticRNN(17,dict(cfg,CTEC_COEF=.1))
        p2=other.init(jax.random.PRNGKey(8),jnp.zeros_like(h),(obs,resets))
        for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(p2)):
            np.testing.assert_array_equal(a,b)

    def test_presets_use_recurrent_map_and_reject_invalid_ctec(self):
        from ppo_experiment_config import Experiment
        import dataclasses
        data=dict(name='test_map_gru',architectures=[dict(name='map_mlp_gru',train_gpu=0,train_gpus=[0,3],evaluation_gpu=1)],
                  ppo=dict(ctec_coef=.1,gru_concat_observation=True,optimistic_resets=False))
        with tempfile.TemporaryDirectory() as d:
            import json
            path=Path(d)/'config.json';path.write_text(json.dumps(data));spec=Experiment.load(path)
            m=spec.metadata(spec.architectures[0],Path.cwd(),'python')
            self.assertTrue(m['recurrent']);self.assertEqual(m['symbolic_encoder'],'map_mlp')
            self.assertIn('--ctec_coef',m['command']);self.assertIn('--map_embedding_size',m['command'])
            with self.assertRaises(ValueError):dataclasses.replace(spec.ppo,ctec_gamma=1.).validate()


def parallel_worker():
    from jax.sharding import Mesh
    from ppo_data_parallel import AXIS,replicate,shard_batch
    cfg=dict(CTEC_WIDTH=16,CTEC_DIM=8,CTEC_BATCH=16,CTEC_UPDATES=2,
        CTEC_COEF=.1,CTEC_WARMUP_STEPS=0,CTEC_RAMP_STEPS=1,
        NUM_ENVS=8,NUM_STEPS=8,TOTAL_TIMESTEPS=1000000,DATA_PARALLEL_DEVICES=2)
    state=initialize(cfg,jax.random.PRNGKey(3))
    obs=jax.random.normal(jax.random.PRNGKey(4),(8,8,1345))
    actions=jnp.arange(64,dtype=jnp.int32).reshape(8,8)%17
    resets=jnp.zeros((8,8),bool).at[4,0].set(True).at[6,7].set(True)
    task=jnp.zeros((8,8)).at[2,0].set(-2).at[1,5].set(1)
    def run(s,o,a,r,w,parallel):
        config=dict(cfg,_DATA_PARALLEL_ACTIVE=parallel)
        s=replicate(s,config);o,a,r,w=shard_batch((o,a,r,w),1,config)
        return update_curiosity(s,o,a,r,w,jax.random.PRNGKey(5),jnp.array(10),config)
    args=(state,obs,actions,resets,task)
    reference=jax.jit(lambda *a:run(*a,False),device=jax.devices()[0])(*args)
    with Mesh(np.asarray(jax.devices()),(AXIS,)):
        actual=jax.jit(lambda *a:run(*a,True))(*args)
        jax.block_until_ready(actual)
    for a,b in zip(jax.tree_util.tree_leaves(reference),jax.tree_util.tree_leaves(actual)):
        np.testing.assert_allclose(a,b,rtol=2e-4,atol=3e-6)
    assert all(a.sharding.is_fully_replicated for a in jax.tree_util.tree_leaves(actual[0]))
    assert float(actual[1].mean())>0


if __name__=='__main__':
    import sys,faulthandler
    faulthandler.dump_traceback_later(120,repeat=False)
    if '--parallel-worker' in sys.argv:parallel_worker()
    else:unittest.main()

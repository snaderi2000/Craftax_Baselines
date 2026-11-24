"""Map-MLP PPO parity and checkpoint checks on four forced CPU devices."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FeedforwardParallelTests(unittest.TestCase):
    def test_supported_vanilla_configuration(self):
        from ppo_data_parallel import validate_data_parallel
        cfg=dict(RECURRENT=False,ENV_NAME='Craftax-Classic-Symbolic-v1',
            SYMBOLIC_ENCODER='map_mlp',NUM_ENVS=12,NUM_STEPS=8,NUM_MINIBATCHES=8,
            DATA_PARALLEL_DEVICES=4,NUM_REPEATS=1)
        self.assertEqual(validate_data_parallel(cfg,4),4)
        for change in (dict(TRAIN_ICM=True),dict(AUX_TASK='inverse'),dict(CULTURE_COEF=.1),
                dict(LEARNING_PROGRAM_GRAPH={}),dict(TRAIN_COMPONENT='actor'),
                dict(SYMBOLIC_ENCODER='mlp'),dict(NUM_STEPS=2),dict(NUM_REPEATS=2)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_data_parallel(dict(cfg,**change),4)

    def test_cpu_one_two_four_device_math_training_and_resume(self):
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=4',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            MPLCONFIGDIR='/tmp/craftax-parallel-test-mpl')
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--cpu-worker'],
            cwd=Path(__file__).resolve().parent,env=env,text=True,capture_output=True,timeout=300)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        print(result.stdout.strip())


def cpu_validation():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    from flax import struct
    from flax.training.train_state import TrainState
    from gymnax.environments import spaces
    from jax.sharding import Mesh
    from models.actor_critic_map_mlp import map_mlp_network
    import ppo
    from ppo_data_parallel import (AXIS,compile_train,execute_train,replicate,shard_batch,
        checkpoint_train_state)
    from types import SimpleNamespace
    import time

    assert len(jax.devices())==4
    devices=jax.devices()
    cfg=dict(RECURRENT=False,ENV_NAME='Craftax-Classic-Symbolic-v1',SYMBOLIC_ENCODER='map_mlp',
        LAYER_SIZE=8,MAP_VIEW_SIZE=9,MAP_EMBEDDING_SIZE=4,CLIP_EPS=.2,VF_COEF=.5,ENT_COEF=.01,
        NUM_ENVS=16,NUM_STEPS=4,NUM_MINIBATCHES=2,UPDATE_EPOCHS=2,NUM_REPEATS=1,
        TRAIN_COMPONENT='both',TRAIN_ICM=False,RESUME_FROM=None,COMPONENT_INIT_FROM=None,
        LR=.001,MAX_GRAD_NORM=.01,ANNEAL_LR=True,LR_HORIZON_TIMESTEPS=10000000000,
        TOTAL_TIMESTEPS=128,GAMMA=.99,GAE_LAMBDA=.8,SEED=50,
        USE_OPTIMISTIC_RESETS=True,OPTIMISTIC_RESET_RATIO=4,
        LAVA_PENALTY=2.,DEATH_PENALTY=0.,DIAMOND_REWARD=1.,PLANT_REWARD=1.,
        DEBUG=False,USE_WANDB=False,SAVE_POLICY=False,SAVE_POLICY_MILESTONES=())
    net=map_mlp_network(cfg,3);n=16;cells=81
    obs=jnp.sin(jnp.arange(n*(1345+2*cells)).reshape(n,-1)*.013)
    obs=obs.at[:,1345:1345+cells].set(jnp.arange(n*cells).reshape(n,cells)%18)
    params=net.init(jax.random.PRNGKey(11),obs)
    pi,value=net.apply(params,obs)
    action=jnp.arange(n)%3;offset=jnp.linspace(-.7,.7,n)
    zeros=jnp.zeros(n)
    traj=ppo.Transition(zeros.astype(bool),action,value+offset,zeros,zeros,zeros,
        pi.log_prob(action)-offset,obs,obs,{})
    advantages=jnp.arange(n,dtype=jnp.float32)-7.1
    targets=value+jnp.cos(jnp.arange(n))
    tx=optax.chain(optax.clip_by_global_norm(.01),optax.adam(.001,eps=1e-5))
    initial=TrainState.create(apply_fn=net.apply,params=params,tx=tx)
    def update(state,traj,adv,target,count):
        c=dict(cfg,DATA_PARALLEL_DEVICES=count,_DATA_PARALLEL_ACTIVE=count>1)
        state=replicate(state,c)
        traj,adv,target=shard_batch((traj,adv,target),0,c)
        (loss,terms),grad=jax.value_and_grad(ppo.feedforward_ppo_loss,has_aux=True)(
            state.params,net,traj,adv,target,c)
        grad=replicate(grad,c)
        return loss,terms,grad,replicate(state.apply_gradients(grads=grad),c)
    reference=jax.jit(lambda *a:update(*a,1),device=devices[0])(initial,traj,advantages,targets)
    assert float(optax.global_norm(reference[2]))>.01
    maximum=0.
    def compare(a,b,tol=5e-5):
        nonlocal maximum
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            x,y=np.asarray(x),np.asarray(y)
            np.testing.assert_allclose(x,y,rtol=tol,atol=3e-6)
            maximum=max(maximum,float(np.max(np.abs(x.astype(float)-y.astype(float)))))
    for count in (2,4):
        mesh=Mesh(np.asarray(devices[:count]),(AXIS,))
        with mesh:
            actual=jax.jit(lambda *a:update(*a,count))(initial,traj,advantages,targets)
            jax.block_until_ready(actual)
        compare(reference,actual)
        assert all(x.sharding.is_fully_replicated for x in jax.tree_util.tree_leaves(actual[3].params))
        # Global shuffle must preserve all flattened rollout samples.
        def shuffle(x):
            x=shard_batch(x,1,dict(DATA_PARALLEL_DEVICES=count,_DATA_PARALLEL_ACTIVE=True))
            flat=x.reshape(-1)
            return jnp.take(flat,jax.random.permutation(jax.random.PRNGKey(8),flat.size))
        ids=jnp.arange(64).reshape(4,16)
        with mesh:shuffled=jax.jit(shuffle)(ids)
        np.testing.assert_array_equal(np.sort(np.asarray(shuffled)),np.arange(64))

    # Full existing PPO train function with real map-memory, lava-cost, logging
    # and global optimistic-reset wrappers around a deterministic cheap world.
    @struct.dataclass
    class Raw:
        map:object
        player_position:object
        elapsed:object
        limit:object
    from craftax.craftax_classic.constants import BlockType
    patch=jax.nn.one_hot(jnp.zeros((7,9),jnp.int32),21).reshape(-1)
    observation=jnp.concatenate((patch,jnp.zeros(22)))
    class Simulator:
        default_params=None
        def action_space(self,params=None):return spaces.Discrete(3)
        def observation_space(self,params=None):return spaces.Box(0.,1.,(1345,),jnp.float32)
        def reset(self,key,params=None):
            return observation,Raw(jnp.zeros((19,19),jnp.int32),jnp.array([9,9]),
                jnp.array(0),jax.random.randint(key,(),2,13))
        def step(self,key,state,action,params=None):
            done=state.elapsed+1>=state.limit
            state=state.replace(elapsed=state.elapsed+1,
                map=state.map.at[9,9].set(jnp.where(done,BlockType.LAVA.value,0)))
            return observation,state,action.astype(jnp.float32)*.1,done,{}
    ppo.make_craftax_env_from_name=lambda *a,**k:Simulator()
    key=jax.random.PRNGKey(50)
    runs={}
    for count in (1,2,4):
        runs[count]=execute_train(dict(cfg,DATA_PARALLEL_DEVICES=count),key)
        assert int(runs[count]['runner_state'][-1])==2
        assert int(runs[count]['runner_state'][0].step)==8
        if count>1:
            out=runs[count]['runner_state'][2]
            assert len(out.addressable_shards)==count
            assert all(s.data.shape[0]==16//count for s in out.addressable_shards)
            compare(runs[1]['runner_state'][0],runs[count]['runner_state'][0],tol=2e-4)
            for x,y in zip(jax.tree_util.tree_leaves(runs[1]['runner_state'][1:3]),
                           jax.tree_util.tree_leaves(runs[count]['runner_state'][1:3])):
                np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    assert int(np.asarray(runs[4]['runner_state'][1].env_state.elapsed).max())>4

    with tempfile.TemporaryDirectory() as folder:
        # Exercise the actual trainer callback with a label larger than int32.
        start=2**31;milestone=start+64
        ppo.wandb.run=SimpleNamespace(dir=folder)
        saving=dict(cfg,DATA_PARALLEL_DEVICES=4,START_TIMESTEPS=start,
            TOTAL_TIMESTEPS=start+128,USE_WANDB=True,SAVE_POLICY=True,
            SAVE_POLICY_MILESTONES=(milestone,),POLICY_MAX_TO_KEEP=3)
        saved_run=execute_train(saving,key)
        destination=Path(folder)/'policies'/str(milestone)
        assert destination.exists() and not (destination.parent/str(np.int32(-2147483584))).exists()
        payload=destination/'default' if (destination/'default').exists() else destination
        restored=ocp.PyTreeCheckpointer().restore(str(payload),item=initial)
        assert int(restored.step)==4
        # Restore actual full checkpoints and run PPO in either direction.
        for count in (1,4):
            resume=dict(cfg,DATA_PARALLEL_DEVICES=count,START_TIMESTEPS=milestone,
                TOTAL_TIMESTEPS=milestone+64,RESUME_FROM=str(destination.parent),RESUME_STEP=milestone)
            continued=execute_train(resume,key)
            assert int(continued['runner_state'][0].step)==8
            if count==1:one=continued
            else:compare(one['runner_state'][0],continued['runner_state'][0],tol=2e-4)
        single_path=Path(folder)/'single'
        manager=ocp.CheckpointManager(str(single_path),ocp.PyTreeCheckpointer(),
            ocp.CheckpointManagerOptions(create=True))
        manager.save(128,checkpoint_train_state(runs[1]['runner_state'][0]))
        manager.wait_until_finished();manager.close()
        for count in (1,4):
            resumed=execute_train(dict(cfg,DATA_PARALLEL_DEVICES=count,RESUME_FROM=str(single_path),
                RESUME_STEP=128,START_TIMESTEPS=128,TOTAL_TIMESTEPS=192),key)
            assert int(resumed['runner_state'][0].step)==12
            if count==1:one=resumed
            else:compare(one['runner_state'][0],resumed['runner_state'][0],tol=2e-4)
        # Check the checkpoint in a genuinely one-visible-device process.
        code="""import sys,jax,numpy as np,orbax.checkpoint as o
s=o.PyTreeCheckpointer().restore(sys.argv[1]);assert int(s['step'])==4
assert len(jax.devices())==1 and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(s['params']))
"""
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=1')
        loaded=subprocess.run([sys.executable,'-c',code,str(payload)],env=env,capture_output=True,text=True,timeout=45)
        assert loaded.returncode==0,loaded.stdout+loaded.stderr
    from training_time_limit import run_update_loop
    mesh=Mesh(np.asarray(devices),(AXIS,))
    def capped(x):
        x=shard_batch(x,0,dict(DATA_PARALLEL_DEVICES=4,_DATA_PARALLEL_ACTIVE=True))
        return run_update_loop(lambda c,_:((c[0]+1,c[1]+1),c[0].mean()),
            (x,jnp.asarray(0)),32,time.time()-1)
    with mesh:stopped=jax.jit(capped)(jnp.zeros(16));jax.block_until_ready(stopped)
    assert int(stopped[0][1])==16
    print(json.dumps(dict(cpu_devices=4,one_two_four_device_gradient_and_clipped_adam_parity=True,
        maximum_absolute_difference=maximum,full_ppo_update_parity=True,global_shuffle_and_map_memory=True,
        four_device_environment_sharding=True,actual_checkpoint_callback_above_int32=True,
        full_optimizer_resume_one_to_four_and_four_to_one=True,deadline_update_boundary=True)))


if __name__=='__main__':
    if '--cpu-worker' in sys.argv:cpu_validation()
    else:unittest.main()

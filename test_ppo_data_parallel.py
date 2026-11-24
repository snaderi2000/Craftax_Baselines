"""Multi-device PPO checks use two forced CPU devices, without occupying GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DataParallelTests(unittest.TestCase):
    def test_rejects_invalid_partitioning_and_policy_repeats(self):
        from ppo_data_parallel import validate_data_parallel
        cfg=dict(NUM_ENVS=1024,NUM_MINIBATCHES=8,MEMORY_TYPE='gru',NUM_REPEATS=1)
        for count in (1,2,4):
            self.assertEqual(validate_data_parallel(dict(cfg,DATA_PARALLEL_DEVICES=count),4),count)
        for changes in (dict(DATA_PARALLEL_DEVICES=3),dict(DATA_PARALLEL_DEVICES=2,NUM_ENVS=24,NUM_MINIBATCHES=8),
                        dict(DATA_PARALLEL_DEVICES=2,NUM_REPEATS=2),dict(DATA_PARALLEL_DEVICES=2,MEMORY_TYPE='transformer')):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_data_parallel(dict(cfg,**changes),4)

    def test_cpu_global_gradient_optimizer_memory_and_checkpoint_parity(self):
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=2',
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/craftax-parallel-test-mpl')
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
    from flax.training.train_state import TrainState
    from jax.sharding import Mesh,PartitionSpec,NamedSharding
    from ppo_data_parallel import (AXIS,replicate,shard_batch,execute_train,
        checkpoint_train_state,save_checkpoint_from_callback)
    from ppo_rnn_og import ActorCriticRNN,ScannedRNN,Transition,recurrent_ppo_loss

    assert len(jax.devices())==2
    devices=jax.devices();mesh=Mesh(np.asarray(devices),(AXIS,))
    cfg=dict(LAYER_SIZE=8,SYMBOLIC_ENCODER='mlp',CLIP_EPS=.2,VF_COEF=.5,ENT_COEF=.01)
    net=ActorCriticRNN(3,cfg);t,n=4,8
    obs=jnp.sin(jnp.arange(t*n*6,dtype=jnp.float32).reshape(t,n,6)*.13)
    done=jnp.zeros((t,n),bool).at[1,0].set(True).at[2,5].set(True)
    hidden=ScannedRNN.initialize_carry(n,8)
    params=net.init(jax.random.PRNGKey(42),hidden,(obs,done))
    _,pi,value=net.apply(params,hidden,(obs,done))
    actions=(jnp.arange(t*n).reshape(t,n)%3).astype(jnp.int32)
    offsets=jnp.linspace(-.7,.7,t*n).reshape(t,n)
    trajectory=Transition(done,actions,value+offsets,jnp.zeros((t,n)),
        pi.log_prob(actions)-offsets,obs,{})
    # Deliberately different per-device distributions expose accidental local
    # advantage normalization; global clipping must follow gradient reduction.
    advantages=jnp.broadcast_to(jnp.asarray([9.,10.,11.,12.,-8.,-7.,-6.,-5.]),(t,n))
    targets=value+jnp.cos(jnp.arange(t*n).reshape(t,n))
    tx=optax.chain(optax.clip_by_global_norm(.01),optax.adam(.001,eps=1e-5))
    initial=TrainState.create(apply_fn=net.apply,params=params,tx=tx)

    def update(state,h,traj,adv,target,parallel):
        configuration=dict(cfg,DATA_PARALLEL_DEVICES=2,_DATA_PARALLEL_ACTIVE=parallel)
        state=replicate(state,configuration)
        h=shard_batch(h,0,configuration)
        traj,adv,target=shard_batch((traj,adv,target),1,configuration)
        loss,gradient=jax.value_and_grad(lambda p:recurrent_ppo_loss(p,net,h,traj,adv,target,cfg)[0])(state.params)
        gradient=replicate(gradient,configuration)
        return loss,gradient,replicate(state.apply_gradients(grads=gradient),configuration)

    reference=jax.jit(lambda *args:update(*args,False),device=devices[0])(initial,hidden,trajectory,advantages,targets)
    with mesh:
        parallel=jax.jit(lambda *args:update(*args,True))(initial,hidden,trajectory,advantages,targets)
        jax.block_until_ready(parallel)
    maximum=0.
    for a,b in zip(jax.tree_util.tree_leaves(reference),jax.tree_util.tree_leaves(parallel)):
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=5e-5,atol=3e-6)
        maximum=max(maximum,float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    assert float(optax.global_norm(reference[1]))>.01
    assert all(x.sharding.is_fully_replicated for x in jax.tree_util.tree_leaves(parallel[2].params))
    assert all(len(x.addressable_shards)==2 for x in jax.tree_util.tree_leaves(parallel[2].params))

    # Global shuffling preserves complete time sequences and their membership.
    permutation=jax.random.permutation(jax.random.PRNGKey(99),n)
    ids=jnp.broadcast_to(jnp.arange(n),(t,n))
    def shuffled(x):
        x=shard_batch(x,1,dict(DATA_PARALLEL_DEVICES=2,_DATA_PARALLEL_ACTIVE=True))
        return jnp.swapaxes(jnp.take(x,permutation,axis=1).reshape(t,2,n//2),0,1)
    with mesh:actual=jax.jit(shuffled)(ids)
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(ids[:,permutation]).reshape(t,2,n//2).swapaxes(0,1))

    # Exercise the real map-memory and global optimistic-reset wrappers using
    # a cheap deterministic simulator. Full Craftax/CNN training is checked
    # separately by the GPU smoke gate.
    from flax import struct
    from symbolic_map_memory import ClassicMapMemoryWrapper
    from wrappers import OptimisticResetVecEnvWrapper
    @struct.dataclass
    class Raw:
        map:object
        player_position:object
        elapsed:object
        limit:object
    patch=jax.nn.one_hot(jnp.zeros((7,9),jnp.int32),21).reshape(-1)
    observation=jnp.concatenate((patch,jnp.zeros(22)))
    class Simulator:
        def reset(self,key,params=None):
            return observation,Raw(jnp.zeros((19,19),jnp.int32),jnp.array([9,9]),jnp.array(0),jnp.array(100))
        def step(self,key,state,action,params=None):
            state=state.replace(elapsed=state.elapsed+1)
            return observation,state,jnp.array(0.),state.elapsed>=state.limit,{}
    env=OptimisticResetVecEnvWrapper(ClassicMapMemoryWrapper(Simulator(),9),n,2)
    initial_obs,initial_state=env.reset(jax.random.PRNGKey(100),None)
    raw=initial_state.env_state.replace(limit=initial_state.env_state.limit.at[0].set(1))
    initial_state=initial_state.replace(env_state=raw,
        terrain=initial_state.terrain.at[:,0,0].set(9))
    def roll_memory(o,state,parallel):
        configuration=dict(DATA_PARALLEL_DEVICES=2,_DATA_PARALLEL_ACTIVE=parallel)
        o,state=shard_batch((o,state),0,configuration)
        def step(carry,i):
            _,s=carry
            o,s,_,_,_=env.step(jax.random.fold_in(jax.random.PRNGKey(101),i),s,jnp.zeros(n,jnp.int32),None)
            return shard_batch((o,s),0,configuration),None
        # Two four-step rollout chunks carry the same map state forward.
        first,_=jax.lax.scan(step,(o,state),jnp.arange(4))
        return jax.lax.scan(step,first,jnp.arange(4,8))[0]
    memory_reference=jax.jit(lambda o,s:roll_memory(o,s,False),device=devices[0])(initial_obs,initial_state)
    with mesh:memory_parallel=jax.jit(lambda o,s:roll_memory(o,s,True))(initial_obs,initial_state)
    for a,b in zip(jax.tree_util.tree_leaves(memory_reference),jax.tree_util.tree_leaves(memory_parallel)):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    state=memory_parallel[1]
    np.testing.assert_array_equal(np.asarray(state.elapsed),[7,8,8,8,8,8,8,8])
    assert int(state.terrain[0,0,0])==0
    assert np.asarray(state.terrain[1:,0,0]).min()==9

    callbacks=[]
    with tempfile.TemporaryDirectory() as folder:
        destination=Path(folder)/'64'
        # Exercise the actual milestone callback path. Calling Orbax directly
        # inside this callback dispatches nested backend barriers and deadlocks
        # on GPUs; the isolated CPU writer must perform the complete save.
        def save_callback(state):
            save_checkpoint_from_callback(folder,64,state,2,timeout=60)
            callbacks.append(64)
        def checkpoint_update(state):
            state=replicate(state,dict(DATA_PARALLEL_DEVICES=2,_DATA_PARALLEL_ACTIVE=True))
            jax.debug.callback(save_callback,state)
            return state
        with mesh:
            saved_state=jax.jit(checkpoint_update)(parallel[2])
            jax.block_until_ready(saved_state)
        jax.effects_barrier()
        assert callbacks==[64]
        assert destination.exists()
        if (destination/'default').exists():destination=destination/'default'
        restored=ocp.PyTreeCheckpointer().restore(str(destination),item=initial)
        assert int(restored.step)==1
        for expected,actual in zip(jax.tree_util.tree_leaves(parallel[2]),jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(np.asarray(expected),np.asarray(actual))
        continued=jax.jit(lambda *args:update(*args,False),device=devices[0])(
            restored,hidden,trajectory,advantages,targets)
        control=jax.jit(lambda *args:update(*args,False),device=devices[0])(
            reference[2],hidden,trajectory,advantages,targets)
        assert int(continued[2].step)==2
        for a,b in zip(jax.tree_util.tree_leaves(continued),jax.tree_util.tree_leaves(control)):
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=5e-5,atol=3e-6)
        # The saved checkpoint must also load in a fresh ONE-device process.
        code="""import sys,jax,jax.numpy as jnp,numpy as np,orbax.checkpoint as ocp
from ppo_rnn_og import ActorCriticRNN,ScannedRNN
s=ocp.PyTreeCheckpointer().restore(sys.argv[1]);assert int(s['step'])==1
n=ActorCriticRNN(3,{'LAYER_SIZE':8,'SYMBOLIC_ENCODER':'mlp'})
_,p,v=n.apply(s['params'],ScannedRNN.initialize_carry(2,8),(jnp.zeros((1,2,6)),jnp.zeros((1,2),bool)))
assert np.isfinite(np.asarray(v)).all() and len(jax.devices())==1
"""
        env_one=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=1')
        loaded=subprocess.run([sys.executable,'-c',code,str(destination)],env=env_one,text=True,capture_output=True,timeout=45)
        assert loaded.returncode==0,loaded.stdout+loaded.stderr

    from training_time_limit import run_update_loop
    import time
    deadline=time.time()-1
    def capped(x):
        x=shard_batch(x,0,dict(DATA_PARALLEL_DEVICES=2,_DATA_PARALLEL_ACTIVE=True))
        def increment(carry,_):
            values,count=carry
            return (values+1,count+1),values.mean()
        return run_update_loop(increment,(x,jnp.asarray(0)),32,deadline)
    with mesh:capped_result=jax.jit(capped)(jnp.zeros(n));jax.block_until_ready(capped_result)
    assert int(capped_result[0][1])==16
    print(json.dumps(dict(cpu_devices=2,global_gradient_and_adam_parity=True,
        global_gradient_clipping_active=True,maximum_absolute_math_difference=maximum,
        global_shuffle_preserves_sequences=True,global_reset_pool_and_map_memory_parity=True,
        single_device_resume_from_parallel_checkpoint=True,
        orbax_checkpoint_saved_inside_parallel_callback=True,
        deadline_stops_at_global_update_boundary=True,
        checkpoint_callback_count=len(callbacks))))


if __name__=='__main__':
    if '--cpu-worker' in sys.argv:cpu_validation()
    else:unittest.main()

"""Global-array data parallelism for PPO on one host.

The environment count, reset pool, minibatches, reductions and optimizer
remain global. GSPMD partitions their array work; no local PPO objective or
per-device optimizer is introduced.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec


AXIS = 'ppo_data'
_ACTIVE_MESH = None


def validate_data_parallel(config, available_devices=None):
    count = config.get('DATA_PARALLEL_DEVICES', 1)
    if isinstance(count, bool) or not isinstance(count, int) or count not in (1, 2, 4, 8):
        raise ValueError('data_parallel_devices must be one of 1, 2, 4, 8')
    if count > 1:
        if jax.process_count() != 1:
            raise ValueError('Data-parallel PPO currently supports one host only')
        envs, batches = config['NUM_ENVS'], config['NUM_MINIBATCHES']
        if config.get('RECURRENT', True):
            if envs % count or envs % batches or (envs // batches) % count:
                raise ValueError('Global environments and each recurrent minibatch must divide evenly across devices')
            if config.get('MEMORY_TYPE', 'gru') != 'gru':
                raise ValueError('Data-parallel recurrent PPO currently supports GRU memory')
        else:
            if envs % count or (envs * config['NUM_STEPS']) % (batches * count):
                raise ValueError('Global environments and each flat PPO minibatch must divide evenly across devices')
            if config.get('SYMBOLIC_ENCODER') != 'map_mlp' or config.get('ENV_NAME') != 'Craftax-Classic-Symbolic-v1':
                raise ValueError('Data-parallel feedforward PPO currently supports Classic symbolic map_mlp')
            unsupported = ('TRAIN_ICM', 'USE_RND', 'USE_E3B', 'CULTURE_PATH',
                'CULTURE_COEF', 'BC_STEPS', 'STONE_PROBE', 'STONE_AUX_COEF',
                'REWARD_SIGN_AUX_COEF', 'MULTI_OUTCOME_AUX_COEF', 'TARGETED_OUTCOME_AUX_COEF',
                'AUX_COEF', 'ID_PRACTICE', 'ID_PRACTICE_ARCHIVE_FROM',
                'CLUSTER_FREE_PRACTICE', 'DIAMOND_DRILLS', 'DIAMOND_CURRICULUM',
                'CULTURE_DIAMOND_ORACLE', 'LEARNING_PROGRAM', 'LEARNING_PROGRAM_GRAPH',
                'COMPONENT_INIT_FROM', 'RESET_SEED_BANK', 'RESUME_RESET_OPTIMIZER')
            if (any(config.get(key) for key in unsupported)
                    or config.get('LEARNING_PROGRAM_GRAPH') is not None
                    or config.get('AUX_TASK', 'none') != 'none'
                    or config.get('TRAIN_COMPONENT', 'both') != 'both'):
                raise ValueError('Data-parallel map_mlp supports vanilla actor/critic PPO only')
        if config.get('NUM_REPEATS', 1) != 1:
            raise ValueError('Data-parallel PPO supports one shared policy per process')
    if available_devices is not None and count > available_devices:
        raise ValueError(f'Requested {count} training devices but only {available_devices} are visible')
    return count


def active(config):
    """Runtime-only flag: saved inference configs never require a Mesh."""
    return bool(config.get('_DATA_PARALLEL_ACTIVE', False))


def shard_batch(tree, axis, config):
    if not active(config):
        return tree

    def constrain(x):
        x = jnp.asarray(x)
        if x.ndim <= axis or x.shape[axis] % config['DATA_PARALLEL_DEVICES']:
            raise ValueError(f'Cannot shard array {x.shape} on batch axis {axis}')
        axes = [None] * x.ndim
        axes[axis] = AXIS
        return jax.lax.with_sharding_constraint(x, PartitionSpec(*axes))

    return jax.tree_util.tree_map(constrain, tree)


def replicate(tree, config):
    if not active(config):
        return tree
    return jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(jnp.asarray(x), PartitionSpec()), tree)


def local_minibatches(batch, rng, num_minibatches, config):
    """Stratified per-device shuffle to avoid cross-GPU observation all-to-all traffic.

    Each device independently permutes its local i.i.d. environments and packs
    them into (num_minibatches, T, local_mb, ...), yielding the exact global
    (num_minibatches, T, global_mb, ...) sharded across devices on axis 2.
    """
    count = config['DATA_PARALLEL_DEVICES']
    mesh = _ACTIVE_MESH if _ACTIVE_MESH is not None else Mesh(np.asarray(jax.local_devices()[:count]), (AXIS,))
    local_envs = config['NUM_ENVS'] // count
    local_mb = local_envs // num_minibatches
    dev_keys = jax.random.split(rng, count)
    in_spec = jax.tree_util.tree_map(lambda x: PartitionSpec(None, AXIS, *([None] * (x.ndim - 2))), batch)
    out_spec = jax.tree_util.tree_map(lambda x: PartitionSpec(None, None, AXIS, *([None] * (x.ndim - 2))), batch)

    def _shuffle(local_batch, key_slice):
        perm = jax.random.permutation(key_slice[0], local_envs)
        shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), local_batch)
        return jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                jnp.reshape(x, [x.shape[0], num_minibatches, local_mb] + list(x.shape[2:])),
                1,
                0,
            ),
            shuffled,
        )

    return shard_map(
        _shuffle,
        mesh=mesh,
        in_specs=(in_spec, PartitionSpec(AXIS)),
        out_specs=out_spec,
        check_rep=False,
    )(batch, dev_keys)


def sharded_value_and_grad_bf16(local_loss_fn, params, init_hstate, traj_batch, advantages, targets, config):
    """Run local forward/backward per device and all-reduce gradients in bfloat16.

    Advantages are normalized globally across the full minibatch before entering
    shard_map, preserving global PPO normalization while halving NCCL traffic.
    """
    count = config['DATA_PARALLEL_DEVICES']
    mesh = _ACTIVE_MESH if _ACTIVE_MESH is not None else Mesh(np.asarray(jax.local_devices()[:count]), (AXIS,))
    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _spec(tree):
        return jax.tree_util.tree_map(lambda x: PartitionSpec(None, AXIS, *([None] * (x.ndim - 2))), tree)

    def _per_device(p, ih, tb, adv, tgt):
        loss, grads = jax.value_and_grad(local_loss_fn, has_aux=True)(
            p, jax.tree_util.tree_map(lambda a: a[0], ih), tb, adv, tgt
        )
        grads = jax.tree_util.tree_map(
            lambda g: jax.lax.pmean(g.astype(jnp.bfloat16), AXIS).astype(jnp.float32),
            grads,
        )
        loss = jax.tree_util.tree_map(lambda x: jax.lax.pmean(x, AXIS), loss)
        return loss, grads

    return shard_map(
        _per_device,
        mesh=mesh,
        in_specs=(PartitionSpec(), _spec(init_hstate), _spec(traj_batch), _spec(norm_advantages), _spec(targets)),
        out_specs=(PartitionSpec(), PartitionSpec()),
        check_rep=False,
    )(params, init_hstate, traj_batch, norm_advantages, targets)


def compile_train(config, devices=None):
    """Compile a conventional unbatched train function for 1/2/4 devices.

    The callable retains global result shapes and ordinary, replicated model
    parameters, so existing checkpoint and evaluation consumers can reuse it.
    """
    if config.get('RECURRENT', True):
        from ppo_rnn_og import make_train
    else:
        from ppo import make_train

    devices = tuple(jax.local_devices() if devices is None else devices)
    count = validate_data_parallel(config, len(devices))
    devices = devices[:count]
    if count == 1:
        return jax.jit(make_train(config), device=devices[0])

    global _ACTIVE_MESH
    mesh = Mesh(np.asarray(devices), (AXIS,))
    execution = dict(config, _DATA_PARALLEL_ACTIVE=True)
    prev_mesh = _ACTIVE_MESH
    _ACTIVE_MESH = mesh
    try:
        with mesh:
            function = make_train(execution)
            compiled = jax.jit(function, in_shardings=NamedSharding(mesh, PartitionSpec()))
    finally:
        _ACTIVE_MESH = prev_mesh
    for name in ('START_TIMESTEPS', 'TOTAL_UPDATES', 'LR_HORIZON_UPDATES',
                 'NUM_UPDATES', 'MINIBATCH_SIZE'):
        config[name] = execution[name]

    def run(rng):
        with mesh:
            return compiled(jax.device_put(rng, NamedSharding(mesh, PartitionSpec())))

    run.mesh = mesh
    run.devices = devices
    return run


def execute_train(config, rng, devices=None):
    result = compile_train(config, devices)(rng)
    jax.block_until_ready(result)
    if config.get('DATA_PARALLEL_DEVICES', 1) > 1:
        # Multi-device ordered debug effects are unsupported by JAX 0.4.23.
        # Wait for the single global checkpoint/log callbacks before the caller
        # writes a final checkpoint or closes the W&B run.
        jax.effects_barrier()
    return result


def checkpoint_train_state(state):
    """Ordinary host arrays keep checkpoint payloads independent of the Mesh."""
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), state)


def save_checkpoint_from_callback(directory, step, state, max_to_keep, timeout=120):
    """Save callback host arrays without dispatching work on the training GPUs.

    Orbax barriers issue JAX computations even on one host. Running Orbax inside
    a GPU callback can therefore deadlock the computation waiting on that
    callback. A fresh CPU-only process owns the complete atomic save instead.
    Only Orbax's serializable numpy tree crosses the process boundary; static
    TrainState apply functions and optimizer transformations never do.
    """
    import os
    import pickle
    from pathlib import Path
    import subprocess
    import sys
    from orbax.checkpoint import utils as checkpoint_utils

    payload = checkpoint_utils.serialize_tree(
        jax.tree_util.tree_map(np.asarray, state), keep_empty_nodes=True)
    environment = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
        XLA_FLAGS='--xla_force_host_platform_device_count=1',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__).resolve()), '--checkpoint-worker',
        str(Path(directory).resolve()), str(int(step)), str(int(max_to_keep))]
    result = subprocess.run(command, input=pickle.dumps(payload, protocol=5),
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout)
    if result.returncode:
        raise RuntimeError('CPU checkpoint writer failed: '+
            result.stderr.decode(errors='replace')[-8000:])


def _checkpoint_worker(directory, step, max_to_keep):
    import pickle
    import sys
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer

    if jax.default_backend() != 'cpu' or jax.process_count() != 1:
        raise RuntimeError('Checkpoint worker requires an isolated single-host CPU backend')
    payload = pickle.load(sys.stdin.buffer)
    manager = CheckpointManager(directory, PyTreeCheckpointer(),
        CheckpointManagerOptions(max_to_keep=int(max_to_keep), create=True))
    try:
        manager.save(int(step), payload)
        manager.wait_until_finished()
    finally:
        manager.close()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5 or sys.argv[1] != '--checkpoint-worker':
        raise SystemExit('This module is launched internally as a CPU checkpoint writer')
    _checkpoint_worker(*sys.argv[2:])

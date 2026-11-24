"""Warm-start the single-map actor and critic on fresh trajectories of the two-map teacher."""
import argparse
import json
from functools import partial
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_reward_shaped_ppo import jax, jnp, ocp, make_craftax_env_from_name
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
from models.actor_critic_map_mlp import map_mlp_network, transfer_map_mlp_params
from symbolic_map_memory import BOARD_OBS_SIZE, GLOBAL_MAP_FEATURES, LOCAL_SIZE, with_map_memory
from wrappers import OptimisticResetVecEnvWrapper
from cap_reward_shaped_runs import try_evaluation_lock


def student_observations(obs, memory, board_env):
    if board_env is not None:
        # Reuse the exact online V6 writer on the teacher's observed memory.
        return jax.vmap(board_env.observe)(obs[:, :LOCAL_SIZE], memory)[0]
    return jnp.concatenate((obs[:, :LOCAL_SIZE], obs[:, -GLOBAL_MAP_FEATURES:]), axis=-1)


def extend_board_state(old, new):
    """Keep the source function and its Adam history when adding units or attention."""
    leaves = lambda state: flatten_dict(serialization.to_state_dict(state), keep_empty_nodes=True)
    src, dst = leaves(old), leaves(new)
    if src.keys() - dst.keys() or any('board_attention' not in k for k in dst.keys() - src.keys()):
        raise ValueError('Unsupported difference between board checkpoint structures')
    for path, original in src.items():
        target = dst[path]
        if np.shape(original) == np.shape(target):
            dst[path] = original
            continue
        if 'actor_critic' not in path or any(a > b for a, b in zip(original.shape, target.shape)):
            raise ValueError(f'Unsupported parameter resize at {path}')
        if original.ndim == 2:
            rows, cols = original.shape
            # New units receive independent inputs. They initially contribute zero to old units.
            target = target.at[rows:, :cols].set(0)
        dst[path] = target.at[tuple(slice(0, n) for n in original.shape)].set(original)
    return serialization.from_state_dict(new, unflatten_dict(dst))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True, help='Source study architecture directory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True, help='Must match CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=208052)
    parser.add_argument('--blocks', type=int, default=256)
    reader = parser.add_mutually_exclusive_group()
    reader.add_argument('--patch-mlp', action='store_true', help='Use position-specific dense patch projections')
    reader.add_argument('--spatial-regions', action='store_true', help='Keep 8×8 regions and the full live observation')
    reader.add_argument('--board-only', action='store_true', help='V6 fixed world board')
    parser.add_argument('--board-attention', action='store_true', help='Add a player-conditioned read of the fixed board')
    parser.add_argument('--student-width', type=int, help='Hidden width for both board decision heads')
    parser.add_argument('--resume-imitation', type=Path,
                        help='V6 imitation snapshot with its Adam state and cumulative decision count')
    parser.add_argument('--rollout-policy', choices=('teacher', 'student'), default='teacher',
                        help='Policy that acts while the frozen teacher supplies the learning targets')
    parser.add_argument('--save-blocks', nargs='+', type=int, default=[],
                        help='Save V6 imitation snapshots at these completed rollout blocks')
    args = parser.parse_args()
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        raise ValueError('CUDA_VISIBLE_DEVICES must match the reserved GPU')
    lock = try_evaluation_lock(args.physical_gpu)
    if lock is None:
        raise RuntimeError('Distillation GPU already reserved')
    teacher_cfg = json.loads((args.teacher/'inference_config.json').read_text())
    if not teacher_cfg.get('MAP_GLOBAL_READOUT') or teacher_cfg.get('MAP_FULL_TERRAIN') or teacher_cfg.get('MAP_SINGLE_READOUT'):
        raise ValueError('Teacher must read a crop plus observed world memory')
    checkpoint = Path(json.loads((args.teacher/'progress.json').read_text())['final_checkpoint'])
    destination = (args.out/'files/policies'/('0' if args.board_only else checkpoint.name)).resolve()
    snapshots = args.out.resolve()/'imitation_checkpoints'
    if args.save_blocks and (not args.board_only or any(b < 1 or b > args.blocks for b in args.save_blocks)):
        raise ValueError('Snapshot blocks must be within a V6 distillation run')
    if args.resume_imitation and not args.board_only:
        raise ValueError('Imitation continuation requires a V6 board-only checkpoint')
    if (args.board_attention or args.student_width is not None) and not args.board_only:
        raise ValueError('Board attention and width options require a fixed board')
    if args.rollout_policy == 'student' and not args.board_only:
        raise ValueError('Student-driven collection is configured for V6')
    if destination.exists() or (args.board_only and snapshots.exists()) or args.blocks < 1:
        raise ValueError('Use a fresh checkpoint destination and a positive number of blocks')
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = ocp.PyTreeCheckpointer().restore(str(checkpoint/'default' if (checkpoint/'default').is_dir() else checkpoint))['params']
    teacher = map_mlp_network(teacher_cfg)
    prior_cfg = {}
    if args.resume_imitation:
        resume = args.resume_imitation.resolve()
        if resume.parent.name != 'imitation_checkpoints' or not resume.name.isdigit():
            raise ValueError('Use a numbered board imitation checkpoint')
        prior_cfg = json.loads((resume.parent.parent/'inference_config.json').read_text())
        if not prior_cfg.get('MAP_BOARD_ONLY'):
            raise ValueError('Source checkpoint must be a fixed board-only policy')
    student_cfg = dict(teacher_cfg, MAP_SINGLE_READOUT=True, MAP_PATCH_MLP=args.patch_mlp,
                       MAP_SPATIAL_REGIONS=args.spatial_regions, MAP_BOARD_ONLY=args.board_only)
    if args.board_only:
        attention = args.board_attention or prior_cfg.get('MAP_BOARD_ATTENTION', False)
        source_width = prior_cfg.get('LAYER_SIZE', teacher_cfg['LAYER_SIZE'])
        width = source_width if args.student_width is None else args.student_width
        if width < 1 or (args.resume_imitation and width < source_width):
            raise ValueError('Student width must be positive and cannot shrink a resumed network')
        student_cfg.update(POLICY_VERSION='V7' if attention else 'V6', LAYER_SIZE=width,
                           MAP_BOARD_ATTENTION=attention, SEED=args.seed, TOTAL_TIMESTEPS=0)
        (args.out/'inference_config.json').write_text(json.dumps(student_cfg, indent=2)+'\n')
    student = map_mlp_network(student_cfg)
    init_key, env_key, key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    obs_size = BOARD_OBS_SIZE if args.board_only else LOCAL_SIZE+GLOBAL_MAP_FEATURES
    template = student.init(init_key, jnp.zeros((1, obs_size)))
    state = TrainState.create(apply_fn=student.apply, params=template if args.board_only else transfer_map_mlp_params(template, source),
                              tx=optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-4)))
    lanes, steps, minibatch = 128, 32, 512
    block_size = lanes * steps
    start_decisions = 0
    start_updates = 0
    if args.resume_imitation:
        start_decisions = int(resume.name)
        original_net = map_mlp_network(prior_cfg)
        changed = width != source_width or attention != prior_cfg.get('MAP_BOARD_ATTENTION', False)
        if changed:
            original_template = TrainState.create(apply_fn=original_net.apply,
                params=original_net.init(init_key, jnp.zeros((1, obs_size))), tx=state.tx)
        else:
            original_template = state
        original = ocp.PyTreeCheckpointer().restore(str(resume), item=original_template)
        state = extend_board_state(original, state) if changed else original
        start_updates = int(state.step)
        if start_updates * minibatch != 2 * start_decisions:
            raise ValueError('Checkpoint Adam step disagrees with its imitation decision count')
        print(f'Resumed {start_decisions:,} decisions and {start_updates:,} Adam updates from {resume}', flush=True)
    raw = make_craftax_env_from_name(teacher_cfg['ENV_NAME'], False)
    board_env = with_map_memory(raw, student_cfg) if args.board_only else None
    env = OptimisticResetVecEnvWrapper(with_map_memory(raw, teacher_cfg), lanes, 16)
    obs, env_state = env.reset(env_key, env.default_params)
    if args.resume_imitation:
        boards = student_observations(obs, env_state, board_env)
        old_pi, old_v = original_net.apply(original.params, boards)
        new_pi, new_v = student.apply(state.params, boards)
        np.testing.assert_allclose(old_pi.probs, new_pi.probs, rtol=1e-5, atol=2e-6)
        np.testing.assert_allclose(old_v, new_v, rtol=1e-5, atol=2e-5)
        check = dict(worlds=len(boards), max_probability_difference=float(jnp.max(abs(old_pi.probs-new_pi.probs))),
                     max_value_difference=float(jnp.max(abs(old_v-new_v))), adam_updates=start_updates,
                     parameters=sum(x.size for x in jax.tree_util.tree_leaves(state.params)))
        (args.out/'warm_start_check.json').write_text(json.dumps(check, indent=2)+'\n')
        print(f'Initial policy and value check passed: {check}', flush=True)

    def rollout(student_params, carry, _):
        obs, env_state, key = carry
        key, act_key, step_key = jax.random.split(key, 3)
        pi, value = teacher.apply(source, obs)
        compact = student_observations(obs, env_state, board_env)
        behavior = student.apply(student_params, compact)[0] if args.rollout_policy == 'student' else pi
        next_obs, next_state, *_ = env.step(step_key, env_state, behavior.sample(seed=act_key), env.default_params)
        return (next_obs, next_state, key), (compact, pi.probs, value)

    def loss(params, obs, targets, values):
        pi, value = student.apply(params, obs)
        kl = optax.kl_divergence(jax.nn.log_softmax(pi.logits), targets).mean()
        value_error = optax.huber_loss(value, values).mean()
        return kl + 0.25*value_error

    @jax.jit
    def train_block(state, carry):
        carry, batch = jax.lax.scan(partial(rollout, state.params), carry, None, length=steps)
        obs, env_state, key = carry
        batch = jax.tree_util.tree_map(lambda x: x.reshape((lanes*steps, *x.shape[2:])), batch)
        def update(state, indices):
            data = jax.tree_util.tree_map(lambda x: x[indices], batch)
            objective, grad = jax.value_and_grad(loss)(state.params, *data)
            return state.apply_gradients(grads=grad), objective
        losses = []
        for _ in range(2):
            key, shuffle_key = jax.random.split(key)
            indices = jax.random.permutation(shuffle_key, lanes*steps).reshape((-1, minibatch))
            state, values = jax.lax.scan(update, state, indices)
            losses.append(values.mean())
        return state, (obs, env_state, key), jnp.mean(jnp.stack(losses))

    carry = (obs, env_state, key)
    if args.board_only:
        snapshots.mkdir()
        if not args.resume_imitation:
            ocp.PyTreeCheckpointer().save(str(snapshots/'0'), {'params': state.params})
    started = time.monotonic()
    save_blocks = set(args.save_blocks) | {args.blocks}
    for block in range(args.blocks):
        # Restart periodically so early survival and crafting remain represented.
        if block and block % 64 == 0:
            key, reset_key = jax.random.split(carry[-1])
            carry = (*env.reset(reset_key, env.default_params), key)
        state, carry, loss_value = train_block(state, carry)
        if block == 0 or (block+1) % 16 == 0 or block+1 in save_blocks:
            value = float(loss_value)
            if not np.isfinite(value):
                raise RuntimeError('Nonfinite distillation loss')
            print(f'{block+1}/{args.blocks} blocks, loss {value:.5f}', flush=True)
            if args.board_only:
                added = (block+1)*block_size
                decisions = start_decisions + added
                if block+1 in save_blocks:
                    ocp.PyTreeCheckpointer().save(str(snapshots/str(decisions)), state)
                progress = dict(block=block+1, blocks=args.blocks, teacher_decisions=decisions,
                                additional_decisions=added, start_decisions=start_decisions,
                                rollout_policy=args.rollout_policy,
                                target_decisions=start_decisions+args.blocks*block_size, optimizer_steps=int(state.step),
                                last_block_loss=value, elapsed_seconds=time.monotonic()-started)
                temp = args.out/'progress.tmp'
                temp.write_text(json.dumps(progress, indent=2)+'\n')
                temp.replace(args.out/'progress.json')
    checkpoint_state = TrainState.create(apply_fn=student.apply, params=state.params, tx=state.tx) if args.spatial_regions or args.board_only else state
    ocp.PyTreeCheckpointer().save(str(destination), checkpoint_state)
    report = dict(teacher=str(checkpoint), student_checkpoint=str(destination), seed=args.seed,
                  imitation_interactions=start_decisions+block_size*args.blocks, optimizer_steps=int(state.step),
                  ppo_steps_during_distillation=0, patch_mlp=args.patch_mlp, spatial_regions=args.spatial_regions,
                  saved_optimizer_steps=int(checkpoint_state.step), final_block_loss=float(loss_value))
    if args.board_only:
        report.update(policy_version=student_cfg['POLICY_VERSION'], map_board_attention=attention, head_width=width,
                      student_initialization='resumed' if args.resume_imitation else 'random', map_board_only=True,
                      source_imitation_checkpoint=str(args.resume_imitation.resolve()) if args.resume_imitation else None,
                      start_decisions=start_decisions, additional_decisions=block_size*args.blocks,
                      additional_optimizer_updates=int(state.step)-start_updates,
                      trajectory_policy=args.rollout_policy, teacher_controls_rollouts=args.rollout_policy=='teacher',
                      imitation_adam_lr=1e-4, minibatch=minibatch,
                      passes_per_rollout=2, envs=lanes, rollout_steps=steps, reset_interval_blocks=64,
                      snapshot_teacher_decisions=([] if args.resume_imitation else [0])+
                                                 [start_decisions+b*block_size for b in sorted(save_blocks)])
    (args.out/'distillation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

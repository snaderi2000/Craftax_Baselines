"""Train only the board-attention adapter against V1's actions, values and first hidden activations."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cap_reward_shaped_runs import try_evaluation_lock
from run_ppo_study import require_free
from run_reward_shaped_architectures import write_json


def v1_targets(teacher, variables, obs):
    import jax.numpy as jnp
    (policy, value), traced = teacher.apply(variables, obs, mutable=['intermediates'],
        capture_intermediates=lambda module, method: module.name in ('Dense_0', 'Dense_4'))
    first = traced['intermediates']['actor_critic']
    hidden = jnp.tanh(jnp.stack([first[k]['__call__'][0] for k in ('Dense_0', 'Dense_4')], axis=-2))
    return policy, value, hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=208055)
    parser.add_argument('--blocks', type=int, default=1221)
    args = parser.parse_args()
    if args.blocks < 1 or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('Set a positive budget and select the indicated physical GPU')
    lock = try_evaluation_lock(args.gpu)
    if lock is None:
        raise RuntimeError('Training GPU is reserved')
    require_free(args.gpu)
    # Craftax can initialize the GPU at import time. Reserve it before importing.
    from experiments.distill_single_map import student_observations
    from evaluate_reward_shaped_ppo import jax, jnp, make_craftax_env_from_name, ocp
    from flax.training.train_state import TrainState
    import optax
    from models.actor_critic import ActorCritic
    from models.actor_critic_map_mlp import map_mlp_network
    from symbolic_map_memory import with_map_memory
    from wrappers import OptimisticResetVecEnvWrapper
    out = args.out.resolve()
    checkpoints = out/'imitation_checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((args.teacher/'inference_config.json').read_text())
    if cfg['SYMBOLIC_ENCODER'] != 'map_mlp' or cfg.get('MAP_GLOBAL_READOUT') or cfg['RECURRENT']:
        raise ValueError('Expected the original feedforward V1 crop policy')
    checkpoint = Path(json.loads((args.teacher/'progress.json').read_text())['final_checkpoint'])
    loader = ocp.PyTreeCheckpointer()
    source = loader.restore(str(checkpoint/'default' if (checkpoint/'default').is_dir() else checkpoint))['params']
    student_cfg = dict(cfg, MAP_SINGLE_READOUT=True, MAP_GLOBAL_READOUT=True, MAP_BOARD_ONLY=True,
                       MAP_V1_ATTENTION_ADAPTER=True, POLICY_VERSION='V1 frozen heads + board attention',
                       SEED=args.seed, TOTAL_TIMESTEPS=0)
    write_json(out/'inference_config.json', student_cfg)
    teacher, student = map_mlp_network(cfg), map_mlp_network(student_cfg)
    raw = make_craftax_env_from_name(cfg['ENV_NAME'], False)
    board_env = with_map_memory(raw, student_cfg)
    lanes, steps, minibatch = 128, 32, 256
    env = OptimisticResetVecEnvWrapper(with_map_memory(raw, cfg), lanes, 16)
    init_key, reset_key, key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    obs, env_state = env.reset(reset_key, env.default_params)
    boards = student_observations(obs, env_state, board_env)
    template = student.init(init_key, boards[:1])['params']
    original = source['params']
    kept = {name: original['actor_critic'][name] for name in template['actor_critic']}
    if set(kept) != {'Dense_1','Dense_2','Dense_3','Dense_5','Dense_6','Dense_7'}:
        raise RuntimeError('Unexpected V1 decision layers')
    frozen = dict(terrain_embedding=original['terrain_embedding'], actor_critic=kept)
    def variables(adapter):
        return {'params': {**frozen, 'board_adapter': adapter}}

    # Only this subtree is passed to Adam or differentiated.
    state = TrainState.create(apply_fn=student.apply, params=template['board_adapter'],
                              tx=optax.chain(optax.clip_by_global_norm(1.), optax.adam(2e-4)))
    pi, v, hidden = v1_targets(teacher, source, obs)
    direct = ActorCritic(17, cfg['LAYER_SIZE'], 'tanh')
    p2, v2 = direct.apply({'params': kept}, hidden, from_first_hidden=True)
    np.testing.assert_allclose(pi.probs, p2.probs, rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(v, v2, rtol=1e-5, atol=2e-5)
    size = lambda tree: sum(x.size for x in jax.tree_util.tree_leaves(tree))
    write_json(out/'transfer_check.json', dict(source=str(checkpoint), worlds=lanes, frozen=size(frozen),
        trainable=size(state.params), trainable_subtree='board_adapter', retained_layers=sorted(kept),
        max_action_probability_error=float(jnp.max(abs(pi.probs-p2.probs))),
        max_critic_error=float(jnp.max(abs(v-v2)))))
    def save(state, decisions):
        loader.save(str(checkpoints/str(decisions)), dict(params=variables(state.params),
                                                         step=state.step, adapter_opt_state=state.opt_state))
    save(state, 0)
    teacher_lane = jnp.arange(lanes) < lanes//2
    @jax.jit
    def block(state, carry):
        def rollout(carry, _):
            obs, memory, rng = carry
            rng, act, step_key = jax.random.split(rng, 3)
            expert, value, targets = v1_targets(teacher, source, obs)
            board = student_observations(obs, memory, board_env)
            learner, _ = student.apply(variables(state.params), board)
            logits = jnp.where(teacher_lane[:,None], expert.logits, learner.logits)
            next_obs, next_mem, *_ = env.step(step_key, memory, jax.random.categorical(act, logits), env.default_params)
            return (next_obs, next_mem, rng), (board, expert.probs, value, targets)
        carry, batch = jax.lax.scan(rollout, carry, None, length=steps)
        batch = jax.tree_util.tree_map(lambda x: x.reshape((lanes*steps, *x.shape[2:])), batch)
        def loss(adapter, board, probs, value, targets):
            pred, v, h = student.apply(variables(adapter), board, method=student.with_hidden)
            kl = optax.kl_divergence(jax.nn.log_softmax(pred.logits), probs).mean()
            value_loss = optax.huber_loss(v, value).mean()
            hidden_mse = jnp.mean((h-targets)**2)
            return kl + .25*value_loss + hidden_mse, jnp.stack((kl, value_loss, hidden_mse))
        def update(state, indices):
            samples = jax.tree_util.tree_map(lambda x: x[indices], batch)
            (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(state.params, *samples)
            return state.apply_gradients(grads=grads), metrics
        obs, mem, rng = carry
        rng, k1, k2 = jax.random.split(rng, 3)
        indices = jnp.stack([jax.random.permutation(k, lanes*steps) for k in (k1, k2)]).reshape(-1, minibatch)
        state, metrics = jax.lax.scan(update, state, indices)
        return state, (obs, mem, rng), metrics.mean(0)

    start = time.monotonic()
    carry = obs, env_state, key
    for completed in range(1, args.blocks+1):
        state, carry, metrics = block(state, carry)
        if completed in (1, 256, args.blocks) or completed%16 == 0:
            metrics = np.asarray(metrics)
            if not np.isfinite(metrics).all():
                raise ValueError('Non-finite training metrics')
            decisions = completed*lanes*steps
            if completed in (256, args.blocks):
                save(state, decisions)
            progress = dict(decisions=decisions, target=args.blocks*lanes*steps, updates=int(state.step),
                            rollout_teacher_fraction=.5, kl=float(metrics[0]), huber=float(metrics[1]),
                            hidden_mse=float(metrics[2]), seconds=time.monotonic()-start)
            write_json(out/'progress.json', progress)
            print(progress, flush=True)
    write_json(out/'distillation.json', dict(**progress, source=str(checkpoint), seed=args.seed,
               ppo_steps=0, trained_subtree='board_adapter', lr=.0002, hidden_loss_weight=1.,
               value_loss_weight=.25, minibatch=minibatch, passes=2))


if __name__ == '__main__':
    main()

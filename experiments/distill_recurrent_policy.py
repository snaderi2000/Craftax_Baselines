"""Distill a frozen local-symbolic SRU into a wider recurrent policy before PPO."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cap_reward_shaped_runs import try_evaluation_lock
from run_ppo_study import require_free
from run_reward_shaped_architectures import ROOT, write_json


def train(args, out):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import orbax.checkpoint as ocp
    import wandb
    from flax.training.train_state import TrainState
    from models.sru_gru import make_optimizer
    from ppo_rnn_og import ActorCriticRNN, initialize_recurrent_carry, make_craftax_env_from_name
    from symbolic_motion_inputs import with_motion_inputs
    from wrappers import AutoResetEnvWrapper, BatchEnvWrapper, ClassicLavaPenaltyWrapper

    teacher_cfg = json.loads(args.teacher_config.read_text())
    student_cfg = json.loads(args.student_config.read_text())
    result = json.loads(args.teacher_evaluation.read_text())
    checkpoint = Path(result['checkpoint']).resolve()
    for cfg in (teacher_cfg, student_cfg):
        if not (cfg['ENV_NAME'] == 'Craftax-Classic-Symbolic-v1' and cfg['SRU_GRU']
                and cfg['MOTION_INPUTS'] and cfg['GRU_CONCAT_OBSERVATION'] and not cfg['WORLD_VIEW']
                and not cfg['USE_OPTIMISTIC_RESETS'] and not cfg['RESET_GRU_EVERY_STEP']
                and cfg['LAVA_PENALTY'] == 2 and cfg['DEATH_PENALTY'] == 0):
            raise ValueError('Expected a standard-reset local symbolic SRU with lava cost 2')
    lanes, steps, minibatches = 512, 64, 8
    blocks = args.blocks_per_phase
    block_size = lanes * steps
    raw = make_craftax_env_from_name(student_cfg['ENV_NAME'], False)
    env = BatchEnvWrapper(AutoResetEnvWrapper(with_motion_inputs(ClassicLavaPenaltyWrapper(raw, 2.), student_cfg)), lanes)
    n_actions = env.action_space(env.default_params).n
    teacher = ActorCriticRNN(n_actions, config=teacher_cfg)
    student = ActorCriticRNN(n_actions, config=student_cfg)
    cp = ocp.PyTreeCheckpointer()
    payload = checkpoint/'default' if (checkpoint/'default').is_dir() else checkpoint
    teacher_params = cp.restore(str(payload))['params']
    key, reset_key = jax.random.split(jax.random.PRNGKey(args.seed))
    obs, worlds = env.reset(reset_key, env.default_params)
    done = jnp.zeros(lanes, bool)
    h_teacher = initialize_recurrent_carry(lanes, teacher_cfg)
    h_student = initialize_recurrent_carry(lanes, student_cfg)
    init_key = jax.random.split(jax.random.split(jax.random.PRNGKey(student_cfg['SEED']), 1)[0])[1]
    params = student.init(init_key, h_student[:1], (obs[None, :1], done[None, :1]))
    previous_teacher = previous_student = 0
    source_checkpoint = None
    if args.continue_from:
        prior = json.loads((args.continue_from/'progress.json').read_text())
        prior_cfg = json.loads((args.continue_from/'distillation_config.json').read_text())
        if prior['state'] != 'complete' or Path(prior_cfg['teacher_checkpoint']).resolve() != checkpoint:
            raise ValueError('Expected a completed imitation run from the same teacher')
        if json.loads((args.continue_from/'inference_config.json').read_text()) != student_cfg:
            raise ValueError('Source student configuration differs')
        source_checkpoint = Path(prior['final_checkpoint']).resolve()
        previous_teacher, previous_student = prior['teacher_led_steps'], prior['student_led_steps']
        if previous_teacher + previous_student != int(source_checkpoint.name) or prior['env_step'] != int(source_checkpoint.name):
            raise ValueError('Inconsistent source imitation counts')
        params = cp.restore(str(source_checkpoint))['params']
    state = TrainState.create(apply_fn=student.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(1.), optax.adam(1e-4)))
    restored_imitation_adam = False
    if source_checkpoint:
        moments = args.continue_from.resolve()/'imitation_optimizer'/source_checkpoint.name
        if moments.exists():
            saved = cp.restore(str(moments), item=dict(step=state.step, opt_state=state.opt_state))
            state = state.replace(**saved)
            restored_imitation_adam = True
    carry = (obs, worlds, done, h_teacher, h_student, key)

    def objective(params, initial_hidden, batch):
        observations, resets, targets, target_values = batch
        _, policy, values = student.apply(params, initial_hidden, (observations, resets))
        kl = optax.kl_divergence(jax.nn.log_softmax(policy.logits), targets).mean()
        value_loss = optax.huber_loss(values, target_values).mean()
        agreement = (policy.logits.argmax(-1) == targets.argmax(-1)).mean()
        return kl + .25 * value_loss, jnp.stack([kl, value_loss, agreement])

    @jax.jit
    def train_block(state, carry, teacher_params, teacher_acts):
        initial_hidden = carry[4]

        def collect(carry, _):
            obs, worlds, done, h_teacher, h_student, key = carry
            inputs = (obs[None], done[None])
            h_teacher, expert, value = teacher.apply(teacher_params, h_teacher, inputs)
            h_student, learner, _ = student.apply(state.params, h_student, inputs)
            key, act_key, env_key = jax.random.split(key, 3)
            logits = jnp.where(teacher_acts, expert.logits[0], learner.logits[0])
            actions = jax.random.categorical(act_key, logits)
            new_obs, worlds, _, new_done, _ = env.step(env_key, worlds, actions, env.default_params)
            targets = (obs, done, expert.probs[0], value[0])
            return (new_obs, worlds, new_done, h_teacher, h_student, key), targets

        carry, batch = jax.lax.scan(collect, carry, None, steps)
        key, shuffle1, shuffle2 = jax.random.split(carry[-1], 3)
        def update(state, ids):
            sequences = jax.tree.map(lambda x: x[:, ids], batch)
            (_, metrics), grad = jax.value_and_grad(objective, has_aux=True)(
                state.params, initial_hidden[ids], sequences)
            return state.apply_gradients(grads=grad), metrics

        ids = jnp.stack([jax.random.permutation(k, lanes) for k in (shuffle1, shuffle2)])
        state, metrics = jax.lax.scan(update, state, ids.reshape((-1, lanes // minibatches)))
        return state, (*carry[:-1], key), metrics.mean(axis=0)

    settings = dict(teacher_checkpoint=str(checkpoint), teacher_all22_pct=result['all_22_success_pct'],
                    teacher_evaluation=str(args.teacher_evaluation.resolve()), envs=lanes,
                    steps=steps, blocks_per_phase=blocks, env_steps_per_phase=blocks*block_size,
                    student_config=str(args.student_config.resolve()), imitation_seed=args.seed,
                    source_student_checkpoint=str(source_checkpoint) if source_checkpoint else None,
                    initial_teacher_steps=previous_teacher, initial_student_steps=previous_student,
                    restored_imitation_adam=restored_imitation_adam,
                    passes=2, minibatches=minibatches, imitation_lr=1e-4, value_loss_weight=.25)
    write_json(out/'distillation_config.json', settings)
    write_json(out/'inference_config.json', student_cfg)
    with wandb.init(project='Craftax_Baselines', entity='synaderi-uc-davis', mode=args.wandb_mode,
                    name=out.name, config=settings) as run:
        print('W&B',run.url,flush=True)
        start = time.time()
        for block in range(1, 2*blocks+1):
            teacher_acts = block <= blocks
            state, carry, metrics = train_block(state, carry, teacher_params, jnp.asarray(teacher_acts))
            if block == 1 or block % 8 == 0 or block in (blocks,blocks+1,2*blocks):
                kl, value_error, agreement = np.asarray(metrics)
                if not np.isfinite([kl, value_error, agreement]).all():
                    raise FloatingPointError('Nonfinite imitation metrics')
                row = dict(env_step=previous_teacher+previous_student+block*block_size,
                           teacher_led_steps=previous_teacher+min(block,blocks)*block_size,
                           student_led_steps=previous_student+max(0,block-blocks)*block_size, policy_kl=float(kl),
                           value_huber=float(value_error), top_action_agreement=float(agreement),
                           phase='teacher' if teacher_acts else 'student', elapsed_seconds=time.time()-start)
                run.log(row)
                write_json(out/'progress.json',dict(state='distilling', wandb_url=run.url, **row))
                print(json.dumps(row),flush=True)
            if block in (blocks, 2*blocks):
                # Both PPO Adam partitions use callable LR schedules in the production trainer.
                # Fresh state needs the matching Optax structure. Restored PPO supplies the
                # actual schedule functions and starts their update counts at zero.
                tx = make_optimizer(student_cfg, lambda _: student_cfg['LR'],
                                    gate_lr=lambda _: student_cfg['SRU_GATE_LR'])
                fresh = TrainState.create(apply_fn=student.apply, params=state.params, tx=tx)
                destination = out/'files/policies'/str(row['env_step'])
                destination.parent.mkdir(exist_ok=True,parents=True)
                cp.save(str(destination), jax.device_get(fresh))
                loaded = cp.restore(str(destination), item=fresh)
                if int(loaded.step) != 0 or not all(np.isfinite(x).all() for x in jax.tree.leaves(jax.device_get(loaded.params))):
                    raise ValueError('Invalid PPO handoff checkpoint')
                moments = out/'imitation_optimizer'/destination.name
                moments.parent.mkdir(exist_ok=True)
                cp.save(str(moments), jax.device_get(dict(step=state.step, opt_state=state.opt_state)))
        if args.evaluate:
            # Evaluator executes in a separate process; release training buffers first
            # via the outer driver before starting it.
            run.summary['evaluation_directory'] = str(out/'evaluation_imitation')
        run.summary['ppo_handoff_checkpoint'] = str(destination)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('teacher-config','teacher-evaluation','student-config','out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--blocks-per-phase',type=int,default=306)
    p.add_argument('--seed',type=int,default=20261124)
    p.add_argument('--continue-from',type=Path,help='Fork from the final checkpoint of a completed imitation run')
    p.add_argument('--wandb-mode',choices=('online','disabled'),default='online')
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    out=args.out.resolve()
    if args.blocks_per_phase<1 or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):
        raise ValueError('Select a positive budget and the indicated physical GPU')
    if args.worker:
        train(args,out)
        return
    lock=try_evaluation_lock(args.gpu)
    if lock is None:
        raise RuntimeError('Imitation GPU is reserved')
    require_free(args.gpu)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'progress.json').exists():
        raise FileExistsError('Use a fresh distillation directory')
    write_json(out/'progress.json',dict(state='initializing',pid=os.getpid()))
    try:
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],check=True)
        progress=json.loads((out/'progress.json').read_text())
        checkpoint=out/'files/policies'/str(progress['env_step'])
        if args.evaluate:
            write_json(out/'progress.json',dict(progress,state='evaluating',latest_checkpoint=int(checkpoint.name)))
            subprocess.run([sys.executable,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),
                            '--checkpoint',str(checkpoint),'--config-json',str(out/'inference_config.json'),
                            '--out',str(out/'evaluation_imitation'),'--episodes','1024','--batch','64',
                            '--seed','24261125','--temperature','1'],check=True)
        write_json(out/'progress.json',dict(progress,state='complete',training_exit_code=0,
                                           latest_checkpoint=int(checkpoint.name),final_checkpoint=str(checkpoint)))
    except BaseException as exc:
        write_json(out/'progress.json',dict(state='failed',error=str(exc)))
        raise


if __name__=='__main__':
    main()

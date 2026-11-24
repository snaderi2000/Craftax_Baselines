"""Score teacher–student agreement on held-out teacher-controlled trajectories."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_reward_shaped_ppo import jax, jnp, make_craftax_env_from_name, ocp, select_policy_actions
from experiments.distill_single_map import student_observations
from fixed_world_seeds import read_world_seeds
from models.actor_critic_map_mlp import map_mlp_network
from symbolic_map_memory import with_map_memory
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--worlds', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads((args.teacher/'inference_config.json').read_text())
    student_settings = json.loads((args.student/'inference_config.json').read_text())
    progress = json.loads((args.teacher/'progress.json').read_text())
    teacher_path = Path(progress['final_checkpoint'])
    raw = make_craftax_env_from_name(settings['ENV_NAME'], False)
    env = with_map_memory(raw, settings)
    board = with_map_memory(raw, student_settings) if student_settings.get('MAP_BOARD_ONLY') else None
    teacher = map_mlp_network(settings)
    student = map_mlp_network(student_settings)
    worlds = jnp.asarray(read_world_seeds(args.worlds), dtype=jnp.uint32)
    batch = len(worlds)

    @jax.jit
    def evaluate(teacher_params, student_params, worlds):
        obs, state = jax.vmap(env.reset, in_axes=(0, None))(jax.vmap(jax.random.PRNGKey)(worlds), env.default_params)
        keys = jax.vmap(lambda i: jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed), i), 2)[1])(jnp.arange(batch))
        def step(carry, _):
            obs, state, keys, active = carry
            streams = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
            expert, _ = teacher.apply(teacher_params, obs)
            pred, _ = student.apply(student_params, student_observations(obs, state, board))
            kl = (expert.probs * (jax.nn.log_softmax(expert.logits) - jax.nn.log_softmax(pred.logits))).sum(-1)
            match = pred.logits.argmax(-1) == expert.logits.argmax(-1)
            stats = jnp.stack((active.sum(), (active & match).sum(), jnp.where(active, kl, 0).sum()))
            actions = select_policy_actions(expert.logits, streams[:, 1], 1.)
            obs, state, _, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(streams[:, 2], state, actions, env.default_params)
            return (obs, state, streams[:, 0], active & ~done), stats
        return jax.lax.scan(step, (obs, state, keys, jnp.ones(batch, bool)), None, length=10000)[1]

    def weights(path):
        payload = path/'default' if (path/'default').is_dir() else path
        return ocp.PyTreeCheckpointer().restore(str(payload.resolve()))['params']

    def summarize(stats):
        decisions, matches, kl_sum = stats.sum(axis=0, dtype=np.float64)
        return dict(decisions=int(decisions), matching_top_actions=int(matches),
                    agreement_pct=100*matches/decisions, mean_teacher_student_kl_nats=kl_sum/decisions)

    stats = np.asarray(evaluate(weights(teacher_path), weights(args.checkpoint), worlds))
    report = dict(teacher=str(teacher_path), student=str(args.checkpoint), world_seeds_file=str(args.worlds),
                  evaluation_seed=args.seed, temperature=1., trajectory_policy='teacher', step_cap=len(stats),
                  all_steps=summarize(stats), first_1000_steps=summarize(stats[:1000]))
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

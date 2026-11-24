"""Fresh PPO architecture comparison with default rewards except lava −2."""
import argparse,hashlib,json,secrets,shutil,subprocess,time
from pathlib import Path
from run_reward_shaped_architectures import (ROOT,PYTHON,ARCHITECTURES,MILESTONES,EVALUATIONS,
    command_for,write_json,inference_config,summarize)

STUDY=ROOT/'concept_mapping/runs/lava_only_architectures_9h_seed50'
SOURCES=('ppo.py','ppo_rnn_og.py','wrappers.py','training_time_limit.py','cap_reward_shaped_runs.py',
    'run_lava_only_architectures.py','run_reward_shaped_architectures.py','evaluate_reward_shaped_ppo.py',
    'models/actor_critic.py','models/conditioned_symbolic.py','smoke_reward_shaped_ppo.py',
    'test_reward_shaped_spatial.py','logz/batch_logging.py')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=STUDY)
    p.add_argument('--launch',action='store_true');args=p.parse_args();study=args.study.resolve()
    if (study/'launch_manifest.json').exists():raise FileExistsError('Study already launched')
    study.mkdir(parents=True,exist_ok=True)
    test_log=study/'smoke/reward_semantics_tests.log'
    if args.launch and (not test_log.exists() or '\nOK\n' not in test_log.read_text()):
        raise RuntimeError('Reward semantics tests must pass before launch')
    smokes={}
    for arch in ARCHITECTURES:
        smoke=study/'smoke'/arch/'summary.json'
        smokes[arch]=json.loads(smoke.read_text()) if smoke.exists() else {}
        if args.launch and not smokes[arch].get('parameters_and_optimizer_finite'):
            raise RuntimeError(f'{arch}: full-size smoke test must pass')
        if args.launch:
            ev=json.loads((study/'smoke'/arch/'evaluation/summary.json').read_text())
            if (ev['episodes']!=8 or ev.get('lava_penalty')!=2. or ev['death_penalty']!=0.
                    or ev['diamond_reward']!=1. or ev['plant_reward']!=1.):
                raise RuntimeError(f'{arch}: checkpoint evaluation/reward configuration smoke failed')
    if args.launch:
        for gpu in range(8):
            free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            if free<14000:raise RuntimeError(f'GPU {gpu} busy')
        if shutil.disk_usage(study).free<20*1024**3:raise RuntimeError('Insufficient checkpoint headroom')
    start=time.time();end=start+9*3600;train_end=end-15*60
    jobs=[]
    for i,arch in enumerate(ARCHITECTURES):
        out=study/arch;out.mkdir(exist_ok=True)
        command=command_for(arch)
        for key,value in [('--death_penalty','0'),('--diamond_reward','1'),('--plant_reward','1'),
                ('--run_name',f'classic-lava-only2-{arch}-9h-seed50')]:command[command.index(key)+1]=value
        command+=['--lava_penalty','2','--training_deadline',str(train_end)]
        metadata=dict(architecture=arch,seed=50,fresh_training=True,train_gpu=i,evaluation_gpu=i+4,
            command=command,requested_training_steps=10000000000,actual_training_steps=None,
            wall_clock_budget_seconds=32400,original_start_time=start,training_deadline=train_end,
            overall_deadline=end,final_evaluation_reserve_seconds=900,
            additional_death_cost=0.,additional_lava_cost=-2.,first_diamond_total_reward=1.,
            first_eat_plant_total_reward=1.,other_achievement_rewards=1.,health_reward_scale=.1,
            environment='Craftax-Classic-Symbolic-v1',max_episode_steps=10000,
            trainable_parameters=smokes[arch].get('parameters'),
            symbolic_encoder='spatial' if arch.startswith('spatial') else 'mlp',
            recurrent=arch in ('gru','spatial_gru'),num_envs=1024,num_steps=64,
            update_epochs=4,num_minibatches=8,gamma=.99,gae_lambda=.8,clip_eps=.2,
            vf_coef=.5,ent_coef=.01,lr=.0002,anneal_lr=True,learning_rate_horizon_steps=10000000000,
            max_grad_norm=1.,training_reset='optimistic ratio 16',requested_checkpoints=list(MILESTONES),
            evaluation=dict(common_checkpoints=[list(x) for x in EVALUATIONS if x[0]!='10b'],
                episodes_per_common_checkpoint=1024,final_episodes=256,seed=18050,
                resets='independent standard',maps='original generator',metrics='All 22 achievement rates, joint all-22 completion, lava, original/lava-adjusted return'),
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES})
        write_json(out/'experiment_metadata.json',metadata)
        write_json(out/'inference_config.json',inference_config(metadata))
        plan=dict(architecture=arch,original_start_time=start,overall_deadline=end,training_deadline=train_end,
            resume_checkpoint=None,previous_local_run=None,wandb_run_id=secrets.token_hex(4),
            command=command,fresh_training=True,optimizer_restored=False)
        write_json(out/'time_budget_plan.json',plan)
        if args.launch:
            with (out/'worker.log').open('w') as log:
                process=subprocess.Popen([PYTHON,'-u',str(ROOT/'cap_reward_shaped_runs.py'),'--worker',arch,'--study',str(study)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            jobs.append(dict(architecture=arch,controller_pid=process.pid,train_gpu=i,
                evaluation_gpu=i+4,wandb_run_id=plan['wandb_run_id']))
    if args.launch:
        for name in SOURCES:
            target=study/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        write_json(study/'launch_manifest.json',dict(time=start,jobs=jobs,seconds=32400,
            training_deadline=train_end,overall_deadline=end,fresh_training=True,replaced_study='reward_shaped_architectures_10b_seed50'))
    summarize(study)
    (study/'README.md').write_text('\n'.join([
        '# Lava-only penalty: four PPO architectures','',
        'Four fresh seed-50 agents: MLP, GRU, spatial CNN, and spatial CNN + GRU. These replace the stopped diamond/plant +5 study; its checkpoints and 100M results remain preserved in the previous folder.','',
        '**Reward change:** subtract 2 only when an episode ends with the player on lava. Ordinary health loss, starvation, combat deaths, time limits, and all 22 achievement rewards retain their default values. Diamond and eat-plant each give the default first-achievement +1. This targets lava entry; it does not establish an agent’s intention or guarantee risk-taking behavior.','',
        'The installed Craftax environment is unchanged. An opt-in wrapper runs before reset and episode logging. Evaluations use the original maps and report both original reward and reward including the lava cost.','',
        '| Model | Training GPU | Evaluation GPU | Parameters |',
        '| --- | ---: | ---: | ---: |',
        *[f'| {arch} | {i} | {i+4} | {smokes[arch].get("parameters",0):,} |' for i,arch in enumerate(ARCHITECTURES)],
        '', 'Each experiment has a nine-hour total wall-clock budget from this fresh launch. Training stops at an update boundary after at most 8h45m, saves the actual step count, then runs a 256-episode final evaluation. Controllers enforce the outer deadline. Common-step frozen evaluations at 100M and 1B use 1,024 episodes where reached. Both the individual achievement rates and all-22 completion rate are retained.','',
        'Shared PPO settings: 1,024 environments, rollout 64, four epochs, eight minibatches, γ=0.99, λ=0.8, clip 0.2, value coefficient 0.5, entropy coefficient 0.01, Adam learning rate 2e−4, gradient clip 1.0. The original 10B-step learning-rate horizon is preserved as an upper bound; stopping is controlled by wall time. Training uses optimistic resets (ratio 16); frozen evaluation uses standard independent resets, seed 18050. Model capacity and parameter sharing differ across architectures, and this initial comparison has one training seed each.','',
        'Logarithmic checkpoints and W&B histories are retained. Look in each model’s `run_manifest.json` for its W&B URL and local checkpoint path, `progress.json` for live status, and `training.log` for training output. Aggregate evaluations appear in `results.md` and `per_checkpoint_results.csv`.','',
        'Checks: actual lava entry receives −2; ordinary starvation death and time-limit endings do not receive extra cost; diamond and plant rewards remain +1; recurrent sequence/reset tests pass; each model completes two full-size training updates with finite parameters and optimizer state. Wall-clock stopping and actual checkpoint labels were tested in the preceding study.','']))
    print(json.dumps(dict(study=str(study),launched=args.launch,jobs=jobs,training_deadline=train_end,overall_deadline=end),indent=2),flush=True)


if __name__=='__main__':main()

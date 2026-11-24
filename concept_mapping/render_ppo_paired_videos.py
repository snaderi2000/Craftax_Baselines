"""Render frozen symbolic PPO/GRU episodes, with clean and annotated copies."""
from contextlib import ExitStack
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

# Select and reserve the physical GPU before any transitive JAX imports.
_gpu_lock=None
if __name__=='__main__':
    boot=argparse.ArgumentParser(add_help=False);boot.add_argument('--gpu',type=int)
    early,_=boot.parse_known_args()
    if early.gpu is not None:
        from run_reward_shaped_architectures import environment
        from run_ppo_study import require_free
        from cap_reward_shaped_runs import try_evaluation_lock
        _gpu_lock=try_evaluation_lock(early.gpu)
        if _gpu_lock is None:raise RuntimeError(f'GPU {early.gpu} is reserved by another job')
        require_free(early.gpu)
        os.environ.update(environment(early.gpu))
from concept_mapping.render_rnn_ppo_episodes import _MP4Stream,_label_frame
from concept_mapping.render_ppo_episodes import _load_wandb_config,_init_network,_files_dir
from concept_mapping.visualize_symbolic_trajectory import render_local_view,_action_name
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import load_all_textures,Achievement,BlockType


def resolve_inputs(args):
    """Resolve a study or legacy run path, including completed resumed saves."""
    from cap_reward_shaped_runs import checkpoints
    if args.study:
        study=args.study.resolve()
        if (study/'inference_config.json').exists():job=study;study=study.parent
        else:
            candidates=list(study.glob('*/inference_config.json'))
            if len(candidates)!=1:
                raise ValueError('Point --study to one architecture folder when the study contains multiple models')
            job=candidates[0].parent
        args.config_json=args.config_json or job/'inference_config.json'
        manifest=job/'run_manifest.json'
        data=json.loads((manifest if manifest.exists() else job/'progress.json').read_text())
        runs=data.get('checkpoint_roots') or [data['local_run']]
    else:
        study=None;runs=[args.run_path or 'wandb/run-20260630_214658-m0mw4end']
    available=checkpoints([str(Path(_files_dir(str(run))).resolve().parent) for run in runs])
    if not available:raise FileNotFoundError('No completed checkpoints found')
    args.checkpoint=args.checkpoint if args.checkpoint is not None else max(available)
    if args.checkpoint not in available:raise FileNotFoundError(f'Checkpoint {args.checkpoint} is absent or incomplete')
    path=available[args.checkpoint];args.run_path=str(path.parents[2])
    if args.out:out=Path(args.out).resolve()
    elif study:
        prefix='world_videos' if args.whole_map else 'videos'
        out=study/f'{prefix}_{args.checkpoint}_seed{args.seed}'
        if out.exists():out=study/f'{prefix}_{args.checkpoint}_seed{args.seed}_{time.time_ns()}'
    else:out=Path('concept_mapping/runs/standard_ppo_10b_paired_videos_seed8050').resolve()
    return path,out


def validate_videos(out,ffmpeg):
    ffprobe=Path(ffmpeg).with_name('ffprobe')
    executable=str(ffprobe) if ffprobe.is_file() else shutil.which('ffprobe')
    if not executable:raise FileNotFoundError('ffprobe is required to validate rendered MP4s')
    summary=json.loads((out/'summary.json').read_text());checks=[]
    for episode in summary['episodes']:
        for version,path in episode['videos'].items():
            stream=json.loads(subprocess.check_output([executable,'-v','error','-select_streams','v:0',
                '-show_entries','stream=codec_name,width,height,nb_frames,duration','-of','json',path],text=True))['streams'][0]
            if stream['codec_name']!='h264' or int(stream['nb_frames'])!=episode['frames'] or abs(float(stream['duration'])-episode['duration_seconds'])>.06:
                raise RuntimeError(f'Video validation failed: {path}')
            checks.append(dict(episode=episode['episode'],version=version,**stream))
    (out/'video_validation.json').write_text(json.dumps(dict(passed=True,checkpoint=summary['checkpoint'],sampling_temperature=summary.get('sampling_temperature',1),videos=checks),indent=2)+'\n')
    print(f'Validated {len(checks)} MP4s. Output: {out}',flush=True)


def render_evaluation(args,path,out,cfg):
    """Replay selected recorded trials with their original batched evaluator."""
    from evaluate_reward_shaped_ppo import build_evaluator
    from fixed_world_seeds import read_world_seeds
    from symbolic_map_memory import with_map_memory
    from symbolic_motion_inputs import with_motion_inputs
    if args.fps<1:raise ValueError('FPS must be positive')
    source=args.evaluation_dir.resolve()
    metadata=json.loads((source/'summary.json').read_text())
    if Path(metadata['checkpoint']).resolve()!=path.resolve():
        raise ValueError('Replay checkpoint does not match the recorded evaluation')
    with (source/'episodes.csv').open() as stream:
        rows={int(r['episode']):r for r in csv.DictReader(stream)}
    episodes=[int(x) for x in args.episode_ids.split(',')]
    if not episodes or len(set(episodes))!=len(episodes) or any(e not in rows for e in episodes):
        raise ValueError('Select distinct episode IDs present in the evaluation CSV')
    if args.evaluation_batch<1:raise ValueError('Evaluation batch must be positive')
    bank=metadata.get('world_seeds_file')
    worlds=jnp.asarray(read_world_seeds(Path(bank)),jnp.uint32) if bank else None
    env=make_craftax_env_from_name(cfg['ENV_NAME'],auto_reset=False)
    env=with_motion_inputs(with_map_memory(env,cfg),cfg)
    recurrent=bool(cfg.get('RECURRENT',cfg.get('MEMORY_TYPE')=='gru'))
    if recurrent:
        from ppo_rnn_og import ActorCriticRNN
        net=ActorCriticRNN(env.action_space(env.default_params).n,cfg)
    else:net=_init_network(cfg,env,env.default_params)
    payload=path/'default' if (path/'default').is_dir() else path
    params=ocp.PyTreeCheckpointer().restore(str(payload))['params']
    settings=dict(seed=metadata['evaluation_seed'],batch=args.evaluation_batch,
        world_seeds=worlds,temperature=metadata['temperature'],max_steps=metadata['max_episode_steps'])
    evaluate=build_evaluator(env,net,params,cfg,**settings,trace=True,
        world_frame_stride=args.frame_stride if args.whole_map else None)
    normal=build_evaluator(env,net,params,cfg,**settings)
    textures=jax.tree_util.tree_map(np.asarray,load_all_textures(7))
    panel_textures=jax.tree_util.tree_map(np.asarray,load_all_textures(28))
    summaries=[];checks=[]
    if args.resume:
        saved=json.loads((out/'summary.json').read_text())
        if (saved['checkpoint']!=args.checkpoint or saved['source_evaluation']!=str(source)
                or saved['episode_ids']!=episodes or saved['fps']!=args.fps
                or saved.get('whole_map',False)!=args.whole_map or saved.get('frame_stride',1)!=args.frame_stride):
            raise ValueError('Resume settings differ from the existing replay')
        summaries=saved['episodes']
        checks=json.loads((out/'replay_validation.json').read_text())['checks']
    for number,episode in enumerate(episodes):
        existing=next((s for s in summaries if s['episode']==episode),None)
        if existing is not None:
            if not all(Path(p).is_file() and Path(p).stat().st_size for p in existing['videos'].values()):
                raise ValueError(f'Episode {episode} has incomplete existing videos')
            print(f'Keeping completed episode {episode}',flush=True)
            continue
        (out/'render_progress.json').write_text(json.dumps(dict(state='replaying',episode=episode,completed=len(summaries)))+'\n')
        first=(episode//args.evaluation_batch)*args.evaluation_batch
        ids=jnp.arange(first,first+args.evaluation_batch)
        print(f'Replaying recorded episode {episode} in original batch {first}:{first+args.evaluation_batch}',flush=True)
        result=jax.device_get(evaluate(ids,jnp.asarray(episode-first,jnp.int32)))
        _,finished,returns,shaped,lengths,ach,lava,death,timeout,trace=result
        matched=[];mismatches=[]
        for i in range(args.evaluation_batch):
            original=rows.get(first+i)
            if original is None:continue
            try:
                assert int(lengths[i])==int(original['length']),f'Episode {first+i}: length mismatch ({int(lengths[i])} versus {original["length"]})'
                np.testing.assert_allclose(returns[i],float(original['original_return']),rtol=0,atol=1e-5)
                np.testing.assert_allclose(shaped[i],float(original['shaped_return']),rtol=0,atol=1e-5)
                for field,values in [('terminated',finished),('lava',lava),('nonlava_death',death),('timeout',timeout)]:
                    assert bool(values[i])==(original[field]=='True'),f'Episode {first+i}: {field} mismatch'
                for a in Achievement:
                    assert bool(ach[i,a.value])==bool(int(original[a.name.lower()])),f'Episode {first+i}: achievement mismatch'
                matched.append(first+i)
            except AssertionError as error:
                if first+i==episode:raise
                mismatches.append(dict(episode=first+i,error=str(error)))
        if number==0 or mismatches:
            baseline=jax.device_get(normal(ids));slot=episode-first
            for actual,expected in zip(result[1:-1],baseline[1:]):
                np.testing.assert_array_equal(actual[slot],expected[slot],err_msg='Trace changed selected episode outcomes')
            if number==0:
                for actual,expected in zip(result[:-1],baseline):
                    np.testing.assert_array_equal(actual,expected,err_msg='Trace changed first-batch evaluation outcomes')
        if mismatches:print(f'Non-selected batch differences (selected episode matches): {mismatches}',flush=True)
        slot=episode-first;length=int(lengths[slot]);state=trace['final_state']
        observations=trace['observations'][:length+1]
        np.savez_compressed(out/f'episode_{episode:02d}_trace.npz',observations=observations,
            actions=trace['actions'][:length],values=trace['values'][:length],
            rewards=trace['rewards'][:length],done=trace['done'][:length])
        checks.append(dict(episode=episode,batch_episodes_matched=matched,
            nonselected_batch_differences=mismatches,steps=length,passed=True))
        (out/'replay_validation.json').write_text(json.dumps(dict(passed=True,evaluation=str(source),
            batch=args.evaluation_batch,default_mode_regression_checked_on_first_batch=True,checks=checks),indent=2)+'\n')
        (out/'render_progress.json').write_text(json.dumps(dict(state='encoding',episode=episode,steps=length,completed=len(summaries)))+'\n')
        kinds=('world',) if args.whole_map else ('annotated','clean')
        paths={kind:str(out/f'episode_{episode:02d}_{kind}.mp4') for kind in kinds}
        if args.whole_map:
            from concept_mapping.visualize_symbolic_trajectory import render_world_frame
            from types import SimpleNamespace
            world_frames=trace['world_frames']
        print(f'Encoding episode {episode}: {length} steps; original outcomes verified',flush=True)
        with ExitStack() as stack:
            streams={k:stack.enter_context(_MP4Stream(paths[k],args.fps,args.ffmpeg)) for k in kinds}
            stream=stack.enter_context((out/f'episode_{episode:02d}_actions.csv').open('w'))
            writer=csv.writer(stream);writer.writerow(['step','action','reward','value','done'])
            for t in range(length+1):
                index=min(t,length-1);action=int(trace['actions'][index]);reward=float(trace['rewards'][index])
                done=bool(trace['done'][index]);value=0. if t==length else float(trace['values'][index])
                if t<length:writer.writerow([t,_action_name(action),reward,value,done])
                if t%args.frame_stride and t!=length:continue
                panel=_label_frame(observations[t],action,reward,done,value,t,episode,7,4,textures,panel_textures)
                if args.whole_map:
                    idx=(t+args.frame_stride-1)//args.frame_stride
                    fields=jax.tree_util.tree_map(lambda x:x[idx],world_frames)
                    for name in ('zombies','skeletons','cows','arrows'):
                        fields[name]=SimpleNamespace(**fields[name])
                    streams['world'].append(render_world_frame(SimpleNamespace(**fields),observations[t],panel,textures))
                else:
                    streams['annotated'].append(panel)
                    streams['clean'].append(render_local_view(observations[t],block_size=7,include_inventory=True,
                        display_scale=4,textures=textures,panel_textures=panel_textures))
            frames=(length-1)//args.frame_stride+2
            assert all(s.frames==frames for s in streams.values())
        present=[a.name.lower() for a in Achievement if bool(ach[slot,a.value])]
        summary=dict(episode=episode,world_seed=int(rows[episode]['world_seed']) if bank else None,
            attempt=int(rows[episode].get('attempt',0)),steps=length,frames=frames,
            duration_seconds=frames/args.fps,total_reward=float(shaped[slot]),
            terminated=bool(finished[slot]),truncated=not bool(finished[slot]),
            initial_observation_sha256=hashlib.sha256(observations[0].tobytes()).hexdigest(),
            collected_diamond=bool(ach[slot,Achievement.COLLECT_DIAMOND.value]),ended_in_lava=bool(lava[slot]),
            final_health=int(state.player_health),final_food=int(state.player_food),
            final_water=int(state.player_drink),final_energy=int(state.player_energy),
            achievements=present,missing_achievements=[a.name.lower() for a in Achievement if a.name.lower() not in present],
            videos=paths)
        summaries.append(summary)
        (out/'summary.json').write_text(json.dumps(dict(run=args.run_path,checkpoint=args.checkpoint,
            architecture='neural_map_gru' if cfg.get('NEURAL_MAP_CHANNELS',0) else ('map_mlp_gru' if recurrent else 'map_mlp') if cfg.get('SYMBOLIC_ENCODER')=='map_mlp' else 'gru' if recurrent else 'feedforward',
            source_evaluation=str(source),seed=metadata['evaluation_seed'],episode_ids=episodes,
            evaluation_batch=args.evaluation_batch,exact_recorded_outcomes_verified=True,
            whole_map=args.whole_map,frame_stride=args.frame_stride,
            sampling_temperature=metadata['temperature'],fps=args.fps,max_steps=metadata['max_episode_steps'],
            render_gpu=args.gpu,episodes=summaries),indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    validate_videos(out,args.ffmpeg)
    (out/'render_progress.json').write_text(json.dumps(dict(state='complete',completed=len(summaries)))+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--run-path',help='Legacy W&B run directory.')
    source.add_argument('--study',type=Path,help='Study root or architecture folder; resolve config/run/latest completed save automatically.')
    p.add_argument('--gpu',type=int,help='Physical GPU to reserve and use for rendering.')
    p.add_argument('--checkpoint',type=int,default=None,help='Defaults to the latest saved checkpoint.')
    p.add_argument('--config-json',type=Path,help='Explicit inference config; otherwise read the saved run config.')
    p.add_argument('--out',help='Output folder; study mode automatically names it by checkpoint and seed.')
    p.add_argument('--whole-map',action='store_true',help='Render episodes with a full-world spectator camera and a local-view inset.')
    p.add_argument('--frame-stride',type=int,default=1,help='In whole-map videos, render every Nth action and the terminal frame.')
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--episode-offset',type=int,default=0,help='First episode index; reproduce or finish a subset with the same seed.')
    p.add_argument('--evaluation-dir',type=Path,help='Replay recorded evaluation episodes with their original world and rollout seeds.')
    p.add_argument('--episode-ids',help='Comma-separated recorded episode IDs; requires --evaluation-dir.')
    p.add_argument('--evaluation-batch',type=int,default=25,help='Original evaluator batch size; required for faithful evaluation replay.')
    p.add_argument('--resume',action='store_true',help='Keep completed recorded-evaluation videos and finish remaining selected IDs.')
    p.add_argument('--seed',type=int,default=8050)
    p.add_argument('--fps',type=int,default=20)
    p.add_argument('--max-steps',type=int,default=10000)
    p.add_argument('--ffmpeg',default='/home/shawheen/miniconda3/bin/ffmpeg')
    args=p.parse_args()
    if args.frame_stride < 1 or (args.frame_stride != 1 and not args.whole_map):p.error('--frame-stride needs a positive value and --whole-map')
    if bool(args.evaluation_dir)!=bool(args.episode_ids):raise ValueError('Use --evaluation-dir and --episode-ids together')
    if args.resume and not args.evaluation_dir:raise ValueError('--resume requires --evaluation-dir')
    if args.evaluation_dir:
        metadata=json.loads((args.evaluation_dir/'summary.json').read_text())
        checkpoint=int(Path(metadata['checkpoint']).name)
        if args.checkpoint is not None and args.checkpoint!=checkpoint:raise ValueError('Checkpoint differs from recorded evaluation')
        args.checkpoint=checkpoint;args.seed=metadata['evaluation_seed']
    path,out=resolve_inputs(args)
    temperature=metadata['temperature'] if args.evaluation_dir else 1
    print(f'Checkpoint: {args.checkpoint}; output: {out}; sampling temperature: {temperature}',flush=True)
    if out.exists() and not args.resume:raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True,exist_ok=args.resume)
    cfg=json.loads(args.config_json.read_text()) if args.config_json else _load_wandb_config(args.run_path)
    cfg={k.upper():v for k,v in cfg.items()}
    if args.evaluation_dir:
        render_evaluation(args,path,out,cfg)
        return
    recurrent=bool(cfg.get('RECURRENT',cfg.get('MEMORY_TYPE') in ('gru','transformer')))
    if recurrent and cfg.get('MEMORY_TYPE','gru')!='gru':raise ValueError('This renderer supports GRU memory')
    if args.episodes<1 or args.max_steps<1 or args.fps<1:raise ValueError('Episodes, max steps and FPS must be positive')
    if args.episode_offset<0:raise ValueError('Episode offset must be nonnegative')
    env=make_craftax_env_from_name(cfg['ENV_NAME'],auto_reset=False)
    from wrappers import ClassicDeathPenaltyWrapper,ClassicLavaPenaltyWrapper,ClassicAchievementRewardWrapper
    if cfg.get('DEATH_PENALTY',0.):env=ClassicDeathPenaltyWrapper(env,cfg['DEATH_PENALTY'])
    if cfg.get('LAVA_PENALTY',0.):env=ClassicLavaPenaltyWrapper(env,cfg['LAVA_PENALTY'])
    if cfg.get('DIAMOND_REWARD',1.)!=1. or cfg.get('PLANT_REWARD',1.)!=1.:
        env=ClassicAchievementRewardWrapper(env,cfg.get('DIAMOND_REWARD',1.),cfg.get('PLANT_REWARD',1.))
    from symbolic_map_memory import display_local_view,with_map_memory,simulator_state
    env=with_map_memory(env,cfg)
    from symbolic_motion_inputs import with_motion_inputs
    env=with_motion_inputs(env,cfg)
    ep=env.default_params
    if recurrent:
        from ppo_rnn_og import ActorCriticRNN,initialize_recurrent_carry
        net=ActorCriticRNN(env.action_space(ep).n,cfg)
    else:net=_init_network(cfg,env,ep)
    if (path/'default').is_dir():path=path/'default'
    params=ocp.PyTreeCheckpointer().restore(str(path))['params']
    reset=jax.jit(env.reset);step_fn=jax.jit(env.step)
    @jax.jit
    def policy(obs,key,hidden):
        if recurrent:
            hidden,pi,v=net.apply(params,hidden,(obs[None,None],jnp.zeros((1,1),bool)))
            value=v[0,0];logits=pi.logits[0,0]
        else:
            pi,v=net.apply(params,obs[None]);value=v[0];logits=pi.logits[0]
        key,k=jax.random.split(key)
        return jax.random.categorical(k,logits),value,key,hidden
    @jax.jit
    def value_only(obs,hidden):
        if recurrent:return net.apply(params,hidden,(obs[None,None],jnp.zeros((1,1),bool)))[2][0,0]
        return net.apply(params,obs[None])[1][0]
    textures=jax.tree_util.tree_map(np.asarray,load_all_textures(7))
    panel_textures=jax.tree_util.tree_map(np.asarray,load_all_textures(28))
    ffmpeg=args.ffmpeg
    summaries=[]
    for episode in range(args.episode_offset,args.episode_offset+args.episodes):
        key=jax.random.fold_in(jax.random.PRNGKey(args.seed),episode)
        key,rk,pk=jax.random.split(key,3)
        obs,state=reset(rk,ep)
        initial_observation_sha256=hashlib.sha256(np.asarray(obs).tobytes()).hexdigest()
        hidden=initialize_recurrent_carry(1,cfg) if recurrent else jnp.zeros((1,0))
        kinds=('world',) if args.whole_map else ('annotated','clean')
        paths={kind:str(out/f'episode_{episode:02d}_{kind}.mp4') for kind in kinds}
        reward_total=0.;done=False
        print(f'Rendering episode {episode+1} ({episode-args.episode_offset+1}/{args.episodes} requested)',flush=True)
        with ExitStack() as stack:
            streams={kind:stack.enter_context(_MP4Stream(paths[kind],args.fps,ffmpeg)) for kind in kinds}
            f=stack.enter_context((out/f'episode_{episode:02d}_actions.csv').open('w'))
            writer=csv.writer(f);writer.writerow(['step','action','reward','value','done'])
            def append(obs,frame_state,action,reward,done,value,t):
                observation=np.asarray(display_local_view(obs,simulator_state(frame_state),cfg))
                panel=_label_frame(observation,action,reward,done,value,t,episode,7,4,textures,panel_textures)
                if args.whole_map:
                    from concept_mapping.visualize_symbolic_trajectory import render_world_frame
                    streams['world'].append(render_world_frame(simulator_state(frame_state),observation,panel,textures))
                else:
                    streams['annotated'].append(panel)
                    streams['clean'].append(render_local_view(observation,block_size=7,include_inventory=True,
                        display_scale=4,textures=textures,panel_textures=panel_textures))
            for t in range(args.max_steps):
                action,value,pk,hidden=policy(obs,pk,hidden)
                key,sk=jax.random.split(key)
                next_obs,next_state,reward,terminal,_=step_fn(sk,state,action,ep)
                a=int(action);r=float(reward);v=float(value);done=bool(terminal)
                if t%args.frame_stride==0:append(obs,state,a,r,done,v,t)
                writer.writerow([t,_action_name(a),r,v,done])
                reward_total+=r;obs,state=next_obs,next_state
                if done:break
                if (t+1)%1000==0: print(f'Episode {episode+1}: {t+1} steps',flush=True)
            append(obs,state,a,r,done,0. if done else float(value_only(obs,hidden)),t+1)
            frames=t//args.frame_stride+2
            assert all(stream.frames==frames for stream in streams.values())
        state=simulator_state(state)
        summary=dict(episode=episode,steps=t+1,frames=frames,duration_seconds=frames/args.fps,
            total_reward=reward_total,terminated=done,truncated=not done,
            initial_observation_sha256=initial_observation_sha256,
            collected_diamond=bool(state.achievements[Achievement.COLLECT_DIAMOND.value]),
            ended_in_lava=bool(state.map[tuple(np.asarray(state.player_position))]==BlockType.LAVA.value),
            final_health=int(state.player_health),final_food=int(state.player_food),
            final_water=int(state.player_drink),final_energy=int(state.player_energy),
            achievements=[a.name.lower() for a in Achievement if bool(state.achievements[a.value])],videos=paths)
        summaries.append(summary)
        (out/'summary.json').write_text(json.dumps(dict(run=args.run_path,checkpoint=args.checkpoint,
            architecture='neural_map_gru' if cfg.get('NEURAL_MAP_CHANNELS',0) else ('map_mlp_gru' if recurrent else 'map_mlp') if cfg.get('SYMBOLIC_ENCODER')=='map_mlp' else ('spatial_map_' if cfg.get('SYMBOLIC_ENCODER')=='spatial_map' else 'spatial_' if cfg.get('SYMBOLIC_ENCODER')=='spatial' else '')+('gru' if recurrent else 'feedforward'),
            explicit_odometry=cfg.get('MOTION_INPUTS',False) or cfg.get('SYMBOLIC_ENCODER') in ('spatial_map','map_mlp'),
            neural_map_channels=cfg.get('NEURAL_MAP_CHANNELS',0),
            neural_map_reset_every_step=cfg.get('NEURAL_MAP_RESET_EVERY_STEP',False),
            gru_concat_observation=cfg.get('GRU_CONCAT_OBSERVATION',False),
            motion_inputs=cfg.get('MOTION_INPUTS',False),
            death_penalty=cfg.get('DEATH_PENALTY',0.),lava_penalty=cfg.get('LAVA_PENALTY',0.),
            diamond_reward=cfg.get('DIAMOND_REWARD',1.),plant_reward=cfg.get('PLANT_REWARD',1.),seed=args.seed,
            independent_episode_seeds=True,episode_offset=args.episode_offset,fps=args.fps,max_steps=args.max_steps,
            sampling_temperature=1,render_gpu=args.gpu,
            whole_map=args.whole_map,frame_stride=args.frame_stride,world_view=cfg.get('WORLD_VIEW'),
            versions='Full-world spectator map alongside local view and inventory.' if args.whole_map else 'Identical frames/timing; clean retains map and inventory, omits white annotation panel.',
            episodes=summaries),indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    validate_videos(out,ffmpeg)


if __name__=='__main__':
    try:main()
    finally:
        if _gpu_lock is not None:_gpu_lock.close()

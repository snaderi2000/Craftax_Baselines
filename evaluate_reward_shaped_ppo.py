"""Frozen-policy evaluation on ordinary Classic maps, reporting all 22 events."""
import argparse,csv,json,time,math
from pathlib import Path
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map,leaves=jax.tree_util.tree_leaves,reduce=jax.tree_util.tree_reduce)
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement,BlockType


def select_policy_actions(logits,keys,temperature=1.):
    """Temperature changes action sampling only; zero selects greedy actions."""
    if not math.isfinite(temperature) or temperature<0:
        raise ValueError('Temperature must be finite and nonnegative')
    if temperature==0:return jnp.argmax(logits,axis=-1)
    return jax.vmap(lambda p,k:jax.random.categorical(k,p/temperature))(logits,keys)


def build_evaluator(env,net,params,cfg,*,seed,batch,world_seeds=None,temperature=1.,max_steps=10000,trace=False,world_frame_stride=None,archive_fields=None,resource_timing=False,coverage_metrics=False):
    """Build the shared batched rollout, optionally recording one episode.

    Trace mode preserves the evaluation batch and PRNG streams. Its dynamic
    trace_index selects a batch lane without recompiling for each episode.
    """
    from symbolic_map_memory import display_local_view, simulator_state
    if trace and archive_fields is not None:
        raise ValueError('Choose one recording mode')
    if resource_timing and (trace or archive_fields is not None):
        raise ValueError('Resource timing is available for untraced evaluations')
    resources = jnp.array([Achievement.COLLECT_DRINK.value, Achievement.COLLECT_DIAMOND.value])
    track_map = resource_timing and cfg.get('MAP_SPATIAL_REGIONS', False)
    recurrent=bool(cfg.get('RECURRENT','MEMORY_TYPE' in cfg))
    def spectator_fields(raw):
        return dict(map=raw.map.astype(jnp.uint8),player_position=raw.player_position,
            achievements=raw.achievements,**{name:dict(position=getattr(raw,name).position,
                mask=getattr(raw,name).mask) for name in ('zombies','skeletons','cows','arrows')})

    @jax.jit
    def evaluate(ids,trace_index=0):
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(seed),i))(ids)
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        reset_keys=(jax.vmap(jax.random.PRNGKey)(world_seeds[ids%len(world_seeds)])
                    if world_seeds is not None else split[:,0])
        obs,state=jax.vmap(env.reset,in_axes=(0,None))(reset_keys,env.default_params)
        from ppo_rnn_og import initialize_recurrent_carry
        hidden=initialize_recurrent_carry(batch,cfg) if recurrent else jnp.zeros((batch,0))
        initial=(jnp.asarray(0),obs,state,split[:,1],jnp.zeros(batch,bool),hidden,jnp.zeros(batch),
            jnp.zeros(batch),jnp.zeros(batch,jnp.int32),jnp.zeros((batch,len(Achievement)),bool),
            jnp.zeros(batch,bool),jnp.zeros(batch,bool),jnp.zeros(batch,bool))
        if trace:
            raw=simulator_state(state)
            selected=jax.tree_util.tree_map(lambda x:x[trace_index],raw)
            live=display_local_view(obs[trace_index],selected,cfg)
            initial+=(dict(observations=jnp.zeros((max_steps+1,1345),obs.dtype).at[0].set(live),
                actions=jnp.zeros(max_steps,jnp.int32),rewards=jnp.zeros(max_steps),
                values=jnp.zeros(max_steps),done=jnp.zeros(max_steps,bool),final_state=selected),)
            if world_frame_stride:
                frame_count=(max_steps-1)//world_frame_stride+2
                initial[-1]['world_frames']=jax.tree_util.tree_map(
                    lambda x:jnp.zeros((frame_count,*x.shape),x.dtype).at[0].set(x),spectator_fields(selected))
        if archive_fields is not None:
            state_fields=archive_fields(obs,simulator_state(state))
            def allocate(x):
                return jnp.zeros((max_steps+1,*x.shape),x.dtype).at[0].set(x)
            frames=jax.tree_util.tree_map(allocate,state_fields)
            transitions=dict(actions=jnp.zeros((max_steps,batch),jnp.uint8),
                rewards=jnp.zeros((max_steps,batch)),original_rewards=jnp.zeros((max_steps,batch)),
                dones=jnp.zeros((max_steps,batch),bool),values=jnp.zeros((max_steps,batch)))
            initial+=((frames,transitions),)
        if resource_timing:
            initial += (jnp.full((batch, 6 if cfg.get('MAP_LANDMARKS') else 2), -1, dtype=jnp.int32),)
        if track_map:
            initial += (jnp.zeros_like(state.terrain, dtype=bool),
                        jnp.full((batch, 3), -1, dtype=jnp.int32))
        if coverage_metrics:
            known = (state.terrain != 0).sum(axis=(1, 2))
            initial += (jnp.stack((known, known), axis=1),)
        def step(c):
            t,obs,state,keys,finished,hidden,returns,shaped_returns,lengths,ach,lava,other_death,timeout=c[:13]
            split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
            if recurrent:
                hidden,pi,value=net.apply(params,hidden,(obs[None],finished[None]))
                logits=pi.logits[0];value=value[0]
            else:
                pi,value=net.apply(params,obs);logits=pi.logits
            action=select_policy_actions(logits,split[:,1],temperature)
            no,ns,r,d,info=jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],state,action,env.default_params)
            active=~finished
            raw=simulator_state(ns);previous=simulator_state(state)
            tile=raw.map[jnp.arange(batch),raw.player_position[:,0],raw.player_position[:,1]]
            died=d&((raw.player_health<=0)|(tile==BlockType.LAVA.value))
            newly=raw.achievements&~previous.achievements
            shaped=(r+(cfg.get('DIAMOND_REWARD',1.)-1.)*newly[:,Achievement.COLLECT_DIAMOND.value]
                +(cfg.get('PLANT_REWARD',1.)-1.)*newly[:,Achievement.EAT_PLANT.value]
                -cfg.get('DEATH_PENALTY',0.)*died
                -cfg.get('LAVA_PENALTY',0.)*(d&(tile==BlockType.LAVA.value)))
            updated=(t+1,no,ns,split[:,0],finished|d,hidden,returns+active*r,
                shaped_returns+active*shaped,lengths+active,
                jnp.where(active[:,None],raw.achievements,ach),jnp.where(active,tile==BlockType.LAVA.value,lava),
                jnp.where(active,d&(raw.player_health<=0)&(tile!=BlockType.LAVA.value),other_death),
                jnp.where(active,d&(raw.player_health>0)&(tile!=BlockType.LAVA.value),timeout))
            if trace:
                recorded=c[13]
                selected=jax.tree_util.tree_map(lambda x:x[trace_index],raw)
                final=jax.tree_util.tree_map(lambda new,old:jnp.where(active[trace_index],new,old),selected,recorded['final_state'])
                live=display_local_view(no[trace_index],selected,cfg)
                recorded=dict(observations=recorded['observations'].at[t+1].set(live),
                    actions=recorded['actions'].at[t].set(action[trace_index]),
                    rewards=recorded['rewards'].at[t].set(shaped[trace_index]),
                    values=recorded['values'].at[t].set(value[trace_index]),
                    done=recorded['done'].at[t].set(d[trace_index]),final_state=final)
                if world_frame_stride:
                    # Reserve the next index for an off-stride terminal observation.
                    index=(t+world_frame_stride)//world_frame_stride
                    save=active[trace_index]&(((t+1)%world_frame_stride==0)|d[trace_index]|(t+1==max_steps))
                    recorded['world_frames']=jax.lax.cond(save,
                        lambda frames:jax.tree_util.tree_map(lambda buf,x:buf.at[index].set(x),
                            frames,spectator_fields(selected)),lambda frames:frames,c[13]['world_frames'])
                updated+=(recorded,)
            if archive_fields is not None:
                frames,transitions=c[13]
                frames=jax.tree_util.tree_map(lambda buf,x:buf.at[t+1].set(x),frames,archive_fields(no,raw))
                fields=dict(actions=action.astype(jnp.uint8),rewards=shaped,original_rewards=r,dones=d,values=value)
                transitions=jax.tree_util.tree_map(lambda buf,x:buf.at[t].set(x),transitions,fields)
                updated+=((frames,transitions),)
            if resource_timing:
                first = c[13]
                events = newly[:, resources]
                if cfg.get('MAP_LANDMARKS'):
                    grid = state.env_state.terrain
                    pick = previous.inventory.iron_pickaxe > 0
                    pending = ~previous.achievements[:,Achievement.COLLECT_DIAMOND.value]
                    visible = obs[:,:1323].reshape(batch,7,9,21)[...,BlockType.DIAMOND.value]
                    distant = (grid == BlockType.DIAMOND.value+1).any((1,2)) & ~visible.any((1,2))
                    adjacent = (visible[:,2,4]+visible[:,4,4]+visible[:,3,3]+visible[:,3,5])>0
                    extra = jnp.stack((pending & pick & distant, pending & pick & adjacent,
                                       newly[:,Achievement.PLACE_PLANT.value],newly[:,Achievement.EAT_PLANT.value]),axis=1)
                    events = jnp.concatenate((events,extra),axis=1)
                reached = active[:, None] & (first < 0) & events
                updated += (jnp.where(reached, t+1, first),)
            if track_map:
                from craftax.craftax_classic.constants import DIRECTIONS
                remembered, first = c[14:16]
                pos = state.position
                outside_rows = jnp.abs(jnp.arange(64)[None,:] - pos[:,0,None]) > 3
                outside_cols = jnp.abs(jnp.arange(64)[None,:] - pos[:,1,None]) > 4
                known = state.terrain == BlockType.DIAMOND.value + 1
                eligible = active & (previous.inventory.iron_pickaxe > 0) & ~previous.achievements[:,Achievement.COLLECT_DIAMOND.value]
                distant = known & (outside_rows[:,:,None] | outside_cols[:,None,:]) & eligible[:,None,None]
                remembered = remembered | distant
                facing = pos + DIRECTIONS[previous.player_direction]
                mined_known = newly[:,Achievement.COLLECT_DIAMOND.value] & remembered[jnp.arange(batch),facing[:,0],facing[:,1]]
                seen = known.any((1,2)) | (ns.terrain == BlockType.DIAMOND.value+1).any((1,2))
                events = jnp.stack((seen, distant.any((1,2)), mined_known), axis=1)
                updated += (remembered, jnp.where(active[:,None] & (first < 0) & events, t+1, first))
            if coverage_metrics:
                new_cells = active * info['map_new_cells']
                updated += (c[-1] + jnp.stack((new_cells, new_cells * (t < 1000)), axis=1),)
            return updated
        result=jax.lax.while_loop(lambda c:(c[0]<max_steps)&~c[4].all(),step,initial)
        outputs = (result[0],result[4],*result[6:13])
        return outputs + ((result[13], *result[15:]) if track_map else tuple(result[13:]))

    return evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reset-neural-map',action='store_true',help='Ablate neural map history, preserving GRU and weights')
    p.add_argument('--map-coverage-metrics',action='store_true',help='Record distinct observed tiles at episode end and at step 1000 or earlier termination.')
    p.add_argument('--run-path');p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config-json',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=1024);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--seed',type=int,default=18050)
    p.add_argument('--world-seeds-file',type=Path,help='JSON world_seeds list; reset with PRNGKey(world_seed).')
    p.add_argument('--repeats',type=int,default=1,help='Independent action/dynamics trials per fixed world.')
    p.add_argument('--temperature',type=float,default=1.,help='Sampling temperature; 0 means greedy argmax, 1 preserves normal sampling.')
    args=p.parse_args()
    if not math.isfinite(args.temperature) or args.temperature<0:raise ValueError('Temperature must be finite and nonnegative')
    world_seeds=None
    if args.world_seeds_file:
        from fixed_world_seeds import read_world_seeds
        world_seeds=jnp.asarray(read_world_seeds(args.world_seeds_file),jnp.uint32)
        args.episodes=len(world_seeds)*args.repeats
    elif args.repeats!=1:raise ValueError('--repeats requires --world-seeds-file')
    if args.repeats<1:raise ValueError('Repeats must be positive')
    if args.config_json:cfg=json.loads(args.config_json.read_text())
    elif args.run_path:
        from concept_mapping.render_ppo_episodes import _load_wandb_config
        cfg=_load_wandb_config(args.run_path)
    else:raise ValueError('Provide --run-path or --config-json')
    cfg={k.upper():v for k,v in cfg.items()}
    if args.reset_neural_map:cfg['NEURAL_MAP_RESET_EVERY_STEP']=True
    recurrent=bool(cfg.get('RECURRENT','MEMORY_TYPE' in cfg))
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    from symbolic_map_memory import with_map_memory,simulator_state
    coverage_metrics = args.map_coverage_metrics or bool(cfg.get('MAP_COVERAGE_BONUS'))
    if coverage_metrics and (cfg.get('SYMBOLIC_ENCODER') != 'map_mlp' or recurrent):
        raise ValueError('Map coverage metrics require a feedforward remembered-map policy')
    env=with_map_memory(env,cfg,track_coverage=coverage_metrics)
    from symbolic_motion_inputs import with_motion_inputs
    env=with_motion_inputs(env,cfg)
    if recurrent:
        from ppo_rnn_og import ActorCriticRNN
        net=ActorCriticRNN(17,cfg)
    elif cfg.get('SYMBOLIC_ENCODER')=='map_mlp':
        from models.actor_critic_map_mlp import map_mlp_network
        net=map_mlp_network(cfg,17)
    elif cfg.get('SYMBOLIC_ENCODER','mlp')=='spatial':
        from models.conditioned_symbolic import ConditionedSymbolicActorCritic
        net=ConditionedSymbolicActorCritic(17)
    else:
        from models.actor_critic import ActorCritic
        net=ActorCritic(17,int(cfg.get('LAYER_SIZE',512)))
    checkpoint=args.checkpoint.resolve()
    payload=checkpoint/'default' if (checkpoint/'default').is_dir() else checkpoint
    params=ocp.PyTreeCheckpointer().restore(str(payload))['params']
    batch=args.batch
    if args.episodes<=0 or batch<=0:raise ValueError('Episodes and batch must be positive')
    args.out.mkdir(parents=True,exist_ok=False)
    began=time.time()

    evaluate=build_evaluator(env,net,params,cfg,seed=args.seed,batch=batch,world_seeds=world_seeds,
        temperature=args.temperature,resource_timing=True,coverage_metrics=coverage_metrics)

    rows=[];executed=0
    for offset in range(0,args.episodes,batch):
        t,finished,returns,shaped,lengths,ach,lava,other_death,timeout,first,*map_times=jax.device_get(evaluate(jnp.arange(offset,offset+batch)))
        if coverage_metrics:
            coverage = map_times.pop()
        executed+=int(t)*batch
        for i in range(min(batch,args.episodes-offset)):rows.append(dict(episode=offset+i,original_return=float(returns[i]),
            shaped_return=float(shaped[i]),length=int(lengths[i]),terminated=bool(finished[i]),
            lava=bool(lava[i]),nonlava_death=bool(other_death[i]),timeout=bool(timeout[i]),all_22=bool(ach[i].all()),
            first_drink_step=int(first[i,0]) if first[i,0]>=0 else None,
            first_diamond_step=int(first[i,1]) if first[i,1]>=0 else None,
            **(dict(seen_tiles=int(coverage[i,0]),seen_tiles_by_1000_or_end=int(coverage[i,1])) if coverage_metrics else {}),
            **({name:(int(tick) if tick >= 0 else None) for name,tick in zip(
                ('diamond_seen_by_step','first_action_offscreen_known_diamond_with_pick',
                 'first_mined_previously_offscreen_diamond_with_pick'),map_times[0][i])} if map_times else {}),
            **({name:(int(tick) if tick >= 0 else None) for name,tick in zip(
                ('first_action_distant_diamond_with_pick','first_action_adjacent_diamond_with_pick',
                 'first_plant_step','first_eat_plant_step'),first[i,2:])} if cfg.get('MAP_LANDMARKS') else {}),
            **({ 'world_seed':int(world_seeds[(offset+i)%len(world_seeds)]),
                 'attempt':(offset+i)//len(world_seeds)} if world_seeds is not None else {}),
            **{a.name.lower():int(ach[i,a.value]) for a in Achievement}))
        print(f'Evaluated {len(rows)}/{args.episodes}',flush=True)
    with (args.out/'episodes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rates={a.name.lower():float(np.mean([r[a.name.lower()] for r in rows])) for a in Achievement}
    def interval(p):
        n=len(rows);z=1.959963984540054;center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        return [100*max(0.,center-half),100*min(1.,center+half)]
    all_rate=float(np.mean([r['all_22'] for r in rows]))
    report=dict(checkpoint=str(checkpoint),training_steps=int(checkpoint.name),
        architecture='neural_map_gru' if cfg.get('NEURAL_MAP_CHANNELS',0) else ('map_mlp_gru' if recurrent else 'map_mlp') if cfg.get('SYMBOLIC_ENCODER')=='map_mlp' else ('spatial_map_' if cfg.get('SYMBOLIC_ENCODER')=='spatial_map' else 'spatial_' if cfg.get('SYMBOLIC_ENCODER')=='spatial' else '')+('gru' if recurrent else 'feedforward'),
        explicit_odometry=cfg.get('MOTION_INPUTS',False) or cfg.get('SYMBOLIC_ENCODER') in ('spatial_map','map_mlp'),
        map_view_size=cfg.get('MAP_VIEW_SIZE',33) if cfg.get('SYMBOLIC_ENCODER') in ('spatial_map','map_mlp') and not cfg.get('MAP_SINGLE_READOUT') else None,
        map_global_readout=cfg.get('MAP_GLOBAL_READOUT',False),
        map_landmarks=cfg.get('MAP_LANDMARKS'),
        map_single_readout=cfg.get('MAP_SINGLE_READOUT',False),
        map_board_only=cfg.get('MAP_BOARD_ONLY',False),
        map_board_attention=cfg.get('MAP_BOARD_ATTENTION',False),
        map_v1_attention_adapter=cfg.get('MAP_V1_ATTENTION_ADAPTER',False),
        internal_crop_size=cfg.get('MAP_VIEW_SIZE',33) if cfg.get('MAP_INTERNAL_CROP') else None,
        map_patch_mlp=cfg.get('MAP_PATCH_MLP',False),
        map_region_shape=[8,8,32] if cfg.get('MAP_SPATIAL_REGIONS') else None,
        map_fine_shortcut=cfg.get('MAP_FINE_SHORTCUT',False),
        map_full_terrain=cfg.get('MAP_FULL_TERRAIN',False),
        neural_map_channels=cfg.get('NEURAL_MAP_CHANNELS',0),
        neural_map_reset_every_step=cfg.get('NEURAL_MAP_RESET_EVERY_STEP',False),
        training_seed=cfg.get('SEED'),episodes=len(rows),evaluation_seed=args.seed,
        world_seeds_file=str(args.world_seeds_file.resolve()) if args.world_seeds_file else None,
        trials_per_world=args.repeats,
        maps='original Classic generator',reset='standard resets on explicit world seeds' if world_seeds is not None else 'independent standard resets',
        action_selection='greedy' if args.temperature==0 else 'categorical',temperature=args.temperature,
        training_temperature=cfg.get('POLICY_TEMPERATURE',1.),training_entropy_coefficient=cfg.get('ENT_COEF'),
        reset_gru_every_step=cfg.get('RESET_GRU_EVERY_STEP',False),
        gru_concat_observation=cfg.get('GRU_CONCAT_OBSERVATION',False),
        sru_gru=cfg.get('SRU_GRU',False),
        world_view=cfg.get('WORLD_VIEW'),local_policy_init=cfg.get('LOCAL_POLICY_INIT'),
        motion_inputs=cfg.get('MOTION_INPUTS',False),
        max_episode_steps=10000,death_penalty=cfg.get('DEATH_PENALTY',0.),
        lava_penalty=cfg.get('LAVA_PENALTY',0.),
        diamond_reward=cfg.get('DIAMOND_REWARD',1.),plant_reward=cfg.get('PLANT_REWARD',1.),
        mean_original_return=float(np.mean([r['original_return'] for r in rows])),
        mean_shaped_return=float(np.mean([r['shaped_return'] for r in rows])),
        mean_length=float(np.mean([r['length'] for r in rows])),
        median_length=float(np.median([r['length'] for r in rows])),
        mean_achievements=float(np.mean([sum(r[a.name.lower()] for a in Achievement) for r in rows])),
        all_22_success_pct=100*all_rate,all_22_ci95_pct=interval(all_rate),
        achievement_success_pct={k:100*v for k,v in rates.items()},
        achievement_ci95_pct={k:interval(v) for k,v in rates.items()},
        minimum_achievement_success_pct=100*min(rates.values()),
        lava_pct=100*float(np.mean([r['lava'] for r in rows])),
        nonlava_death_pct=100*float(np.mean([r['nonlava_death'] for r in rows])),
        early_death_before_1000_pct=100*float(np.mean([(r['lava'] or r['nonlava_death']) and r['length']<1000 for r in rows])),
        timeout_pct=100*float(np.mean([r['timeout'] for r in rows])),
        evaluation_executed_transitions=executed,seconds=time.time()-began,
        caveat='One training seed; episode confidence intervals do not measure training-seed variability. Observed 100% is not a guarantee on unseen episodes.')
    if coverage_metrics:
        report.update(mean_seen_tiles=float(np.mean([r['seen_tiles'] for r in rows])),
                      mean_seen_tiles_by_1000_or_end=float(np.mean([r['seen_tiles_by_1000_or_end'] for r in rows])),
                      training_map_coverage_bonus=cfg.get('MAP_COVERAGE_BONUS') or 0.,
                      evaluation_map_coverage_bonus=0.)
    if cfg.get('MAP_SPATIAL_REGIONS'):
        opportunities = [r for r in rows if r['first_action_offscreen_known_diamond_with_pick'] is not None]
        report['diamond_memory'] = dict(
            worlds_never_saw_diamond=sum(r['diamond_seen_by_step'] is None for r in rows),
            worlds_with_eligible_offscreen_diamond=len(opportunities),
            worlds_mined_an_eligible_tile=sum(r['first_mined_previously_offscreen_diamond_with_pick'] is not None
                                               for r in opportunities))
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

"""Replay recorded early failures, attribute health losses, and find same-world controls.

Diagnostic simulator state is used only for auditing, never as policy input.
The existing evaluator supplies actions; every failure must reproduce its CSV
outcome. A second action replay must reproduce every recorded observation before
its movement and health accounting are accepted.
"""
import argparse,csv,json,os,sys,time,hashlib
from pathlib import Path


def read_rows(path):
    with (path/'episodes.csv').open() as f:return list(csv.DictReader(f))


def longest_run(flags):
    best=run=0
    for flag in flags:
        run=run+1 if flag else 0;best=max(best,run)
    return best


def summarize_trace(data,length,window=200):
    import numpy as np
    n=min(length,len(data['combat_damage']));sl=slice(max(0,n-window),n)
    vitals=data['vitals'][sl];combat=data['combat_damage'][sl];intrinsic=data['intrinsic_delta'][sl]
    blocked=data['blocked'][sl].astype(bool);exits=data['available_exits'][sl]
    last=data['vitals'][n-1]
    deprived=[name for name,k in [('food',1),('water',2),('energy',3)] if last[k]<=0 and (k!=3 or not data['sleeping_after'][n-1])]
    terminal=('mixed_combat_and_deprivation' if data['combat_damage'][n-1]>0 and data['intrinsic_delta'][n-1]<0
        else 'combat' if data['combat_damage'][n-1]>0 else 'deprivation' if data['intrinsic_delta'][n-1]<0 else 'other')
    return dict(observed_steps=n,window_steps=min(n,window),terminal_health_loss_mechanism=terminal,
        depleted_at_end=deprived,final_vitals=[int(x) for x in last],
        combat_damage_last_window=int(combat.sum()),deprivation_damage_last_window=int(-np.minimum(intrinsic,0).sum()),
        healing_last_window=int(np.maximum(intrinsic,0).sum()),
        blocked_moves_last_window=int(blocked.sum()),longest_blocked_streak_last_window=longest_run(blocked),
        blocked_moves_with_other_exit=int((blocked&(exits>0)).sum()),
        food_zero_steps=int((vitals[:,1]<=0).sum()),water_zero_steps=int((vitals[:,2]<=0).sum()),
        energy_zero_awake_steps=int(((vitals[:,3]<=0)&~data['sleeping_after'][sl]).sum()),
        sleeping_steps=int(data['sleeping_before'][sl].sum()),
        combat_damage_while_sleeping=int((combat*data['sleeping_before'][sl]).sum()),
        unique_positions_last_window=len(np.unique(data['position'][sl],axis=0)))


def build_diagnostic(raw_env,horizon):
    import jax,jax.numpy as jnp
    from craftax.craftax_classic import game_logic as g
    from craftax.craftax_classic.constants import Action,DIRECTIONS
    params=raw_env.default_params;static=raw_env.static_env_params
    @jax.jit
    def replay(world,seed,episode,actions):
        obs,state=raw_env.reset(jax.random.PRNGKey(world),params)
        key=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),episode),2)[1]
        def step(carry,action):
            s,key=carry;split=jax.random.split(key,3);envkey=split[2]
            effective=jnp.where(s.is_sleeping,Action.NOOP.value,action)
            before=g.do_crafting(s,effective)
            k,k_action=jax.random.split(envkey)
            before=g.do_action(k_action,before,effective,static)
            before=g.place_block(before,effective,static)
            moved=g.move_player(before,effective)
            _,k_mob=jax.random.split(k)
            mobs=g.update_mobs(k_mob,moved,params,static)
            no,ns,_,_,_=raw_env.step(envkey,s,action,params)
            move=(effective>=1)&(effective<=4)
            blocked=move&jnp.all(ns.player_position==s.player_position)
            exits=jax.vmap(lambda d:g.is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava(s,s.player_position+d))(DIRECTIONS[1:5]).sum()
            row=dict(observations=no,position=ns.player_position,
                vitals=jnp.array([ns.player_health,ns.player_food,ns.player_drink,ns.player_energy]),
                combat_damage=before.player_health-mobs.player_health,
                intrinsic_delta=ns.player_health-mobs.player_health,
                blocked=blocked,available_exits=exits,sleeping_before=s.is_sleeping,sleeping_after=ns.is_sleeping,
                facing_before=s.player_direction,action=action,
                attempted_tile=s.map[(s.player_position+DIRECTIONS[effective])[0],(s.player_position+DIRECTIONS[effective])[1]])
            return (ns,split[0]),row
        _,data=jax.lax.scan(step,(state,key),actions[:horizon])
        return obs,data
    return replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations',type=Path,nargs='+',required=True)
    p.add_argument('--config-json',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2);p.add_argument('--batch',type=int,default=25)
    p.add_argument('--threshold',type=int,default=1000);p.add_argument('--window',type=int,default=200)
    p.add_argument('--success-repeats',type=int,default=4);p.add_argument('--control-seed',type=int,default=69150)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    from cap_reward_shaped_runs import try_evaluation_lock
    from run_reward_shaped_architectures import environment
    lock=try_evaluation_lock(a.gpu)
    if lock is None:raise RuntimeError(f'GPU {a.gpu} reserved; retry after its evaluation completes')
    os.environ.update(environment(a.gpu))
    import numpy as np,jax,jax.numpy as jnp,orbax.checkpoint as ocp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.constants import Achievement
    from symbolic_map_memory import with_map_memory
    from models.actor_critic_map_mlp import map_mlp_network
    from evaluate_reward_shaped_ppo import build_evaluator
    from fixed_world_seeds import read_world_seeds
    cfg={k.upper():v for k,v in json.loads(a.config_json.read_text()).items()}
    if cfg.get('RECURRENT') or cfg.get('SYMBOLIC_ENCODER')!='map_mlp':raise ValueError('This audit currently supports feedforward map-MLP checkpoints')
    raw=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    env=with_map_memory(raw,cfg);net=map_mlp_network(cfg)
    diagnostic=build_diagnostic(raw,a.threshold)
    report=dict(started=time.time(),threshold=a.threshold,window=a.window,evaluation_batch=a.batch,
        note='Health attribution is exact stage accounting on a verified action replay. Matched successful rerolls are selected diagnostic controls, not an unbiased success estimate. No causal intervention is performed.',
        sources=[],episodes=[],controls=[])
    checkpoint=None;params=None
    def save():
        (a.out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    def matched(result,row,slot):
        _,finished,returns,shaped,lengths,ach,lava,death,timeout,*_=result
        assert int(lengths[slot])==int(row['length']),(int(lengths[slot]),row['length'])
        np.testing.assert_allclose(returns[slot],float(row['original_return']),rtol=0,atol=1e-5)
        np.testing.assert_allclose(shaped[slot],float(row['shaped_return']),rtol=0,atol=1e-5)
        for field,values in [('terminated',finished),('lava',lava),('nonlava_death',death),('timeout',timeout)]:assert bool(values[slot])==(str(row[field])=='True'),field
        for achievement in Achievement:assert bool(ach[slot,achievement.value])==bool(int(row[achievement.name.lower()])),achievement.name
    def record_trace(result,slot,world,seed,episode,prefix,original_length,is_failure):
        trace=result[-1];length=min(int(result[4][slot]),a.threshold)
        actions=np.zeros(a.threshold,np.int32);actions[:length]=trace['actions'][:length]
        initial,details=jax.device_get(diagnostic(jnp.uint32(world),jnp.uint32(seed),jnp.uint32(episode),jnp.asarray(actions)))
        np.testing.assert_array_equal(initial,trace['observations'][0],err_msg='Initial world mismatch')
        np.testing.assert_array_equal(details['observations'][:length],trace['observations'][1:length+1],err_msg='Action replay diverged; diagnostic attribution rejected')
        obs=np.concatenate((initial[None],details.pop('observations')[:length]))
        details={k:np.asarray(v)[:length] for k,v in details.items()}
        np.savez_compressed(a.out/(prefix+'.npz'),observations=obs,**details)
        row=dict(trace=prefix+'.npz',episode=int(episode),world_seed=int(world),seed=int(seed),
            recorded_length=int(original_length),is_failure=is_failure,action_replay_observations_exact=True,
            initial_observation_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),**summarize_trace(details,length,a.window))
        if not is_failure:row['terminal_health_loss_mechanism']='not_terminal_prefix'
        return row
    for dataset,source in enumerate(a.evaluations):
        source=source.resolve();metadata=json.loads((source/'summary.json').read_text());rows=read_rows(source)
        failures=[r for r in rows if r['all_22']=='False' and int(r['length'])<a.threshold]
        if not failures:continue
        if checkpoint!=metadata['checkpoint']:
            checkpoint=metadata['checkpoint'];path=Path(checkpoint);payload=path/'default' if (path/'default').exists() else path
            params=ocp.PyTreeCheckpointer().restore(str(payload))['params']
        worlds=jnp.asarray(read_world_seeds(Path(metadata['world_seeds_file'])),jnp.uint32)
        settings=dict(seed=metadata['evaluation_seed'],batch=a.batch,world_seeds=worlds,temperature=metadata['temperature'])
        short=build_evaluator(env,net,params,cfg,**settings,max_steps=a.threshold,trace=True)
        full=None
        report['sources'].append(dict(evaluation=str(source),checkpoint=checkpoint,early_failures=len(failures),
            csv_sha256=hashlib.sha256((source/'episodes.csv').read_bytes()).hexdigest()))
        for row in failures:
            episode=int(row['episode']);first=episode//a.batch*a.batch;slot=episode-first
            result=jax.device_get(short(jnp.arange(first,first+a.batch),jnp.int32(slot)))
            try:matched(result,row,slot)
            except AssertionError:
                if full is None:full=build_evaluator(env,net,params,cfg,**settings,max_steps=10000,trace=True)
                result=jax.device_get(full(jnp.arange(first,first+a.batch),jnp.int32(slot)));matched(result,row,slot)
            entry=record_trace(result,slot,int(row['world_seed']),metadata['evaluation_seed'],episode,
                f'dataset{dataset}_failure{episode}',int(row['length']),True)
            entry['dataset']=dataset;entry['recorded_outcome_exact']=True;report['episodes'].append(entry);save()
            print(json.dumps(entry),flush=True)
        # Same policy, same worlds, independent action/dynamics streams. Keep
        # the first successful trial per world, with no success-rate inference.
        control_worlds=jnp.asarray([int(r['world_seed']) for r in failures],jnp.uint32)
        control_seed=a.control_seed+dataset*10000
        control_settings=dict(seed=control_seed,batch=a.batch,world_seeds=control_worlds,temperature=metadata['temperature'],max_steps=10000)
        control_eval=build_evaluator(env,net,params,cfg,**control_settings)
        winners={};trials=[]
        for first in range(0,len(failures)*a.success_repeats,a.batch):
            result=jax.device_get(control_eval(jnp.arange(first,first+a.batch)))
            _,finished,returns,shaped,lengths,ach,lava,death,timeout=result
            for slot in range(min(a.batch,len(failures)*a.success_repeats-first)):
                eid=first+slot;world=int(control_worlds[eid%len(failures)])
                row=dict(episode=eid,world_seed=world,length=int(lengths[slot]),all_22=bool(ach[slot].all()),
                    original_return=float(returns[slot]),shaped_return=float(shaped[slot]),terminated=bool(finished[slot]),
                    lava=bool(lava[slot]),nonlava_death=bool(death[slot]),timeout=bool(timeout[slot]),
                    **{x.name.lower():int(ach[slot,x.value]) for x in Achievement})
                trials.append(row)
                if row['all_22'] and world not in winners:winners[world]=row
        (a.out/f'dataset{dataset}_control_trials.json').write_text(json.dumps(trials,indent=2)+'\n')
        traced=build_evaluator(env,net,params,cfg,**control_settings,trace=True)
        for world,row in winners.items():
            eid=row['episode'];first=eid//a.batch*a.batch;slot=eid-first
            result=jax.device_get(traced(jnp.arange(first,first+a.batch),jnp.int32(slot)));matched(result,row,slot)
            entry=record_trace(result,slot,world,control_seed,eid,f'dataset{dataset}_success{eid}',row['length'],False)
            entry['dataset']=dataset;entry['all_22_outcome_verified']=True;report['controls'].append(entry);save()
            print('Matched successful control',dataset,world,flush=True)
        report['sources'][-1]['worlds_with_successful_control']=len(winners)
        report['sources'][-1]['worlds_without_successful_control']=[int(r['world_seed']) for r in failures if int(r['world_seed']) not in winners]
        save()
    report['completed']=time.time();report['complete']=True;save()
    lock.close()


if __name__=='__main__':main()

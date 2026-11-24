"""Replay the three saved diamond5 RNN videos and audit memory/rewards/resources."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concept_mapping.render_rnn_ppo_episodes import (
    _load_wandb_config,_restore_train_state,ActorCriticRNN,ScannedRNN)
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement
from wrappers import ClassicDeathPenaltyWrapper,ClassicDiamondRewardWrapper

def main():
    run='wandb/run-20261030_005629-blyguexq'
    folder=Path('concept_mapping/runs/classic_rnn_death_penalty0p5_diamond5_10b_seed50/videos_final10b_seed8050')
    cfg=_load_wandb_config(run)
    env=make_craftax_env_from_name(cfg['ENV_NAME'],auto_reset=False)
    ep=env.default_params
    env=ClassicDiamondRewardWrapper(ClassicDeathPenaltyWrapper(env,.5),5.)
    net=ActorCriticRNN(17,cfg)
    train,_=_restore_train_state(run,cfg,net,env,ep,9999941632)
    reset=jax.jit(env.reset);step_fn=jax.jit(env.step)
    @jax.jit
    def policy(h,obs,done,key):
        h,pi,v=net.apply(train.params,h,(obs[None,None],done.reshape(1,1)))
        key,k=jax.random.split(key)
        return h,pi.sample(seed=k)[0,0],v[0,0],key,pi.logits[0,0]
    summaries=[]
    for episode in range(3):
        records=list(csv.DictReader((folder/f'episode_{episode:02d}_actions.txt').open()))
        key=jax.random.fold_in(jax.random.PRNGKey(8050),episode)
        key,rk,pk=jax.random.split(key,3)
        obs,state=reset(rk,ep)
        initial=ScannedRNN.initialize_carry(1,512);h=initial;done=jnp.asarray(False)
        rows=[];observations=[];logits=[];values=[];hidden_states=[];max_reward_error=0.;max_value_error=0.
        for i,record in enumerate(records):
            observations.append(np.asarray(obs));before=state
            hidden_states.append(h)
            h,action,v,pk,l=policy(h,obs,done,pk)
            assert int(action)==int(record['action']),('action mismatch',episode,i)
            logits.append(np.asarray(l));values.append(float(v))
            key,sk=jax.random.split(key)
            obs,state,reward,done,_=step_fn(sk,state,action,ep)
            max_reward_error=max(max_reward_error,abs(float(reward)-float(record['reward'])))
            max_value_error=max(max_value_error,abs(float(v)-float(record['value'])))
            flags=np.asarray(state.achievements)&~np.asarray(before.achievements)
            expected=flags.sum()+4*flags[Achievement.COLLECT_DIAMOND.value]+.1*float(state.player_health-before.player_health)-.5*bool(done)
            assert abs(float(reward)-expected)<1e-5
            rows.append(dict(step=i,action=record['action_name'],health=int(before.player_health),
                food=int(before.player_food),water=int(before.player_drink),energy=int(before.player_energy),
                recover=float(before.player_recover),sleeping=bool(before.is_sleeping),
                next_health=int(state.player_health),next_food=int(state.player_food),next_water=int(state.player_drink),
                next_energy=int(state.player_energy),next_recover=float(state.player_recover),
                reward=float(reward),value=float(v),done=bool(done),
                diamond_new=bool(flags[Achievement.COLLECT_DIAMOND.value])))
        # Training replays complete sequences; check its output against online inference.
        flags=jnp.zeros((len(rows),1),bool)
        _,pi,vs=net.apply(train.params,initial,(jnp.asarray(np.stack(observations))[:,None],flags))
        logit_error=float(np.max(np.abs(np.asarray(pi.logits)[:,0]-np.stack(logits))))
        sequence_value_error=float(np.max(np.abs(np.asarray(vs)[:,0]-values)))
        probability_error=float(np.max(np.abs(np.asarray(jax.nn.softmax(pi.logits[:,0]))-np.asarray(jax.nn.softmax(jnp.asarray(logits))))))
        # Actual PPO replays 64-step chunks from their factual starting carry.
        chunk_value_error=0.;chunk_probability_error=0.
        for start in range(0,len(rows),64):
            ob=jnp.asarray(np.stack(observations[start:start+64]))[:,None]
            _,chunk_pi,chunk_v=net.apply(train.params,hidden_states[start],(ob,jnp.zeros(ob.shape[:2],bool)))
            chunk_value_error=max(chunk_value_error,float(np.max(np.abs(np.asarray(chunk_v)[:,0]-values[start:start+64]))))
            chunk_probability_error=max(chunk_probability_error,float(np.max(np.abs(np.asarray(jax.nn.softmax(chunk_pi.logits[:,0]))-np.asarray(jax.nn.softmax(jnp.asarray(logits[start:start+64])))))))
        with (folder/f'episode_{episode:02d}_survival_audit.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        first_zero={k:next((r['step'] for r in rows if r[k]==0),None) for k in ('food','water','energy')}
        negatives=[r for r in rows if r['next_health']<r['health']]
        summary=dict(episode=episode,steps=len(rows),all_saved_actions_match=True,
            max_saved_reward_error=max_reward_error,max_saved_value_error=max_value_error,
            max_sequence_vs_online_logit_error=logit_error,max_sequence_vs_online_value_error=sequence_value_error,
            max_sequence_vs_online_probability_error=probability_error,
            max_64step_vs_online_value_error=chunk_value_error,max_64step_vs_online_probability_error=chunk_probability_error,
            first_zero=first_zero,health_loss_transitions=len(negatives),
            health_loss_units=sum(r['health']-r['next_health'] for r in negatives),
            health_loss_with_any_resource_zero=sum(any(r[k]==0 or r['next_'+k]==0 for k in ('food','water','energy')) for r in negatives),
            recovery_accumulator_reset_health_losses=sum(r['recover']<0 and r['next_recover']==0 for r in negatives),
            sleep_actions=sum(r['action']=='SLEEP' for r in rows),
            first_diamond=next((r['step'] for r in rows if r['diamond_new']),None),
            terminal=rows[-1],last_health_losses=negatives[-4:])
        summaries.append(summary);print(json.dumps(summary),flush=True)
        (folder/'survival_audit_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()

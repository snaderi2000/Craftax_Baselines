"""Fresh normal episodes: stone behavior and predictor labels from actual actions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from orbax.checkpoint import CheckpointManager,CheckpointManagerOptions,PyTreeCheckpointer
from craftax.craftax_env import make_craftax_env_from_name
from models.actor_critic import ActorCritic
from wrappers import LogWrapper,AutoResetEnvWrapper,BatchEnvWrapper
from stone_outcome import StoneOutcomeWrapper,predict_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=256)
    p.add_argument('--seed',type=int,default=2050)
    args = p.parse_args()
    manager = CheckpointManager(str(Path(args.run).resolve()/'files/policies'),PyTreeCheckpointer(),CheckpointManagerOptions(create=False))
    step = max(manager.all_steps())
    weights = manager.restore(step)['params']
    net = ActorCritic(17,512)
    env = BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(StoneOutcomeWrapper(
        make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',False)))),32)
    params = env.default_params
    key,rk = jax.random.split(jax.random.PRNGKey(args.seed))
    obs,state = env.reset(rk,params)
    def transition(carry,_):
        key,obs,state = carry
        key,ak,ek = jax.random.split(key,3)
        pi,_ = net.apply(weights,obs)
        actions = pi.sample(seed=ak)
        probability = jax.nn.sigmoid(predict_logits(weights,obs,actions))
        next_obs,next_state,_,done,info = env.step(ek,state,actions,params)
        info['prediction'] = probability
        info['mining_probability'] = jax.nn.softmax(pi.logits_parameter(),axis=-1)[:,5]
        info['chose_mine'] = actions == 5
        return (key,next_obs,next_state),(done,info)
    block = jax.jit(lambda c:jax.lax.scan(transition,c,None,length=64))
    quotas = np.full(32,args.episodes//32)
    quotas[:args.episodes%32] += 1
    completed = np.zeros(32,int)
    length = np.zeros(32,int)
    first = np.full(32,-1,int)
    rows,attempts = [],[]
    opportunities=mined=0
    probability_sum=0.
    carry=(key,obs,state)
    while len(rows)<args.episodes:
        carry,(done,info)=block(carry)
        done=np.asarray(done)
        info=jax.tree_util.tree_map(np.asarray,info)
        for t in range(64):
            for e in range(32):
                if completed[e]>=quotas[e]:continue
                if info['stone_success'][t,e] and first[e]<0:first[e]=length[e]
                if info['stone_attempt'][t,e]:
                    attempts.append(dict(worker=e,episode=int(completed[e]),timestep=int(length[e]),
                        success=bool(info['stone_success'][t,e]),has_pickaxe=bool(info['stone_has_pickaxe'][t,e]),
                        sleeping=bool(info['stone_sleeping'][t,e]),blocked=bool(info['stone_blocked'][t,e]),
                        stone_inventory=int(info['stone_inventory'][t,e]),
                        predicted_success=float(info['prediction'][t,e])))
                if info['stone_actionable'][t,e]:
                    opportunities+=1
                    mined+=int(info['chose_mine'][t,e])
                    probability_sum+=float(info['mining_probability'][t,e])
                length[e]+=1
                if done[t,e]:
                    rows.append(dict(worker=e,episode=int(completed[e]),length=int(length[e]),
                                     first_stone_step=int(first[e]),return_=float(info['returned_episode_returns'][t,e])))
                    completed[e]+=1
                    length[e]=0
                    first[e]=-1
    summary=dict(episodes=len(rows),evaluation_seed=args.seed,checkpoint=step,
        collected_stone=sum(r['first_stone_step']>=0 for r in rows),
        stone_collection_rate=np.mean([r['first_stone_step']>=0 for r in rows]),
        median_first_stone_step=float(np.median([r['first_stone_step'] for r in rows if r['first_stone_step']>=0])) if any(r['first_stone_step']>=0 for r in rows) else None,
        actionable_steps=opportunities,mining_fraction_when_actionable=mined/opportunities if opportunities else None,
        mean_mining_probability_when_actionable=probability_sum/opportunities if opportunities else None,
        observed_attempts=len(attempts),mean_episode_return=float(np.mean([r['return_'] for r in rows])))
    summary['failed_attempts_without_pickaxe']=sum(not r['success'] and not r['has_pickaxe'] for r in attempts)
    for label,name in [(True,'success'),(False,'failure')]:
        samples=[r for r in attempts if r['success']==label]
        summary[f'{name}_attempts']=len(samples)
        summary[f'{name}_prediction_accuracy']=float(np.mean([(r['predicted_success']>=.5)==label for r in samples])) if samples else None
    if attempts:
        summary['prediction_brier_score']=float(np.mean([(r['predicted_success']-r['success'])**2 for r in attempts]))
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('episodes',rows),('attempts',attempts)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            if data:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

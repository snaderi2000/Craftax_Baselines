"""Build the expert-experience pilot memory, without collecting new rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
from orbax.checkpoint import CheckpointManager,CheckpointManagerOptions,PyTreeCheckpointer
from culture import CultureMemory,CultureSelector,encode


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='concept_mapping/runs/master_symbolic_10b_5000episodes')
    p.add_argument('--out',required=True)
    p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--candidates',type=int,default=5000)
    p.add_argument('--fragment_length',type=int,default=20)
    p.add_argument('--max_size',type=int,default=512)
    p.add_argument('--duplicate_threshold',type=float,default=.995)
    p.add_argument('--representation',choices=['critic_hidden','actor_hidden','raw_flattened_observation'],default='critic_hidden')
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--seed',type=int,default=50)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dataset=Path(args.dataset)
    source=json.loads((dataset/'dataset_metadata.json').read_text())
    paths=sorted((dataset/'episodes').glob('episode_*.npz'))
    rng=np.random.default_rng(args.seed)
    paths=[paths[i] for i in sorted(rng.choice(len(paths),min(args.episodes,len(paths)),replace=False))]
    rows=[]
    for path in paths:
        with np.load(path) as d:
            obs=d['observations'];actions=d['actions'];rewards=d['rewards'];dones=d['dones']
            future=np.zeros(len(rewards),np.float32); acc=0.
            for t in range(len(rewards)-1,-1,-1):
                acc=float(rewards[t])+args.gamma*acc*(1-int(dones[t]));future[t]=acc
            for t in range(0,len(actions),5):
                length=min(args.fragment_length,len(actions)-t)
                ao=np.zeros((args.fragment_length,obs.shape[-1]),np.float32)
                aa=np.zeros(args.fragment_length,np.int32);rr=np.zeros(args.fragment_length,np.float32)
                mask=np.arange(args.fragment_length)<length
                ao[:length]=obs[t:t+length];aa[:length]=actions[t:t+length];rr[:length]=rewards[t:t+length]
                final=obs[t+length] if t+length<len(obs) else d['terminal_observation']
                rows.append((obs[t].copy(),ao,aa,rr,mask,float(future[t]),final.copy(),
                             int(d['episode_id']),t,length))
    if len(rows)>args.candidates:
        rows=[rows[i] for i in rng.choice(len(rows),args.candidates,replace=False)]
    columns=list(zip(*rows))
    arrays=dict(observations=np.stack(columns[0]),fragment_observations=np.stack(columns[1]),
                actions=np.stack(columns[2]),rewards=np.stack(columns[3]),mask=np.stack(columns[4]),
                returns=np.asarray(columns[5],np.float32),final_observations=np.stack(columns[6]),
                episode_id=np.asarray(columns[7]),timestep=np.asarray(columns[8]),
                lengths=np.asarray(columns[9]),generation_id=np.zeros(len(rows),np.int32))
    manager=CheckpointManager(str(Path(source['run_path'])/'files/policies'),PyTreeCheckpointer(),
                              CheckpointManagerOptions(create=False,max_to_keep=1))
    params=manager.restore(source['checkpoint'])['params']
    encoder_path=out/'encoder_params.npz'
    np.savez_compressed(encoder_path,**{f'{k}_{leaf}':np.asarray(value) for k,v in params['params'].items()
                                      for leaf,value in v.items()})
    encoder=jax.jit(lambda x:encode(params,x,args.representation))
    for dest,key in [('embeddings','observations'),('final_embeddings','final_observations')]:
        arrays[dest]=np.concatenate([np.asarray(encoder(arrays[key][i:i+256])) for i in range(0,len(rows),256)])
    scores,outcomes,novelty=CultureSelector.score_fragment(arrays['returns'],arrays['embeddings'])
    arrays.update(scores=scores,outcomes=outcomes,novelty=novelty)
    metadata=dict(source_dataset=str(dataset.resolve()),source_policy=source['run_path'],
                  source_checkpoint=source['checkpoint'],representation=args.representation,
                  encoder_path=str(encoder_path.resolve()),episodes=len(paths),candidate_count=len(rows),
                  gamma=args.gamma,seed=args.seed,fragment_length=args.fragment_length,generation_id=0,
                  expert_pilot=True,ancestral_data_cost_not_in_student_budget=True)
    memory=CultureMemory(arrays,metadata)
    memory.consolidate(args.max_size,args.duplicate_threshold,args.max_size)
    memory.metadata['memory_size']=len(memory.arrays['actions'])
    memory.save(out/'culture.npz')
    (out/'metadata.json').write_text(json.dumps(memory.metadata,indent=2)+'\n')
    print(json.dumps(memory.metadata,indent=2),flush=True)


if __name__=='__main__':main()

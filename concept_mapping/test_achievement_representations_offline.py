"""Frozen achievement-event retrieval on an episode-disjoint precollected batch.

No PPO/encoder training, clustering fit, simulator stepping or dataset mutation.
Semantic labels select a balanced evaluation reference bank and score retrieval;
they never enter the observation encoder. This is a labeled retrieval diagnostic,
not an implementation of Achievement Distillation or label-free discovery.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np

DATASETS=(ROOT/'concept_mapping/runs/master_symbolic_10b_5000episodes',
          ROOT/'concept_mapping/runs/master_symbolic_10b_5000episodes_seed51')


def extract_events(observations, achievements, actions, terminal_observation, terminal_achievements):
    """Keep single first-completion events; final action uses actual terminal state."""
    obs=np.asarray(observations,np.float32)
    flags=np.asarray(achievements,bool)
    final_flags=np.asarray(terminal_achievements,bool)
    if obs.ndim!=2 or len(obs)==0 or flags.shape!=(len(obs),22) or len(actions)!=len(obs):
        raise ValueError('Malformed episode')
    after_flags=np.concatenate([flags[1:],final_flags[None]],axis=0)
    if np.any(flags & ~after_flags):raise ValueError('Achievement flags are not monotonic')
    unlocked=after_flags & ~flags
    sizes=unlocked.sum(axis=1)
    steps=np.flatnonzero(sizes==1)
    labels=unlocked[steps].argmax(axis=1).astype(np.int16)
    after=np.empty((len(steps),obs.shape[1]),np.float32)
    continuing=steps+1<len(obs)
    after[continuing]=obs[steps[continuing]+1]
    after[~continuing]=np.asarray(terminal_observation,np.float32)
    return dict(before=obs[steps],after=after,labels=labels,actions=np.asarray(actions,np.int32)[steps],
                steps=steps,compound_events=int((sizes>1).sum()),terminal_events=int((~continuing).sum()))


def normalize(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def select_reference(labels, split, cap, seed):
    rng=np.random.default_rng(seed)
    selected=[]
    for label in range(22):
        candidates=np.flatnonzero((labels==label)&(split==0))
        selected.extend(rng.choice(candidates,min(cap,len(candidates)),replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected,np.int32)


def _load_episode(job):
    dataset,episode,split,path=job
    with np.load(path,allow_pickle=False) as d:
        result=extract_events(d['observations'],d['achievements'],d['actions'],
                              d['terminal_observation'],d['terminal_achievements'])
    result.update(dataset=dataset,episode=episode,split=split,path=str(path))
    return result


def read_batch(roots, episodes_per_dataset, seed, workers, out):
    metadata=[json.loads((p/'dataset_metadata.json').read_text()) for p in roots]
    for m in metadata:
        if m['run_path']!=metadata[0]['run_path'] or m['checkpoint']!=metadata[0]['checkpoint']:
            raise ValueError('Dataset policies/checkpoints differ')
    jobs=[]
    for dataset,p in enumerate(roots):
        with (p/'episode_manifest.csv').open() as f:entries=list(csv.DictReader(f))
        if len(entries)<episodes_per_dataset:raise ValueError('Insufficient episodes')
        rng=np.random.default_rng(np.random.SeedSequence([seed,dataset]))
        chosen=rng.choice(len(entries),episodes_per_dataset,replace=False)
        for n,i in enumerate(chosen):
            row=entries[i];path=p/row['file']
            if not path.is_file():raise FileNotFoundError(path)
            jobs.append((dataset,int(row['episode_id']),0 if n<int(.7*episodes_per_dataset) else 1,path))
    with (out/'episode_selection.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['dataset','episode','split','source_npz'])
        w.writerows((d,e,'reference' if s==0 else 'test',str(p)) for d,e,s,p in jobs)
    chunks=[];compound=terminal=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,r in enumerate(pool.map(_load_episode,jobs),1):
            count=len(r['labels'])
            r['episode_uid']=np.full(count,r['dataset']*100000+r['episode'],np.int32)
            r['dataset_id']=np.full(count,r['dataset'],np.int16)
            r['episode_id']=np.full(count,r['episode'],np.int32)
            r['split_id']=np.full(count,r['split'],np.int8)
            compound+=r['compound_events'];terminal+=r['terminal_events'];chunks.append(r)
            if n%100==0:print(f'Extracted {n}/{len(jobs)} episodes',flush=True)
    keys=('before','after','labels','actions','steps','episode_uid','dataset_id','episode_id','split_id')
    bank={k:np.concatenate([r[k] for r in chunks]) for k in keys}
    if set(bank['episode_uid'][bank['split_id']==0])&set(bank['episode_uid'][bank['split_id']==1]):
        raise AssertionError('Episode leakage')
    np.savez_compressed(out/'event_bank.npz',**bank)
    return bank,metadata,dict(episodes=len(jobs),reference_episodes=sum(s==0 for _,_,s,_ in jobs),
        test_episodes=sum(s==1 for _,_,s,_ in jobs),events=len(bank['labels']),
        compound_transitions_excluded=compound,terminal_events_included=terminal)


def actor_hidden(params,obs,activation):
    import jax
    import jax.numpy as jnp
    p=params['params'] if 'params' in params else params
    nonlinearity=jax.nn.relu if activation=='relu' else jnp.tanh
    h=obs
    for name in ('Dense_0','Dense_1','Dense_2'):
        h=nonlinearity(h@p[name]['kernel']+p[name]['bias'])
    return h,h@p['Dense_3']['kernel']+p['Dense_3']['bias']


def embed_bank(bank,metadata,batch_size,seed):
    import jax
    import jax.numpy as jnp
    from orbax.checkpoint import CheckpointManager,CheckpointManagerOptions,PyTreeCheckpointer
    from concept_mapping.render_ppo_episodes import _load_wandb_config,_init_network
    from concept_mapping.critic_representation import critic_pre_value_representation
    from craftax.craftax_env import make_craftax_env_from_name
    run=Path(metadata['run_path'])
    if not run.is_absolute():run=ROOT/run
    config=_load_wandb_config(str(run))
    activation=config.get('ACTIVATION','tanh')
    env=make_craftax_env_from_name(config['ENV_NAME'],auto_reset=False)
    network=_init_network(config,env,env.default_params)
    manager=CheckpointManager(str(run/'files/policies'),PyTreeCheckpointer(),CheckpointManagerOptions(create=False))
    params=manager.restore(metadata['checkpoint'])['params']
    random_params=network.init(jax.random.PRNGKey(seed+1000),jnp.zeros((1,1345),jnp.float32))
    @jax.jit
    def embeddings(obs):
        actor,_=actor_hidden(params,obs,activation)
        critic,_=critic_pre_value_representation(params,obs,activation)
        random_actor,_=actor_hidden(random_params,obs,activation)
        return actor,critic,random_actor
    probe=jnp.asarray(bank['before'][:32])
    pi,value=network.apply(params,probe)
    _,logits=actor_hidden(params,probe,activation)
    _,reconstructed=critic_pre_value_representation(params,probe,activation)
    actor_error=float(jnp.max(jnp.abs(jax.nn.softmax(logits)-jax.nn.softmax(pi.logits))))
    critic_error=float(jnp.max(jnp.abs(value-reconstructed)))
    if actor_error>1e-5 or critic_error>1e-5:raise AssertionError('Hidden extraction validation failed')
    differences=[[],[],[]]
    for i in range(0,len(bank['before']),batch_size):
        before=embeddings(jnp.asarray(bank['before'][i:i+batch_size]))
        after=embeddings(jnp.asarray(bank['after'][i:i+batch_size]))
        for k,(a,b) in enumerate(zip(before,after)):differences[k].append(np.asarray(b-a))
    inventory=(bank['after'][:,1323:1335]-bank['before'][:,1323:1335])*10
    action=np.eye(17,dtype=np.float32)[bank['actions']]
    features=dict(action_only=action,inventory_delta=inventory,
        action_inventory_delta=np.concatenate([action,normalize(inventory)],axis=1),
        raw_observation_delta=bank['after']-bank['before'])
    features.update({name:np.concatenate(parts) for name,parts in zip(
        ('actor_delta','critic_delta','random_actor_delta'),differences)})
    return features,dict(actor_probability_max_error=actor_error,critic_value_max_error=critic_error)


def retrieval_scores(features,bank,reference,test,batch_size):
    """Expected top-1 correctness under uniform tie-breaking; zeros stay zeros.

    Squared Euclidean distance between normalized vectors handles zero-change
    inventory events consistently. Exact/near ties are averaged rather than
    allowing arbitrary array order to favor one achievement.
    """
    import jax
    import jax.numpy as jnp
    x=normalize(features)
    gallery=jnp.asarray(x[reference]);gallery_labels=bank['labels'][reference]
    onehot=jax.nn.one_hot(jnp.asarray(gallery_labels),22)
    @jax.jit
    def search(query):
        distances=jnp.maximum((query**2).sum(1)[:,None]+(gallery**2).sum(1)[None]-
            2*jnp.matmul(query,gallery.T,precision=jax.lax.Precision.HIGHEST),0)
        nearest=distances.min(axis=1)
        ties=distances<=nearest[:,None]+1e-6
        probabilities=ties.astype(jnp.float32)@onehot/jnp.maximum(1,ties.sum(axis=1))[:,None]
        return probabilities,jnp.argmin(distances,axis=1),ties.sum(axis=1)
    probs=[];nearest=[];ties=[]
    for i in range(0,len(test),batch_size):
        p,n,t=search(jnp.asarray(x[test[i:i+batch_size]]))
        probs.append(np.asarray(p));nearest.append(reference[np.asarray(n)]);ties.append(np.asarray(t))
    probs=np.concatenate(probs);nearest=np.concatenate(nearest);ties=np.concatenate(ties)
    correct=probs[np.arange(len(test)),bank['labels'][test]]
    return correct,probs,nearest,ties,float((np.linalg.norm(features[test],axis=1)<1e-8).mean())


def bootstrap_micro(scores,episode_uids,seed,samples=2000):
    unique,inverse=np.unique(episode_uids,return_inverse=True)
    counts=np.bincount(inverse)
    totals=np.stack([np.bincount(inverse,weights=s,minlength=len(unique)) for s in scores])
    rng=np.random.default_rng(seed)
    draws=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=samples)
    estimates=(draws@totals.T)/(draws@counts)[:,None]
    return np.percentile(estimates,[2.5,97.5],axis=0).T,estimates


def render_confusions(confusions,names,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})
    np.savez_compressed(out/'retrieval_confusions.npz',**confusions)
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    for ax,method in zip(axes,('action_inventory_delta','critic_delta')):
        im=ax.imshow(confusions[method],vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(22),names,rotation=90,fontsize=7)
        ax.set_yticks(range(22),names,fontsize=7)
        ax.set_title(method.replace('_',' '));ax.set_xlabel('Retrieved achievement');ax.set_ylabel('True achievement')
    fig.subplots_adjust(left=.12,bottom=.3,right=.88,wspace=.5)
    cax=fig.add_axes([.92,.30,.015,.55])
    fig.colorbar(im,cax=cax,label='Retrieval probability')
    for extension in ('pdf','png'):fig.savefig(out/f'retrieval_confusions.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'concept_mapping/runs/achievement_representation_offline_10b_batch2000')
    p.add_argument('--episodes_per_dataset',type=int,default=1000)
    p.add_argument('--seed',type=int,default=50)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch_size',type=int,default=512)
    p.add_argument('--reference_per_achievement',type=int,default=128)
    args=p.parse_args();out=args.out
    if out.exists():raise RuntimeError(f'Refusing overwrite {out}')
    out.mkdir(parents=True)
    started=time.time()
    def progress(stage,**extra):(out/'progress.json').write_text(json.dumps(dict(stage=stage,elapsed_seconds=time.time()-started,**extra),indent=2)+'\n')
    progress('extracting')
    bank,metadata,description=read_batch(DATASETS,args.episodes_per_dataset,args.seed,args.workers,out)
    reference=select_reference(bank['labels'],bank['split_id'],args.reference_per_achievement,args.seed+1)
    test=np.flatnonzero(bank['split_id']==1)
    names=[s.lower() for s in metadata[0]['achievement_names']]
    print('Batch',description,'reference events',len(reference),'test events',len(test),flush=True)
    progress('embedding',**description)
    features,validation=embed_bank(bank,metadata[0],args.batch_size,args.seed)
    np.savez_compressed(out/'embeddings.npz',**features,reference_indices=reference,test_indices=test)
    scores=[];methods=[];summaries=[];per_class=[];confusions={}
    for method,feature in features.items():
        progress('retrieval',method=method)
        correct,probs,nearest,ties,zero_fraction=retrieval_scores(feature,bank,reference,test,args.batch_size)
        scores.append(correct);methods.append(method)
        class_scores=[]
        confusion=np.zeros((22,22))
        for c,name in enumerate(names):
            mask=bank['labels'][test]==c
            n=int(mask.sum());gallery_n=int((bank['labels'][reference]==c).sum())
            accuracy=float(correct[mask].mean()) if n else None
            if accuracy is not None:class_scores.append(accuracy)
            if n:confusion[c]=probs[mask].mean(axis=0)
            per_class.append(dict(method=method,achievement=name,reference_n=gallery_n,test_n=n,accuracy=accuracy))
        confusions[method]=confusion
        summaries.append(dict(method=method,micro_accuracy=float(correct.mean()),macro_accuracy=float(np.mean(class_scores)),
            test_n=len(test),reference_n=len(reference),zero_test_fraction=zero_fraction,mean_tie_count=float(ties.mean())))
        with (out/f'{method}_per_event.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['test_event','episode_uid','step','achievement','correct_probability',
                'nearest_reference_event','nearest_reference_episode','nearest_reference_achievement','tie_count'])
            for j,idx in enumerate(test):w.writerow([idx,bank['episode_uid'][idx],bank['steps'][idx],names[bank['labels'][idx]],
                float(correct[j]),int(nearest[j]),int(bank['episode_uid'][nearest[j]]),names[bank['labels'][nearest[j]]],int(ties[j])])
        print(method,summaries[-1],flush=True)
    intervals,draws=bootstrap_micro(scores,bank['episode_uid'][test],args.seed+2)
    for i,s in enumerate(summaries):s['micro_episode_bootstrap_95_ci']=intervals[i].tolist()
    baseline_index=methods.index('action_inventory_delta')
    for i,s in enumerate(summaries):
        s['micro_difference_vs_action_inventory']=float(scores[i].mean()-scores[baseline_index].mean())
        s['difference_episode_bootstrap_95_ci']=np.percentile(draws[:,i]-draws[:,baseline_index],[2.5,97.5]).tolist()
    with (out/'per_achievement.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_class[0]));w.writeheader();w.writerows(per_class)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    payload=dict(**description,analysis_seed=args.seed,checkpoint=metadata[0]['checkpoint'],run_path=metadata[0]['run_path'],
        collection_seeds=[m['seed'] for m in metadata],encoder_trained=False,policy_modified=False,new_rollouts=False,
        labels_used_for='Event boundary identification, reference balancing, and retrieval scoring only; not encoder training.',
        reference_per_achievement=args.reference_per_achievement,split='70/30 episode split within each collection seed',
        retrieval='Top-1 normalized-vector Euclidean retrieval, average correctness across ties within 1e-6',
        bootstrap='2000 paired episode-level samples; intervals concern rollout episodes of one frozen policy, not training seeds',
        validation=validation,methods=summaries,elapsed_seconds=time.time()-started,
        limitations=['Observational transitions include incidental scene and health changes.',
            'These are already-symbolic observations; action/inventory shortcuts may identify achievements.',
            'Retrieval does not prove causal prerequisites, hierarchy, future prediction, or policy improvement.',
            'No test on ordinary non-achievement transitions; no claim of automatic event detection.'])
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    render_confusions(confusions,names,out)
    progress('complete',**description)
    print('Complete:',out,flush=True)


if __name__=='__main__':main()

"""Probe the V5 actor input for last-observed water, using fresh on-policy worlds."""
import argparse
import json
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

from evaluate_reward_shaped_ppo import jax, jnp  # Shared Optax/JAX compatibility setup.
from craftax.craftax_env import make_craftax_env_from_name
from models.actor_critic import ActorCritic
from models.actor_critic_map_mlp import GlobalMapReader, map_mlp_network
from symbolic_map_memory import LOCAL_SIZE, with_map_memory

RUNS = Path(__file__).resolve().parent / 'runs'
STUDY = RUNS / 'spatial_regions_from75m_lr6e5_300m/map_mlp'
OUT = STUDY.parent / 'out_of_view_water_probe'
SEED = 1118427
BATCH = 64
WORLDS = 1024
TRAIN = 768
TICK = 1024


def model():
    cfg = json.loads((STUDY/'inference_config.json').read_text())
    assert cfg['MAP_SPATIAL_REGIONS'] and not cfg['MAP_FULL_TERRAIN']
    path = Path(json.loads((STUDY/'progress.json').read_text())['final_checkpoint'])
    params = ocp.PyTreeCheckpointer().restore(str(path/'default'))['params']
    return cfg, path, params


def readers(cfg, params):
    reader = GlobalMapReader(cfg['LAYER_SIZE'], features_only=True, spatial_regions=True)
    read_params = {'params':params['params']['global_reader']}
    heads = ActorCritic(17, cfg['LAYER_SIZE'], 'tanh')
    head_params = {'params':params['params']['actor_critic']}
    net = map_mlp_network(cfg)

    def read(obs):
        regions, captured = reader.apply(read_params, obs[LOCAL_SIZE:],
            capture_intermediates=lambda module, method: module.name == 'patches' and method == '__call__')
        patches = jax.nn.relu(captured['intermediates']['patches']['__call__'][0])
        joined = jnp.concatenate((obs[:LOCAL_SIZE], regions.ravel()))
        reconstructed, value = heads.apply(head_params, joined)
        actual, actual_value = net.apply(params, obs)
        error = jnp.maximum(jnp.abs(actual.logits-reconstructed.logits).max(), jnp.abs(value-actual_value))
        return patches, joined, error
    return read


def collect():
    cfg, path, params = model()
    net = map_mlp_network(cfg)
    env = with_map_memory(make_craftax_env_from_name(cfg['ENV_NAME'], False), cfg)
    read = readers(cfg, params)

    def rollout(world):
        reset, key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEED), world))
        obs, s = env.reset(reset, env.default_params)
        initial = s.env_state.map.astype(jnp.uint8)
        def advance(carry):
            obs, s, key, tick, _ = carry
            key, act, dyn = jax.random.split(key, 3)
            policy, _ = net.apply(params, obs)
            nxt, ns, _, done, _ = env.step(dyn, s, policy.sample(seed=act), env.default_params)
            # Keep the final live pre-action state if the episode ends early.
            return (jnp.where(done, obs, nxt), jax.tree_util.tree_map(lambda a,b:jnp.where(done,a,b),s,ns),
                    key,tick+1,done)
        obs,s,*_ = jax.lax.while_loop(lambda c:(c[3]<TICK) & ~c[4], advance,
                                         (obs,s,key,jnp.int32(0),jnp.bool_(False)))
        patches, joined, error = read(obs)
        return dict(memory=s.terrain, last_seen=s.last_seen.astype(jnp.int16), pos=s.position, tick=s.elapsed,
                    live=joined[:LOCAL_SIZE], regions=joined[LOCAL_SIZE:].reshape(8,8,32),
                    patches=patches, error=error, initial=initial)
    collect_batch = jax.jit(jax.vmap(rollout))
    chunks = []
    for start in range(0, WORLDS, BATCH):
        chunks.append(jax.device_get(collect_batch(jnp.arange(start,start+BATCH,dtype=jnp.uint32))))
        if start % 256 == 0:
            print(f'Collected {start+BATCH}/{WORLDS}', flush=True)
    data = {k:np.concatenate([b[k] for b in chunks]) for k in chunks[0]}
    assert len({m.tobytes() for m in data.pop('initial')}) == WORLDS
    assert np.max(data['error']) < 1e-4, data['error'].max()
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'snapshots.npz', **data, checkpoint=str(path))
    print('Snapshots complete', flush=True)


def blocks(grid):
    return grid.reshape(-1,8,8,8,8).transpose(0,1,3,2,4).reshape(-1,64,64)


def paired_scores(labels, scores, mask):
    """Probability a water tile outranks a known dry tile from the very same region."""
    from scipy.stats import rankdata
    n_pos = (labels & mask).sum(-1)
    n_neg = (~labels & mask).sum(-1)
    ranks = rankdata(np.where(mask, scores, np.inf), axis=-1)
    wins = (ranks*(labels & mask)).sum(-1) - n_pos*(n_pos+1)/2
    pairs = n_pos*n_neg
    return wins, pairs


def interval(hits, count):
    rng = np.random.default_rng(41907)
    samples = rng.integers(len(hits), size=(2000,len(hits)))
    total = count[samples].sum(1)
    draws = hits[samples].sum(1)[total>0] / total[total>0]
    return dict(score=float(hits.sum()/count.sum()), interval95=np.quantile(draws,[.025,.975]).tolist(),
                worlds=int((count>0).sum()), denominator=int(count.sum()))


def score_set(y, scores, mask):
    from scipy.stats import rankdata
    actual = y[mask]; predicted=scores[mask]
    pos=actual.sum(); neg=len(actual)-pos
    auc=(rankdata(predicted)[actual].sum()-pos*(pos+1)/2)/(pos*neg)
    wins, pairs = paired_scores(y,scores,mask)
    return dict(tiles=int(mask.sum()),water=int(pos),pooled_auc=float(auc),
                region_pairs=interval(wins.sum(1),pairs.sum(1)))


def train_probes():
    import flax.linen as nn
    import optax
    from flax.training.train_state import TrainState
    with np.load(OUT/'snapshots.npz') as archive:
        d=dict(archive)
    assert len(d['memory'])==WORLDS
    row,col=np.indices((64,64))
    delta_r=row[None]-d['pos'][:,0,None,None]
    delta_c=col[None]-d['pos'][:,1,None,None]
    fov=(abs(delta_r)<=3)&(abs(delta_c)<=4)
    known=d['last_seen']>=0
    water=d['memory']==4  # the stored code is BlockType.WATER (3) plus one
    assert np.array_equal((water & fov).sum((1,2)),d['live'][:,:1323].reshape(-1,63,21)[:,:,3].sum(1))
    labels=blocks(water)
    eligible=blocks(known)&~blocks(fov).any(-1,keepdims=True)
    old=blocks(d['tick'][:,None,None]-d['last_seen'])>=100
    no_visible=~(water&fov).any((1,2))
    patch=d['patches'].reshape(-1,8,2,8,2,8).transpose(0,1,3,2,4,5).reshape(-1,64,32)
    final=d['regions'].reshape(-1,64,32)
    coords=np.stack(np.unravel_index(np.arange(64),(8,8)),axis=-1).astype(np.float32)*8+3.5
    geometry=np.concatenate((np.broadcast_to(coords[None],(WORLDS,64,2)),
                             coords[None]-d['pos'][:,None]),axis=-1).astype(np.float32)/63.
    trw,trb=np.nonzero(eligible[:TRAIN].any(-1))
    testw,testb=np.indices((WORLDS-TRAIN,64)).reshape(2,-1)
    testw=testw+TRAIN
    class_weight=(eligible[:TRAIN]&~labels[:TRAIN]).sum()/(eligible[:TRAIN]&labels[:TRAIN]).sum()

    class Probe(nn.Module):
        @nn.compact
        def __call__(self,live,spatial,geo):
            live=nn.relu(nn.Dense(32)(live))
            memory=nn.relu(nn.Dense(96)(jnp.concatenate((spatial,geo),-1)))
            joint=nn.relu(nn.Dense(96)(jnp.concatenate((memory,live,geo),-1)))
            return nn.Dense(64)(joint)+nn.Dense(64)(spatial)

    probe=Probe()
    @jax.jit
    def update(state,live,x,geo,y,mask):
        def loss(p):
            scores=probe.apply(p,live,x,geo)
            weights=mask*jnp.where(y,class_weight,1.)
            ce=optax.sigmoid_binary_cross_entropy(scores,y.astype(scores.dtype))
            return (ce*weights).sum()/jnp.maximum(weights.sum(),1.)
        return state.apply_gradients(grads=jax.grad(loss)(state.params))
    predict=jax.jit(probe.apply)
    outputs={}
    paired={}
    preds={}
    for name, spatial, use_live in [('local_only',final*0,True),('first_conv_plus_local',patch,True),
                                    ('final_conv_only',final,False),('concatenated',final,True)]:
        avg=spatial[trw,trb].mean(0)
        std=spatial[trw,trb].std(0).clip(.001)
        feature=(spatial-avg)/std
        live=d['live'] if use_live else np.zeros_like(d['live'])
        state=TrainState.create(apply_fn=probe.apply,params=probe.init(jax.random.PRNGKey(147),
                    live[:1],feature[0,:1],geometry[0,:1]),tx=optax.adam(.0009))
        rng=np.random.default_rng(5047)
        for epoch in range(48):
            order=rng.permutation(len(trw))
            for ix in np.array_split(order, int(np.ceil(len(order)/256))):
                w,b=trw[ix],trb[ix]
                state=update(state,live[w],feature[w,b],geometry[w,b],labels[w,b],eligible[w,b])
        def apply(w,b):
            return np.asarray(predict(state.params,live[w],feature[w,b],geometry[w,b]))
        scores=np.concatenate([apply(testw[ix],testb[ix]) for ix in np.array_split(np.arange(len(testw)),64)])
        scores=scores.reshape(-1,64,64)
        preds[name]=scores
        targets=labels[TRAIN:]
        subset=eligible[TRAIN:]
        masks=dict(all=subset,water_not_visible=subset & no_visible[TRAIN:,None,None],
                   age100=subset & old[TRAIN:])
        outputs[name]={key:score_set(targets,scores,mask) for key,mask in masks.items()}
        wins,pairs=paired_scores(targets,scores,subset)
        paired[name]=wins.sum(1),pairs.sum(1)
        print(name, outputs[name]['all'], flush=True)
        if name=='concatenated':
            outputs['water_grass_swap']=swap_test(d,state.params,avg,std,geometry,predict,eligible)
    a, count = paired['concatenated']
    deltas={name:interval(a-h, count) for name,(h,_) in paired.items() if name!='concatenated'}
    write_summary=dict(checkpoint=str(d['checkpoint']),world_seed=SEED,training_worlds=TRAIN,
                 held_out_worlds=WORLDS-TRAIN,snapshot_max_tick=TICK,train_regions=len(trw),
                 heldout_without_visible_water=int(no_visible[TRAIN:].sum()),epochs=48,
                 max_actor_logit_or_value_reconstruction_error=float(d['error'].max()),
                 results=outputs,paired_difference_from_other_probe=deltas,raw_remembered_tile_ceiling=1.)
    (OUT/'results.json').write_text(json.dumps(write_summary,indent=2)+'\n')
    np.savez_compressed(OUT/'heldout_predictions.npz',labels=labels[TRAIN:],eligible=eligible[TRAIN:],
                        no_visible_water=no_visible[TRAIN:],age100=old[TRAIN:],**preds)


def swap_test(d,weights,avg,std,geometry,predict,eligible):
    cfg,path,params=model()
    reader=GlobalMapReader(cfg['LAYER_SIZE'], features_only=True,spatial_regions=True)
    extract=jax.jit(lambda x:reader.apply({'params':params['params']['global_reader']},x))
    rng=np.random.default_rng(987)
    # One pair per eligible held-out world. Preserve the number of water tiles in every region.
    world=[]; region=[]; water_slot=[]; grass_slot=[]
    tile=blocks(d['memory'])
    for w in range(TRAIN,WORLDS):
        water=(tile[w]==4)&eligible[w]; grass=(tile[w]==3)&eligible[w]
        possible=np.flatnonzero(water.any(-1)&grass.any(-1))
        if not len(possible):continue
        b=int(rng.choice(possible))
        world.append(w);region.append(b)
        water_slot.append(rng.choice(np.flatnonzero(water[b])));grass_slot.append(rng.choice(np.flatnonzero(grass[b])))
    w,b,wi,gi=map(np.asarray,(world,region,water_slot,grass_slot))
    rr,cc=np.divmod(b,8)
    age=np.log1p(np.where(d['last_seen'][w]>=0,d['tick'][w,None,None]-d['last_seen'][w],10000))/np.log(10001.)
    def maps(terrain):
        x=np.concatenate((terrain.reshape(len(w),-1),age.reshape(len(w),-1),d['pos'][w]/63.),1).astype('float32')
        return np.concatenate([np.asarray(extract(chunk)) for chunk in np.array_split(x,4)]).reshape(-1,64,32)
    base=maps(d['memory'][w])
    assert np.allclose(base,d['regions'][w].reshape(-1,64,32),atol=1e-4)
    terrain=d['memory'][w].copy()
    rows=np.arange(len(w))
    terrain[rows,rr*8+wi//8,cc*8+wi%8]=3
    terrain[rows,rr*8+gi//8,cc*8+gi%8]=4
    def margin(f):
        s=np.asarray(predict(weights,d['live'][w],(f[rows,b]-avg)/std,geometry[w,b]))
        return s[rows,wi]-s[rows,gi]
    before=margin(base); after=margin(maps(terrain))
    return dict(worlds=len(w),both_sites_in_same_offscreen_region=True,
                correct_direction=interval((before>after).astype(float)+.5*(before==after),np.ones(len(w))),
                original_preference=float((before>0).mean()),swapped_preference=float((after<0).mean()),
                correct_before_and_after=float(((before>0)&(after<0)).mean()))


def plot_results():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.special import expit
    report=json.loads((OUT/'results.json').read_text())
    with np.load(OUT/'heldout_predictions.npz') as p:
        y, valid, logits = p['labels'], p['eligible'], p['concatenated']
        # First held-out world with enough off-screen water and none in the live view.
        i=int(np.flatnonzero(p['no_visible_water'] & ((y & valid).sum((1,2)) >= 32))[0])
        def grid(data):
            return data[i].reshape(8,8,8,8).transpose(0,2,1,3).reshape(64,64)
        truth, mask, scores=grid(y),grid(valid),grid(expit(logits))
    with np.load(OUT/'snapshots.npz') as snap:
        row, col=snap['pos'][TRAIN+i]
        tick=snap['tick'][TRAIN+i]
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.6),gridspec_kw={'width_ratios':[1.2,1,1]},layout='constrained')
    names=['local_only','first_conv_plus_local','final_conv_only','concatenated']
    display=['Local view only','First CNN layer + local','Final CNN layer only','Final vector + local' ]
    display[-1]='Final concatenated vector'
    for j,key in enumerate(names):
        m=report['results'][key]['all']['region_pairs']
        val=m['score']; lo,hi=m['interval95']
        axes[0].errorbar(val,j,xerr=[[val-lo],[hi-val]],capsize=3,fmt='o',color='#187677')
        axes[0].text(hi+.012,j,f'{val:.1%}',va='center',fontsize=9)
    ax=axes[0];ax.axvline(.5,color='#969e9e',ls='--',lw=1)
    ax.set(yticks=range(4),yticklabels=display,xlim=(.45,1.06),ylim=(3.6,-.6),
           xticks=[.5,.6,.7,.8,.9,1.],xlabel='Water tile ranked above dry tile',title='Off-screen tile pairs')
    ax.spines[['top','right','left']].set_visible(False)
    cmap=plt.colormaps['Blues'].copy();cmap.set_bad('#e0dfda')
    for axis,field,title in zip(axes[1:],(truth.astype(float),scores),('Remembered water','Frozen-feature probe score')):
        axis.imshow(np.ma.masked_where(~mask,field),cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        axis.add_patch(Rectangle((col-4.5,row-3.5),9,7,fill=False,ec='#d06a27',lw=1.4))
        axis.set_title(title);axis.set_xticks([]);axis.set_yticks([])
    fig.suptitle('Water memory after spatial compression',weight='bold',fontsize=17)
    fig.supxlabel(f'Intervals resample held-out worlds.   |   Example world {TRAIN+i}, step {tick}. Orange outlines the live view. Gray cells were excluded.',fontsize=10)
    fig.savefig(OUT/'water_probe.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--collect',action='store_true')
    mode.add_argument('--probe',action='store_true')
    mode.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    if args.collect:collect()
    elif args.probe:train_probes()
    else:plot_results()

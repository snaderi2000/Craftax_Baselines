"""Counterbalanced two-landmark writes on actual held-out movement/view sequences."""
import argparse
import csv
from pathlib import Path

import numpy as np

from concept_mapping.probe_plant_diamond_gate import initialize, model, pilot, CODES, REPLACEMENT, OUT, SOURCE
import jax
import jax.numpy as jnp


def periods(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def choose(path,rng):
    # Consecutive sightings of A separated by enough time to encounter B. A and B
    # are each placed on a tile within views the agent actually received.
    steps=len(path)
    for _ in range(32):
        a=path[rng.integers(min(steps,128))]+rng.integers(-2,3,2)
        visible_a=(abs(path-a)<=[3,4]).all(1)
        runs=periods(visible_a)
        gaps=[(lo,end,start,hi) for (lo,end),(start,hi) in zip(runs,runs[1:])
              if end-lo>=5 and start-end>=96 and hi-start>=5 and hi+12<steps]
        rng.shuffle(gaps)
        for lo,end,start,hi in gaps:
            for _ in range(12):
                b=path[rng.integers(end+10,start-20)]+rng.integers(-2,3,2)
                if (a==b).all():continue
                visible_b=(abs(path-b)<=[3,4]).all(1)
                for bstart,bend in periods(visible_b[end:start]):
                    bon=end+bstart
                    boff=end+bend
                    if bon<end+5 or boff-bon<5 or boff+8>=start:continue
                    # Score both while offscreen before A returns, and again soon
                    # after updated A leaves. Reject intervening resightings of A.
                    joint=boff+7
                    if visible_b[boff:joint+1].any():continue
                    post=np.flatnonzero(~visible_a[hi:hi+32]&~visible_b[hi:hi+32])
                    if not post.size:continue
                    after=hi+int(post[0])
                    return a,b,lo,bon,start,joint,after
    return None


def prepare():
    with np.load(SOURCE/"test.npz") as archive:
        tiles,motion,length=[archive[k] for k in ("tiles","motion","length")]
    rows=[]
    for i,n in enumerate(length):
        path=motion[i,:n].astype(int).cumsum(0)
        chosen=choose(path,np.random.default_rng([78913,i]))
        if chosen is None:continue
        a,b,lo,bon,start,joint,after=chosen
        # Border IDs cannot be replaced by landmarks.
        for cell,tick in ((a,lo),(b,bon)):
            row,col=cell-path[tick]
            if tiles[i,tick,(row+3)*9+col+4] in (0,1):break
        else:
            rows.append([i,*a,*b,lo,bon,start,joint,after])
    np.save(OUT/"controlled_schedule.npy",np.asarray(rows))
    print("Matched held-out paths",len(rows),flush=True)


def replay(arm,seed):
    schedule=np.load(OUT/"controlled_schedule.npy")
    with np.load(SOURCE/"test.npz") as arc:
        original,motion=arc["tiles"],arc["motion"]
    with np.load(OUT/f"{arm}_{seed}.npz") as fit:
        p={k:jnp.asarray(fit[k]) for k in initialize(seed)}
    freq=np.asarray((p["direction"][0]-p["direction"][1])/2)
    def decode(state,offset):
        phase=offset@freq
        keys=np.concatenate((np.cos(phase),np.sin(phase)),axis=-1)/np.sqrt(pilot.FREQ)
        return keys@state
    # Event probes can fall on any frame; group them as single-frame blocks
    # by requesting the scan carry for each actual update.
    from concept_mapping.probe_plant_diamond_gate import step_components
    def eval_chunk(h,v,d):
        step=step_components(p,arm=="tile")
        def update(h,x):
            h,_=step(h,x)
            return h,h
        return jax.lax.scan(update,h,(v.swapaxes(0,1),d.swapaxes(0,1)))
    scan=jax.jit(eval_chunk)
    records=[]
    for r in range(2):
        for group in range(0,len(schedule),pilot.BATCH):
            rows=schedule[group:group+pilot.BATCH]
            ids=rows[:,0]
            v=original[ids].copy()
            d=motion[ids]
            path=d.astype(int).cumsum(1)
            for i,(_,ar,ac,br,bc,lo,bon,start,joint,after) in enumerate(rows):
                for cell,begin,code,is_first in [((ar,ac),lo,CODES[r],True),((br,bc),bon,CODES[1-r],False)]:
                    xy=np.array(cell)-path[i]
                    ticks=np.flatnonzero((np.abs(xy)<=[3,4]).all(1)&(np.arange(len(xy))>=begin))
                    col=(xy[ticks,0]+3)*9+xy[ticks,1]+4
                    v[i,ticks,col]=np.where((ticks>=start)&is_first,REPLACEMENT[r],code)
                for t0 in (joint,after):
                    assert (abs(path[i,t0]-[ar,ac])>[3,4]).any() and (abs(path[i,t0]-[br,bc])>[3,4]).any()
            h=jnp.zeros((len(rows),2*pilot.FREQ,17))
            scores=np.empty((len(rows),2,2,17))
            for t in range(0,int(rows[:,-1].max())+1,pilot.CHUNK):
                h,states=scan(h,v[:,t:t+pilot.CHUNK],d[:,t:t+pilot.CHUNK])
                for stage in range(2):
                    ticks=rows[:,-2+stage]
                    ix=np.flatnonzero((ticks>=t)&(ticks<t+pilot.CHUNK))
                    if len(ix):
                        current=np.asarray(states[ticks[ix]-t,ix])
                        for j,i in enumerate(ix):
                            offsets=rows[i,1:5].reshape(2,2)-path[i,ticks[i]]
                            scores[i,stage]=decode(current[j],offsets)
            for row,s in zip(rows,scores):
                pre,post=s
                a,b=CODES[r],CODES[1-r]
                def hit(x):return float(x>0)+.5*float(x==0)
                keep_a=hit(pre[0,a]-pre[1,a])
                add_b=hit(pre[1,b]-pre[0,b])
                revise_a=hit(post[0,REPLACEMENT[r]]-post[0,a])
                keep_b=hit(post[1,b]-post[0,b])
                records.append([row[0],r,keep_a,add_b,keep_a*add_b,revise_a,keep_b,keep_a*add_b*revise_a*keep_b])
    with (OUT/f"{arm}_{seed}_controlled.csv").open("w") as f:
        w=csv.writer(f)
        w.writerow(["world","first_resource","retain_first","add_second","both_before_revision",
                    "revise_first","retain_second","complete_sequence"])
        w.writerows(records)



def summarize():
    import pandas as pd
    schedule = np.load(OUT / "controlled_schedule.npy")
    keys = [(w, r) for w in schedule[:, 0] for r in (0, 1)]
    metrics = ["retain_first", "add_second", "both_before_revision",
               "revise_first", "retain_second", "complete_sequence"]
    arms = []
    for arm in ("scene", "tile"):
        seeds = []
        for seed in pilot.SEEDS:
            df = pd.read_csv(OUT / f"{arm}_{seed}_controlled.csv").set_index(["world", "first_resource"])
            assert df.index.is_unique and set(df.index) == set(keys)
            seeds.append(df.loc[keys, metrics].to_numpy().reshape(len(schedule), 2, len(metrics)))
        arms.append(np.mean(seeds, axis=0))
    scene, tile = arms
    sample = np.random.default_rng(90123).integers(len(schedule), size=(10000,len(schedule)))
    rows = []
    for k, metric in enumerate(metrics):
        for order, x, y in (("plant first", scene[:,0,k], tile[:,0,k]),
                            ("diamond first", scene[:,1,k], tile[:,1,k]),
                            ("both orders", scene[:,:,k].mean(1), tile[:,:,k].mean(1))):
            diff = y-x
            lo, hi = np.quantile(diff[sample].mean(1), [.025,.975])
            rows.append(dict(metric=metric,order=order,worlds=len(schedule),scene=x.mean(),
                             tile=y.mean(),gain=diff.mean(),low=lo,high=hi))
    pd.DataFrame(rows).to_csv(OUT / "controlled_results.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","scene","tile","report"))
    parser.add_argument("--seed",type=int,choices=pilot.SEEDS)
    args=parser.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="report":summarize()
    else:replay(args.action,args.seed)

"""Compare scene-wide and tile-wise write strengths in matched reversible memories."""
import argparse
import csv
import json
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from concept_mapping import probe_spatial_transport as pilot
from concept_mapping.plant_diamond_gate_data import CODES, OUT, REPLACEMENT, SOURCE
from concept_mapping.spatial_transport_queries import LOCAL, STRIDE

NAMES = ("ripe plant","diamond")


def initialize(seed):
    return dict(pilot.init(seed), error_weight=jnp.zeros(17))


def step_components(params, per_tile):
    freq = (params["direction"][0] - params["direction"][1])/2
    directions = jnp.stack([freq,-freq])
    local = pilot.address(jnp.asarray(LOCAL,jnp.float32), freq)
    scale = jnp.linalg.norm(local.T @ local) + 1e-6
    def step(old, frame):
        view, delta = frame
        moved = pilot.move(old, delta.astype(jnp.float32), directions)
        predicted = local @ moved
        residual = jax.nn.one_hot(view, 17) - predicted
        surprise = 1 - jnp.take_along_axis(predicted,view[...,None].astype(jnp.int32),axis=-1)[...,0]
        logit = params["gate"][view] + params["error_weight"][view] * surprise
        if not per_tile:
            logit = logit.mean(-1,keepdims=True)
        gate = jax.nn.sigmoid(logit)
        return moved + local.T @ (gate[...,None] * residual) / scale, None
    return step


def model(params, memory, tiles, motion, query, per_tile, return_states=False):
    freq = (params["direction"][0] - params["direction"][1])/2
    step = step_components(params, per_tile)
    def block(state, xs):
        tile_chunk, delta_chunk, offsets = xs
        state, _ = jax.lax.scan(step,state,(tile_chunk,delta_chunk))
        if return_states:
            return state,state
        keys = pilot.address(offsets.astype(jnp.float32),freq)
        logits = jnp.einsum("barpk,bkr->barp",keys,state[...,jnp.array(CODES)])
        return state,(logits[...,0]-logits[...,1])*jax.nn.softplus(params["gain"])
    def group(x):
        return x.swapaxes(0,1).reshape((-1,STRIDE,x.shape[0],*x.shape[2:]))
    return jax.lax.scan(block,memory,(group(tiles),group(motion),query.swapaxes(0,1)))


def check():
    p = initialize(32150)
    view = jax.random.randint(jax.random.PRNGKey(7),(2,32,63),0,17,dtype=jnp.int32)
    motion = jnp.zeros((2,32,2))
    q = jnp.ones((2,1,4,2,2,2))
    blank = jnp.zeros((2,2*pilot.FREQ,17))
    state = model(p,blank,view,motion,q,False)[0]
    np.testing.assert_allclose(state,model(p,blank,view,motion,q,True)[0],atol=1e-6)
    np.testing.assert_allclose(state,pilot.forward(p,blank,view,motion,q,True)[0],atol=1e-5)


def diagnostics(out, arm, seed):
    with np.load(out/f"{arm}_{seed}.npz") as fitted:
        p = {k:jnp.asarray(fitted[k]) for k in initialize(seed)}
    freq = np.asarray((p["direction"][0]-p["direction"][1])/2)
    query_events = defaultdict(list)
    for ev in json.loads((out/"events.json").read_text()):
        query_events[ev["world"]].append(ev)
    with np.load(out/"test.npz") as a:
        tiles,motions=a["tiles"],a["motion"]
    get_states=jax.jit(lambda h,v,m,q:model(p,h,v,m,q,arm=="tile",True))
    out_rows = []
    def score(hidden, xy):
        angles = np.asarray(xy) @ freq
        key = np.concatenate((np.cos(angles),np.sin(angles)),axis=-1)/np.sqrt(pilot.FREQ)
        return key @ hidden
    def hit(margin):
        return float(margin>0)+.5*float(margin==0)
    for start in range(0,len(tiles),pilot.BATCH):
        ids=range(start,start+pilot.BATCH)
        if not any(query_events[i] for i in ids):
            continue
        carry=jnp.zeros((pilot.BATCH,2*pilot.FREQ,17))
        states=[]
        for t in range(0,tiles.shape[1],pilot.CHUNK):
            q=jnp.zeros((pilot.BATCH,pilot.CHUNK//STRIDE))
            carry,h=get_states(carry, tiles[start:start+pilot.BATCH,t:t+pilot.CHUNK],
                               motions[start:start+pilot.BATCH,t:t+pilot.CHUNK],q)
            states.append(np.asarray(h))
        states=np.concatenate(states)
        for lane,i in enumerate(ids):
            for ev in query_events[i]:
                r=ev["r"]
                state=states[ev["new"],lane]
                if ev["task"]=="both":
                    h=[]
                    for xy,code in zip(ev["probes"],(CODES[r],CODES[1-r])):
                        a,b=score(state,xy)[:,code]
                        h.append(hit(a-b))
                else:
                    old=score(states[ev["old"],lane],ev["xy_old"])
                    new=score(state,ev["xy_new"])
                    a,b=CODES[r],REPLACEMENT[r]
                    h=[hit(old[a]-old[b]),hit(new[b]-new[a])]
                out_rows.append([i,ev["new"],ev["task"],r,*h,h[0]*h[1]])
    with (out/f"{arm}_{seed}_events.csv").open("w") as f:
        w=csv.writer(f)
        w.writerow(["world","block","task","first_resource","first_correct","second_correct","both_correct"])
        w.writerows(out_rows)


def report(out):
    import pandas as pd
    arms=("scene","tile")
    summary=[]
    def compare(name,r,ages,values,n):
        if not len(n): return
        delta=values[1]-values[0]
        rng=np.random.default_rng(41089)
        boots=rng.choice(delta,(10000,len(delta))).mean(1)
        low,high=np.quantile(boots,[.025,.975])
        summary.append(dict(task=name,resource=NAMES[r],delay=ages,worlds=len(n),queries=int(n.sum()),
                            scene=values[0].mean(),tile=values[1].mean(),gain=delta.mean(),low=low,high=high))
    counts=np.stack([[np.load(out/f"{arm}_{seed}.npz")["counts"] for seed in pilot.SEEDS] for arm in arms])
    n=counts[0,0,...,1]
    assert (counts[...,1]==n).all()
    hits=counts[...,0].mean(1)
    for r in range(2):
        for b,(lo,hi) in enumerate(pilot.BANDS):
            den=n[:,b,r]
            ok=den>0
            compare("recall",r,f"{lo}-{hi}",hits[:,ok,b,r]/den[ok],den[ok])
    frames={(a,s):pd.read_csv(out/f"{a}_{s}_events.csv") for a in arms for s in pilot.SEEDS}
    for task in ("both","revision"):
        for r in range(2):
            for metric in ("first_correct","second_correct","both_correct"):
                values=[]
                for arm in arms:
                    seed_values=[]
                    for seed in pilot.SEEDS:
                        f=frames[arm,seed]
                        g=f[(f.task==task)&(f.first_resource==r)].groupby("world")[metric]
                        seed_values.append(g.mean().to_numpy())
                        den=g.size().to_numpy()
                    values.append(np.mean(seed_values,axis=0))
                compare(task+"_"+metric,r,"natural",np.asarray(values),den)
    pd.DataFrame(summary).to_csv(out/"results.csv",index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=["scene","tile","report"])
    parser.add_argument("--seed",choices=pilot.SEEDS,type=int,default=pilot.SEEDS[0])
    args=parser.parse_args()
    if args.arm=="report":
        report(OUT)
    else:
        check()
        pilot.run(OUT,args.arm,args.seed,initializer=initialize,
                  memory_forward=lambda p,h,t,m,q,_: model(p,h,t,m,q,args.arm=="tile"))
        diagnostics(OUT,args.arm,args.seed)

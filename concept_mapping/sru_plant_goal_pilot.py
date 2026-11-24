"""Frozen 12.5B SRU, paired plant-goal pilot. Run from the repository root as a module."""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import serialization
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Achievement, BlockType
from ppo_rnn_og import ActorCriticRNN, initialize_recurrent_carry
from symbolic_motion_inputs import with_motion_inputs
from collect_recovery_bank import select
from evaluate_reward_shaped_ppo import select_policy_actions

RUNS = Path(__file__).resolve().parent / "runs"
OUT = RUNS / "sru_12p5b_plant_goal_pilot"
SOURCE = RUNS / "sru_gru_map_reconstruction_to12p5b_seed50/gru"
FLAG = Achievement.EAT_PLANT.value
LIMIT = 2048
BATCH = 16
HORIZON = 64
TOTAL_CAP = 1099776


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def distance(obs):
    ripe = obs[:, :1323].reshape(-1,7,9,21)[...,BlockType.RIPE_PLANT.value] > .5
    d = abs(jnp.arange(7)[:,None]-3)+abs(jnp.arange(9)[None,:]-4)
    return jnp.where(ripe,d,99).min((1,2))


def zero_head():
    return dict(actor=jnp.zeros((512,17)), bias=jnp.zeros(17), value=jnp.zeros(512), offset=jnp.array(0.))


def logits_for(goal, head, features, logits):
    return logits + goal * (features @ head["actor"] + head["bias"])


def main(args):
    out = RUNS / "sru_12p5b_offscreen_plant_goal" if args.offscreen else OUT
    if args.offscreen and args.stage == "screen-train":
        out = out / "visible_sources"
    out.mkdir(exist_ok=True, parents=True)
    cfg = json.loads((SOURCE/"inference_config.json").read_text())
    manifest = json.loads((SOURCE/"progress.json").read_text())
    checkpoint = Path(manifest["local_run"])/"files/policies/12500008960/default"
    assert cfg["RECURRENT"] and cfg["SRU_GRU"] and cfg["MOTION_INPUTS"] and cfg["SRU_MAP_COEF"] > 0
    params = ocp.PyTreeCheckpointer().restore(str(checkpoint))["params"]
    net = ActorCriticRNN(17,cfg)
    env = with_motion_inputs(make_craftax_env_from_name(cfg["ENV_NAME"],False),cfg)
    reset = jax.jit(jax.vmap(lambda key:env.reset(key,env.default_params)))
    step = jax.vmap(lambda k,s,a:env.step(k,s,a,env.default_params))
    vkey = jax.vmap(jax.random.PRNGKey)
    split = jax.vmap(lambda k:jax.random.split(k,3))
    def actor(obs,h):
        nh,pi,_=net.apply(params,h,(obs[None],jnp.zeros((1,len(obs)),bool)))
        return nh,pi.logits[0]
    @jax.jit
    def screen(world_seeds,stream_seeds):
        obs,state=reset(vkey(world_seeds))
        h=initialize_recurrent_carry(len(obs),cfg)
        def active(c):
            (o,s,_),_,done,n,_=c
            d=distance(o)
            return ~done & ~s.env_state.achievements[:,FLAG] & ~((d>=2)&(d<99)) & (n<LIMIT)
        def slide(c):
            (o,s,h),keys,done,n,t=c
            live=active(c)
            ks=split(keys)
            nh,logits=actor(o,h)
            a=select_policy_actions(logits,ks[:,1])
            no,ns,_,nd,_=step(ks[:,2],s,a)
            nxt=select(live,(no,ns,nh),(o,s,h))
            return nxt,ks[:,0],done|(live&nd),n+live,t+1
        init=((obs,state,h),vkey(stream_seeds),jnp.zeros(len(obs),bool),jnp.zeros(len(obs),jnp.int32),jnp.int32(0))
        (obs,state,h),_,done,n,t=jax.lax.while_loop(lambda c:active(c).any(),slide,init)
        d=distance(obs)
        ok=~done & ~state.env_state.achievements[:,FLAG] & (d>=2)&(d<99)
        return (obs,state,h),ok,n,d,t

    def bank_path(part):return out/f"{part}_bank.msgpack"
    def bank(part, directory=None):
        template=reset(vkey(jnp.array([123],jnp.uint32)))
        tree=(*template,initialize_recurrent_carry(1,cfg))
        path = (directory or out) / f"{part}_bank.msgpack"
        return jax.tree_util.tree_map(jnp.asarray,serialization.from_bytes(tree,path.read_bytes()))

    if args.stage.startswith("prepare-"):
        from concept_mapping.offscreen_plant_starts import build_offscreen_bank
        assert args.offscreen
        part = args.stage.split("-")[1]
        source = OUT if part == "test" else out / "visible_sources"
        build_offscreen_bank(part, source, out, bank(part, source), actor, step, FLAG)
        return

    if args.stage in ("screen-test","screen-train"):
        part=args.stage.split("-")[1]
        target=100 if part=="test" else args.worlds
        info=out/f"{part}_bank.json"
        if bank_path(part).exists():
            if not info.exists():raise RuntimeError("Missing bank metadata")
            print(info.read_text()[:250])
            return
        # Disjoint world seeds for trial and training and separate policy RNG.
        table=np.random.default_rng(168521).choice(2**31,20000,replace=False).astype(np.uint32)
        table=table[:10000] if part=="test" else table[10000:]
        picked=[]
        records=[]
        ledger=[]
        actual=slots=0
        for i in range(0,len(table),BATCH):
            if slots + BATCH*LIMIT > args.screen_cap:break
            seeds=table[i:i+BATCH]
            snaps,ok,n,dist,t=jax.device_get(screen(seeds,seeds ^ np.uint32(32195)))
            actual+=int(n.sum())
            slots+=int(t)*BATCH
            for lane,seed in enumerate(seeds):
                success=bool(ok[lane])
                row=dict(world=int(seed),tick=int(n[lane]),distance=int(dist[lane]),eligible=success)
                ledger.append(row)
                if success and len(picked)<target:
                    records.append(row)
                    picked.append(jax.tree_util.tree_map(lambda x:x[lane],snaps))
            write_json(out/f"{part}_screen_progress.json",dict(screened=len(ledger),eligible=len(picked),
                active_steps=actual,simulator_slots=slots))
            print(part,len(ledger),len(picked),actual,slots,flush=True)
            if len(picked)>=target:break
        if not picked:raise RuntimeError("No eligible states")
        saved=jax.tree_util.tree_map(lambda *x:np.stack(x),*picked)
        # Verify the stored observation is the factual observation of that state.
        from craftax.craftax_classic.renderer import render_craftax_symbolic
        np.testing.assert_array_equal(np.asarray(jax.jit(jax.vmap(render_craftax_symbolic))(saved[1].env_state)),
                                      saved[0][:,:1345])
        assert np.isfinite(saved[2]).all() and np.all(np.linalg.norm(saved[2],axis=1)>0)
        payload=serialization.to_bytes(saved)
        restored=serialization.from_bytes(saved,payload)
        for a,b in zip(jax.tree_util.tree_leaves(saved),jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(a,b)
        bank_path(part).write_bytes(payload)
        write_json(info,dict(checkpoint=str(checkpoint),records=records,screened=len(ledger),active_steps=actual,
                             simulator_slots=slots,selection="First pre-harvest view with nearest ripe plant >=2 tiles, by step 2048",
                             screening=ledger))
        return

    def roll(starts,keys,head,goal,targets,phase=None):
        obs,s,h=starts
        def tick(c,_):
            obs,s,h,keys,ended,harvested,seen=c
            ks=split(keys)
            features,baseline=actor(obs,h)
            gate = goal if phase is None else jnp.where(seen, phase[1], phase[0])[:, None]
            logits=logits_for(gate,head,features,baseline)
            a=select_policy_actions(logits,ks[:,1])
            no,ns,_,done,_=step(ks[:,2],s,a)
            raw=ns.env_state
            tile=raw.map[jnp.arange(len(obs)),raw.player_position[:,0],raw.player_position[:,1]]
            death=done&((raw.player_health<=0)|(tile==BlockType.LAVA.value))
            reward=~s.env_state.achievements[:,FLAG]&raw.achievements[:,FLAG]&~ended
            if args.offscreen:
                indices = (jnp.arange(len(obs)), targets[:,0], targets[:,1])
                reward = ((s.env_state.map[indices] == BlockType.RIPE_PLANT.value)
                          & (raw.map[indices] == BlockType.PLANT.value) & ~ended & ~harvested)
            value=features@head["value"]+head["offset"]
            logp=jnp.take_along_axis(jax.nn.log_softmax(logits),a[:,None],axis=-1)[:,0]
            measurements = (features,baseline,a,logp,value,reward,~ended,done,death&~ended)
            if phase is not None:
                # Bank starts are offscreen. Latch on the successor frame, before its action.
                offset = targets - raw.player_position
                in_view = (abs(offset[:,0]) <= 3) & (abs(offset[:,1]) <= 4)
                patch = no[:, :1323].reshape(-1, 7, 9, 21)
                ripe = patch[jnp.arange(len(no)), (3+offset[:,0]).clip(0,6),
                             (4+offset[:,1]).clip(0,8), BlockType.RIPE_PLANT.value] > .5
                first_sight = in_view & ripe & ~ended & ~done & ~seen
                seen = seen | first_sight
                measurements = (first_sight, reward, death & ~ended, ~ended)
            return (no,ns,features,ks[:,0],ended|done,harvested|reward,seen), measurements
        initial_flag = jnp.zeros(len(obs),bool)
        _,records=jax.lax.scan(tick,(obs,s,h,keys,initial_flag,initial_flag,initial_flag),None,length=HORIZON)
        return records

    if args.stage == "phase-replay":
        from concept_mapping.plant_goal_phase_replay import evaluate_phases
        assert args.offscreen
        head = serialization.from_bytes(zero_head(), (out/"plant_head.msgpack").read_bytes())
        evaluate_phases(out, bank("test"), head, roll)
        return

    if args.stage in ("normal","plant"):
        world_bank=bank("test")
        normal=args.stage=="normal"
        head=zero_head() if normal else serialization.from_bytes(zero_head(),(out/"plant_head.msgpack").read_bytes())
        @jax.jit
        def evaluate(snaps,keys,targets):
            data=roll(snaps,keys,head,int(not normal),targets)
            return data[5].any(0),data[8].any(0),data[6].sum(0)
        meta=json.loads((out/"test_bank.json").read_text())["records"]
        assert len(meta)>0 and (args.offscreen or len(meta)==100)
        targets = jnp.asarray([r.get("target", [0,0]) for r in meta])
        features,raw_logits=jax.jit(actor)(world_bank[0],world_bank[2])
        for g in (0,1):
            np.testing.assert_array_equal(logits_for(g,zero_head(),features,raw_logits),raw_logits)
        np.testing.assert_array_equal(logits_for(0,head,features,raw_logits),raw_logits)
        rows=[]
        ids=np.repeat(np.arange(len(meta)),4)
        for i in range(0,len(ids),40):
            ix=ids[i:i+40]
            snaps=jax.tree_util.tree_map(lambda x:x[ix],world_bank)
            keys=jax.vmap(lambda n:jax.random.fold_in(jax.random.PRNGKey(182194),n))(jnp.arange(i,i+len(ix)))
            hits,deaths,steps=jax.device_get(evaluate(snaps,keys,targets[ix]))
            for k,(j,a,b,n) in enumerate(zip(ix,hits,deaths,steps)):
                rows.append(dict(world=meta[j]["world"],repeat=(i+k)%4,harvest=int(a),death=int(b),steps=int(n)))
        with (out/f"{args.stage}.csv").open("w") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(out/f"{args.stage}.json",dict(harvest=sum(r["harvest"] for r in rows),
                  death=sum(r["death"] for r in rows),trials=len(rows),steps=sum(r["steps"] for r in rows),
                  simulator_slots=len(rows)*64))
        print((out/f"{args.stage}.json").read_text(),flush=True)
        return

    train_bank=bank("train")
    train_meta=json.loads((out/"train_bank.json").read_text())["records"]
    targets = jnp.asarray([r.get("target", [0,0]) for r in train_meta])
    if args.offscreen:
        prior_slots = sum(json.loads(p.read_text())["simulator_slots"] for p in
                          (out/"test_bank.json",out/"train_bank.json",out/"visible_sources/train_bank.json",out/"normal.json"))
        heldout = json.loads((out/"normal.json").read_text())
        assert heldout["harvest"]/heldout["trials"] <= .9
        assert not set(r["world"] for r in train_meta).intersection(
            r["world"] for r in json.loads((out/"test_bank.json").read_text())["records"])
        args.updates = min(args.updates, (TOTAL_CAP - prior_slots - heldout["simulator_slots"])//65536)
        assert args.updates > 0
        print("Budgeted updates", args.updates, flush=True)
    adam=optax.chain(optax.clip_by_global_norm(1.),optax.adam(2e-4))
    head=zero_head()
    opt=adam.init(head)
    sample=jax.jit(roll)
    @jax.jit
    def update(head,opt,records,key):
        feat,base,action,oldlog,value,reward,active,done,_=records
        def backward(c,x):
            next_adv,next_v=c
            r,v,d=x
            adv=r+.99*(~d)*next_v-v+.99*.8*(~d)*next_adv
            return (adv,v),adv
        # The task is harvest within 64 steps; the imposed horizon terminates the task.
        zero=jnp.zeros_like(value[0])
        _,advantages=jax.lax.scan(backward,(zero,zero),(reward.astype(float),value,done|~active),reverse=True)
        flat=lambda x:x.reshape((-1,*x.shape[2:]))
        data=tuple(map(flat,(feat,base,action,oldlog,value,advantages,advantages+value,active)))
        def epoch(c,k):
            def mini(c,index):
                head,opt=c
                f,b,a,lp,v,adv,target,mask=(x[index] for x in data)
                mean=lambda x:(x*mask).sum()/mask.sum().clip(1)
                adv=(adv-mean(adv))/jnp.sqrt(mean((adv-mean(adv))**2)+1e-8)
                def loss(h):
                    ll=jax.nn.log_softmax(logits_for(1,h,f,b))
                    ratio=jnp.exp(jnp.take_along_axis(ll,a[:,None],axis=-1)[:,0]-lp)
                    pred=f@h["value"]+h["offset"]
                    vf=jnp.maximum((pred-target)**2,(v+jnp.clip(pred-v,-.2,.2)-target)**2)
                    return mean(-jnp.minimum(ratio*adv,ratio.clip(.8,1.2)*adv)+.25*vf+.01*(jnp.exp(ll)*ll).sum(-1))
                grad=jax.grad(loss)(head)
                diff,opt=adam.update(grad,opt,head)
                return (optax.apply_updates(head,diff),opt),None
            ix=jax.random.permutation(k,action.size).reshape(8,-1)
            return jax.lax.scan(mini,c,ix)[0],None
        return jax.lax.scan(epoch,(head,opt),jax.random.split(key,4))[0]
    for u in range(args.updates):
        rng=jax.random.fold_in(jax.random.PRNGKey(99817),u)
        choice,stream,learn=jax.random.split(rng,3)
        ix=jax.random.randint(choice,(1024,),0,len(train_bank[0]))
        # Smaller rollout batches avoid replicating 1024 large simulator maps in compilation.
        keys=jax.random.split(stream,1024)
        parts=[]
        for j in range(0,1024,128):
            part=jax.tree_util.tree_map(lambda x:x[ix[j:j+128]], train_bank)
            parts.append(sample(part,keys[j:j+128],head,1,targets[ix[j:j+128]]))
        records=jax.tree_util.tree_map(lambda *xs:jnp.concatenate(xs,axis=1),*parts)
        head,opt=update(head,opt,records,learn)
        for x in head.values():assert np.isfinite(np.asarray(x)).all()
        print(u+1,float(records[5].any(0).mean()),flush=True)
        (out/"plant_head.msgpack").write_bytes(serialization.to_bytes(head))
        write_json(out/"training.json",dict(updates=u+1,simulator_slots=(u+1)*65536,training_worlds=len(train_bank[0])))
        

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("screen-test","screen-train","prepare-test","prepare-train","normal","train","plant","phase-replay"))
    p.add_argument("--offscreen",action="store_true")
    p.add_argument("--screen-cap",type=int,default=800000)
    p.add_argument("--worlds",type=int,default=128)
    p.add_argument("--updates",type=int,default=16)
    main(p.parse_args())

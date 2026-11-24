"""Matched arrow-proximity probe of the frozen feedforward death-cost critic.

Usage: CUDA_VISIBLE_DEVICES=7 python concept_mapping/probe_arrow_proximity.py
Horizontal approaches keep every tested distance inside the 7x9 observation.
"""
import csv
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
if not hasattr(jax,'tree'):
    from types import SimpleNamespace
    jax.tree=SimpleNamespace(map=jax.tree_util.tree_map)
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import BlockType, OBS_DIM
from craftax.craftax_classic.renderer import render_craftax_symbolic, render_craftax_pixels
from models.actor_critic import ActorCritic


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-only',action='store_true',help='Two strips from one reset scene; no rollouts or aggregate analysis')
    parser.add_argument('--entity',choices=('arrow','zombie','two_zombies','three_zombies'),default='arrow')
    parser.add_argument('--health',type=int,nargs='+',default=[1,2],choices=range(1,10))
    args=parser.parse_args()
    if args.entity!='arrow' and not args.images_only:
        parser.error('Zombie probes require --images-only')
    root = Path(__file__).resolve().parents[1]
    out = root/f'concept_mapping/runs/{args.entity}_proximity_death_penalty2_10b'
    out.mkdir(parents=True, exist_ok=True)
    run = root/'wandb/run-20261006_194739-uj5n3za2'
    checkpoint = 9999941632
    params = ocp.PyTreeCheckpointer().restore(str(run/'files/policies'/str(checkpoint)/'default'))['params']
    net = ActorCritic(17,512)
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=True)
    if args.images_only:
        render_simple_images(out,env,net,params,args.entity,tuple(args.health))
        return
    seeds = np.arange(9300,9364)
    keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0,None)))
    obs, states = reset(keys, env.default_params)

    @jax.jit
    def collect(obs, states, keys):
        def body(carry, _):
            obs, states, keys = carry
            split = jax.vmap(lambda k: jax.random.split(k,3))(keys)
            policy,_ = net.apply(params,obs)
            actions = jax.vmap(lambda logits,k: jax.random.categorical(k,logits))(policy.logits,split[:,1])
            obs,states,_,_,_ = jax.vmap(env.step,in_axes=(0,0,0,None))(split[:,2],states,actions,env.default_params)
            return (obs,states,split[:,0]),None
        return jax.lax.scan(body,(obs,states,keys),None,length=128)[0]

    _,states,_ = collect(obs,states,keys)
    positions=np.asarray(states.player_position)
    valid=np.flatnonzero((positions[:,1]>=4)&(positions[:,1]<states.map.shape[2]-4))
    seeds=seeds[valid]
    states=jax.tree_util.tree_map(lambda a:a[valid],states)
    value = jax.jit(lambda obs: net.apply(params,obs)[1])
    render = jax.jit(lambda s: render_craftax_pixels(s,7))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',17)
    rows, all_obs, example_states = [], [], []
    effects = np.empty((len(seeds),2,2,4))
    for i,seed in enumerate(seeds):
        factual = jax.tree_util.tree_map(lambda a:a[i],states)
        y,x = map(int,factual.player_position)
        # Shared edits isolate one approaching projectile in a clear visible lane.
        common = factual.replace(
            map=factual.map.at[y,x-4:x+5].set(BlockType.PATH.value),
            mob_map=jnp.zeros_like(factual.mob_map),is_sleeping=False,
            zombies=factual.zombies.replace(mask=jnp.zeros_like(factual.zombies.mask)),
            skeletons=factual.skeletons.replace(mask=jnp.zeros_like(factual.skeletons.mask)),
            cows=factual.cows.replace(mask=jnp.zeros_like(factual.cows.mask)),
            arrows=factual.arrows.replace(mask=jnp.zeros_like(factual.arrows.mask)))
        for hi,health in enumerate((1,2)):
            base = common.replace(player_health=health)
            for si,side in enumerate((-1,1)):
                branches=[base]
                for distance in (1,2,3,4):
                    branches.append(base.replace(arrows=base.arrows.replace(
                        mask=base.arrows.mask.at[0].set(True),
                        position=base.arrows.position.at[0].set(jnp.array([y,x+side*distance]))),
                        arrow_directions=base.arrow_directions.at[0].set(jnp.array([0,-side]))))
                observations=np.stack([render_craftax_symbolic(s) for s in branches])
                map_end=OBS_DIM[0]*OBS_DIM[1]*(len(BlockType)+4)
                for d in (1,2,3,4):
                    # All non-arrow inputs must match the no-arrow control.
                    delta=observations[d]-observations[0]
                    assert np.count_nonzero(delta)==1
                    idx=np.flatnonzero(delta)[0]
                    assert idx<map_end and idx%(len(BlockType)+4)==len(BlockType)+3
                vals=np.asarray(value(jnp.asarray(observations))).reshape(-1)
                effects[i,hi,si]=vals[1:]-vals[0]
                all_obs.append(observations)
                for d,val in enumerate(vals):
                    rows.append(dict(scene_id=i,seed=int(seed),factual_timestep=int(factual.timestep),
                        health=health,approach='left' if side<0 else 'right',distance=d,
                        arrow_present=d>0,value=float(val),no_arrow_value=float(vals[0]),
                        value_minus_no_arrow=float(val-vals[0])))
                if i<3 and side==1:
                    example_states.append((i,health,branches,vals))
    with (out/'per_state_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'counterfactual_observations.npz',observations=np.stack(all_obs),effects=effects)
    rng=np.random.default_rng(9300)
    boot=rng.integers(0,len(seeds),size=(10000,len(seeds)))
    summaries=[]
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for hi,(health,color) in enumerate(((1,'#e52222'),(2,'#2461eb'))):
        per_scene=effects[:,hi].mean(axis=1)
        mean=per_scene.mean(axis=0)
        low,high=np.percentile(per_scene[boot].mean(axis=1),[2.5,97.5],axis=0)
        ax.plot([4,3,2,1],mean[::-1],'-o',color=color,label=f'Health {health}')
        ax.fill_between([4,3,2,1],low[::-1],high[::-1],color=color,alpha=.15)
        adjacent=effects[:,hi,:,0]-effects[:,hi,:,3]
        summaries.append(dict(health=health,mean_effect_by_distance_1_to_4=mean.tolist(),
            mean_adjacent_minus_distance4=float(adjacent.mean()),
            ci95_adjacent_minus_distance4=np.percentile(adjacent.mean(axis=1)[boot].mean(axis=1),[2.5,97.5]).tolist(),
            fraction_adjacent_lower_than_distance4=float((adjacent<0).mean()),
            fraction_strictly_decreasing_toward_agent=float((np.diff(effects[:,hi],axis=-1)>0).all(axis=-1).mean()),
            fraction_adjacent_lower_than_no_arrow=float((effects[:,hi,:,0]<0).mean())))
    ax.axhline(0,color='black',lw=1,ls=':')
    ax.set(xlabel='Arrow distance (tiles; approaching →)',ylabel='Value − no-arrow value',
           title='Arrow proximity: frozen −2 death-cost critic',xticks=[4,3,2,1])
    ax.invert_xaxis();ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    for ext in ('png','pdf','jpeg'):fig.savefig(out/f'arrow_proximity.{ext}',dpi=220)
    for scene_id,health,branches,vals in example_states:
        panels=[]
        for d in (0,4,3,2,1):
            pixels=np.asarray(render(branches[d])).clip(0,255).astype(np.uint8)
            img=Image.fromarray(pixels).resize((pixels.shape[1]*3,pixels.shape[0]*3),Image.Resampling.NEAREST)
            panel=Image.new('RGB',(img.width,img.height+60),'white');panel.paste(img,(0,0))
            name='No arrow' if d==0 else f'Distance {d}'
            ImageDraw.Draw(panel).text((4,img.height+4),f'{name}\nvalue: {vals[d]:.4f}',font=font,fill='black')
            panels.append(panel)
        sheet=Image.new('RGB',(sum(p.width for p in panels)+12*4,panels[0].height),'white')
        for k,p in enumerate(panels):sheet.paste(p,(k*(p.width+12),0))
        sheet.save(out/f'example_{scene_id:02d}_health{health}.png')
    metadata=dict(run=str(run),checkpoint=checkpoint,parameters=2438162,death_cost=-2,
        scenes=len(seeds),seeds=seeds.tolist(),initial_seed_range=[9300,9363],natural_context_collection_steps=128,
        health_levels=[1,2],approaches=['left','right'],distances=[1,2,3,4],
        total_evaluations=len(rows),synthetic_counterfactuals=True,
        shared_edits='Clear horizontal 9-tile path; remove other mobs/arrows; awake; set health. Preserve other inventory and resources.',
        validation='Exactly one symbolic arrow-occupancy feature changes from no-arrow control.',
        bootstrap='10000 scene-level resamples, averaging both approaches within scene; seed9300.',
        interpretation='Distance-sensitive threat valuation, not inference of arrow direction or guaranteed death: policy may dodge.',
        results=summaries)
    (out/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


def render_simple_images(out,env,net,params,entity='arrow',health_levels=(1,2)):
    _,state=jax.jit(env.reset)(jax.random.PRNGKey(9300),env.default_params)
    y,x=map(int,state.player_position)
    state=state.replace(map=state.map.at[y,x:x+5].set(BlockType.PATH.value),
        mob_map=jnp.zeros_like(state.mob_map),
        zombies=state.zombies.replace(mask=jnp.zeros_like(state.zombies.mask)),
        skeletons=state.skeletons.replace(mask=jnp.zeros_like(state.skeletons.mask)),
        cows=state.cows.replace(mask=jnp.zeros_like(state.cows.mask)),
        arrows=state.arrows.replace(mask=jnp.zeros_like(state.arrows.mask)))
    multiple_zombies=entity in ('two_zombies','three_zombies')
    distances=(3,2,1) if multiple_zombies else (4,3,2,1)
    if multiple_zombies:
        state=state.replace(map=state.map.at[y,x-3:x+1].set(BlockType.PATH.value)
            .at[y:y+4,x].set(BlockType.PATH.value))
    render=jax.jit(lambda s:render_craftax_pixels(s,7))
    evaluate=jax.jit(lambda obs:net.apply(params,obs)[1])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',19)
    rows=[]
    for health in health_levels:
        branches=[]
        for d in distances:
            position=jnp.array([y,x+d])
            branch=state.replace(player_health=health)
            if entity=='arrow':
                branch=branch.replace(arrows=state.arrows.replace(
                    mask=state.arrows.mask.at[0].set(True),
                    position=state.arrows.position.at[0].set(position)),
                    arrow_directions=state.arrow_directions.at[0].set(jnp.array([0,-1])))
            elif entity=='zombie':
                branch=branch.replace(zombies=state.zombies.replace(
                    mask=state.zombies.mask.at[0].set(True),
                    position=state.zombies.position.at[0].set(position),
                    health=state.zombies.health.at[0].set(env.default_params.zombie_health),
                    attack_cooldown=state.zombies.attack_cooldown.at[0].set(0)),
                    mob_map=state.mob_map.at[y,x+d].set(True))
            else:
                positions=jnp.array([[y+d,x],[y,x-d]])
                if entity=='three_zombies':
                    positions=jnp.concatenate([positions,jnp.array([[y,x+d]])],axis=0)
                n=positions.shape[0]
                branch=branch.replace(zombies=state.zombies.replace(
                    mask=state.zombies.mask.at[:n].set(True),
                    position=state.zombies.position.at[:n].set(positions),
                    health=state.zombies.health.at[:n].set(env.default_params.zombie_health),
                    attack_cooldown=state.zombies.attack_cooldown.at[:n].set(0)),
                    mob_map=state.mob_map.at[positions[:,0],positions[:,1]].set(True))
                assert bool(jnp.all(jnp.abs(positions-state.player_position).sum(axis=1)==d))
            branches.append(branch)
        observations=np.stack([render_craftax_symbolic(s) for s in branches])
        values=np.asarray(evaluate(jnp.asarray(observations))).reshape(-1)
        panels=[]
        for d,s,v in zip(distances,branches,values):
            pix=np.asarray(render(s)).clip(0,255).astype(np.uint8)
            img=Image.fromarray(pix).resize((pix.shape[1]*4,pix.shape[0]*4),Image.Resampling.NEAREST)
            panel=Image.new('RGB',(img.width,img.height+65),'white');panel.paste(img,(0,0))
            ImageDraw.Draw(panel).text((4,img.height+5),f'Distance {d}\nvalue: {v:.4f}',font=font,fill='black')
            panels.append(panel)
            rows.append(dict(seed=9300,health=health,distance=d,value=float(v)))
        sheet=Image.new('RGB',(panels[0].width*len(panels)+12*(len(panels)-1),panels[0].height),'white')
        for i,p in enumerate(panels):sheet.paste(p,(i*(p.width+12),0))
        path=out/f'simple_health{health}.png';sheet.save(path);print(path,flush=True)
    suffix='' if health_levels==(1,2) else '_health'+'_'.join(map(str,health_levels))
    with (out/f'simple_values{suffix}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/f'simple_setup{suffix}.json').write_text(json.dumps(dict(seed=9300,health=list(health_levels),distances=list(distances),
        checkpoint='wandb/run-20261006_194739-uj5n3za2/files/policies/9999941632',
        entity=entity,
        approaches=(['bottom','left','right'] if entity=='three_zombies' else
                    ['bottom','left'] if entity=='two_zombies' else ['right']),
        setup=f'One synthetic reset scene; shared clear paths and other mobs removed. Only {entity} positions change within each strip. No rollouts or training.'),indent=2)+'\n')
    print(rows,flush=True)


if __name__=='__main__':main()

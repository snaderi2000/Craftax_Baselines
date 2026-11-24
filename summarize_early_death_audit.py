"""Summarize verified traces, aligned successful controls, and observed-map routes."""
import argparse,csv,json,html
from pathlib import Path
from collections import Counter,deque
import numpy as np
from audit_ppo_early_deaths import summarize_trace

# Classic's public terrain semantics. Unknown cells never become traversable.
SOLID={1,3,4,5,8,9,10,11,12,15,16};LAVA=14;WATER=3
DIRECTIONS=[(0,-1),(0,1),(-1,0),(1,0)]


def observed_routes(data,frames):
    """Accumulate only recorded local observations; no simulator map access."""
    obs=data['observations'];positions=np.concatenate((np.array([[32,32]]),data['position']))
    memory=np.full((64,64),-1,np.int16);result={}
    for t in range(min(len(obs),max(frames)+1)):
        center=positions[t];tiles=obs[t,:1323].reshape(7,9,21)[...,:17].argmax(-1)
        for i in range(7):
            for j in range(9):
                r,c=center+[i-3,j-4]
                if 0<=r<64 and 0<=c<64:memory[r,c]=tiles[i,j]
        if t not in frames:continue
        known_water=np.argwhere(memory==WATER)
        near=known_water[np.max(np.abs(known_water-center),axis=1)<=16] if len(known_water) else known_water
        start=tuple(center);q=deque([(start,0)]);seen={start};distance=None;first_action=None
        hops={start:None}
        while q:
            (r,c),dist=q.popleft()
            if any(0<=r+dr<64 and 0<=c+dc<64 and memory[r+dr,c+dc]==WATER for dr,dc in DIRECTIONS):
                distance=dist;first_action=hops[(r,c)];break
            for action,(dr,dc) in enumerate(DIRECTIONS,1):
                nxt=(r+dr,c+dc)
                if not (0<=nxt[0]<64 and 0<=nxt[1]<64) or nxt in seen:continue
                tile=int(memory[nxt])
                if tile<0 or tile in SOLID or tile==LAVA:continue
                seen.add(nxt);hops[nxt]=action if dist==0 else hops[(r,c)];q.append((nxt,dist+1))
        patch=obs[t,:1323].reshape(7,9,21)
        hostile=np.argwhere((patch[...,17]+patch[...,19]+patch[...,20])>0)
        result[t]=dict(frame=t,known_water_cells=int(len(known_water)),known_water_in_33_crop=int(len(near)),
            known_walkable_steps_to_water_edge=distance,first_route_move=first_action,
            nearest_visible_hostile_manhattan=int(np.abs(hostile-[3,4]).sum(-1).min()) if len(hostile) else None,
            note='Observed terrain only; route excludes unknown/lava/solid tiles, ignores dynamic mobs and mining; not a safety guarantee.')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args()
    root=a.audit;report=json.loads((root/'audit.json').read_text())
    if not a.partial and not report.get('complete'):raise RuntimeError('Audit is still running')
    controls={(x['dataset'],x['world_seed']):x for x in report['controls']};rows=[];paired=[]
    for episode in report['episodes']:
        data=dict(np.load(root/episode['trace']));n=episode['recorded_length']
        vitals=np.rint(data['observations'][:,1335:1339]*10).astype(int)
        np.testing.assert_array_equal(np.diff(vitals[:,0]),-data['combat_damage']+data['intrinsic_delta'])
        assert (data['combat_damage']>=0).all()
        checkpoints={n-1,max(0,n-200)}
        for k in (1,2,3):
            indices=np.flatnonzero(vitals[:n,k]<=0)
            if len(indices):checkpoints.add(int(indices[0]))
        routes=observed_routes(data,checkpoints)
        selected={k:v for k,v in episode.items() if k not in ('initial_observation_sha256',)}
        selected['map_snapshots']=list(routes.values());rows.append(selected)
        control=controls.get((episode['dataset'],episode['world_seed']))
        if control:
            assert control['initial_observation_sha256']==episode['initial_observation_sha256']
            other=dict(np.load(root/control['trace']));aligned=summarize_trace(other,n,report['window'])
            aligned['terminal_health_loss_mechanism']='successful_control_at_failure_time'
            paired.append(dict(dataset=episode['dataset'],world_seed=episode['world_seed'],failure_episode=episode['episode'],
                failure_time=n,successful_control_trace=control['trace'],successful_control_full_length=control['recorded_length'],
                identical_initial_observation=True,failure=summarize_trace(data,n,report['window']),control=aligned))
    mechanisms=dict(Counter(r['terminal_health_loss_mechanism'] for r in rows))
    summary=dict(complete=report.get('complete',False),failures=len(rows),paired_successful_worlds=len(paired),terminal_mechanisms=mechanisms,
        final_depleted_needs={k:sum(k in r['depleted_at_end'] for r in rows) for k in ('food','water','energy')},
        failures_with_combat_damage_last200=sum(r['combat_damage_last_window']>0 for r in rows),
        failures_with_deprivation_damage_last200=sum(r['deprivation_damage_last_window']>0 for r in rows),
        failures_with_blocked_streak_at_least20=sum(r['longest_blocked_streak_last_window']>=20 for r in rows),
        failures_with_sleeping_combat_damage=sum(r['combat_damage_while_sleeping']>0 for r in rows),
        total_combat_damage_last_windows=sum(r['combat_damage_last_window'] for r in rows),
        total_deprivation_damage_last_windows=sum(r['deprivation_damage_last_window'] for r in rows),
        episodes=rows,paired=paired,
        limitations=['Selected early-failure cohort; not overall policy failure frequencies.',
            'Successful controls are selected from independent same-world rerolls of the same checkpoint.',
            'Pairing controls world layout, not trajectories; this comparison is observational, not causal.',
            'Terminal mechanism differs from underlying cause. Resource and combat damage can interact.'])
    for metric in ('combat_damage_last_window','deprivation_damage_last_window','energy_zero_awake_steps','water_zero_steps','blocked_moves_last_window'):
        summary[metric+'_paired_means']={group:float(np.mean([r[group][metric] for r in paired])) if paired else None for group in ('failure','control')}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['dataset','episode','world_seed','recorded_length','terminal_health_loss_mechanism','depleted_at_end',
        'combat_damage_last_window','deprivation_damage_last_window','water_zero_steps','energy_zero_awake_steps',
        'blocked_moves_last_window','longest_blocked_streak_last_window','unique_positions_last_window']
    with (root/'failures.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:(', '.join(r[k]) if isinstance(r[k],list) else r[k]) for k in fields})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,5,figsize=(18,9),sharex=True,sharey=True)
    for ax,episode in zip(axes.flat,report['episodes']):
        data=dict(np.load(root/episode['trace']));n=episode['recorded_length'];lo=max(0,n-200)
        v=np.rint(data['observations'][:,1335:1339]*10)
        for k,(label,color) in enumerate(zip(('Health','Food','Water','Energy'),('#d62728','#2ca02c','#1f77b4','#9467bd'))):
            ax.plot(np.arange(lo,n+1)-n,v[lo:n+1,k],label=label,color=color,lw=1.4)
        ax.set_title(f"Set {episode['dataset']} / episode {episode['episode']} / {n} steps",fontsize=9);ax.set_ylim(-2,10);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Steps before death')
    axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle('Verified early failures: final 200 steps',fontsize=15);fig.tight_layout()
    fig.savefig(root/'final_200_steps.png',dpi=160);plt.close(fig)
    body=['<!doctype html><meta charset="utf-8"><title>Craftax early-death audit</title>',
      '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccc}img{width:100%}</style>',
      '<h1>Craftax early-death audit</h1>',f'<p>{len(rows)} exact recorded failure replays; {len(paired)} successful same-world controls from the same 12B checkpoint.</p>',
      '<p>Every diagnostic action replay reproduces every recorded observation. Health losses are attributed using the installed simulator’s combat and intrinsic-needs update stages.</p>',
      '<p>Terminal mechanisms: '+html.escape(str(mechanisms))+'</p>','<img src="final_200_steps.png" alt="Health, food, water and energy before each death">',
      '<h2>Failure details</h2><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>']
    for r in rows:body.append('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in fields)+'</tr>')
    body+=['</table><h2>Interpretation limits</h2><ul>']+['<li>'+html.escape(s)+'</li>' for s in summary['limitations']]+['</ul><p><a href="summary.json">Detailed results and aligned comparisons</a> · <a href="failures.csv">CSV</a></p>']
    (root/'report.html').write_text('\n'.join(body))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('episodes','paired')},indent=2))


if __name__=='__main__':main()

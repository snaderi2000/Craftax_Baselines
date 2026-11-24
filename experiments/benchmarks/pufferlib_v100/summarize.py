"""Summarize completed PufferLib timing repeats without conflating policies."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args()
d=json.loads((a.out/'results.json').read_text());rows=[]
for variant in d['config']['variants']:
 r=[x for x in d['results'] if x['variant']==variant and x['exit_code']==0]
 if not r:continue
 wall=[x['end_to_end_sps'] for x in r]
 warm=[statistics.median(x['metrics']['SPS'][2:-2]) for x in r]
 rows.append(dict(variant=variant,completed_repeats=len(r),parameters=r[0]['parameter_count'],wall_sps_median=statistics.median(wall),wall_sps_range=[min(wall),max(wall)],warmed_sps_median=statistics.median(warm),warmed_sps_range=[min(warm),max(warm)],all_checkpoints_finite=all(x['checkpoint_finite'] for x in r)))
summary={'hardware':d['gpu'],'commit':d['commit'],'precision':d['precision'],'rows':rows,'reference_jax_pilot_sps':{'map_gru_control':28720.250100003108,'richer_map_gru':20475.673941601486},'limitations':d['limitations'],'reload_evaluations':d.get('reload_evaluations',[])}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# PufferLib Classic on one V100','',f'Commit: `{d["commit"]}`. Precision: float32.','', '| Configuration | Parameters | Median process SPS | Median warmed SPS | Repeats |','|---|---:|---:|---:|---:|']
for r in rows:lines.append(f'| {r["variant"]} | {r["parameters"]:,} | {r["wall_sps_median"]:,.0f} | {r["warmed_sps_median"]:,.0f} | {r["completed_repeats"]} |')
lines+=['','Process SPS includes initialization and checkpoint saving; warmed SPS is the median of the interior 16 reported timing bins, then median over repeats. Frozen reload evaluation is excluded.','', 'Historical JAX pilot rates: 28,720 SPS for map+GRU control; 20,476 SPS with richer-map inputs. These are workload comparisons, not measurements of the same policy ported between backends.','','## Limits','']+[f'- {x}' for x in d['limitations']]
lines+=['- Similar parameter count does not imply equal computation or capability.','- Same-seed repeats measure timing repeatability; no multi-seed learning comparison.','- Float32 limits applicability to the default bfloat16 backend on newer GPUs.','- CPU load, observation construction, reward configuration and model architecture differ from historical JAX runs.','','Reload evaluations passed: '+str(len(d.get('reload_evaluations',[])))+'.']
(a.out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))

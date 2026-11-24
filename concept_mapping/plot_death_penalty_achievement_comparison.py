"""Compare seed-matched normal/death-cost PPO training achievement histories."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore
from google.protobuf.message import DecodeError
from concept_mapping.plot_smoothed_diamond_success import _centered_rolling_mean

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'wandb/run-20260630_214658-m0mw4end'
CACHES={
    'diamond':ROOT/'concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end/merged/diamond_success_curve/diamond_success_smoothed.csv',
    'plant':ROOT/'concept_mapping/runs/plant_collectability_10b_10000episodes_ripe_vs_unripe/emergence_over_checkpoints/behavior_success_smoothed.csv'}
METRICS={'diamond':'Achievements/collect_diamond','plant':'Achievements/eat_plant'}


def config(run):
    raw=yaml.safe_load((run/'files/config.yaml').read_text())
    return {k:v.get('value',v) if isinstance(v,dict) else v for k,v in raw.items()}


def history(run,cfg):
    if run==BASELINE and all(p.exists() for p in CACHES.values()):
        points={}
        for task,path in CACHES.items():
            rows=list(csv.DictReader(path.open()))
            rawkey='diamond_success_raw_pct' if task=='diamond' else 'success_raw_pct'
            points[task]=np.asarray([(int(r['environment_steps']),float(r[rawkey])) for r in rows])
        if all(v[-1,0] >= .99*cfg['TOTAL_TIMESTEPS'] for v in points.values()):
            return points,dict(type='previously cached full W&B history',files={k:str(v) for k,v in CACHES.items()})
    files=list(run.glob('run-*.wandb'))
    if len(files)!=1:raise RuntimeError(f'Expected one local W&B history in {run}')
    store=DataStore();store.open_for_scan(str(files[0]))
    points={k:{} for k in METRICS}
    wanted=set(METRICS.values())|{'env_step','_step'}
    print(f'Reading {files[0]}',flush=True)
    while True:
        try: scanned=store.scan_record()
        except (IndexError,OSError):break
        if scanned is None:break
        record=wandb_internal_pb2.Record()
        try:record.ParseFromString(scanned[1])
        except DecodeError:continue
        if record.WhichOneof('record_type')!='history':continue
        row={}
        for item in record.history.item:
            key=item.key or '/'.join(item.nested_key)
            if key in wanted:row[key]=json.loads(item.value_json)
        if '_step' not in row:continue
        step=int(row.get('env_step',(row['_step']+1)*cfg['NUM_ENVS']*cfg['NUM_STEPS']))
        for task,metric in METRICS.items():
            if metric in row and np.isfinite(row[metric]):points[task][step]=float(row[metric])
    arrays={k:np.asarray(sorted(v.items())) for k,v in points.items()}
    if any(len(v)==0 for v in arrays.values()):raise RuntimeError('Missing achievement history')
    if any(v[-1,0]<.99*cfg['TOTAL_TIMESTEPS'] for v in arrays.values()):
        import wandb
        run_id=run.name.rsplit('-',1)[-1]
        ref=f'synaderi-uc-davis/Craftax_Baselines/{run_id}'
        remote=wandb.Api(timeout=60).run(ref)
        print(f'Local history incomplete; reading full synced history: {ref}',flush=True)
        points={k:{} for k in METRICS}
        for row in remote.scan_history(keys=['_step',*METRICS.values()],page_size=10000):
            step=int((row['_step']+1)*cfg['NUM_ENVS']*cfg['NUM_STEPS'])
            for task,metric in METRICS.items():
                if row.get(metric) is not None and np.isfinite(row[metric]):points[task][step]=float(row[metric])
        arrays={k:np.asarray(sorted(v.items())) for k,v in points.items()}
        if any(len(v)==0 or v[-1,0]<.99*cfg['TOTAL_TIMESTEPS'] for v in arrays.values()):
            raise RuntimeError(f'Synced history also incomplete for {ref}; refusing a misleading full-budget plot')
        return arrays,dict(type='synced W&B history',run=ref)
    return arrays,dict(type='local W&B history',file=str(files[0]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline_run',default=str(BASELINE))
    p.add_argument('--penalty_run',default=str(ROOT/'wandb/run-20261006_194739-uj5n3za2'))
    p.add_argument('--out',default=str(ROOT/'concept_mapping/runs/death_penalty2_vs_baseline_10b_seed50'))
    p.add_argument('--window_updates',type=int,default=1024)
    p.add_argument('--xscale',choices=['linear','log'],default='log')
    p.add_argument('--reuse_cached_history',action='store_true',help='Re-render from the saved raw CSV histories without fetching W&B again.')
    args=p.parse_args()
    runs={'baseline':Path(args.baseline_run).resolve(),'death_penalty2':Path(args.penalty_run).resolve()}
    configs={k:config(v) for k,v in runs.items()}
    for field in ['SEED','NUM_ENVS','NUM_STEPS','TOTAL_TIMESTEPS','GAMMA','GAE_LAMBDA',
                  'UPDATE_EPOCHS','NUM_MINIBATCHES','LR','ANNEAL_LR','CLIP_EPS','VF_COEF','ENT_COEF','MAX_GRAD_NORM']:
        if configs['baseline'][field]!=configs['death_penalty2'][field]:raise ValueError(f'Unmatched {field}')
    assert configs['baseline']['SEED']==50 and configs['baseline'].get('DEATH_PENALTY',0)==0
    assert configs['death_penalty2']['DEATH_PENALTY']==2
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    histories,sources={},{}
    if args.reuse_cached_history:
        previous=json.loads((out/'metadata.json').read_text())
        if previous['runs']!={k:str(v) for k,v in runs.items()}:raise ValueError('Cached histories belong to different runs')
        sources=previous['sources']
        for name in runs:
            histories[name]={}
            for task in METRICS:
                with (out/f'{task}_{name}_history.csv').open() as f:rows=list(csv.DictReader(f))
                histories[name][task]=np.asarray([(int(r['environment_steps']),float(r['success_raw_pct'])) for r in rows])
    else:
        for name,run in runs.items():histories[name],sources[name]=history(run,configs[name])
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
                         'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.default':'bf'})
    summary={}
    for task,title in [('diamond','Diamond'),('plant','Plant')]:
        fig,ax=plt.subplots(figsize=(5.0,3.5),layout='constrained')
        summary[task]={}
        for name,label,color,style in [('baseline','Baseline PPO','#2563eb','-'),
                                        ('death_penalty2','PPO with death cost −2','#dc2626','-')]:
            data=histories[name][task]
            steps,raw=data[:,0],data[:,1]
            if np.any((raw < -1e-3)|(raw > 100+1e-3)):
                raise ValueError(f'History is not in percent units: {name}/{task} range {raw.min()}–{raw.max()}')
            raw=np.clip(raw,0,100) # Float32 achievement means can exceed 100 by roundoff.
            smooth=_centered_rolling_mean(raw,min(args.window_updates,len(raw)))
            with (out/f'{task}_{name}_history.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['environment_steps','success_raw_pct','success_smoothed_pct'])
                writer.writerows(zip(steps.astype(np.int64),raw,smooth))
            ax.plot(steps,smooth,label=label,color=color,linestyle=style,
                    linewidth=2.0 if name=='baseline' else 1.8)
            summary[task][name]=dict(history_points=len(raw),last_environment_step=int(steps[-1]),
                last_smoothed_success_pct=float(smooth[-1]),tail_1024_update_mean_pct=float(raw[-1024:].mean()))
        ax.set(title=f'{title} (smoothed)',xlabel='Training environment steps',ylabel='Success rate (%)',ylim=(-2,102))
        ax.set_yticks([0,20,40,60,80,100])
        if args.xscale=='linear':ax.set_xlim(0,configs['baseline']['TOTAL_TIMESTEPS'])
        else:
            ax.set_xscale('log')
            ax.set_xlim(1e5,1e10)
            ax.xaxis.set_major_locator(FixedLocator([10.0**k for k in range(5,11)]))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.grid(alpha=.22)
        for label in (*ax.get_xticklabels(),*ax.get_yticklabels()):
            label.set_fontweight('bold')
        ax.legend(frameon=False,prop={'size':8,'weight':'bold'},loc='upper left')
        for suffix in ['pdf','png','jpeg']:
            kwargs={'pil_kwargs':{'quality':95}} if suffix=='jpeg' else {}
            fig.savefig(out/f'{task}_success_comparison.{suffix}',dpi=300,**kwargs)
        plt.close(fig)
    metadata=dict(runs={k:str(v) for k,v in runs.items()},sources=sources,seed=50,
                  smoothing='centered rolling mean over logged PPO update success rates; edge windows shortened',
                  window_updates=args.window_updates,
                  window_environment_steps=args.window_updates*configs['baseline']['NUM_ENVS']*configs['baseline']['NUM_STEPS'],
                  xscale=args.xscale,summary=summary,
                  interpretation='training episode achievement percentages, NOT frozen-policy evaluation or pooled episode-weighted rates')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

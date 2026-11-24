"""Parallel frozen-checkpoint MCTS evaluations and paired episode summaries."""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PYTHON='/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
STUDY=ROOT/'concept_mapping/runs/ppo_guided_oracle_mcts_1m'
RUN=ROOT/'wandb/run-20261102_235443-pmbb8pb7'
SETTINGS=[('search32_depth5',0,32,5,8,1.),('search64_depth10',1,64,10,17,1.),
          ('search128_depth15',2,128,15,17,1.),('search128_depth15_greedy',3,128,15,17,0.)]


def summarize(study):
    baseline=study/'baseline_sampled'
    b=json.loads((baseline/'summary.json').read_text())
    with (baseline/'episodes.csv').open() as f:base={int(r['episode']):float(r['total_reward']) for r in csv.DictReader(f)}
    rows=[dict(name='PPO sampled',simulations=0,depth=0,candidates=0,gumbel_scale=None,
        return_=b['mean_return'],achievements=b['mean_achievements'],paired_gain=0.,ci_low=0.,ci_high=0.,
        mean_actual_max_depth=None,mean_root_actions=None,depth_limit_hit_pct=None,
        simulator_search_calls=0,seconds=b['seconds'])]
    rates=[dict(name='PPO sampled',**b['achievement_success_pct'])]
    for name,gpu,sims,depth,candidates,gumbel in SETTINGS:
        path=study/name
        if not (path/'summary.json').exists():continue
        d=json.loads((path/'summary.json').read_text())
        with (path/'episodes.csv').open() as f:episodes=list(csv.DictReader(f))
        gains=np.array([float(r['total_reward'])-base[int(r['episode'])] for r in episodes])
        rng=np.random.default_rng(50)
        means=gains[rng.integers(len(gains),size=(10000,len(gains)))].mean(-1)
        low,high=np.quantile(means,[.025,.975]);stats=d['search_diagnostics']
        rows.append(dict(name=name,simulations=sims,depth=depth,candidates=candidates,gumbel_scale=gumbel,
            return_=d['mean_return'],achievements=d['mean_achievements'],paired_gain=float(gains.mean()),
            ci_low=float(low),ci_high=float(high),mean_actual_max_depth=stats['max_depth_reached'],
            mean_root_actions=stats['root_actions_visited'],depth_limit_hit_pct=100*stats['hit_depth_limit'],
            simulator_search_calls=d['executed_search_transitions'],seconds=d['seconds']))
        rates.append(dict(name=name,**d['achievement_success_pct']))
    for filename,data in [('comparison.csv',rows),('achievement_success_rates.csv',rates)]:
        with (study/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Frozen 1M PPO with actual-simulator MCTS','',
        'One frozen seed-50 checkpoint, 128 common standard-reset evaluation episodes (seed 15050). No training or parameter updates. Confidence intervals are 10,000 paired episode-bootstrap resamples, conditional on this checkpoint; they do not measure training-seed variability. Search has privileged simulator-state access and extra simulator cost.','',
        '| Setting | Return | Achievements | Gain over PPO [95% CI] | Mean maximum depth reached |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        reached='—' if r['mean_actual_max_depth'] is None else f"{r['mean_actual_max_depth']:.2f}"
        lines.append(f"| {r['name']} | {r['return_']:.3f} | {r['achievements']:.3f} | {r['paired_gain']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {reached} |")
    lines+=['','Search settings are exploratory comparisons on the same worlds. The largest observed gain is not an independent validation result. All achievement rates, root coverage, actual depth, depth-cap hits, wall time, and executed simulator calls are in the CSVs and per-setting summaries.']
    (study/'results.md').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summarize-only',action='store_true');args=p.parse_args()
    if args.summarize_only:summarize(STUDY);return
    children=[]
    for name,gpu,sims,depth,candidates,gumbel in SETTINGS:
        if (STUDY/name).exists():raise FileExistsError(STUDY/name)
        free=int(subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        if free<14000:raise RuntimeError(f'GPU {gpu} occupied; existing jobs will not be interrupted')
    commands=[]
    for name,gpu,sims,depth,candidates,gumbel in SETTINGS:
        command=[PYTHON,'-u',str(ROOT/'evaluate_ppo_mcts.py'),'--run',str(RUN),'--out',str(STUDY/name),
            '--simulations',str(sims),'--max-depth',str(depth),'--candidates',str(candidates),'--gumbel-scale',str(gumbel)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
            OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        log=(STUDY/f'{name}.log').open('w')
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((name,proc,log));commands.append(dict(name=name,gpu=gpu,pid=proc.pid,command=command))
    (STUDY/'launch_manifest.json').write_text(json.dumps(dict(time=time.time(),jobs=commands),indent=2)+'\n')
    pending=children.copy();failed=[]
    while pending:
        for child in pending.copy():
            name,proc,log=child;code=proc.poll()
            if code is None:continue
            log.close();pending.remove(child)
            if code:failed.append(dict(name=name,exit_code=code))
            else:summarize(STUDY)
        (STUDY/'progress.json').write_text(json.dumps(dict(state='evaluating' if pending else ('failed' if failed else 'complete'),
            running=[c[0] for c in pending],failed=failed,time=time.time()),indent=2)+'\n')
        if pending:time.sleep(15)
    summarize(STUDY)
    if failed:raise RuntimeError(failed)


if __name__=='__main__':main()

"""Prepare a fixed-world fine-tune and evaluate practiced/unseen worlds.

Training itself is launched by run_ppo_study.py. This tool organizes the
selection, independent retries, and paired checkpoint evaluations.
"""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
from run_reward_shaped_architectures import ROOT,PYTHON,environment,write_json,checkpoint_ready


def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def setup(study):
    selected=rows(study/'selection_baseline/episodes.csv')
    failed=[r for r in selected if r['all_22']!='True']
    seeds=[int(r['world_seed']) for r in failed]
    if not seeds:raise RuntimeError('All 100 worlds succeeded; no failed-world bank to train on')
    write_json(study/'failed_worlds.json',dict(world_seeds=seeds,selected_from=100,
        selection_seed=49150,criterion='not all 22 achievements in one observed rollout'))
    source=ROOT/'wandb/run-20261006_194739-uj5n3za2/files/policies/9999941632'
    start=int(source.name);extra=50000000
    spec=dict(name='death2_10b_failed_world_finetune_seed50',seed=50,resume_from=str(source),
        resume_reset_optimizer=True,
        architectures=[dict(name='mlp',train_gpu=3,evaluation_gpu=7)],
        rewards=dict(death_penalty=2),
        ppo=dict(lr=.00002,optimistic_resets=False,reset_seed_bank=str((study/'failed_worlds.json').resolve())),
        budget=dict(max_steps=start+extra,max_hours=9),
        checkpoint_milestones=[start+x for x in (1000000,5000000,10000000,20000000)],
        evaluation=dict(common_checkpoints={},common_episodes=100,final_episodes=100,batch=20,seed=49152))
    write_json(study/'finetune_config.json',spec)
    report(study)
    return len(failed)


def report(study):
    selection=rows(study/'selection_baseline/episodes.csv')
    failed=[r for r in selection if r['all_22']!='True']
    datasets={}
    for p in sorted(study.glob('evaluation_*/episodes.csv')):
        datasets[p.parent.name.removeprefix('evaluation_')]=rows(p)
    fields=list(json.loads((study/'selection_baseline/summary.json').read_text())['achievement_success_pct'])
    per_world=[]
    for r in failed:
        seed=int(r['world_seed'])
        row=dict(world_seed=seed,initial_achievements=sum(int(r[k]) for k in fields),
                 initial_missing=';'.join(k for k in fields if not int(r[k])))
        for name,data in datasets.items():
            if not name.endswith('_failed'):continue
            matches=[x for x in data if int(x['world_seed'])==seed]
            row[name+'_successes']=sum(x['all_22']=='True' for x in matches)
            row[name+'_trials']=len(matches)
        per_world.append(row)
    if per_world:
        with (study/'per_world_results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(per_world[0]));writer.writeheader();writer.writerows(per_world)
    aggregate=[]
    for name,data in datasets.items():
        aggregate.append(dict(condition=name,episodes=len(data),
            all22_pct=100*sum(x['all_22']=='True' for x in data)/len(data),
            mean_achievements=sum(sum(int(x[k]) for k in fields) for x in data)/len(data)))
    if aggregate:
        with (study/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    lines=['# Failed-world memorization pilot','',
        f'Source: feedforward PPO seed 50, checkpoint 9,999,941,632; general death cost −2, default achievement rewards. Initial fixed-world evaluation: {len(selection)-len(failed)}/{len(selection)} completed all 22; {len(failed)} worlds selected for practice.','',
        'World seeds come from a fixed random draw (generator seed 41150); failed worlds are selected by one stochastic rollout (selection action/dynamics seed 49150). The normal generator, initial inventory, mechanics and 10,000-step cap are unchanged. No seed ID or achievement flags are added to policy observations.','',
        'Train only on the selected worlds, sampled uniformly on each standard reset. Start from the original actor/critic weights; reset Adam and anneal a lower initial LR 2e−5 over 50M additional interactions. All other PPO settings and rewards are retained; independent resets replace the historical optimistic reset scheme. This is a selected-world fine-tuning pilot, not a controlled estimate of the benefit over equal-budget normal-world training.','',
        'Before/after evaluations retry each failed world five times with independent action/dynamics randomness (seed 49151). Those evaluation streams are never used for training. The same 100 disjoint held-out worlds use seed 49152. The final original-100 evaluation repeats the initial seed 49150. A failure in the selection rollout does not establish that a map is intrinsically hard or impossible.','',
        '| Evaluation | Episodes | All 22 (%) | Mean achievements |',
        '| --- | ---: | ---: | ---: |',
        *[f'| {r["condition"]} | {r["episodes"]} | {r["all22_pct"]:.2f} | {r["mean_achievements"]:.3f} |' for r in aggregate],
        '', 'Per-world outcomes and initially missing achievements are in `per_world_results.csv`. `finetune/` contains the training manifest, W&B link, checkpoints and progress. Checkpoint labels include the original 10B training; evaluation directory labels denote additional interactions.','']
    (study/'README.md').write_text('\n'.join(lines))


def evaluate(study,label,checkpoint,bank,repeats,seed,gpu,config):
    out=study/f'evaluation_{label}'
    if (out/'summary.json').exists():return
    if out.exists():raise RuntimeError(f'Incomplete previous evaluation retained: {out}')
    command=[PYTHON,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),'--checkpoint',str(checkpoint),
        '--config-json',str(config),'--world-seeds-file',str(bank),'--repeats',str(repeats),
        '--seed',str(seed),'--batch','64','--out',str(out)]
    with (study/f'evaluation_{label}.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
    report(study)


def watch(study,gpu):
    spec=json.loads((study/'finetune_config.json').read_text())
    source=Path(spec['resume_from']);start=int(source.name)
    source_cfg=study/'source_inference_config.json'
    evaluate(study,'baseline_failed',source,study/'failed_worlds.json',5,49151,gpu,source_cfg)
    evaluate(study,'baseline_heldout',source,study/'heldout_worlds.json',1,49152,gpu,source_cfg)
    pending={name:start+(step+65535)//65536*65536 for name,step in [('1m',1000000),('5m',5000000),('20m',20000000)]}
    while True:
        progress_path=study/'finetune/mlp/progress.json'
        manifest_path=study/'finetune/mlp/run_manifest.json'
        if not progress_path.exists() or not manifest_path.exists():time.sleep(10);continue
        progress=json.loads(progress_path.read_text());manifest=json.loads(manifest_path.read_text())
        local=Path(manifest['local_run'])/'files/policies'
        config=study/'finetune/mlp/inference_config.json'
        for label,step in list(pending.items()):
            checkpoint=local/str(step)
            if checkpoint_ready(checkpoint):
                evaluate(study,label+'_failed',checkpoint,study/'failed_worlds.json',5,49151,gpu,config)
                del pending[label]
        if progress.get('final_checkpoint'):
            checkpoint=Path(progress['final_checkpoint'])
            evaluate(study,'final_failed',checkpoint,study/'failed_worlds.json',5,49151,gpu,config)
            evaluate(study,'final_heldout',checkpoint,study/'heldout_worlds.json',1,49152,gpu,config)
            evaluate(study,'final_original100',checkpoint,study/'selection_worlds.json',1,49150,gpu,config)
            write_json(study/'analysis_complete.json',dict(time=time.time(),checkpoint=str(checkpoint),
                additional_training_steps=int(checkpoint.name)-start))
            report(study);return
        if progress.get('state')=='wall_clock_complete' or time.time()>progress['overall_deadline']:
            report(study);return
        time.sleep(10)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--setup',action='store_true')
    p.add_argument('--watch',action='store_true');p.add_argument('--gpu',type=int,default=6)
    args=p.parse_args();study=args.study.resolve()
    if args.setup:print('Selected failed worlds:',setup(study),flush=True)
    if args.watch:watch(study,args.gpu)


if __name__=='__main__':main()

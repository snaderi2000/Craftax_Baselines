"""Compare matched 11B and 12.5B SRU checkpoints as they become available."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from cap_reward_shaped_runs import try_evaluation_lock
from run_ppo_study import require_free
from run_reward_shaped_architectures import ROOT, checkpoint_ready, environment, write_json

RUNS = ROOT / 'concept_mapping/runs'
STUDIES = {'regular': RUNS/'sru_gru_motion_to12p5b_seed50/gru',
           'reconstruction': RUNS/'sru_gru_map_reconstruction_to12p5b_seed50/gru'}
CHECKPOINTS = {'plus400m': 11000020992, 'wallclock': 12500008960}


def delta_interval(hits_a, counts_a, hits_b, counts_b):
    """Paired bootstrap over independent worlds; retain within-world query correlations."""
    rng = np.random.default_rng(90514)
    a, n, b, m = map(np.asarray, (hits_a, counts_a, hits_b, counts_b))
    rows = rng.integers(len(a), size=(2000, len(a)))
    na, nb = n[rows].sum(1), m[rows].sum(1)
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return None
    draws = a[rows].sum(1)[ok]/na[ok] - b[rows].sum(1)[ok]/nb[ok]
    return dict(difference=float(a.sum()/n.sum() - b.sum()/m.sum()),
                world_bootstrap_95=np.quantile(draws, [.025, .975]).tolist())


def read(path):
    return json.loads(path.read_text())


def compare_gameplay(label, step):
    tables = {}
    for arm, study in STUDIES.items():
        p = study / ('evaluation_'+label)
        if not (p/'summary.json').exists():
            return None
        summary = read(p/'summary.json')
        recorded = Path(str(summary['checkpoint'])).name
        if int(recorded) != step:
            raise ValueError(f'{arm} evaluation ended at {recorded}, expected {step}')
        with (p/'episodes.csv').open() as f:
            tables[arm] = {int(row['world_seed']): row['all_22']=='True' for row in csv.DictReader(f)}
    worlds = sorted(tables['regular'])
    if len(worlds)!=100 or set(worlds)!=set(tables['reconstruction']):
        raise ValueError('Unmatched gameplay worlds')
    reg, aux = (np.array([tables[arm][k] for k in worlds]) for arm in STUDIES)
    counts = np.ones(len(reg))
    return dict(worlds=len(worlds), regular_all22=int(reg.sum()), reconstruction_all22=int(aux.sum()),
                gained=int((aux & ~reg).sum()), lost=int((reg & ~aux).sum()),
                **delta_interval(aux, counts, reg, counts))


def run(out, gpu):
    out.mkdir(parents=True, exist_ok=True)
    pending = dict(CHECKPOINTS)
    while pending:
        for label, step in list(pending.items()):
            folder = out/label
            for arm, study in STUDIES.items():
                dest = folder/(arm+'.json')
                progress = study/'progress.json'
                if dest.exists() or not progress.exists():
                    continue
                status = read(progress)
                ckpt = Path(status.get('local_run',''))/'files/policies'/str(step)
                if not checkpoint_ready(ckpt):
                    if status.get('state') in ('failed','complete') and int(status.get('latest_checkpoint') or 0) < step:
                        raise RuntimeError(f'{arm} ended before {step}: inspect {progress}')
                    continue
                lock = try_evaluation_lock(gpu)
                if lock is None:
                    continue
                with lock:
                    try:
                        require_free(gpu)
                    except RuntimeError:
                        continue
                    subprocess.run([sys.executable,'-u','-m','concept_mapping.probe_sru_map_recall',
                        '--config',str(study/'inference_config.json'),'--checkpoint',str(ckpt),
                        '--out',str(dest)],cwd=ROOT,env=environment(gpu),check=True)
            if not all((folder/(arm+'.json')).exists() for arm in STUDIES):
                continue
            reports = {a:read(folder/(a+'.json'))['per_test_world'] for a in STUDIES}
            stats = {}
            for key, observed in reports['reconstruction'].items():
                ref = reports['regular'][key]
                stats[key] = delta_interval(observed['hits'],observed['count'],ref['hits'],ref['count'])
            gameplay = compare_gameplay(label, step)
            write_json(folder/'comparison.json', dict(step=step, reconstruction_minus_regular=stats,gameplay=gameplay))
            if gameplay:
                del pending[label]
        if pending:
            time.sleep(60)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    run(STUDIES['reconstruction'].parent/'paired_comparison', args.gpu)

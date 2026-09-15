"""Run seven deterministic collection shards and maintain aggregate progress."""
import csv
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "concept_mapping/runs/master_symbolic_10b_5000episodes"


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--out_dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = OUT / 'dataset_metadata.json'
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous['seed'] != args.seed or previous['checkpoint'] != 9999941632:
            raise ValueError('Output metadata does not match seed/checkpoint; use a separate directory')
    os.chdir(ROOT)
    processes, logs = [], []
    started = time.time()
    try:
        for worker in range(7):
            directory = OUT / "workers" / f"worker_{worker}"
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "collection.log").open("a")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker + 1),
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       MPLCONFIGDIR=f"/tmp/mpl-master10b-worker{worker}")
            command = [sys.executable, "-u", "concept_mapping/collect_master_symbolic_dataset.py",
                       "--run_path", "wandb/run-20260630_214658-m0mw4end",
                       "--timestep", "9999941632", "--episodes", "5000",
                       "--max_steps", "10000", "--seed", str(args.seed), "--out_dir", str(OUT),
                       "--worker_index", str(worker), "--num_workers", "7"]
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [p.poll() for p in processes]
            paths = sorted((OUT / "episodes").glob("episode_*.npz"))
            progress = {"completed_episodes": len(paths), "requested_episodes": 5000,
                        "seed": args.seed,
                        "num_workers": 7, "gpu_ids": list(range(1, 8)),
                        "worker_exit_codes": codes, "elapsed_seconds": time.time() - started,
                        "updated_unix_time": time.time()}
            temp = OUT / ".parallel_progress.tmp.json"
            temp.write_text(json.dumps(progress, indent=2) + "\n")
            temp.replace(OUT / "collection_progress.json")
            if all(code is not None for code in codes):
                if any(code != 0 for code in codes):
                    raise RuntimeError(f"Workers failed: {codes}; see workers/*/collection.log")
                break
            time.sleep(10)
        # Rebuild the manifest from saved episodes, including the earlier serial
        # collector's files, so interruptions cannot duplicate or omit rows.
        rows = []
        action_names = ['NOOP', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'DO', 'SLEEP',
                        'PLACE_STONE', 'PLACE_TABLE', 'PLACE_FURNACE', 'PLACE_PLANT',
                        'MAKE_WOOD_PICKAXE', 'MAKE_STONE_PICKAXE', 'MAKE_IRON_PICKAXE',
                        'MAKE_WOOD_SWORD', 'MAKE_STONE_SWORD', 'MAKE_IRON_SWORD']
        for path in paths:
            with np.load(path) as data:
                action = int(data['actions'][-1])
                rows.append(dict(episode_id=int(data['episode_id']), checkpoint=int(data['checkpoint']),
                                 n_pre_action_states=len(data['actions']), terminated=int(data['dones'][-1]),
                                 final_reward=float(data['rewards'][-1]), final_action=action,
                                 final_action_name=action_names[action], file=str(path.relative_to(OUT))))
        if {r['episode_id'] for r in rows} != set(range(5000)):
            raise RuntimeError('Episode IDs do not cover exactly 0 through 4999')
        temp = OUT / '.episode_manifest.tmp.csv'
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(OUT / 'episode_manifest.csv')
        print('Complete: 5,000 episodes validated and manifest merged.', flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()

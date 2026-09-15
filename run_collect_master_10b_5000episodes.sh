#!/usr/bin/env bash
set -euo pipefail
cd /home/shawheen/Craftax_Baselines
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/mpl-master-10b-5000
OUT=concept_mapping/runs/master_symbolic_10b_5000episodes
mkdir -p "$OUT"
exec /home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u \
  concept_mapping/collect_master_symbolic_dataset.py \
  --run_path wandb/run-20260630_214658-m0mw4end \
  --timestep 9999941632 --episodes 5000 --max_steps 10000 --seed 50 \
  --out_dir "$OUT" >> "$OUT/collection.log" 2>&1

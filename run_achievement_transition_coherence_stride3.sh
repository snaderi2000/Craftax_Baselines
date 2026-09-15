#!/usr/bin/env bash
set -euo pipefail

cd /home/shawheen/Craftax_Baselines

PYTHON=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
SCRIPT=concept_mapping/analyze_achievement_transition_coherence.py
OUT=concept_mapping/runs/master_symbolic_1p5b_1000episodes/achievement_transition_coherence_stride3
mkdir -p "$OUT/logs"

declare -a worker_pids
for worker in 0 1 2 3 4 5 6 7; do
    MPLCONFIGDIR="/tmp/mpl-achievement-coherence-w${worker}" \
        CUDA_VISIBLE_DEVICES="$worker" \
        OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        "$PYTHON" -u "$SCRIPT" \
        --checkpoint_stride 3 \
        --worker_index "$worker" \
        --num_workers 8 \
        > "$OUT/logs/worker_${worker}.log" 2>&1 &
    worker_pids[$worker]=$!
done

for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid"
done

MPLCONFIGDIR=/tmp/mpl-achievement-coherence-aggregate \
    "$PYTHON" -u "$SCRIPT" \
    --checkpoint_stride 3 \
    --aggregate_only \
    > "$OUT/logs/aggregate.log" 2>&1

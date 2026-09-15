#!/usr/bin/env bash
set -euo pipefail

cd /home/shawheen/Craftax_Baselines

PYTHON=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
SCRIPT=concept_mapping/analyze_achievement_stabilized_effect_over_checkpoints.py
OUT=concept_mapping/runs/master_symbolic_1p5b_1000episodes/achievement_stabilized_effect_over_checkpoints_stride3
mkdir -p "$OUT/logs"

for worker in 0 1 2 3 4 5 6 7; do
    MPLCONFIGDIR="/tmp/mpl-achievement-w${worker}" \
        CUDA_VISIBLE_DEVICES="$worker" \
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

MPLCONFIGDIR=/tmp/mpl-achievement-aggregate \
    "$PYTHON" -u "$SCRIPT" \
    --checkpoint_stride 3 \
    --aggregate_only \
    > "$OUT/logs/aggregate.log" 2>&1

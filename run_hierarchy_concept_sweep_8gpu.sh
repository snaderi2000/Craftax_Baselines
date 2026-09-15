#!/usr/bin/env bash
# Resumable 8-GPU checkpoint sweep. Each worker owns disjoint checkpoint rows.
set -euo pipefail

ROOT_DIR="/home/shawheen/Craftax_Baselines"
PYTHON_BIN="/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python"
OUT_DIR="concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/checkpoint_sweep"
mkdir -p "$ROOT_DIR/$OUT_DIR/logs"
cd "$ROOT_DIR"

for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u concept_mapping/analyze_hierarchy_concept_sweep.py \
    --worker_index "$gpu" --num_workers 8 > "$OUT_DIR/logs/worker_${gpu}.log" 2>&1 &
done
wait

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u concept_mapping/analyze_hierarchy_concept_sweep.py --aggregate_only \
  > "$OUT_DIR/logs/aggregate.log" 2>&1

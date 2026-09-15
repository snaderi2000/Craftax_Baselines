#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="/home/shawheen/Craftax_Baselines"
PYTHON_BIN="/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python"
OUT_DIR="concept_mapping/runs/master_symbolic_1p5b_1000episodes/hierarchy_concept_banks/progression_vs_pickaxe_value/progression_checkpoint_sweep_stride2"
mkdir -p "$ROOT_DIR/$OUT_DIR/logs"
cd "$ROOT_DIR"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u concept_mapping/analyze_progression_heuristic_over_checkpoints.py \
    --checkpoint_stride 2 --worker_index "$gpu" --num_workers 8 \
    > "$OUT_DIR/logs/worker_${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u concept_mapping/analyze_progression_heuristic_over_checkpoints.py \
  --checkpoint_stride 2 --aggregate_only > "$OUT_DIR/logs/aggregate.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=/home/shawheen/Craftax_Baselines
PYTHON_BIN=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
OUT_ROOT=concept_mapping/runs/inventory_progress_hierarchy_3gpu_m0mw4end
cd "$REPO_ROOT"; mkdir -p "$OUT_ROOT"
PIDS=()
for SPEC in "2:0:101" "3:1:202" "4:2:303"; do
  IFS=: read -r GPU SHARD SEED <<< "$SPEC"
  mkdir -p "$OUT_ROOT/shard_$SHARD"
  BASE_ARGS=()
  if [[ -f "$OUT_ROOT/shard_$SHARD/base_states.npz" ]]; then
    BASE_ARGS=(--base_states "$OUT_ROOT/shard_$SHARD/base_states.npz")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u concept_mapping/inventory_progress_counterfactual.py \
    --out_dir "$OUT_ROOT/shard_$SHARD" --target_states 500 --seed "$SEED" \
    "${BASE_ARGS[@]}" \
    > "$OUT_ROOT/shard_$SHARD.log" 2>&1 &
  PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done
"$PYTHON_BIN" concept_mapping/analyze_inventory_progress_counterfactual.py \
  --shard_root "$OUT_ROOT" --out_dir "$OUT_ROOT/merged"

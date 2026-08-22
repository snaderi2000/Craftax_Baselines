#!/usr/bin/env bash
set -euo pipefail

# Collect 5,000 fresh, clean diamond/pickaxe counterfactual base states in
# four independent GPU shards, then merge their results.
REPO_ROOT=/home/shawheen/Craftax_Baselines
PYTHON_BIN=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
RUN_PATH=wandb/run-20260630_214658-m0mw4end
TIMESTEP=9999941632
OUT_ROOT=concept_mapping/runs/diamond_affordance_clean_5000_4gpu_m0mw4end
TARGET_PER_SHARD=1250

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"

PIDS=()
for SPEC in "1:0" "2:1" "3:2" "4:3"; do
  GPU_ID=${SPEC%%:*}
  SHARD_ID=${SPEC##*:}
  SHARD_DIR="$OUT_ROOT/shard_$SHARD_ID"
  LOG_FILE="$OUT_ROOT/shard_$SHARD_ID.log"
  mkdir -p "$SHARD_DIR"

  echo "Launching shard $SHARD_ID on GPU $GPU_ID; log: $LOG_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u \
    concept_mapping/diamond_affordance_clean_filter.py \
    --run_path "$RUN_PATH" \
    --out_dir "$SHARD_DIR" \
    --timestep "$TIMESTEP" \
    --target_states "$TARGET_PER_SHARD" \
    --fresh \
    --max_extra_episodes 30000 \
    --max_steps 4096 \
    --lookback 5 \
    --batch_size 512 \
    --render_examples 0 \
    --seed "$((7 + SHARD_ID))" \
    --extra_seed "$((107 + SHARD_ID))" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  wait "$PID"
done

"$PYTHON_BIN" concept_mapping/merge_diamond_affordance_shards.py \
  --shard_root "$OUT_ROOT" \
  --out_dir "$OUT_ROOT/merged"

echo "Finished. Merged results: $OUT_ROOT/merged"

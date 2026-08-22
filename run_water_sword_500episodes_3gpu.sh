#!/usr/bin/env bash
set -euo pipefail

# Exactly 500 frozen-policy episodes, split across GPUs 2, 3, and 4.
REPO_ROOT=/home/shawheen/Craftax_Baselines
PYTHON_BIN=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
RUN_PATH=wandb/run-20260630_214658-m0mw4end
TIMESTEP=9999941632
OUT_ROOT=concept_mapping/runs/water_sword_500episodes_3gpu_m0mw4end

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"

PIDS=()
for SPEC in "2:0:167" "3:1:167" "4:2:166"; do
  IFS=: read -r GPU_ID SHARD_ID EPISODES <<< "$SPEC"
  SHARD_DIR="$OUT_ROOT/shard_$SHARD_ID"
  LOG_FILE="$OUT_ROOT/shard_$SHARD_ID.log"
  mkdir -p "$SHARD_DIR"
  echo "Launching shard $SHARD_ID: $EPISODES episodes on GPU $GPU_ID; log: $LOG_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u \
    concept_mapping/collect_water_sword_counterfactuals.py \
    --run_path "$RUN_PATH" \
    --out_dir "$SHARD_DIR" \
    --timestep "$TIMESTEP" \
    --max_episodes "$EPISODES" \
    --water_target 501 \
    --sword_target 501 \
    --max_per_episode_per_task 3 \
    --min_spacing 256 \
    --lookback 5 \
    --max_steps 4096 \
    --batch_size 512 \
    --render_examples 0 \
    --run_all_episodes \
    --allow_partial \
    --seed "$((211 + SHARD_ID))" \
    > "$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done

for PID in "${PIDS[@]}"; do
  wait "$PID"
done

"$PYTHON_BIN" concept_mapping/merge_water_sword_shards.py \
  --shard_root "$OUT_ROOT" \
  --out_dir "$OUT_ROOT/merged"

echo "Finished. Merged data: $OUT_ROOT/merged"

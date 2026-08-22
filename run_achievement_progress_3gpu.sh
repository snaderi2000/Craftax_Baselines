#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/shawheen/Craftax_Baselines
PYTHON_BIN=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
RUN_PATH=wandb/run-20260630_214658-m0mw4end
OUT_ROOT=concept_mapping/runs/achievement_progress_logcheckpoints_m0mw4end

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"
PIDS=()
for SPEC in "2:0:65536,131072,1048576" "3:1:10027008,100007936" "4:2:1000013824,9999941632"; do
  IFS=: read -r GPU_ID SHARD_ID CHECKPOINTS <<< "$SPEC"
  SHARD_DIR="$OUT_ROOT/shard_$SHARD_ID"
  LOG_FILE="$OUT_ROOT/shard_$SHARD_ID.log"
  mkdir -p "$SHARD_DIR"
  echo "Launching $CHECKPOINTS on GPU $GPU_ID; log: $LOG_FILE"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u concept_mapping/evaluate_achievements_at_checkpoints.py \
    --run_path "$RUN_PATH" --out_dir "$SHARD_DIR" --checkpoints "$CHECKPOINTS" \
    --episodes 59 --max_steps 10000 --seed 50 --greedy \
    > "$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done
"$PYTHON_BIN" concept_mapping/merge_achievement_checkpoint_evaluations.py \
  --shard_root "$OUT_ROOT" --out_dir "$OUT_ROOT/merged"
echo "Finished. Outputs: $OUT_ROOT/merged"

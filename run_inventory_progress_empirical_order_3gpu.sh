#!/usr/bin/env bash
# Evaluate the existing 1,500 frozen-policy base states along their empirically
# observed first-acquisition order. No new environment data are collected.
set -euo pipefail

REPO_ROOT=/home/shawheen/Craftax_Baselines
PYTHON_BIN=/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python
SOURCE_ROOT=concept_mapping/runs/inventory_progress_hierarchy_3gpu_m0mw4end
OUT_ROOT=concept_mapping/runs/inventory_progress_empirical_order_3gpu_m0mw4end
ORDER=sapling,wood,wood_pickaxe,stone,wood_sword,stone_pickaxe,coal,iron,stone_sword,iron_sword,iron_pickaxe,diamond

cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"
PIDS=()
for SPEC in "2:0:101" "3:1:202" "4:2:303"; do
  IFS=: read -r GPU SHARD SEED <<< "$SPEC"
  mkdir -p "$OUT_ROOT/shard_$SHARD"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u concept_mapping/inventory_progress_counterfactual.py \
    --out_dir "$OUT_ROOT/shard_$SHARD" \
    --base_states "$SOURCE_ROOT/shard_$SHARD/base_states.npz" \
    --seed "$SEED" --order "$ORDER" \
    > "$OUT_ROOT/shard_$SHARD.log" 2>&1 &
  PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done
"$PYTHON_BIN" concept_mapping/analyze_inventory_progress_counterfactual.py \
  --shard_root "$OUT_ROOT" --out_dir "$OUT_ROOT/merged"
"$PYTHON_BIN" concept_mapping/analyze_inventory_item_additions.py \
  --shard_root "$OUT_ROOT" --out_dir "$OUT_ROOT/merged"

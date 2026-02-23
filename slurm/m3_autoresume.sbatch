#!/usr/bin/env bash
#SBATCH --job-name=m3_autoresume
#SBATCH --gpus=7g.80gb:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail

# -------------------------------
# User-configurable settings
# -------------------------------
CONDA_ENV="${CONDA_ENV:-craftax}"
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_DIR/checkpoints}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"

# Optional wandb overrides:
# export WANDB_PROJECT=craftax-mbrl
# export WANDB_ENTITY=your_entity
WANDB_PROJECT="${WANDB_PROJECT:-craftax-mbrl}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "$PROJECT_DIR/slurm"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
unset LD_LIBRARY_PATH || true

latest_checkpoint_dir() {
  local newest
  newest="$(ls -1d "$CKPT_DIR"/checkpoint_* 2>/dev/null | awk -F_ '{print $NF " " $0}' | sort -n | tail -n1 | cut -d' ' -f2- || true)"
  echo "$newest"
}

latest_checkpoint_step() {
  local ckpt
  ckpt="$(latest_checkpoint_dir)"
  if [[ -z "${ckpt}" ]]; then
    echo 0
    return
  fi
  basename "$ckpt" | awk -F_ '{print $2}'
}

START_STEP="$(latest_checkpoint_step)"
RESUME_CKPT="$(latest_checkpoint_dir)"

echo "[$(date)] Starting job ${SLURM_JOB_ID:-N/A} from step ${START_STEP}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "[$(date)] Resuming from checkpoint: ${RESUME_CKPT}"
else
  echo "[$(date)] No checkpoint found; starting from scratch."
fi

CMD=(python ppo_mbrl_m3.py
  --n_iters_tok 500
  --n_iters_twm 500
  --n_mb_wm 3
  --n_iters_ac 150
  --dyna_enable_real_updates
  --n_iters_real_ac 1
  --background_planning_start 200000
  --total_timesteps "${TOTAL_TIMESTEPS}"
  --use_wandb
  --wandb_project "${WANDB_PROJECT}"
  --checkpoint_freq 50000
  --checkpoint_save_buffers
  --checkpoint_compress_buffers
  --checkpoint_gzip_level 1
  --max_checkpoints 1
  --use_binary_reward_target
  --binary_reward_threshold 0.5
  --burnin_horizon 5
  --burnin_oversample_factor 4
  --burnin_max_sampling_attempts 4
  --burnin_require_no_done_context
  --burnin_require_start_not_done
  --burnin_allow_fallback
  --patch_size 7
  --obs_image_size 63
  --vqvae_codebook_size 512
  --vqvae_embed_dim 128
  --patch_encoder_hidden_dim 128
  --vqvae_lambda_l1 0.1
  --vqvae_lambda_l2 1.0
  --vqvae_lambda_codebook 1.0
  --vqvae_lambda_commitment 0.02
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume_from "${RESUME_CKPT}")
fi

echo "[$(date)] Launching training..."
echo "[$(date)] Command: ${CMD[*]}"

run_status=0
"${CMD[@]}" || run_status=$?

END_STEP="$(latest_checkpoint_step)"
echo "[$(date)] Job ended with status=${run_status}, latest checkpoint step=${END_STEP}"

if (( END_STEP >= TOTAL_TIMESTEPS )); then
  echo "[$(date)] Target reached (${END_STEP} >= ${TOTAL_TIMESTEPS}). Exiting."
  exit 0
fi

if (( run_status != 0 )); then
  echo "[$(date)] Training failed (status=${run_status}); not auto-requeueing."
  exit "${run_status}"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[$(date)] Time-sliced run ended before target; requeueing job ${SLURM_JOB_ID}."
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi

echo "[$(date)] Not running under Slurm; please re-submit manually."
exit 0


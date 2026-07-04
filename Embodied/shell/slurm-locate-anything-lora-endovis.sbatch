#!/bin/bash
#SBATCH -A IscrC_FLAC
#SBATCH -p boost_usr_prod
#SBATCH --time 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --job-name=locany-endovis
#SBATCH --output=jobs/out/locany-endovis.%j.out
#SBATCH --error=jobs/out/locany-endovis.%j.err

set -euo pipefail

date

# Adjust these paths for the cluster checkout and environment.
WORKDIR="${WORKDIR:-/leonardo_work/IscrC_FLAC/Eagle}"
VENV_PATH="${VENV_PATH:-$WORKDIR/Embodied/.venv}"
HF_HOME="${HF_HOME:-/leonardo_scratch/large/userexternal/mnunzian/HF_DOWNLOADS}"

# Use a local checkpoint path here if LocateAnything is not already cached in HF_HOME.
MODEL_PATH="${MODEL_PATH:-nvidia/LocateAnything-3B}"
META_PATH="${META_PATH:-$WORKDIR/data/endovis_locany_single/endovis_recipe.json}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/locany_lora_endovis_single}"
WANDB_PROJECT="${WANDB_PROJECT:-locany-endovis}"
WANDB_RUN_ID="${WANDB_RUN_ID:-locany-lora-endovis-${SLURM_JOB_ID:-local}}"
WANDB_NAME="${WANDB_NAME:-$WANDB_RUN_ID}"

echo "Loading necessary modules"
ml purge
ml load cuda/12.2
ml load gcc/12.2.0

export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LOCANY_VISION_ATTN="${LOCANY_VISION_ATTN:-sdpa}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT WANDB_RUN_ID WANDB_NAME
export WANDB_RESUME="${WANDB_RESUME:-allow}"

echo "Working directory: $WORKDIR"
cd "$WORKDIR"
mkdir -p Embodied/jobs/out

echo "Activating virtual environment: $VENV_PATH"
source "$VENV_PATH/bin/activate"

echo "Job configuration"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  META_PATH=$META_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "  LOCANY_VISION_ATTN=$LOCANY_VISION_ATTN"
echo "  WANDB_MODE=$WANDB_MODE"
echo "  WANDB_PROJECT=$WANDB_PROJECT"
echo "  WANDB_RUN_ID=$WANDB_RUN_ID"

test -f "$META_PATH"

cd Embodied

GPUS=1 \
MODEL_PATH="$MODEL_PATH" \
META_PATH="$META_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}" \
MAX_NUM_TOKENS_PER_SAMPLE="${MAX_NUM_TOKENS_PER_SAMPLE:-4096}" \
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-4096}" \
MAX_STEPS="${MAX_STEPS:-1000}" \
SAVE_STEPS="${SAVE_STEPS:-100}" \
WARMUP_STEPS="${WARMUP_STEPS:-50}" \
PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-16}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}" \
REPORT_TO="${REPORT_TO:-wandb}" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_RUN_ID="$WANDB_RUN_ID" \
WANDB_NAME="$WANDB_NAME" \
WANDB_MODE="$WANDB_MODE" \
WANDB_RESUME="$WANDB_RESUME" \
bash shell/locate-anything-lora-endovis.sh 1 "$OUTPUT_DIR"

date

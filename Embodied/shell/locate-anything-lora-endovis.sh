#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GPUS=${GPUS:-1}
NNODES=${1:-1}
OUTPUT_DIR=${2:-"work_dirs/locany_lora_endovis"}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

MODEL_PATH=${MODEL_PATH:-"nvidia/LocateAnything-3B"}
META_PATH=${META_PATH:-"../data/endovis_locany/endovis_recipe.json"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"deepspeed_configs/zero_stage1_config.json"}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-"sdpa"}
LOCANY_VISION_ATTN=${LOCANY_VISION_ATTN:-"sdpa"}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-1}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-100}
LR=${LR:-2e-5}
WARMUP_STEPS=${WARMUP_STEPS:-50}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-4096}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-4096}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-16}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}

USE_LLM_LORA=${USE_LLM_LORA:-64}
USE_BACKBONE_LORA=${USE_BACKBONE_LORA:-0}
FREEZE_LLM=${FREEZE_LLM:-True}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-True}
FREEZE_MLP=${FREEZE_MLP:-False}
REPORT_TO=${REPORT_TO:-"wandb"}

WANDB_MODE=${WANDB_MODE:-"offline"}
WANDB_PROJECT=${WANDB_PROJECT:-"locany-endovis"}
WANDB_RUN_ID=${WANDB_RUN_ID:-"locany-lora-endovis"}
WANDB_NAME=${WANDB_NAME:-"$WANDB_RUN_ID"}
WANDB_RESUME=${WANDB_RESUME:-"allow"}
WANDB_DIR=${WANDB_DIR:-"$OUTPUT_DIR/wandb"}

if [[ ! -f "$META_PATH" ]]; then
  echo "Missing META_PATH=$META_PATH. Run tools/prepare_endovis_locany.py first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$WANDB_DIR"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LOCANY_VISION_ATTN
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export WANDB_MODE WANDB_PROJECT WANDB_RUN_ID WANDB_NAME WANDB_RESUME WANDB_DIR

python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --causal_attn False \
  --freeze_llm "$FREEZE_LLM" \
  --freeze_mlp "$FREEZE_MLP" \
  --freeze_backbone "$FREEZE_BACKBONE" \
  --use_llm_lora "$USE_LLM_LORA" \
  --use_backbone_lora "$USE_BACKBONE_LORA" \
  --vision_select_layer -1 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACC" \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --learning_rate "$LR" \
  --weight_decay 0.01 \
  --warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --sample_log_interval 10 \
  --packing_buffer_size "$PACKING_BUFFER_SIZE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_num_tokens_per_sample "$MAX_NUM_TOKENS_PER_SAMPLE" \
  --max_num_tokens "$MAX_NUM_TOKENS" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --report_to "$REPORT_TO" \
  --run_name "locany-lora-endovis" \
  --use_onelogger False \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

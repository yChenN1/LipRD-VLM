#!/bin/bash
#SBATCH --job-name=liprd_qwen3vl_lora

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=./slurm_run/%x.%j.out
#SBATCH --error=./slurm_run/%x.%j.err

set -eo pipefail

mkdir -p slurm_run

# ====== Environment ======
# 修改为你的环境加载方式

# ====== Runtime Config ======
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

# W&B (可按需修改)
export WANDB_PROJECT=liprd-qwen3vl
# export WANDB_ENTITY=your_team

# ====== Train Config ======
MODEL_NAME_OR_PATH="Qwen/Qwen3-VL-4B-Instruct"
DATE_TAG="$(date +%m%d)"
OUTPUT_DIR="/scratch/u5ge/yuncy.u5ge/LipRD/results/liprd_qwen3vl_lora_${DATE_TAG}_2e-5"

TRAIN_ROOT="/scratch/u5ge/yuncy.u5ge/LipRD/dataset/trainval"
VAL_ROOT="/scratch/u5ge/yuncy.u5ge/LipRD/dataset/test"
TRAIN_FILE_LIST="/scratch/u5ge/yuncy.u5ge/LipRD/dataset/file_filter.list"
VAL_FILE_LIST="/scratch/u5ge/yuncy.u5ge/LipRD/dataset/test_file_filter.list"

RUN_NAME="liprd-qwen3vl-lora-${DATE_TAG}-2e-5"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"
MASTER_PORT=29666

# ====== Ablation (optional) ======
# 开启后，故意打乱 video-text 对应关系，用于验证视觉信息是否起作用。
# ABLATION_FLAGS="--unpaired_text_for_train"
ABLATION_FLAGS=""

# ====== Vision LoRA (optional) ======
# 开启后，对 vision encoder 的 q/k/v/o(含常见命名变体) 注入 LoRA。
VISION_LORA_FLAGS="--lora_vision_qkvo"
# VISION_LORA_FLAGS=""

DEEPSPEED_CONFIG="configs/deepspeed_zero2.json"

torchrun \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_port="${MASTER_PORT}" \
  train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_file_list "${TRAIN_FILE_LIST}" \
  --val_file_list "${VAL_FILE_LIST}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_root "${TRAIN_ROOT}" \
  --val_root "${VAL_ROOT}" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_frames 64 \
  --save_steps 500 \
  --eval_steps 200 \
  --logging_steps 10 \
  --report_to wandb \
  --run_name "${RUN_NAME}" \
  --wandb_project "${WANDB_PROJECT}" \
  --bf16 \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  ${ABLATION_FLAGS} \
  ${VISION_LORA_FLAGS}

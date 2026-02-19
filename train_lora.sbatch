#!/bin/bash
#SBATCH --job-name=liprd_qwen3vl_lora
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=./slurm_run/%x.%j.out
#SBATCH --error=./slurm_run/%x.%j.err

set -eo pipefail

cd /mnt/fast/nobackup/users/yc01815/code/LipRD-VLM
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
OUTPUT_DIR="/mnt/fast/nobackup/scratch4weeks/yc01815/ckpt/liprd_qwen3vl_lora_${DATE_TAG}_3e-4"

TRAIN_ROOT="/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/trainval"
VAL_ROOT="/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/test"
TRAIN_FILE_LIST="/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/file_filter.list"
VAL_FILE_LIST="/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/test_file_filter.list"

RUN_NAME="liprd-qwen3vl-lora-${DATE_TAG}-3e-4"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"
MASTER_PORT=29666

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
  --learning_rate 3e-4 \
  --save_steps 500 \
  --eval_steps 500 \
  --logging_steps 10 \
  --report_to wandb \
  --run_name "${RUN_NAME}" \
  --wandb_project "${WANDB_PROJECT}" \
  --bf16 \

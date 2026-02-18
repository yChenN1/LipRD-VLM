# LipRD_VLLM

Qwen3-VL LoRA finetuning code for lip reading, split into `dataset`, `train`, and `utils`.

## Structure

```text
LipRD_VLLM/
  dataset.py
  train.py
  utils/
    __init__.py
    args.py
    common.py
    modeling.py
    collator.py
  requirements.txt
```

## Data Format

Default mode scans folders recursively:

- Train: `/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/trainval`
- Val: `/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/test`

Each sample is paired by same stem:
- `xxx.mp4` as video input
- `xxx.txt` as text GT
- `xxx.npy` and `xxx.wav` are ignored by loader

Optional mode still supports annotation files `.json`, `.jsonl`, `.csv`, `.tsv` via `--train_data --val_data`.

Annotation required fields:
- `video` or `video_path`
- `text` or `label`

Example `train.jsonl`:

```json
{"video":"./data/sample1.mp4","text":"hello world","instruction":"Please read the lips and output the sentence."}
{"video":"./data/sample2.mp4","text":"how are you"}
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
cd LipRD_VLLM
python train.py \
  --model_name_or_path /path/to/Qwen3-VL \
  --output_dir /path/to/ckpt_qwen3vl_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --bf16 \
  --gradient_checkpointing
```

Enable W&B logging:

```bash
python train.py \
  --model_name_or_path /path/to/Qwen3-VL \
  --output_dir /path/to/ckpt_qwen3vl_lora \
  --report_to wandb \
  --run_name liprd-qwen3vl-lora-exp1 \
  --wandb_project liprd-qwen3vl \
  --wandb_entity your_wandb_team
```

## Notes

- This script saves LoRA adapter weights via `Trainer.save_model`.
- It evaluates on `test` split during training (default `--eval_steps 500`).
- If your environment does not support `bf16`, switch to `--fp16` or disable both.
- If your model has different LoRA target module names, set `--lora_target_modules`.

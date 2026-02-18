import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-VL LoRA Finetuning for Lip Reading")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--train_root",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/trainval",
        help="Train root folder. Recursively scans *.mp4 and matches same-name *.txt.",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default="/mnt/fast/nobackup/scratch4weeks/yc01815/Voicecraft_dub/samples/test",
        help="Validation root folder. Recursively scans *.mp4 and matches same-name *.txt.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Optional annotation file (json/jsonl/csv/tsv). If set, it overrides --train_root.",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="",
        help="Optional annotation file (json/jsonl/csv/tsv). If set, it overrides --val_root.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--default_instruction", type=str, default="Please read the speaker's lips and transcribe the speech.")

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=32)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb", help="none|tensorboard|wandb")
    parser.add_argument("--run_name", type=str, default="qwen3vl-liprd-lora")
    parser.add_argument("--wandb_project", type=str, default="liprd-qwen3vl")
    parser.add_argument("--wandb_entity", type=str, default="")
    return parser.parse_args()

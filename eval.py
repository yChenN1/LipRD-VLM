import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from dataset import LipReadingDataset
from utils import set_seed

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:  # pragma: no cover
    process_vision_info = None
    _QWEN_VL_UTILS_IMPORT_ERROR = exc
else:
    _QWEN_VL_UTILS_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Qwen3-VL LoRA Eval for Lip Reading")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
        help="LoRA adapter directory from training output. If empty, evaluates base model only.",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default="/scratch/u5ge/yuncy.u5ge/LipRD/dataset/test",
        help="Validation root folder. Recursively scans *.mp4 and matches same-name *.txt.",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="",
        help="Optional annotation file (json/jsonl/csv/tsv). If set, it overrides --val_root.",
    )
    parser.add_argument(
        "--default_instruction",
        type=str,
        default="Please read the speaker's lips and transcribe the speech.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--save_predictions_path",
        type=str,
        default="",
        help="If set, saves per-sample predictions as jsonl.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def levenshtein_distance(seq_a: List[str], seq_b: List[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev_row = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current_row = [i]
        for j, token_b in enumerate(seq_b, start=1):
            ins_cost = current_row[j - 1] + 1
            del_cost = prev_row[j] + 1
            sub_cost = prev_row[j - 1] + (0 if token_a == token_b else 1)
            current_row.append(min(ins_cost, del_cost, sub_cost))
        prev_row = current_row
    return prev_row[-1]


def compute_cer_and_wer(prediction: str, reference: str) -> Tuple[float, float]:
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)

    pred_chars = list(pred_norm)
    ref_chars = list(ref_norm)
    char_dist = levenshtein_distance(pred_chars, ref_chars)
    cer = char_dist / max(1, len(ref_chars))

    pred_words = pred_norm.split()
    ref_words = ref_norm.split()
    word_dist = levenshtein_distance(pred_words, ref_words)
    wer = word_dist / max(1, len(ref_words))
    return cer, wer


def load_model_and_processor(args: argparse.Namespace):
    torch_dtype = "auto"
    if args.bf16:
        torch_dtype = torch.bfloat16
    if args.fp16:
        torch_dtype = torch.float16

    processor_path = args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    return model, processor


@torch.inference_mode()
def predict_one(
    model,
    processor,
    sample: Dict[str, str],
    max_length: int,
    max_new_tokens: int,
    fps: float,
    max_frames: int,
) -> str:
    if process_vision_info is None:
        raise ImportError(
            "qwen-vl-utils is required for video input processing."
            f" Original error: {_QWEN_VL_UTILS_IMPORT_ERROR}"
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": sample["video_path"], "fps": fps, "max_frames": max_frames},
                {"type": "text", "text": sample["instruction"]},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(messages)
    if len(video_inputs) != 1:
        raise ValueError("Each sample should contain exactly one video input.")

    inputs = processor(
        text=[prompt_text],
        videos=[video_inputs[0]],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = generated[:, prompt_len:]
    prediction = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return prediction.strip()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = LipReadingDataset(
        ann_path=args.val_data,
        data_root="" if args.val_data else args.val_root,
        default_instruction=args.default_instruction,
    )
    model, processor = load_model_and_processor(args)

    records: List[Dict[str, str]] = []
    total_cer = 0.0
    total_wer = 0.0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        prediction = predict_one(
            model=model,
            processor=processor,
            sample=sample,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        cer, wer = compute_cer_and_wer(prediction, sample["text"])
        total_cer += cer
        total_wer += wer

        records.append(
            {
                "index": idx,
                "video_path": sample["video_path"],
                "instruction": sample["instruction"],
                "reference": sample["text"],
                "prediction": prediction,
                "cer": cer,
                "wer": wer,
            }
        )

        print(
            f"[{idx + 1}/{len(dataset)}] CER={cer:.4f} WER={wer:.4f}\n"
            f"  REF: {sample['text']}\n"
            f"  PRED:{prediction}"
        )

    avg_cer = total_cer / max(1, len(dataset))
    avg_wer = total_wer / max(1, len(dataset))
    print(f"\nFinal Results on {len(dataset)} samples")
    print(f"Average CER: {avg_cer:.6f}")
    print(f"Average WER: {avg_wer:.6f}")

    if args.save_predictions_path:
        output_path = Path(args.save_predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()

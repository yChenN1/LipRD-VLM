from typing import Tuple

from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def build_model_and_processor(args) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.print_trainable_parameters()
    return model, processor

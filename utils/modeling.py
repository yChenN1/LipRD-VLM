from typing import List, Tuple

import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def _collect_vision_qkvo_modules(model) -> List[str]:
    targets = []
    leaf_names = {"q_proj", "k_proj", "v_proj", "o_proj", "q", "k", "v", "o", "out_proj"}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lower_name = name.lower()
        if "vision" not in lower_name and "visual" not in lower_name:
            continue

        leaf = name.split(".")[-1]
        if leaf in leaf_names:
            targets.append(name)
            continue
        if leaf == "proj" and ".attn." in lower_name:
            targets.append(name)
            continue
        if leaf == "qkv" and ".attn." in lower_name:
            targets.append(name)

    return sorted(set(targets))


def build_model_and_processor(args) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    target_modules = list(args.lora_target_modules)
    if args.lora_vision_qkvo:
        vision_targets = _collect_vision_qkvo_modules(model)
        if not vision_targets:
            raise ValueError(
                "No vision q/k/v/o projection layers found for LoRA. "
                "Please check model architecture or disable --lora_vision_qkvo."
            )
        target_modules.extend(vision_targets)
        target_modules = sorted(set(target_modules))
        print(f"[LoRA] Added {len(vision_targets)} vision q/k/v/o modules.")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.print_trainable_parameters()
    return model, processor

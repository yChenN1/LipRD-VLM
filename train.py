import os

from transformers import Trainer, TrainingArguments

from dataset import LipReadingDataset
from utils import build_model_and_processor, parse_args, set_seed
from utils.collator import LipReadingDataCollator


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

    train_dataset = LipReadingDataset(
        ann_path=args.train_data,
        data_root="" if args.train_data else args.train_root,
        file_list="" if args.train_data else args.train_file_list,
        default_instruction=args.default_instruction,
        unpaired_text=args.unpaired_text_for_train,
        unpaired_shuffle_seed=args.unpaired_shuffle_seed,
    )
    val_dataset = LipReadingDataset(
        ann_path=args.val_data,
        data_root="" if args.val_data else args.val_root,
        file_list="" if args.val_data else args.val_file_list,
        default_instruction=args.default_instruction,
        unpaired_text=args.unpaired_text_for_val,
        unpaired_shuffle_seed=args.unpaired_shuffle_seed + 1,
    )
    model, processor = build_model_and_processor(args)
    collator = LipReadingDataCollator(
        processor=processor,
        max_length=args.max_length,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        run_name=args.run_name,
        dataloader_num_workers=0,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed or None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

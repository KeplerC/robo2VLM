#!/usr/bin/env python3
"""
Train a single model, saving checkpoints at specified sample intervals.

Instead of retraining from scratch for each sample size, this trains once
and saves checkpoints after processing 10k, 20k, 30k... samples.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_single.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --max-samples 100000 \
        --checkpoint-intervals 10000,20000,30000,40000,50000,60000,70000,80000,90000,100000 \
        --output-dir outputs
"""

import os
import sys
import argparse
import torch
import wandb
from datasets import load_from_disk
from tqdm import tqdm

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig


# Training hyperparameters (fixed across all models)
TRAINING_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-6,
    "weight_decay": 0.01,
    "warmup_steps": 5,
    "lr_scheduler_type": "linear",
    "lora_r": 128,
    "lora_alpha": 256,
    "lora_dropout": 0,
    "max_seq_length": 2048,
    "logging_steps": 10,
    "random_state": 3407,
}

# Dataset path
DATASET_PATH = "/home/syx/robo2VLM/merged_dataset"

# Default checkpoint intervals
DEFAULT_INTERVALS = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]


def convert_to_conversation(sample):
    """Convert a VQA sample to conversation format."""
    question = sample["question"]
    choices = sample["choices"]
    correct_answer = sample["correct_answer"]

    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    correct_idx = letter_to_idx.get(correct_answer, 0)

    formatted_question = f"{question}\nChoices:\n"
    for i, choice in enumerate(choices):
        formatted_question += f"{chr(65 + i)}. {choice}\n"

    answer = f"{correct_answer}. {choices[correct_idx]}"

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": formatted_question},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            },
        ]
    }


def get_slug(model_name: str) -> str:
    """Get model slug for directory names."""
    return model_name.replace("/", "_")


class CheckpointCallback:
    """Callback to save checkpoints at specific sample intervals."""

    def __init__(self, model, tokenizer, output_dir: str, model_slug: str,
                 intervals: list, batch_size: int, grad_accum: int):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.model_slug = model_slug
        self.intervals = sorted(intervals)
        self.samples_per_step = batch_size * grad_accum
        self.saved_intervals = set()

    def check_and_save(self, step: int):
        """Check if we've hit an interval and save checkpoint."""
        samples_processed = step * self.samples_per_step

        for interval in self.intervals:
            if interval not in self.saved_intervals and samples_processed >= interval:
                self._save_checkpoint(interval)
                self.saved_intervals.add(interval)

    def _save_checkpoint(self, num_samples: int):
        """Save checkpoint for this sample count."""
        checkpoint_dir = os.path.join(
            self.output_dir, self.model_slug, f"{num_samples}samples", "checkpoint-final"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"\n>>> Saving checkpoint at {num_samples} samples: {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def save_final(self, total_samples: int):
        """Save final checkpoint if not already saved."""
        # Find the largest interval <= total_samples that wasn't saved
        for interval in reversed(self.intervals):
            if interval <= total_samples and interval not in self.saved_intervals:
                self._save_checkpoint(interval)
                self.saved_intervals.add(interval)
                break


def train(
    model_name: str,
    max_samples: int,
    checkpoint_intervals: list,
    output_dir: str,
    use_wandb: bool = True
):
    """Train a model, saving checkpoints at specified intervals."""
    model_slug = get_slug(model_name)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Max samples: {max_samples}")
    print(f"Checkpoint intervals: {checkpoint_intervals}")
    print(f"Output: {output_dir}/{model_slug}/")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"{'='*60}\n")

    # Check which checkpoints already exist
    remaining_intervals = []
    for interval in checkpoint_intervals:
        if interval > max_samples:
            continue
        checkpoint_path = os.path.join(
            output_dir, model_slug, f"{interval}samples", "checkpoint-final"
        )
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint exists, skipping: {interval} samples")
        else:
            remaining_intervals.append(interval)

    if not remaining_intervals:
        print("All checkpoints already exist. Skipping training.")
        return

    print(f"Will save checkpoints at: {remaining_intervals}")

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        use_gradient_checkpointing="unsloth",
    )

    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        bias="none",
        random_state=TRAINING_CONFIG["random_state"],
        use_rslora=False,
    )

    # Load dataset
    print(f"\nLoading dataset from: {DATASET_PATH}")
    dataset_dict = load_from_disk(DATASET_PATH)
    train_dataset = dataset_dict["train"]

    # Limit to max_samples
    if max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(max_samples))
    print(f"Training samples: {len(train_dataset)}")

    # Convert to conversation format
    print("Converting to conversation format...")
    train_data = [convert_to_conversation(s) for s in tqdm(train_dataset, desc="Converting")]

    # Small validation set
    val_data = train_data[:100]

    # Prepare for training
    FastVisionModel.for_training(model)

    # Initialize wandb
    if use_wandb:
        run_name = f"{model_slug}_train"
        wandb.init(project="finetune-pipeline", name=run_name)

    # Calculate steps for checkpoints
    batch_size = TRAINING_CONFIG["per_device_train_batch_size"]
    grad_accum = TRAINING_CONFIG["gradient_accumulation_steps"]
    samples_per_step = batch_size * grad_accum

    # Determine save_steps to hit our intervals
    # We'll save at each interval
    min_interval = min(remaining_intervals)
    save_steps = max(1, min_interval // samples_per_step)

    # Create checkpoint callback
    checkpoint_cb = CheckpointCallback(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        model_slug=model_slug,
        intervals=remaining_intervals,
        batch_size=batch_size,
        grad_accum=grad_accum,
    )

    # Custom training loop with checkpoint saving
    from transformers import TrainingArguments, Trainer

    # Calculate total steps
    total_steps = len(train_data) // samples_per_step

    # Create trainer with custom callback
    from transformers import TrainerCallback

    class IntervalCheckpointCallback(TrainerCallback):
        def __init__(self, checkpoint_cb):
            self.checkpoint_cb = checkpoint_cb

        def on_step_end(self, args, state, control, **kwargs):
            self.checkpoint_cb.check_and_save(state.global_step)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_data,
        eval_dataset=val_data,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=TRAINING_CONFIG["logging_steps"],
            optim="adamw_8bit",
            weight_decay=TRAINING_CONFIG["weight_decay"],
            lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
            seed=TRAINING_CONFIG["random_state"],
            output_dir=os.path.join(output_dir, model_slug, "training_tmp"),
            save_strategy="no",  # We handle saving manually
            report_to=["wandb"] if use_wandb else [],
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=TRAINING_CONFIG["max_seq_length"],
        ),
        callbacks=[IntervalCheckpointCallback(checkpoint_cb)],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final checkpoint
    checkpoint_cb.save_final(len(train_data))

    if use_wandb:
        wandb.finish()

    # Cleanup
    del model, tokenizer, trainer
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Checkpoints saved: {sorted(checkpoint_cb.saved_intervals)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train a model with interval checkpoints")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Maximum training samples")
    parser.add_argument("--checkpoint-intervals", type=str,
                        default=",".join(map(str, DEFAULT_INTERVALS)),
                        help="Comma-separated checkpoint intervals (e.g., 10000,20000,30000)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb")

    args = parser.parse_args()

    # Parse intervals
    intervals = [int(x) for x in args.checkpoint_intervals.split(",")]

    train(
        model_name=args.model,
        max_samples=args.max_samples,
        checkpoint_intervals=intervals,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()

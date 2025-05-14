"""
Training module for fine-tuning vision-language models using Unsloth.
"""

import os
import torch
import wandb
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from config import Config, default_config
from dataset import load_training_dataset

def load_model(config):
    """
    Load a vision-language model from HuggingFace with Unsloth optimization.
    
    Args:
        config: Configuration with model parameters
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Get distributed training info from environment
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1
    
    # Print distributed training info
    if local_rank <= 0:
        print(f"Distributed training: {is_distributed}, World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    
    # Let PyTorch/Unsloth handle the distributed initialization
    model, tokenizer = FastVisionModel.from_pretrained(
        config.model.model_name,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        device_map={'': local_rank} if is_distributed else "auto",
    )
    
    print(f"Loaded model: {config.model.model_name}")
    
    # Apply LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.model.finetune_vision_layers,
        finetune_language_layers=config.model.finetune_language_layers,
        finetune_attention_modules=config.model.finetune_attention_modules,
        finetune_mlp_modules=config.model.finetune_mlp_modules,
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        bias=config.model.lora_bias,
        random_state=config.model.random_state,
        use_rslora=config.model.use_rslora,
        loftq_config=None,
    )
    
    return model, tokenizer

def train_model(config=None):
    """
    Train a vision-language model with the specified configuration.
    
    Args:
        config: Configuration for training. If None, uses default_config
        
    Returns:
        Trained model
    """
    if config is None:
        config = default_config
    
    # Get distributed training info
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    is_distributed = world_size > 1
    
    # Load model and tokenizer
    model, tokenizer = load_model(config)
    
    # Load datasets
    train_split, val_split = load_training_dataset(config)
    
    # Prepare model for training
    FastVisionModel.for_training(model)
    
    # Initialize wandb only on main process
    if not is_distributed or rank == 0:
        if "wandb" in config.report_to:
            model_name = os.path.basename(config.model.model_name)
            run_name = config.training.run_name or f"{model_name}_vqa"
            wandb.init(project="vision-model-finetuning", name=run_name)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_split,
        eval_dataset=val_split,
        args=SFTConfig(
            per_device_train_batch_size=config.training.per_device_train_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            warmup_steps=config.training.warmup_steps,
            num_train_epochs=config.training.num_train_epochs,
            learning_rate=config.training.learning_rate,
            fp16=config.fp16 and not is_bf16_supported(),
            bf16=config.bf16 and is_bf16_supported(),
            logging_steps=config.training.logging_steps,
            optim="adamw_8bit",
            weight_decay=config.training.weight_decay,
            lr_scheduler_type=config.training.lr_scheduler_type,
            seed=config.model.random_state,
            output_dir=config.training.output_dir,
            eval_strategy="steps",
            eval_steps=config.training.eval_steps,
            save_strategy="steps",
            save_steps=config.training.save_steps,
            report_to=config.report_to,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=config.model.max_seq_length,
            # Let the trainer handle distributed training
            ddp_find_unused_parameters=False,
            dataloader_num_workers=4,
        ),
    )
    
    # Start training
    trainer.train()
    
    # Save model only on main process
    if not is_distributed or rank == 0:
        output_dir = config.training.output_dir
        checkpoint_dir = os.path.join(output_dir, "checkpoint-latest")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Model saved to {checkpoint_dir}")
        
        # Finish wandb run after saving the model
        if "wandb" in config.report_to:
            wandb.finish()
    
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a vision-language model")
    parser.add_argument("--model_name", type=str, help="Name of the model to train")
    parser.add_argument("--output_dir", type=str, help="Directory to save the model")
    parser.add_argument("--run_name", type=str, help="Name of the run for wandb")
    parser.add_argument("--batch_size", type=int, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = default_config
    if args.model_name:
        config.model.model_name = args.model_name
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.run_name:
        config.training.run_name = args.run_name
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_train_epochs = args.num_epochs
    
    train_model(config) 
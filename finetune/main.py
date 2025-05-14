#!/usr/bin/env python
"""
Main script for vision-language model fine-tuning and evaluation.
"""

import os
import argparse
from config import default_config
from train import train_model
from evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Vision-Language Model Fine-tuning and Evaluation")
    
    # General parameters
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "train-and-evaluate"], 
                        default="train-and-evaluate", help="Operation mode")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, 
                        help="Model name (e.g., 'llama3.2-11b', 'qwen2-7b', or full HF path)")
    parser.add_argument("--finetune_vision_layers", action="store_true", 
                        help="Whether to fine-tune vision layers")
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, help="Directory to save model outputs")
    parser.add_argument("--run_name", type=str, help="Name for the training run (used in wandb)")
    parser.add_argument("--batch_size", type=int, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_train_samples", type=int, help="Maximum number of training samples to use")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    
    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, help="HuggingFace dataset name")
    
    # Evaluation parameters
    parser.add_argument("--checkpoint_path", type=str, 
                        help="Path to checkpoint for evaluation (if not provided, will find latest)")
    parser.add_argument("--max_test_samples", type=int, help="Maximum number of test samples")
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update configuration based on command line arguments."""
    
    # Convert namespace to dictionary
    args_dict = vars(args)
    
    # Update model configuration
    if args.model_name:
        config.model.model_name = args.model_name
    
    if args.finetune_vision_layers is not None:
        config.model.finetune_vision_layers = args.finetune_vision_layers
    
    # Update training configuration
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
    
    if args.max_train_samples:
        config.training.max_train_samples = args.max_train_samples
    
    if args.dataset_name:
        config.training.dataset_name = args.dataset_name
    
    # Update evaluation configuration
    if args.checkpoint_path:
        config.evaluation.checkpoint_path = args.checkpoint_path
    
    if args.max_test_samples:
        config.evaluation.max_test_samples = args.max_test_samples
    
    # Handle wandb configuration
    if args.no_wandb and "wandb" in config.report_to:
        config.report_to.remove("wandb")
    
    return config

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get configuration and update with command line args
    config = default_config
    config = update_config_from_args(config, args)
    
    # Print configuration summary
    print("\nConfiguration:")
    print(f"  Model: {config.model.model_name}")
    print(f"  Output directory: {config.training.output_dir}")
    print(f"  Mode: {args.mode}")
    
    # Create output directory if it doesn't exist
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Execute based on mode
    if args.mode in ["train", "train-and-evaluate"]:
        print("\n=== Training Model ===")
        model, tokenizer = train_model(config)
        print("\nTraining completed.")
    
    if args.mode in ["evaluate", "train-and-evaluate"]:
        print("\n=== Evaluating Model ===")
        results = evaluate(config)
        print("\nEvaluation completed.")
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    main() 
"""
Configuration settings for vision model fine-tuning and evaluation.
Contains all parameters for models, training, and evaluation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

@dataclass
class ModelConfig:
    """Configuration for the model."""
    model_name: str = "meta-llama/Llama-3.2-11B-Vision"  # Model identifier from HuggingFace
    use_gradient_checkpointing: str = "unsloth"
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0
    lora_bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    max_seq_length: int = 2048

@dataclass
class TrainingConfig:
    """Configuration for training."""
    dataset_name: str = "keplerccc/ManipulationVQA"
    dataset_split: str = "train"
    validation_split: float = 0.05
    max_train_samples: Optional[int] = None  # Maximum number of training samples to use
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 1
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    logging_steps: int = 10
    eval_steps: int = 5000
    save_steps: int = 1000
    output_dir: str = "outputs"
    run_name: Optional[str] = None
    
    def __post_init__(self):
        # Generate unique output directory if not provided
        if self.output_dir == "outputs":
            model_name = os.path.basename(self.model_name) if hasattr(self, 'model_name') else "vision_model"
            self.output_dir = os.path.join("outputs", f"{model_name}_{self.run_name or 'run'}")

@dataclass
class APIConfig:
    """Configuration for external API models."""
    openai_api_key: Optional[str] = None
    openai_model_name: str = "gpt-4o"
    google_api_key: Optional[str] = None
    gemini_model_name: str = "gemini-pro-vision"
    use_cache: bool = True

@dataclass
class ModelEvalConfig:
    """Configuration for a single model to evaluate."""
    type: str = "internal"  # "internal", "openai", or "gemini"
    name: str = ""  # Display name for the model
    model_name: Optional[str] = None  # Model identifier or name
    base_model_name: Optional[str] = None  # Base model name for fine-tuned models
    is_finetuned: bool = False  # Whether this is a fine-tuned model
    checkpoint_path: Optional[str] = None  # Path to checkpoint (for fine-tuned models)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    max_test_samples: int = 10000
    checkpoint_path: Optional[str] = None  # Path to fine-tuned model checkpoint
    generate_visualizations: bool = True
    test_split: str = "test"
    fallback_to_train: bool = True  # Whether to use part of train split if test doesn't exist
    
    # Generation parameters
    max_new_tokens: int = 50
    temperature: float = 0.7
    do_sample: bool = True
    
    # Multi-model evaluation
    models_to_evaluate: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"type": "internal", "name": "Base Model", "is_finetuned": False},
        {"type": "internal", "name": "Fine-tuned Model", "is_finetuned": True}
    ])
    
    def __post_init__(self):
        # If checkpoint_path is not provided, construct from output_dir
        if self.checkpoint_path is None and hasattr(self, 'output_dir'):
            self.checkpoint_path = os.path.join(self.output_dir, "checkpoint-latest")

@dataclass
class Config:
    """Main configuration combining model, training and evaluation settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    apis: APIConfig = field(default_factory=APIConfig)
    
    # Distributed training settings
    use_distributed: bool = True
    
    # Precision settings
    fp16: bool = False
    bf16: bool = True
    
    # Logging
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    def __post_init__(self):
        # Link fields that need to be shared
        self.evaluation.checkpoint_path = os.path.join(self.training.output_dir, "checkpoint-latest")
        self.training.model_name = self.model.model_name
        
    def update_from_args(self, args: Dict[str, Any]) -> None:
        """Update config from command line arguments."""
        for k, v in args.items():
            if hasattr(self.model, k):
                setattr(self.model, k, v)
            elif hasattr(self.training, k):
                setattr(self.training, k, v)
            elif hasattr(self.evaluation, k):
                setattr(self.evaluation, k, v)
            elif hasattr(self.apis, k):
                setattr(self.apis, k, v)
            elif hasattr(self, k):
                setattr(self, k, v)

# Default configuration
default_config = Config() 
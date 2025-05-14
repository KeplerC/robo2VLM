# Vision-Language Model Fine-tuning

This repository contains modular code for fine-tuning and evaluating vision-language models (VLMs) on Visual Question Answering (VQA) tasks using Unsloth for optimization. The code supports various models including Llama-3.2-Vision, Qwen2-VL, and others.

## Structure

The code has been refactored into a modular structure:

- `config.py` - Central configuration for all model and training parameters
- `dataset.py` - Dataset loading and preparation functions
- `train.py` - Model loading and training functionality
- `evaluate.py` - Model evaluation and comparison
- `main.py` - Main script that brings everything together

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- datasets
- transformers
- unsloth
- trl
- matplotlib
- pandas
- tqdm
- pillow
- accelerate
- bitsandbytes
- peft

## Usage

### Basic Usage

Train and evaluate a model with default settings:

```bash
python main.py
```

### Selecting a Model

You can choose between different pre-defined models:

```bash
# Use Llama 3.2 Vision (11B)
python main.py --model_name llama3.2-11b

# Use Qwen2 Vision (7B)
python main.py --model_name qwen2-7b

# Use a custom model from HuggingFace
python main.py --model_name "organization/model-name"
```

### Training Configuration

Configure the training process:

```bash
python main.py \
  --mode train \
  --model_name qwen2-7b \
  --output_dir outputs/qwen2_run1 \
  --run_name "qwen2_vqa_run1" \
  --batch_size 4 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --max_train_samples 10000  # Limit the number of training samples
```

### Training with Different Dataset Sizes

You can use the `--max_train_samples` parameter to limit the number of training samples:

```bash
# Train with 10,000 samples
python main.py --max_train_samples 10000 --output_dir outputs/10k_samples

# Train with 20,000 samples
python main.py --max_train_samples 20000 --output_dir outputs/20k_samples
```

A script is available in the repo (`train_scale.sh`) to train models with increasing dataset sizes.

### Evaluation Only

Evaluate a previously fine-tuned model:

```bash
python main.py \
  --mode evaluate \
  --model_name qwen2-7b \
  --checkpoint_path outputs/qwen2_run1/checkpoint-1000 \
  --max_test_samples 100
```

### Distributed Training

For distributed training with multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 TRANSFORMERS_NO_TP=1 torchrun --nproc_per_node=4 main.py \
  --model_name llama3.2-11b \
  --output_dir outputs/llama3_run1
```

## Configuration

The default configuration is defined in `config.py`. You can modify it directly or override values using command-line arguments.

Key configuration options:

- **Model parameters**:
  - Model name/path
  - Which layers to fine-tune (vision, language, attention, MLP)
  - LoRA parameters (rank, alpha, dropout)

- **Training parameters**:
  - Dataset name
  - Batch size
  - Learning rate
  - Number of epochs
  - Output directory
  - Evaluation frequency

- **Evaluation parameters**:
  - Number of test samples
  - Checkpoint path
  - Visualization options

## Examples

### Fine-tune Llama 3.2 Vision on VQA dataset

```bash
python main.py \
  --model_name llama3.2-11b \
  --output_dir outputs/llama3_vqa \
  --batch_size 4 \
  --learning_rate 1e-6 \
  --num_epochs 1
```

### Fine-tune Qwen2 Vision without WandB

```bash
python main.py \
  --model_name qwen2-7b \
  --output_dir outputs/qwen2_vqa \
  --no_wandb
```

### Evaluate only a specific checkpoint

```bash
python main.py \
  --mode evaluate \
  --model_name qwen2-7b \
  --checkpoint_path outputs/qwen2_vqa/checkpoint-500 \
  --max_test_samples 200
```

## License

[Specify your license]

# Dataset ID Fixer for keplerccc/ManipulationVQA

This script finds question IDs in the keplerccc/ManipulationVQA dataset and processes them in two ways:

1. For IDs without proper prefixes: It adds both language instructions and metadata tags
2. For IDs with proper prefixes: It adds only the metadata tags to maintain consistency

The script uses Ray for parallel processing to significantly improve performance.

## Prerequisites

- Python 3.7+
- Google Cloud credentials configured
- Required Python packages (install with `pip install -r requirements.txt`)
- Ray for parallel processing

## Setup

1. Make sure you have Google Cloud SDK installed and configured with access to the `kych-openx-vqa` bucket
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Authenticate with Google Cloud:
   ```
   gcloud auth application-default login
   ```

## Usage

Run the script:

```
python fix_dataset_ids.py
```

The script will:
1. Load the keplerccc/ManipulationVQA dataset
2. Identify IDs with and without proper prefixes
3. Download vqa_data.json files in parallel using Ray
4. Process the dataset in parallel batches:
   - For IDs without prefixes: Create a new ID with format `droid_language_instruction_tag_id_questionidx`
   - For IDs with prefixes: Preserve the existing structure and add the metadata tag
5. Save the processed datasets to disk

## Output

The script saves the fixed datasets to:
- `fixed_dataset_train` for the train split
- `fixed_dataset_test` for the test split
- `fixed_dataset_combined` if splits are not available

You can load these datasets using:
```python
from datasets import load_from_disk
dataset = load_from_disk("fixed_dataset_train")
``` 
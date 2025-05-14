# Model Evaluation Guide

This guide explains how to use the updated evaluation script to compare multiple vision-language models on VQA tasks.

## Features

- Evaluate multiple models in a single run
- Support for local models (internal) and external API models (OpenAI, Gemini)
- Result caching to avoid re-calling APIs for the same questions
- Detailed analytics and visualizations
- Fair comparisons with the same test set

## Configuration

The evaluation is controlled via a YAML configuration file. A sample configuration is provided in `sample_eval_config.yaml`.

### Model Types

The evaluation script supports three types of models:

1. **Internal models** - Models loaded with FastVisionModel, either base or fine-tuned
2. **OpenAI models** - Models accessed via OpenAI's API (e.g., GPT-4o)
3. **Gemini models** - Models accessed via Google's Gemini API

### Configuration File Structure

```yaml
# Model configuration (for internal models)
model:
  model_name: "meta-llama/Llama-3.2-11B-Vision"
  # ... other model parameters

# Evaluation configuration
evaluation:
  max_test_samples: 100
  # ... other evaluation parameters
  
  # Define which models to evaluate
  models_to_evaluate:
    # Base model example
    - type: "internal"
      name: "Base-Model"
      is_finetuned: false
      
    # Fine-tuned model example
    - type: "internal"
      name: "Fine-tuned-Model" 
      is_finetuned: true
      checkpoint_path: "outputs/checkpoint-latest"
      
    # OpenAI API model example
    - type: "openai"
      name: "GPT-4o"
      model_name: "gpt-4o"

# API keys for external models
apis:
  openai_api_key: "your-openai-key"
  google_api_key: "your-google-key"
  use_cache: true  # Enable/disable result caching
```

## Running the Evaluation

Run the evaluation script with a configuration file:

```bash
python evaluate.py --config_file sample_eval_config.yaml
```

Or use command-line arguments to override specific settings:

```bash
python evaluate.py --config_file sample_eval_config.yaml --max_test_samples 50
```

## Result Caching

The script caches evaluation results in a `.cache` directory to avoid re-calling APIs for the same question-model pairs. This saves time and API costs when re-running evaluations.

To disable caching, set `apis.use_cache: false` in your configuration.

## Output

The evaluation generates:

1. **Console output** with detailed results and comparisons
2. **JSON file** with complete evaluation data
3. **Visualizations** (if enabled) showing accuracy comparisons and other metrics

Look for files named like `comparison_results_[models]_[timestamp].json` for the raw evaluation data.

## Important Notes

- For API models, you need valid API keys set in the configuration
- The script will automatically use the GPU if available
- Cached results are stored in the `.cache` directory
- You can evaluate just one model or any combination of multiple models 
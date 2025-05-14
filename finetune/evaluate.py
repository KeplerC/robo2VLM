"""
Evaluation module for vision-language models.
Compares model performance on VQA tasks.
"""

import os
import json
import re
import time
import torch
import base64
import requests
from datetime import datetime
import numpy as np
from PIL import Image
from unsloth import FastVisionModel
import ray
from functools import partial
from tqdm import tqdm

from config import Config, default_config
from dataset import load_evaluation_dataset

# For external API models
try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI package not found. Gemini models will not be available.")
    genai = None

def load_pretrained_model(config):
    """
    Load base model without fine-tuning.
    
    Args:
        config: Configuration with model parameters
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        config.model.model_name,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
    )
    FastVisionModel.for_inference(model)
    print(f"Loaded base model: {config.model.model_name}")
    return model, tokenizer

def load_finetuned_model(config):
    """
    Load fine-tuned model from checkpoint.
    
    Args:
        config: Configuration with model and checkpoint parameters
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # First load the base model
    base_model_name = config.model.model_name
    print(f"Loading base model: {base_model_name} for fine-tuned model")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model_name,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
    )
    
    # Get PEFT model with same configuration as used in training
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
    
    # Load fine-tuned weights
    checkpoint_path = config.evaluation.checkpoint_path
    adapter_name = "default"
    
    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} not found.")
        print("Looking for checkpoints in output directory...")
        
        # Try to find the latest checkpoint in output directory
        output_dir = config.training.output_dir
        checkpoints = [d for d in os.listdir(output_dir) 
                       if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")]
        
        if checkpoints:
            # Sort checkpoints to find the latest one
            checkpoints.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0, reverse=True)
            checkpoint_path = os.path.join(output_dir, checkpoints[0])
            print(f"Found checkpoint: {checkpoint_path}")
        else:
            print(f"No checkpoints found in {output_dir}. Using base model.")
            FastVisionModel.for_inference(model)
            return model, tokenizer
    
    # Load the adapter
    model.load_adapter(checkpoint_path, adapter_name)
    print(f"Loaded fine-tuned adapter from {checkpoint_path} with name '{adapter_name}'")
    
    # Set the model to use the loaded adapter
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    
    FastVisionModel.for_inference(model)
    return model, tokenizer

# External API model handlers
def encode_image_to_base64(image_path):
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image string
    """
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_image

def perform_vqa_openai(image_path, question, config):
    """
    Perform VQA using OpenAI's API.
    
    Args:
        image_path: Path to the image
        question: Question text
        config: Configuration with API parameters
        
    Returns:
        The model's answer as text
    """
    api_key = config.apis.openai_api_key
    model_name = config.apis.openai_model_name
    
    base64_image = encode_image_to_base64(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Answer the following multiple choice question by selecting the letter (A, B, C, or D) only. {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": config.evaluation.max_new_tokens
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                                headers=headers, 
                                json=payload)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            return answer.strip()
        else:
            print(f"OpenAI Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"OpenAI Exception: {str(e)}")
        return None

def perform_vqa_gemini(image_path, question, config):
    """
    Perform VQA using Google's Gemini API.
    
    Args:
        image_path: Path to the image
        question: Question text
        config: Configuration with API parameters
        
    Returns:
        The model's answer as text
    """
    if genai is None:
        print("Gemini API not available. Please install the google-generativeai package.")
        return None
    
    try:
        # Configure the API
        genai.configure(api_key=config.apis.google_api_key)
        
        # Read the image
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
        
        # Initialize the model
        model = genai.GenerativeModel(config.apis.gemini_model_name)
        
        # Create prompt
        prompt = f"Answer the following multiple choice question by selecting the letter (A, B, C, or D) only. {question}"
        
        # Make API request
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": image_data}])
        
        if response:
            answer = response.text
            return answer.strip()
        else:
            print("Empty response from Gemini")
            return None
    except Exception as e:
        print(f"Gemini Exception: {str(e)}")
        return None

def extract_letter_answer(predicted_answer):
    """Extract the letter answer (A, B, C, D, E) from model output."""
    match = re.search(r"[A-E]", predicted_answer, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None

def perform_vqa(model_type, model_params, image_path, question, config):
    """
    Perform VQA using the specified model type.
    
    Args:
        model_type: Type of model ("internal", "openai", "gemini")
        model_params: Model-specific parameters (model, tokenizer for internal)
        image_path: Path to the image
        question: Question text including multiple choice options
        config: Configuration with generation parameters
        
    Returns:
        The model's answer as text
    """
    if model_type == "internal":
        model, tokenizer = model_params
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Answer the following multiple choice question by selecting the letter (A, B, C, or D) only. {question}"}
                ]}
            ]
            
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            # Get the current device from the model
            device = next(model.parameters()).device
            
            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.evaluation.max_new_tokens,
                    do_sample=config.evaluation.do_sample,
                    temperature=config.evaluation.temperature,
                    use_cache=True
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"VQA Exception: {str(e)}")
            return None
    elif model_type == "openai":
        return perform_vqa_openai(image_path, question, config)
    elif model_type == "gemini":
        return perform_vqa_gemini(image_path, question, config)
    else:
        print(f"Unknown model type: {model_type}")
        return None

def load_or_create_results_cache(model_name, cache_dir=".cache"):
    """
    Load existing results cache or create a new one for a specific model.
    
    Args:
        model_name: Name of the model to load cache for
        cache_dir: Directory to store cache files
        
    Returns:
        Dictionary with cached results
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Create model-specific cache file name
    model_slug = model_name.replace("/", "_").replace(":", "_")
    cache_file = os.path.join(cache_dir, f"evaluation_results_{model_slug}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache for {model_name}: {str(e)}")
            return {}
    else:
        return {}

def save_results_cache(cache, model_name, cache_dir=".cache"):
    """
    Save results cache to disk for a specific model.
    
    Args:
        cache: Cache dictionary to save
        model_name: Name of the model for the cache
        cache_dir: Directory to store cache files
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Create model-specific cache file name
    model_slug = model_name.replace("/", "_").replace(":", "_")
    cache_file = os.path.join(cache_dir, f"evaluation_results_{model_slug}.json")
    
    try:
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error saving cache for {model_name}: {str(e)}")

def get_cache_key(model_id, question_id):
    """
    Generate a unique cache key for a model-question pair.
    
    Args:
        model_id: Identifier for the model
        question_id: Identifier for the question
        
    Returns:
        Cache key string
    """
    return f"{model_id}::{question_id}"

@ray.remote(num_cpus=10)
def process_test_case(test_case, model_type, model_params, model_name, config, use_cache=True, cache_dir=".cache"):
    """
    Process a single test case with the given model.
    This function is designed to be used as a Ray remote task.
    
    Args:
        test_case: Tuple of (image_path, question, expected_answer, tag, question_id)
        model_type: Type of model ("internal", "openai", "gemini")
        model_params: Model-specific parameters
        model_name: Name of the model for reporting
        config: Configuration with evaluation parameters
        use_cache: Whether to use cached results
        cache_dir: Directory to store cache files
        
    Returns:
        Dictionary with result information
    """
    image_path, question, expected_answer_letter, tag, question_id = test_case
    
    # Load cache for this model (each worker loads its own copy)
    if use_cache:
        model_slug = model_name.replace("/", "_").replace(":", "_")
        cache_file = os.path.join(cache_dir, f"evaluation_results_{model_slug}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        else:
            cache = {}
            
        # Generate cache key
        cache_key = f"{model_name}::{question_id}"
        
        # Check cache for existing result
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return {
                "question": question,
                "expected": expected_answer_letter,
                "model": model_name,
                "tag": tag,
                "response_time": cached_result["response_time"],
                "question_id": question_id,
                "predicted": cached_result["predicted"],
                "predicted_letter": cached_result["predicted_letter"],
                "correct": cached_result["correct"],
                "from_cache": True
            }
    
    # If we get here, we need to run the model
    start_time = time.time()
    predicted_answer = perform_vqa(model_type, model_params, image_path, question, config)
    end_time = time.time()
    response_time = end_time - start_time
    
    # Store response details
    result = {
        "question": question,
        "expected": expected_answer_letter,
        "model": model_name,
        "tag": tag,
        "response_time": response_time,
        "question_id": question_id,
        "from_cache": False
    }

    if predicted_answer is not None:
        predicted_letter = extract_letter_answer(predicted_answer)
        result["predicted"] = predicted_answer
        result["predicted_letter"] = predicted_letter
        
        is_correct = (predicted_letter == expected_answer_letter.upper())
        result["correct"] = is_correct
        
        # Cache the result locally - each worker will maintain its own version
        # The main process will aggregate these later
        if use_cache and question_id:
            local_cache_update = {
                cache_key: {
                    "predicted": predicted_answer,
                    "predicted_letter": predicted_letter,
                    "response_time": response_time,
                    "correct": is_correct
                }
            }
            
            # Include the cache update in the result
            result["cache_update"] = local_cache_update
    else:
        result["predicted"] = None
        result["predicted_letter"] = None
        result["correct"] = False
    
    return result

def evaluate_model(model_type, model_params, model_name, test_cases, tags, config, use_cache=True):
    """
    Evaluate model on test cases using Ray for parallelization.
    
    Args:
        model_type: Type of model ("internal", "openai", "gemini")
        model_params: Model-specific parameters
        model_name: Name of the model for reporting
        test_cases: List of test cases (image_path, question, expected_answer, tag, question_id)
        tags: List of unique tags/categories
        config: Configuration with evaluation parameters
        use_cache: Whether to use cached results
        
    Returns:
        Tuple of (accuracy, tag_results, responses)
    """
    # Make sure Ray is initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    correct = 0
    total = len(test_cases)
    
    # Initialize tag-based tracking
    tag_results = {}
    for tag in tags:
        tag_results[tag] = {"correct": 0, "total": 0}
    
    # Track response times
    total_time = 0
    responses = []
    
    # Load cache for reference (but will be updated by workers)
    cache = load_or_create_results_cache(model_name) if use_cache else {}
    cache_updated = False
    
    print(f"Submitting {total} test cases for parallel evaluation with {model_name}...")
    
    # For internal models, we need to handle differently to avoid serializing the model
    if model_type == "internal" and model_params is not None:
        # We can't parallelize easily with internal models due to GPU memory constraints
        # So we'll process these sequentially
        for idx, test_case in enumerate(test_cases):
            image_path, question, expected_answer_letter, tag, question_id = test_case
            print(f"Evaluating {idx + 1}/{total} with {model_name}...")
            
            # Check cache first
            cache_key = get_cache_key(model_name, question_id)
            cached_result = cache.get(cache_key) if use_cache else None
            
            if cached_result:
                print(f"Using cached result for question {question_id}")
                result = {
                    "question": question,
                    "expected": expected_answer_letter,
                    "model": model_name,
                    "tag": tag,
                    "response_time": cached_result["response_time"],
                    "question_id": question_id,
                    "predicted": cached_result["predicted"],
                    "predicted_letter": cached_result["predicted_letter"],
                    "correct": cached_result["correct"],
                    "from_cache": True
                }
            else:
                # Measure response time
                start_time = time.time()
                predicted_answer = perform_vqa(model_type, model_params, image_path, question, config)
                end_time = time.time()
                response_time = end_time - start_time
                
                # Store response details
                result = {
                    "question": question,
                    "expected": expected_answer_letter,
                    "model": model_name,
                    "tag": tag,
                    "response_time": response_time,
                    "question_id": question_id,
                    "from_cache": False
                }

                if predicted_answer is not None:
                    predicted_letter = extract_letter_answer(predicted_answer)
                    result["predicted"] = predicted_answer
                    result["predicted_letter"] = predicted_letter
                    
                    is_correct = (predicted_letter == expected_answer_letter.upper())
                    result["correct"] = is_correct
                    
                    # Cache the result
                    if use_cache and question_id:
                        cache[cache_key] = {
                            "predicted": predicted_answer,
                            "predicted_letter": predicted_letter,
                            "response_time": response_time,
                            "correct": is_correct
                        }
                        cache_updated = True
                else:
                    print("Failed to get prediction.")
                    result["predicted"] = None
                    result["predicted_letter"] = None
                    result["correct"] = False
            
            # Process and show the result
            display_and_accumulate_result(result, tag_results)
            responses.append(result)
            total_time += result["response_time"]
            if result["correct"]:
                correct += 1
    else:
        # For API models, we can parallelize
        # Submit all test cases as Ray tasks
        futures = [
            process_test_case.remote(
                test_case, model_type, None, model_name, config, use_cache
            )
            for test_case in test_cases
        ]
        
        # Process results as they complete
        for future in tqdm(ray.get(futures), total=len(futures), desc=f"Evaluating {model_name}"):
            result = future
            
            # Update cache if needed
            if use_cache and "cache_update" in result:
                for key, value in result["cache_update"].items():
                    cache[key] = value
                    cache_updated = True
                # Remove cache_update from result
                del result["cache_update"]
            
            # Process and show the result
            display_and_accumulate_result(result, tag_results)
            responses.append(result)
            total_time += result["response_time"]
            if result["correct"]:
                correct += 1

    # Save cache if updated
    if use_cache and cache_updated:
        save_results_cache(cache, model_name)

    # Calculate overall accuracy
    accuracy = correct / total * 100 if total > 0 else 0
    avg_response_time = total_time / total if total > 0 else 0
    
    print(f"\n{model_name.upper()} Evaluation Complete")
    print(f"Correct Answers: {correct}/{total} ({accuracy:.2f}% accuracy)")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    
    # Print tag-based breakdown
    print(f"\n{model_name.upper()} Success Rate Breakdown by Tag:")
    print("-" * 60)
    print(f"{'Tag':<30} | {'Accuracy':<10} | {'Correct/Total':<15} | {'Avg Time':<10}")
    print("-" * 60)
    
    for tag, result in tag_results.items():
        if result["total"] > 0:
            # Calculate average time for this tag
            tag_times = [r["response_time"] for r in responses if r["tag"] == tag]
            avg_tag_time = sum(tag_times) / len(tag_times) if tag_times else 0
            
            tag_accuracy = (result["correct"] / result["total"]) * 100
            print(f"{tag:<30} | {tag_accuracy:>8.2f}% | {result['correct']}/{result['total']} | {avg_tag_time:>8.2f}s")
    
    return accuracy, tag_results, responses

def display_and_accumulate_result(result, tag_results):
    """Helper function to display and accumulate results."""
    tag = result["tag"]
    tag_results[tag]["total"] += 1
    if result["correct"]:
        tag_results[tag]["correct"] += 1
    
    # Output result information
    print(f"Question: {result['question']}")
    print(f"Predicted Answer: {result.get('predicted', 'None')} (Extracted: {result.get('predicted_letter', 'None')})")
    print(f"Expected Answer: {result['expected']}")
    print(f"Tag: {tag}")
    print(f"Response Time: {result['response_time']:.2f}s")
    
    if result.get("correct", False):
        print("✓ Correct")
    else:
        print("✗ Incorrect")
    
    print("-" * 40)

def compare_models(results_dict, config):
    """
    Compare performance between multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to their evaluation results
        config: Configuration with visualization parameters
        
    Returns:
        Summary of results
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    # Extract model names
    models = list(results_dict.keys())
    
    # Table headers
    print(f"{'Model':<30} | {'Overall Accuracy':<20} | {'Avg Response Time':<20}")
    print("-"*75)
    
    # Calculate average response times and print summary
    for model_name in models:
        model_results = results_dict[model_name]
        accuracy = model_results[0]
        responses = model_results[2]
        
        avg_time = sum(r["response_time"] for r in responses) / len(responses) if responses else 0
        
        print(f"{model_name:<30} | {accuracy:>18.2f}% | {avg_time:>18.2f}s")
    
    # Compare tag performance
    print("\nTAG-BASED COMPARISON:")
    print("-"*100)
    
    header = f"{'Tag':<30}"
    for model_name in models:
        header += f" | {model_name[:15]:<15}"
    print(header)
    print("-"*100)
    
    # Get all unique tags across models
    all_tags = set()
    for model_name in models:
        tag_results = results_dict[model_name][1]
        all_tags.update(tag_results.keys())
    
    # Print tag-based comparison
    tag_data = []
    for tag in sorted(all_tags):
        row = f"{tag:<30}"
        tag_row = {"tag": tag}
        
        for model_name in models:
            tag_results = results_dict[model_name][1]
            if tag in tag_results and tag_results[tag]["total"] > 0:
                tag_acc = (tag_results[tag]["correct"] / tag_results[tag]["total"]) * 100
                row += f" | {tag_acc:>13.2f}%"
                tag_row[model_name] = tag_acc
            else:
                row += f" | {'N/A':>13}"
                tag_row[model_name] = 0
        
        print(row)
        tag_data.append(tag_row)
    
    # Analyze disagreements between pairs of models
    print("\nDISAGREEMENTS ANALYSIS:")
    print("-"*80)
    
    model_pairs = [(a, b) for i, a in enumerate(models) for b in models[i+1:]]
    disagreement_data = {}
    
    for model_a, model_b in model_pairs:
        print(f"\n{model_a} vs {model_b}:")
        responses_a = results_dict[model_a][2]
        responses_b = results_dict[model_b][2]
        
        disagreements = []
        for resp_a, resp_b in zip(responses_a, responses_b):
            if resp_a["predicted_letter"] != resp_b["predicted_letter"]:
                disagreements.append({
                    "question": resp_a["question"],
                    "expected": resp_a["expected"],
                    model_a: resp_a["predicted_letter"],
                    model_b: resp_b["predicted_letter"],
                    f"{model_a}_correct": resp_a["correct"],
                    f"{model_b}_correct": resp_b["correct"],
                    "tag": resp_a["tag"]
                })
        
        print(f"Total disagreements: {len(disagreements)} out of {len(responses_a)}")
        if len(responses_a) > 0:
            print(f"Disagreement percentage: {len(disagreements)/len(responses_a)*100:.2f}%")
        
        # Count where each model was correct when the other was wrong
        a_better = sum(1 for d in disagreements if d[f"{model_a}_correct"] and not d[f"{model_b}_correct"])
        b_better = sum(1 for d in disagreements if d[f"{model_b}_correct"] and not d[f"{model_a}_correct"])
        both_wrong = sum(1 for d in disagreements if not d[f"{model_a}_correct"] and not d[f"{model_b}_correct"])
        
        print(f"{model_a} correct / {model_b} wrong: {a_better}")
        print(f"{model_b} correct / {model_a} wrong: {b_better}")
        print(f"Both models wrong but disagreed: {both_wrong}")
        
        pair_key = f"{model_a}_vs_{model_b}"
        disagreement_data[pair_key] = {
            "total": len(disagreements),
            "percentage": len(disagreements)/len(responses_a)*100 if len(responses_a) > 0 else 0,
            f"{model_a}_better": a_better,
            f"{model_b}_better": b_better,
            "both_wrong": both_wrong,
            "details": disagreements
        }
    
    # Save results to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct results summary
    results = {
        "summary": {
            "models": models,
            "accuracies": {model: results_dict[model][0] for model in models},
            "avg_times": {model: sum(r["response_time"] for r in results_dict[model][2])/len(results_dict[model][2]) 
                         if results_dict[model][2] else 0 for model in models},
            "total_examples": len(results_dict[models[0]][2]) if models and results_dict[models[0]][2] else 0
        },
        "tag_comparison": tag_data,
        "responses": {model: results_dict[model][2] for model in models},
        "disagreements": disagreement_data
    }
    
    # Save results to file
    model_names_slug = "_".join([m.split("/")[-1] for m in models])
    if len(model_names_slug) > 100:  # Limit filename length
        model_names_slug = model_names_slug[:100]
    filename = f"comparison_results_{model_names_slug}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {filename}")
    
    return results

def process_config_from_yaml(yaml_config, default_config):
    """Process YAML config and properly set values in dataclass config objects."""
    if "model" in yaml_config and isinstance(yaml_config["model"], dict):
        for k, v in yaml_config["model"].items():
            setattr(default_config.model, k, v)
    
    if "training" in yaml_config and isinstance(yaml_config["training"], dict):
        for k, v in yaml_config["training"].items():
            setattr(default_config.training, k, v)
    
    if "evaluation" in yaml_config and isinstance(yaml_config["evaluation"], dict):
        for k, v in yaml_config["evaluation"].items():
            if k == "models_to_evaluate" and isinstance(v, list):
                # Process the model configurations
                models_to_evaluate = []
                for model_config in v:
                    # Ensure all model config items are properly processed
                    # Explicitly handle the new base_model_name field
                    processed_model = {
                        "type": model_config.get("type", "internal"),
                        "name": model_config.get("name", ""),
                        "model_name": model_config.get("model_name"),
                        "base_model_name": model_config.get("base_model_name"),
                        "is_finetuned": model_config.get("is_finetuned", False),
                        "checkpoint_path": model_config.get("checkpoint_path")
                    }
                    models_to_evaluate.append(processed_model)
                default_config.evaluation.models_to_evaluate = models_to_evaluate
            else:
                setattr(default_config.evaluation, k, v)
    
    if "apis" in yaml_config and isinstance(yaml_config["apis"], dict):
        for k, v in yaml_config["apis"].items():
            setattr(default_config.apis, k, v)
            
    # Update any top-level attributes
    for k, v in yaml_config.items():
        if k not in ["model", "training", "evaluation", "apis"]:
            if hasattr(default_config, k):
                setattr(default_config, k, v)
    
    # Link some fields that need to be shared
    default_config.evaluation.checkpoint_path = default_config.evaluation.checkpoint_path or os.path.join(default_config.training.output_dir, "checkpoint-latest")
    
    return default_config

@ray.remote(num_gpus=0.25)
def evaluate_gpu_model(model_config, model_name, test_cases, all_tags, config):
    """
    Ray remote wrapper for GPU model evaluation.
    
    Args:
        model_config: Configuration for the model
        model_name: Name of the model
        test_cases: List of test cases
        all_tags: List of unique tags/categories
        config: Configuration with evaluation parameters
        
    Returns:
        Evaluation results for the model
    """
    print(f"Started GPU evaluation for model: {model_name}")
    
    try:
        # This is a standard model that we load with FastVisionModel
        if model_config.get("is_finetuned", False):
            # Use the fine-tuned model loader
            model, tokenizer = load_finetuned_model(config)
        else:
            # Use the base model loader
            model, tokenizer = load_pretrained_model(config)
        
        # Evaluate the model
        model_params = (model, tokenizer)
        use_cache = config.apis.use_cache
        results = evaluate_model("internal", model_params, model_name, test_cases, all_tags, config, use_cache)
        
        # Clear GPU memory to avoid OOM issues
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return results
    except Exception as e:
        print(f"Error evaluating model {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@ray.remote(num_cpus=2)
def evaluate_api_model_remote(model_config, model_type, model_name, test_cases, all_tags, config):
    """
    Ray remote wrapper for API model evaluation.
    
    Args:
        model_config: Configuration for the model
        model_type: Type of model ("openai", "gemini")
        model_name: Name of the model
        test_cases: List of test cases
        all_tags: List of unique tags/categories
        config: Configuration with evaluation parameters
        
    Returns:
        Tuple of (model_name, evaluation_results)
    """
    print(f"Started API evaluation for model: {model_name} (type: {model_type})")
    
    # For API-based models, we don't need to load any model
    model_params = None
    use_cache = config.apis.use_cache
    
    if model_type == "openai":
        # This is an OpenAI API model
        if not config.apis.openai_api_key:
            print(f"Error: OpenAI API key not found for {model_name}. Skipping.")
            return model_name, None
            
        # Set the model name if provided
        if "model_name" in model_config and model_config["model_name"]:
            config.apis.openai_model_name = model_config["model_name"]
    
    elif model_type == "gemini":
        # This is a Google Gemini API model
        if not config.apis.google_api_key:
            print(f"Error: Google API key not found for {model_name}. Skipping.")
            return model_name, None
            
        # Set the model name if provided
        if "model_name" in model_config and model_config["model_name"]:
            config.apis.gemini_model_name = model_config["model_name"]
    
    # Evaluate the model
    results = evaluate_model(model_type, model_params, model_name, test_cases, all_tags, config, use_cache)
    
    return model_name, results

def evaluate(config=None):
    """
    Run evaluation on multiple models in parallel using Ray.
    
    Args:
        config: Configuration for evaluation. If None, uses default_config
        
    Returns:
        Comparison results
    """
    if config is None:
        config = default_config
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Load test dataset
    test_cases, all_tags = load_evaluation_dataset(config)
    
    if not test_cases:
        print("Error: No test cases found. Please check the dataset configuration.")
        return None
    
    print(f"Loaded {len(test_cases)} test cases with {len(all_tags)} tags")
    
    # Store results for all models
    all_results = {}
    
    # Evaluate models according to configuration
    models_to_evaluate = config.evaluation.models_to_evaluate
    
    # Separate internal and API models
    internal_models = []
    api_models = []
    
    for model_config in models_to_evaluate:
        model_type = model_config.get("type", "internal")
        model_name = model_config.get("name", "unknown_model")
        
        if model_type == "internal":
            internal_models.append((model_config, model_name))
        else:
            api_models.append((model_config, model_type, model_name))
    
    # Process internal models in batches (assuming 4 models can fit in GPU memory)
    MAX_MODELS_PER_BATCH = 4
    
    # Process internal models in parallel batches
    for i in range(0, len(internal_models), MAX_MODELS_PER_BATCH):
        batch = internal_models[i:i+MAX_MODELS_PER_BATCH]
        print(f"\n{'='*50}")
        print(f"EVALUATING BATCH OF {len(batch)} INTERNAL MODELS (GPU)")
        print(f"{'='*50}")
        
        # Save original model name to restore later
        original_model_name = config.model.model_name
        
        # Create Ray actors for parallel GPU evaluation
        futures = []
        for model_config, model_name in batch:
            # Make a copy of the config for this model
            model_config_copy = config.copy() if hasattr(config, "copy") else config
            
            # Configure model-specific settings
            if model_config.get("is_finetuned", False):
                # Set checkpoint path if provided
                if "checkpoint_path" in model_config and model_config["checkpoint_path"]:
                    model_config_copy.evaluation.checkpoint_path = model_config["checkpoint_path"]
                
                # Set base model name if provided, otherwise use the config's model_name
                if "base_model_name" in model_config and model_config["base_model_name"]:
                    model_config_copy.model.model_name = model_config["base_model_name"]
                elif "model_name" in model_config and model_config["model_name"]:
                    model_config_copy.model.model_name = model_config["model_name"]
            else:
                # Use the base model loader
                if "model_name" in model_config and model_config["model_name"]:
                    model_config_copy.model.model_name = model_config["model_name"]
            
            # Evaluate using a dedicated GPU resource
            future = evaluate_gpu_model.remote(
                model_config, model_name, test_cases, all_tags, model_config_copy
            )
            futures.append((model_name, future))
        
        # Collect results from each model in this batch
        for model_name, future in futures:
            results = ray.get(future)
            if results is not None:
                all_results[model_name] = results
        
        # Restore original model name
        config.model.model_name = original_model_name
        
        # Explicitly clear GPU memory after each batch
        torch.cuda.empty_cache()
    
    # Process API models in parallel with Ray
    if api_models:
        print(f"\n{'='*50}")
        print(f"EVALUATING {len(api_models)} API MODELS IN PARALLEL")
        print(f"{'='*50}")
        
        # Submit all API models as Ray tasks
        futures = []
        for model_config, model_type, model_name in api_models:
            future = evaluate_api_model_remote.remote(
                model_config, model_type, model_name, test_cases, all_tags, config
            )
            futures.append(future)
        
        # Process results as they complete
        for model_name, results in ray.get(futures):
            if results is not None:
                all_results[model_name] = results
    
    # Compare results if we have more than one model
    if len(all_results) > 1:
        print("\nComparing model performance...")
        results = compare_models(all_results, config)
        return results
    elif len(all_results) == 1:
        model_name = list(all_results.keys())[0]
        print(f"\nOnly evaluated one model: {model_name}")
        return all_results[model_name]
    else:
        print("No models were successfully evaluated.")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate vision-language models")
    parser.add_argument("--config_file", type=str, help="Path to configuration file")
    parser.add_argument("--ray_address", type=str, help="Ray cluster address (e.g. 'auto' or 'localhost:6379')")
    parser.add_argument("--num_cpus", type=int, help="Number of CPU cores to use")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Initialize Ray with provided configuration
    ray_init_args = {"ignore_reinit_error": True}
    if args.ray_address:
        ray_init_args["address"] = args.ray_address
    if args.num_cpus:
        ray_init_args["num_cpus"] = args.num_cpus
    if args.num_gpus:
        ray_init_args["num_gpus"] = args.num_gpus
        
    # Initialize Ray
    ray.init(**ray_init_args)
    
    # Load config from file if specified
    if args.config_file:
        with open(args.config_file, 'r') as f:
            import yaml
            config_data = yaml.safe_load(f)
            # Process the nested YAML config structure
            config = process_config_from_yaml(config_data, default_config)
    else:
        config = default_config
    
    try:
        evaluate(config)
    finally:
        # Shutdown Ray when done
        ray.shutdown() 
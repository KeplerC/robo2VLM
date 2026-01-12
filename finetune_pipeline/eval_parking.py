#!/usr/bin/env python3
"""
Evaluate a single checkpoint on the parking benchmark.
Designed to be run as a standalone process with CUDA_VISIBLE_DEVICES set.

Usage:
    # Evaluate finetuned model
    CUDA_VISIBLE_DEVICES=0 python eval_parking.py \
        --checkpoint outputs/Qwen_Qwen2.5-VL-3B-Instruct/10000samples/checkpoint-final \
        --output-file results_parking/Qwen_3B_10k.json

    # Evaluate base model (no checkpoint)
    CUDA_VISIBLE_DEVICES=0 python eval_parking.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --output-file results_parking/Qwen_3B_base.json
"""

import os
import sys
import json
import argparse
import re
import shutil
from dataclasses import dataclass, asdict
from typing import Optional, List
from tqdm import tqdm

import torch
from PIL import Image

from unsloth import FastVisionModel


# Examples output directory
EXAMPLES_DIR = "/home/syx/robo2VLM/finetune_pipeline/parking_examples"


# Parking benchmark configuration
PARKING_BENCHMARK_DIR = "/home/syx/robo2VLM/finetune_pipeline/parking_benchmark"
PARKING_DATASET_FILE = os.path.join(PARKING_BENCHMARK_DIR, "dataset", "vqa_items.jsonl")


@dataclass
class EvalResult:
    """Single evaluation result."""
    item_id: str
    question: str
    correct_answer: str
    model_response: str
    is_correct: bool
    choices: Optional[List[str]] = None


def load_parking_benchmark():
    """Load parking benchmark dataset from JSONL file."""
    items = []
    with open(PARKING_DATASET_FILE, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"Loaded {len(items)} items from parking benchmark")
    return items


def get_model_short_name(model_name: Optional[str], checkpoint_path: Optional[str]) -> str:
    """Get a short model name for organizing examples."""
    if checkpoint_path:
        # Extract from path like outputs/Qwen_Qwen2.5-VL-3B-Instruct_cot/10000samples/checkpoint-final
        parts = checkpoint_path.split("/")
        for i, p in enumerate(parts):
            if p == "outputs" and i + 2 < len(parts):
                model_part = parts[i + 1]  # e.g., Qwen_Qwen2.5-VL-3B-Instruct_cot
                samples_part = parts[i + 2]  # e.g., 10000samples
                return f"{model_part}_{samples_part}"
        return checkpoint_path.replace("/", "_")
    elif model_name:
        # e.g., Qwen/Qwen2.5-VL-3B-Instruct -> Qwen_Qwen2.5-VL-3B-Instruct_0samples
        return model_name.replace("/", "_") + "_0samples"
    return "unknown_model"


def save_example(item_id: str, image_path: str, question: str, model_name: str, response: str,
                 correct_answer: str, is_correct: bool):
    """Save an example to the parking_examples directory."""
    # Create item directory
    item_dir = os.path.join(EXAMPLES_DIR, item_id)
    os.makedirs(item_dir, exist_ok=True)

    # Copy image if not already there
    dest_image = os.path.join(item_dir, "image.jpg")
    if not os.path.exists(dest_image) and os.path.exists(image_path):
        shutil.copy(image_path, dest_image)

    # Save/update question file if not exists
    question_file = os.path.join(item_dir, "question.txt")
    if not os.path.exists(question_file):
        with open(question_file, "w") as f:
            f.write(f"Correct Answer: {correct_answer}\n\n")
            f.write(question)

    # Save model response
    response_file = os.path.join(item_dir, f"response_{model_name}.txt")
    with open(response_file, "w") as f:
        f.write(f"Correct: {is_correct}\n")
        f.write(f"Expected: {correct_answer}\n\n")
        f.write(response)


def format_parking_sample(item):
    """Format a parking benchmark item for evaluation."""
    question = item["question"]
    choices = item["choices"]
    correct_choice_id = item["correct_choice_id"]

    # Build formatted question with choices
    formatted_question = f"{question}\nChoices:\n"
    choice_texts = []
    for choice in choices:
        formatted_question += f"{choice['id']}. {choice['text']}\n"
        choice_texts.append(choice['text'])

    # Load image - try both original and resized paths
    image_path = item["image_path"]
    # Convert path like "images/vqa/S1_3.jpg" to use resized version
    if "images/vqa/" in image_path:
        resized_path = image_path.replace("images/vqa/", "images/vqa_resized/")
        full_path = os.path.join(PARKING_BENCHMARK_DIR, resized_path)
        if not os.path.exists(full_path):
            # Fallback to original path
            full_path = os.path.join(PARKING_BENCHMARK_DIR, image_path)
    else:
        full_path = os.path.join(PARKING_BENCHMARK_DIR, image_path)

    return {
        "item_id": item["item_id"],
        "question": formatted_question,
        "correct_answer": correct_choice_id,
        "choices": choice_texts,
        "image_path": full_path,
    }


def extract_cot_answer(model_response: str) -> str:
    """Extract answer from CoT format response (<think>...</think><answer>...</answer>)."""
    # Try to extract from <answer> tags
    answer_match = re.search(r'<answer>\s*([A-E])\s*</answer>', model_response, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Fallback: look for answer patterns after </think>
    after_think = model_response.split('</think>')[-1] if '</think>' in model_response else model_response

    # Try common patterns
    patterns = [
        r"(?:the answer is|my answer is|final answer is|correct answer is)[:\s]*\(?([A-E])\)?",
        r"(?:answer|option)[:\s]*\(?([A-E])\)?",
        r"\*\*([A-E])\*\*",
        r"^([A-E])[.\s]",  # Answer at start of line
    ]

    for pattern in patterns:
        match = re.search(pattern, after_think, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Last resort: find any standalone letter
    letters = re.findall(r'\b([A-E])\b', after_think.upper())
    if letters:
        return letters[-1]

    return ""


def extract_zero_shot_answer(model_response: str) -> str:
    """Extract answer letter from zero-shot response."""
    response = model_response.strip()
    response_upper = response.upper()

    # If response starts with a letter A-E followed by punctuation/space, that's the answer
    if response_upper and response_upper[0] in 'ABCDE':
        if len(response_upper) == 1 or response_upper[1] in '.):, \t\n':
            return response_upper[0]

    # Look for explicit answer patterns
    patterns = [
        r"(?:the answer is|my answer is|final answer is|correct answer is)[:\s]*\(?([A-E])\)?",
        r"(?:therefore|thus|so)[,\s]+(?:the answer is)?[:\s]*\(?([A-E])\)?",
        r"(?:answer|option)[:\s]*\(?([A-E])\)?",
        r"\*\*([A-E])\*\*",  # Bold markdown **A**
        r"\n([A-E])[.\s\)]",  # Letter at start of new line
    ]

    for pattern in patterns:
        match = re.search(pattern, response_upper, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Look for "A. choice text" or "B) choice text" pattern anywhere
    choice_pattern = r'(?:^|\n)\s*([A-E])[.\)]\s*\w'
    matches = re.findall(choice_pattern, response_upper, re.MULTILINE)
    if matches:
        return matches[-1]

    return ""


def check_answer(model_response: str, correct_answer: str, mode: str = "zero_shot") -> bool:
    """Check if model response matches correct answer."""
    correct = correct_answer.strip().upper()

    if mode == "cot":
        extracted = extract_cot_answer(model_response)
        return extracted == correct
    else:
        extracted = extract_zero_shot_answer(model_response)
        return extracted == correct


def evaluate(
    model_name: Optional[str],
    checkpoint_path: Optional[str],
    output_file: str,
    mode: str = "zero_shot",
):
    """Evaluate model on parking benchmark."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name or checkpoint_path}")
    print(f"Mode: {mode}")
    print(f"Dataset: parking_benchmark")
    print(f"Output: {output_file}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"{'='*60}\n")

    # Set max_new_tokens based on mode
    max_new_tokens = 2048 if mode == "cot" else 128

    # Load model
    if checkpoint_path:
        print(f"Loading finetuned model from: {checkpoint_path}")
        model, tokenizer = FastVisionModel.from_pretrained(
            checkpoint_path,
            use_gradient_checkpointing=False,
        )
    else:
        print(f"Loading base model: {model_name}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            use_gradient_checkpointing=False,
        )

    FastVisionModel.for_inference(model)

    # Get model short name for saving examples
    model_short_name = get_model_short_name(model_name, checkpoint_path)
    print(f"Model short name for examples: {model_short_name}")

    # Create examples directory
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    # Load parking benchmark
    items = load_parking_benchmark()

    # Evaluate
    results = []
    correct_count = 0

    for item in tqdm(items, desc="Evaluating on parking_benchmark"):
        try:
            formatted = format_parking_sample(item)

            # Check if image exists
            if not os.path.exists(formatted["image_path"]):
                print(f"Warning: Image not found: {formatted['image_path']}")
                continue

            # Create message
            question_text = formatted["question"]
            if mode == "cot":
                question_text = (
                    f"{formatted['question']}\n"
                    "Think step by step about this question. "
                    "Put your reasoning inside <think>...</think> tags, "
                    "then provide your final answer inside <answer>...</answer> tags."
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question_text},
                    ]
                }
            ]

            # Apply chat template
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Load and prepare image - resize to avoid token count issues
            image = Image.open(formatted["image_path"]).convert("RGB")
            # Resize large images to max 1024 on longest side
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)

            # Tokenize
            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)

            # Generate
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    use_cache=True,
                )

            # Decode
            response = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Check answer
            is_correct = check_answer(response, formatted["correct_answer"], mode=mode)
            if is_correct:
                correct_count += 1

            results.append(EvalResult(
                item_id=formatted["item_id"],
                question=formatted["question"][:500],
                correct_answer=formatted["correct_answer"],
                model_response=response,
                is_correct=is_correct,
                choices=formatted["choices"],
            ))

            # Save example to parking_examples directory
            save_example(
                item_id=formatted["item_id"],
                image_path=formatted["image_path"],
                question=formatted["question"],
                model_name=model_short_name,
                response=response,
                correct_answer=formatted["correct_answer"],
                is_correct=is_correct,
            )

        except Exception as e:
            print(f"Error processing item {item.get('item_id', 'unknown')}: {e}")
            continue

    # Calculate accuracy
    accuracy = correct_count / len(results) if results else 0.0

    # Save results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    output_data = {
        "model": model_name or checkpoint_path,
        "checkpoint": checkpoint_path,
        "dataset": "parking_benchmark",
        "mode": mode,
        "total_samples": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results: {correct_count}/{len(results)} = {accuracy*100:.2f}%")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on parking benchmark")
    parser.add_argument("--model", type=str, help="Base model name (for base model evaluation)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path (for finetuned model)")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "cot"],
                        help="Evaluation mode: zero_shot (direct answer) or cot (chain-of-thought)")

    args = parser.parse_args()

    if not args.model and not args.checkpoint:
        parser.error("Either --model or --checkpoint must be provided")

    evaluate(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        output_file=args.output_file,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

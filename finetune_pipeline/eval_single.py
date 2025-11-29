#!/usr/bin/env python3
"""
Evaluate a single checkpoint on a single dataset.
Designed to be run as a standalone process with CUDA_VISIBLE_DEVICES set.

Usage:
    # Evaluate finetuned model
    CUDA_VISIBLE_DEVICES=0 python eval_single.py \
        --checkpoint outputs/Qwen_Qwen2.5-VL-3B-Instruct/10000samples/checkpoint-final \
        --dataset merged_dataset \
        --output-file results/Qwen_3B_10k_merged.json

    # Evaluate base model (no checkpoint)
    CUDA_VISIBLE_DEVICES=0 python eval_single.py \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --dataset merged_dataset \
        --output-file results/Qwen_3B_base_merged.json
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List
from tqdm import tqdm

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image

from unsloth import FastVisionModel


# Dataset configurations
DATASETS = {
    "merged_dataset": {
        "path": "/home/syx/robo2VLM/merged_dataset",
        "split": "test",
        "is_local": True,
    },
    "ERQA": {
        "path": "FlagEval/ERQA",
        "split": "test",
        "is_local": False,
    },
    "CV-Bench": {
        "path": "nyu-visionx/CV-Bench",
        "split": "test",
        "is_local": False,
    },
}


@dataclass
class EvalResult:
    """Single evaluation result."""
    question: str
    correct_answer: str
    model_response: str
    is_correct: bool
    choices: Optional[List[str]] = None


def load_eval_dataset(dataset_name: str, max_samples: Optional[int] = None):
    """Load evaluation dataset."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    config = DATASETS[dataset_name]
    print(f"Loading dataset: {dataset_name} from {config['path']}")

    if config["is_local"]:
        dataset_dict = load_from_disk(config["path"])
        dataset = dataset_dict[config["split"]]
    else:
        dataset = load_dataset(config["path"], split=config["split"])

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Loaded {len(dataset)} samples")
    return dataset, dataset_name


def format_sample(sample, dataset_name: str):
    """Format sample based on dataset type."""
    # Detect format based on columns
    columns = set(sample.keys())

    # merged_dataset format: question, choices, correct_answer, image
    if "choices" in columns and "correct_answer" in columns:
        question = sample["question"]
        choices = sample["choices"]
        correct_answer = sample["correct_answer"]
        image = sample["image"]

        formatted_question = f"{question}\nChoices:\n"
        for i, choice in enumerate(choices):
            formatted_question += f"{chr(65 + i)}. {choice}\n"

        return {
            "question": formatted_question,
            "correct_answer": correct_answer,
            "choices": choices,
            "image": image,
        }

    # ERQA format: question with embedded choices, answer
    elif "answer" in columns and "question" in columns:
        question = sample["question"]
        answer = sample.get("answer", sample.get("correct_answer", ""))
        image = sample.get("image", sample.get("images", None))

        # Handle image list
        if isinstance(image, list) and len(image) > 0:
            image = image[0]

        return {
            "question": question,
            "correct_answer": str(answer),
            "choices": None,
            "image": image,
        }

    # CV-Bench format
    elif "answer" in columns:
        question = sample.get("question", sample.get("prompt", ""))
        answer = sample["answer"]
        image = sample.get("image", sample.get("images", None))

        if isinstance(image, list) and len(image) > 0:
            image = image[0]

        return {
            "question": question,
            "correct_answer": str(answer),
            "choices": None,
            "image": image,
        }

    else:
        raise ValueError(f"Unknown dataset format. Columns: {columns}")


import re


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


def check_answer(model_response: str, correct_answer: str, mode: str = "zero_shot") -> bool:
    """Check if model response matches correct answer.

    Args:
        model_response: Raw model output
        correct_answer: Expected correct answer letter
        mode: "zero_shot" or "cot" - determines how to extract answer from response
    """
    correct = correct_answer.strip().upper()

    if mode == "cot":
        # Extract answer from CoT format
        extracted = extract_cot_answer(model_response)
        return extracted == correct
    else:
        # Zero-shot mode: original logic
        response = model_response.strip().upper()

        # Direct match
        if response == correct:
            return True

        # Check if response starts with correct letter (e.g., "A. xxx")
        if response and correct and response[0] == correct[0]:
            return True

        # Check if correct answer is contained in response
        if correct in response:
            return True

        return False


def evaluate(
    model_name: Optional[str],
    checkpoint_path: Optional[str],
    dataset_name: str,
    output_file: str,
    max_samples: Optional[int] = None,
    mode: str = "zero_shot",
):
    """Evaluate model on dataset.

    Args:
        mode: "zero_shot" or "cot" - determines answer extraction method and max_new_tokens
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name or checkpoint_path}")
    print(f"Mode: {mode}")
    print(f"Dataset: {dataset_name}")
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

    # Load dataset
    dataset, ds_name = load_eval_dataset(dataset_name, max_samples)

    # Evaluate
    results = []
    correct_count = 0

    for sample in tqdm(dataset, desc=f"Evaluating on {dataset_name}"):
        try:
            formatted = format_sample(sample, ds_name)

            # Skip if no image
            if formatted["image"] is None:
                continue

            # Create message
            question_text = formatted["question"]
            if mode == "cot":
                # Add CoT instruction to match training format
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

            # Prepare image
            image = formatted["image"]
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")

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
                question=formatted["question"][:500],  # Truncate for storage
                correct_answer=formatted["correct_answer"],
                model_response=response,
                is_correct=is_correct,
                choices=formatted["choices"],
            ))

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    # Calculate accuracy
    accuracy = correct_count / len(results) if results else 0.0

    # Save results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    output_data = {
        "model": model_name or checkpoint_path,
        "checkpoint": checkpoint_path,
        "dataset": dataset_name,
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
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--model", type=str, help="Base model name (for base model evaluation)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path (for finetuned model)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()),
                        help="Dataset to evaluate on")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--max-samples", type=int, help="Max samples to evaluate")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "cot"],
                        help="Evaluation mode: zero_shot (direct answer) or cot (chain-of-thought)")

    args = parser.parse_args()

    if not args.model and not args.checkpoint:
        parser.error("Either --model or --checkpoint must be provided")

    evaluate(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        output_file=args.output_file,
        max_samples=args.max_samples,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

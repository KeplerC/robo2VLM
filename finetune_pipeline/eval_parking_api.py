#!/usr/bin/env python3
"""
Evaluate API-based models (OpenAI GPT, Google Gemini) on the parking benchmark.

Usage:
    # Evaluate GPT-4o
    python eval_parking_api.py \
        --model gpt-4o \
        --output-file results_parking/gpt-4o_cot.json \
        --mode cot

    # Evaluate Gemini
    python eval_parking_api.py \
        --model gemini-1.5-pro \
        --output-file results_parking/gemini-1.5-pro_cot.json \
        --mode cot

Environment variables:
    OPENAI_API_KEY: Required for OpenAI models
    GOOGLE_API_KEY: Required for Gemini models
"""

import os
import sys
import json
import argparse
import re
import shutil
import base64
import time
from dataclasses import dataclass, asdict
from typing import Optional, List
from tqdm import tqdm
from PIL import Image
import io

# API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Examples output directory
EXAMPLES_DIR = "/home/syx/robo2VLM/finetune_pipeline/parking_examples"

# Parking benchmark configuration
PARKING_BENCHMARK_DIR = "/home/syx/robo2VLM/finetune_pipeline/parking_benchmark"
PARKING_DATASET_FILE = os.path.join(PARKING_BENCHMARK_DIR, "dataset", "vqa_items.jsonl")

# Model configurations
OPENAI_MODELS = [
    # GPT-5 family (latest)
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.1",
    "gpt-5",
    # GPT-4 family
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    # Reasoning models
    "o1",
    "o3-mini",
]
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"]

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # seconds between API calls


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


def get_model_short_name(model_name: str, mode: str) -> str:
    """Get a short model name for organizing examples."""
    # Clean model name (replace slashes and special chars)
    clean_name = model_name.replace("/", "_").replace("-", "_")
    return f"{clean_name}_{mode}"


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


def encode_image_base64(image_path: str, max_size: int = 1024) -> str:
    """Load image, resize if needed, and encode as base64."""
    image = Image.open(image_path).convert("RGB")

    # Resize large images
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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


class OpenAIClient:
    """OpenAI API client for vision models."""

    def __init__(self, model: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, image_base64: str, question: str, mode: str) -> str:
        """Generate response for an image+question pair."""
        if mode == "cot":
            prompt = (
                f"{question}\n"
                "Think step by step about this question. "
                "Put your reasoning inside <think>...</think> tags, "
                "then provide your final answer inside <answer>...</answer> tags."
            )
        else:
            prompt = f"{question}\nProvide only the letter of the correct answer (A, B, C, D, or E)."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        max_tokens = 4096 if mode == "cot" else 1024

        # GPT-5 and reasoning models use max_completion_tokens and reasoning parameter
        if self.model.startswith("gpt-5"):
            # GPT-5 requires reasoning.effort to be set (default is "none" which produces empty output)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens + 1024,  # Extra tokens for reasoning
                # reasoning={"effort": "low"},  # Enable reasoning output
            )
        elif self.model.startswith("o1") or self.model.startswith("o3"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )

        return response.choices[0].message.content or ""


class GeminiClient:
    """Google Gemini API client for vision models."""

    def __init__(self, model: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    def generate(self, image_path: str, question: str, mode: str) -> str:
        """Generate response for an image+question pair."""
        if mode == "cot":
            prompt = (
                f"{question}\n"
                "Think step by step about this question. "
                "Put your reasoning inside <think>...</think> tags, "
                "then provide your final answer inside <answer>...</answer> tags."
            )
        else:
            prompt = f"{question}\nProvide only the letter of the correct answer (A, B, C, D, or E)."

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize if needed
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Configure generation
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=2048 if mode == "cot" else 128,
            temperature=0,
        )

        response = self.model.generate_content(
            [image, prompt],
            generation_config=generation_config,
        )

        return response.text


def evaluate(
    model_name: str,
    output_file: str,
    mode: str = "zero_shot",
):
    """Evaluate API model on parking benchmark."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Mode: {mode}")
    print(f"Dataset: parking_benchmark")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Initialize client based on model
    if model_name in OPENAI_MODELS:
        client = OpenAIClient(model_name)
        use_base64 = True
    elif model_name in GEMINI_MODELS:
        client = GeminiClient(model_name)
        use_base64 = False
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: {OPENAI_MODELS + GEMINI_MODELS}")

    # Get model short name for saving examples
    model_short_name = get_model_short_name(model_name, mode)
    print(f"Model short name for examples: {model_short_name}")

    # Create examples directory
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    # Load parking benchmark
    items = load_parking_benchmark()

    # Evaluate
    results = []
    correct_count = 0
    errors = []

    for item in tqdm(items, desc=f"Evaluating {model_name}"):
        try:
            formatted = format_parking_sample(item)

            # Check if image exists
            if not os.path.exists(formatted["image_path"]):
                print(f"Warning: Image not found: {formatted['image_path']}")
                continue

            # Generate response
            if use_base64:
                image_data = encode_image_base64(formatted["image_path"])
                response = client.generate(image_data, formatted["question"], mode)
            else:
                response = client.generate(formatted["image_path"], formatted["question"], mode)

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

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            error_msg = f"Error processing item {item.get('item_id', 'unknown')}: {e}"
            print(error_msg)
            errors.append(error_msg)
            # Continue with exponential backoff on rate limit errors
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(5)
            continue

    # Calculate accuracy
    accuracy = correct_count / len(results) if results else 0.0

    # Save results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    output_data = {
        "model": model_name,
        "checkpoint": None,
        "dataset": "parking_benchmark",
        "mode": mode,
        "total_samples": len(results),
        "correct": correct_count,
        "accuracy": accuracy,
        "errors": errors,
        "results": [asdict(r) for r in results],
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results: {correct_count}/{len(results)} = {accuracy*100:.2f}%")
    if errors:
        print(f"Errors: {len(errors)}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}\n")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate API models on parking benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name. Supported: {OPENAI_MODELS + GEMINI_MODELS}")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "cot"],
                        help="Evaluation mode: zero_shot (direct answer) or cot (chain-of-thought)")

    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        output_file=args.output_file,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()

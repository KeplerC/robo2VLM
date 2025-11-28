#!/usr/bin/env python3
"""
Generate Chain-of-Thought (CoT) reasoning for VQA dataset using Qwen2.5-VL-72B.

Uses SGLang OpenAI-compatible server for inference. Similar to Video-R1's approach.

Usage:
    1. Start sglang server:
       ./launch_server.sh --model Qwen/Qwen2.5-VL-72B-Instruct --tp 4

    2. Run reasoning generation:
       python generate_reasoning.py --server-url http://localhost:30000/v1

    3. Resume from checkpoint:
       python generate_reasoning.py --resume
"""

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict, Dataset

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ReasoningConfig:
    """Configuration for reasoning generation."""
    # Server settings
    server_url: str = "http://localhost:30000/v1"
    api_key: str = "EMPTY"

    # Model name (for metadata - actual model runs on server)
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"

    # Dataset settings
    input_path: str = "/home/syx/robo2VLM/merged_dataset"
    output_path: str = "/home/syx/robo2VLM/reasoning_dataset"
    checkpoint_dir: str = "/home/syx/robo2VLM/reasoning_checkpoints"

    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024

    # Concurrency
    num_workers: int = 16

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Limits - only for training set
    max_train_samples: int = 100000  # Max samples to process from training set


# ============================================================================
# Prompt Template
# ============================================================================

COT_SYSTEM_PROMPT = """You are a helpful assistant that carefully analyzes images and generates detailed reasoning for visual question answering.

You will be given a question, multiple choices, and the correct answer. Your task is to generate a detailed chain-of-thought reasoning process that explains WHY the given answer is correct.

Your thinking process should:
- Describe what you observe in the image relevant to the question
- Analyze each answer choice and explain why it is or isn't correct
- Use natural expressions like "let me think", "I notice that", "this suggests", "wait", "hmm", "looking at the image"
- Show your reasoning process naturally, as if thinking out loud
- Build a logical argument that leads to the correct answer

Format your response as:
<think>
[Your detailed reasoning process that explains why the correct answer is right]
</think>
<answer>
[The correct answer letter provided]
</answer>"""


def build_prompt(question: str, choices: List[str], correct_answer: str, hint: str = "") -> str:
    """Build the user prompt for a single VQA item with the correct answer and optional hint."""
    choice_letters = ['A', 'B', 'C', 'D', 'E']
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i < len(choice_letters):
            formatted_choices.append(f"{choice_letters[i]}. {choice}")
    choices_text = "\n".join(formatted_choices)

    # Include hint section if hint is provided
    hint_section = ""
    if hint and hint.strip():
        hint_section = f"""
Context and Hints:
{hint}

"""

    return f"""Look at this image and generate reasoning for the following question.

Question: {question}

Choices:
{choices_text}

The correct answer is: {correct_answer}
{hint_section}
Please think about this question as if you were a human pondering deeply. 
Include self-reflection or verification in your reasoning process. 
Explain step by step why {correct_answer} is the correct answer."""


# ============================================================================
# Response Parsing
# ============================================================================

def extract_reasoning_and_answer(response: str) -> Tuple[str, str]:
    """Extract reasoning and answer from model response.

    Returns:
        tuple: (reasoning_text, answer_letter)
    """
    if not response or response.startswith("ERROR:"):
        return "", ""

    # Extract thinking/reasoning
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""

    # If no <think> tags, try to get everything before <answer>
    if not reasoning:
        answer_start = response.find('<answer>')
        if answer_start > 0:
            reasoning = response[:answer_start].strip()
        else:
            # No structured output, use the whole response as reasoning
            reasoning = response.strip()

    # Extract answer
    answer_match = re.search(r'<answer>\s*([A-E])\s*</answer>', response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).upper()
    else:
        # Try to find patterns indicating final answer
        patterns = [
            r"(?:the answer is|my answer is|final answer is|correct answer is)[:\s]*\(?([A-E])\)?",
            r"(?:answer|option)[:\s]*\(?([A-E])\)?[.\s]*$",
            r"\b([A-E])\b[.\s]*$",
            r"\*\*([A-E])\*\*",
        ]
        answer = ""
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
                break

        if not answer:
            # Fallback: find any letter in response
            letters = re.findall(r'\b([A-E])\b', response.upper())
            if letters:
                answer = letters[-1]

    return reasoning, answer


# ============================================================================
# SGLang Client
# ============================================================================

class SGLangReasoningClient:
    """Client for generating reasoning using SGLang OpenAI-compatible server."""

    def __init__(self, config: ReasoningConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.server_url,
            api_key=config.api_key,
        )
        self._validate_connection()

    def _validate_connection(self) -> bool:
        """Validate connection to the server."""
        try:
            models = self.client.models.list()
            print(f"Connected to server. Available models: {[m.id for m in models.data]}")
            return True
        except Exception as e:
            print(f"Warning: Could not connect to server at {self.config.server_url}: {e}")
            return False

    def _encode_image(self, image) -> str:
        """Encode image to base64 string."""
        # Handle different image formats
        if isinstance(image, dict) and "bytes" in image:
            pil_image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.open(BytesIO(image))

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        return base64.b64encode(image_bytes).decode('utf-8')

    def generate_reasoning(
        self,
        image,
        question: str,
        choices: List[str],
        correct_answer: str,
        hint: str = "",
    ) -> Tuple[str, str, float]:
        """
        Generate reasoning for an image-question pair given the correct answer.

        Args:
            image: The image to analyze
            question: The question text
            choices: List of answer choices
            correct_answer: The correct answer letter
            hint: Optional hint with task context and reasoning guidance

        Returns:
            Tuple of (reasoning, answer, response_time)
        """
        image_base64 = self._encode_image(image)
        prompt = build_prompt(question, choices, correct_answer, hint)

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                elapsed = time.time() - start_time

                content = response.choices[0].message.content or ""
                reasoning, answer = extract_reasoning_and_answer(content)

                return reasoning, answer, elapsed

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"ERROR: {str(e)}", "", 0.0


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"processed_ids": [], "results": {}}


def save_checkpoint(checkpoint_path: str, checkpoint_data: Dict):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    # Write to temp file first, then rename (atomic operation)
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f)
    os.rename(temp_path, checkpoint_path)


# ============================================================================
# Main Processing
# ============================================================================

def process_single_item(
    client: SGLangReasoningClient,
    item: Dict,
    item_id: str,
) -> Dict:
    """Process a single dataset item."""
    correct_answer = item.get('correct_answer', '')
    hint = item.get('hint', '')  # Extract hint if available

    reasoning, answer, elapsed = client.generate_reasoning(
        image=item['image'],
        question=item['question'],
        choices=item['choices'],
        correct_answer=correct_answer,
        hint=hint,
    )

    return {
        "id": item_id,
        "reasoning": reasoning,
        "generated_answer": answer,
        "response_time": elapsed,
        "correct_answer": correct_answer,
        "is_correct": answer.upper() == correct_answer.upper() if answer and correct_answer else False,
    }


def process_split(
    client: SGLangReasoningClient,
    dataset,
    split_name: str,
    checkpoint_path: str,
    config: ReasoningConfig,
) -> List[Dict]:
    """Process a dataset split and generate reasoning."""

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    processed_ids = set(checkpoint["processed_ids"])
    results = checkpoint["results"]

    print(f"\nProcessing {split_name} split: {len(dataset)} samples")
    print(f"Already processed: {len(processed_ids)} samples")

    # Collect items to process
    items_to_process = []
    for idx, item in enumerate(dataset):
        item_id = item.get('id', str(idx))
        if item_id not in processed_ids:
            items_to_process.append((idx, item_id, item))

    print(f"Remaining to process: {len(items_to_process)} samples")

    if not items_to_process:
        # Return existing results in order
        return [
            results.get(item.get('id', str(idx)), {"reasoning": "", "generated_answer": ""})
            for idx, item in enumerate(dataset)
        ]

    # Process with concurrent workers
    correct_count = 0
    total_processed = 0

    def process_item(args):
        idx, item_id, item = args
        return process_single_item(client, item, item_id)

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_item, args): args[1]
            for args in items_to_process
        }

        pbar = tqdm(total=len(futures), desc=f"Processing {split_name}")
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                result = future.result()
                results[item_id] = result
                processed_ids.add(item_id)
                total_processed += 1

                if result.get("is_correct"):
                    correct_count += 1

                # Update progress bar
                accuracy = correct_count / total_processed * 100 if total_processed > 0 else 0
                pbar.set_postfix({"accuracy": f"{accuracy:.1f}%"})
                pbar.update(1)

                # Save checkpoint periodically (every 100 items)
                if total_processed % 100 == 0:
                    checkpoint["processed_ids"] = list(processed_ids)
                    checkpoint["results"] = results
                    save_checkpoint(checkpoint_path, checkpoint)

            except Exception as e:
                print(f"Error processing {item_id}: {e}")
                results[item_id] = {"reasoning": f"ERROR: {str(e)}", "generated_answer": ""}
                processed_ids.add(item_id)

        pbar.close()

    # Final checkpoint save
    checkpoint["processed_ids"] = list(processed_ids)
    checkpoint["results"] = results
    save_checkpoint(checkpoint_path, checkpoint)

    # Return results in original order
    final_results = []
    for idx, item in enumerate(dataset):
        item_id = item.get('id', str(idx))
        final_results.append(results.get(item_id, {"reasoning": "", "generated_answer": ""}))

    return final_results


def add_reasoning_columns(dataset, results: List[Dict]) -> Dataset:
    """Add reasoning columns to dataset."""

    def add_columns(example, idx):
        result = results[idx]
        example['reasoning'] = result.get('reasoning', '')
        example['generated_answer'] = result.get('generated_answer', '')
        return example

    return dataset.map(add_columns, with_indices=True)


def main():
    parser = argparse.ArgumentParser(description="Generate CoT reasoning for VQA dataset")
    parser.add_argument("--server-url", type=str, default="http://localhost:30000/v1",
                        help="SGLang server URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-72B-Instruct",
                        help="Model name (for metadata)")
    parser.add_argument("--input", type=str, default="/home/syx/robo2VLM/merged_dataset",
                        help="Path to input dataset")
    parser.add_argument("--output", type=str, default="/home/syx/robo2VLM/reasoning_dataset",
                        help="Path to output dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="/home/syx/robo2VLM/reasoning_checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of concurrent workers")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens to generate")
    parser.add_argument("--max-train-samples", type=int, default=100000,
                        help="Max samples to randomly sample from training set (default: 100000)")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HuggingFace repo to push to (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    # Create config
    config = ReasoningConfig(
        server_url=args.server_url,
        model_name=args.model,
        input_path=args.input,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_train_samples=args.max_train_samples,
    )

    print("=" * 60)
    print("CoT Reasoning Generation Pipeline")
    print("=" * 60)
    print(f"Server URL: {config.server_url}")
    print(f"Model: {config.model_name}")
    print(f"Input dataset: {config.input_path}")
    print(f"Output dataset: {config.output_path}")
    print(f"Max train samples: {config.max_train_samples}")
    print(f"Workers: {config.num_workers}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset_dict = load_from_disk(config.input_path)
    print(f"Dataset loaded: {dataset_dict}")

    # Initialize client
    print("\nConnecting to SGLang server...")
    client = SGLangReasoningClient(config)

    processed_datasets = {}

    # Process ONLY training set with reasoning generation
    if 'train' in dataset_dict:
        train_dataset = dataset_dict['train']
        print(f"\nOriginal training set size: {len(train_dataset)}")

        # Randomly sample if training set is larger than max_train_samples
        if len(train_dataset) > config.max_train_samples:
            print(f"Randomly sampling {config.max_train_samples} samples from training set (seed={args.seed})")
            import random
            random.seed(args.seed)
            indices = random.sample(range(len(train_dataset)), config.max_train_samples)
            indices.sort()  # Keep sorted for better cache locality
            train_dataset = train_dataset.select(indices)
            print(f"Sampled training set size: {len(train_dataset)}")

        checkpoint_path = os.path.join(config.checkpoint_dir, "train_checkpoint.json")

        # Generate reasoning for training set
        results = process_split(
            client=client,
            dataset=train_dataset,
            split_name="train",
            checkpoint_path=checkpoint_path,
            config=config,
        )

        # Add reasoning columns to training set
        print(f"\nAdding reasoning columns to train split...")
        processed_datasets['train'] = add_reasoning_columns(train_dataset, results)
    else:
        print("Warning: No 'train' split found in dataset")

    # Keep validation and test sets UNCHANGED (no reasoning generation)
    for split in ['validation', 'test']:
        if split in dataset_dict:
            print(f"\nKeeping {split} set unchanged (no reasoning generation)")
            # Add empty reasoning columns to maintain schema consistency
            def add_empty_reasoning(example):
                example['reasoning'] = ''
                example['generated_answer'] = ''
                return example
            processed_datasets[split] = dataset_dict[split].map(add_empty_reasoning)
            print(f"  {split}: {len(processed_datasets[split])} samples")

    # Create final dataset
    print("\nCreating final dataset...")
    final_dataset = DatasetDict(processed_datasets)

    # Save dataset
    print(f"\nSaving dataset to {config.output_path}...")
    os.makedirs(config.output_path, exist_ok=True)
    final_dataset.save_to_disk(config.output_path, num_proc=32)
    print("Dataset saved!")

    # Push to hub if specified
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace: {args.push_to_hub}")
        final_dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Dataset available at: https://huggingface.co/datasets/{args.push_to_hub}")

    print("\n" + "=" * 60)
    print("REASONING GENERATION COMPLETE!")
    print("=" * 60)

    # Print statistics
    for split in processed_datasets:
        ds = processed_datasets[split]
        if 'reasoning' in ds.column_names:
            non_empty = sum(1 for r in ds['reasoning'] if r and not r.startswith("ERROR"))
            print(f"{split}: {len(ds)} samples, {non_empty} with reasoning ({100*non_empty/len(ds):.1f}%)")
        else:
            print(f"{split}: {len(ds)} samples (no reasoning)")


if __name__ == "__main__":
    main()

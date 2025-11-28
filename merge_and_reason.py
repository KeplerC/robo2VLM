#!/usr/bin/env python3
"""
Refactored pipeline: Sample -> Generate Reasoning -> Merge -> Save locally

Workflow:
1. Load ManipulationVQA-60k from HuggingFace and sample
2. Load local Agibot VQA and sample
3. Generate reasoning for both sampled datasets
4. Merge datasets
5. Save HuggingFace dataset to local merged_dataset

Usage:
    python merge_and_reason.py --manipulation-samples 30000 --agibot-samples 50000
    python merge_and_reason.py --skip-reasoning  # Skip reasoning generation
"""

import argparse
import base64
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as HFImage, load_from_disk

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Data sources
    local_vqa_path: str = "/home/syx/robo2VLM/vqa_output14000new3"
    hf_source_repo: str = "keplerccc/ManipulationVQA-60k"

    # Output paths
    output_path: str = "/home/syx/robo2VLM/merged_dataset"
    checkpoint_dir: str = "/home/syx/robo2VLM/reasoning_checkpoints"

    # Sampling settings
    manipulation_samples: int = 30000  # Samples from ManipulationVQA-60k
    agibot_samples: int = 50000  # Samples from local Agibot VQA

    # Split settings
    val_size: int = 5000
    test_size: int = 5000

    # Reasoning settings
    server_url: str = "http://localhost:30000/v1"
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    temperature: float = 0.7
    max_tokens: int = 10240
    num_workers: int = 16
    max_retries: int = 3

    # Random seed
    seed: int = 42


# ============================================================================
# Reasoning Generation
# ============================================================================

COT_SYSTEM_PROMPT = """You are a helpful assistant that carefully analyzes images and generates detailed reasoning for visual question answering.

You will be given a question, multiple choices, and the correct answer. Your task is to generate a detailed chain-of-thought reasoning process that explains WHY the given answer is correct.

Your thinking process should:
- Describe what you observe in the image relevant to the question
- Analyze each answer choice and explain why it is or isn't correct
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
    """Build the user prompt for reasoning generation."""
    choice_letters = ['A', 'B', 'C', 'D', 'E']
    formatted_choices = []
    for i, choice in enumerate(choices):
        if i < len(choice_letters):
            formatted_choices.append(f"{choice_letters[i]}. {choice}")
    choices_text = "\n".join(formatted_choices)

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


def extract_reasoning_and_answer(response: str) -> Tuple[str, str]:
    """Extract reasoning and answer from model response."""
    if not response or response.startswith("ERROR:"):
        return "", ""

    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""

    if not reasoning:
        answer_start = response.find('<answer>')
        if answer_start > 0:
            reasoning = response[:answer_start].strip()
        else:
            reasoning = response.strip()

    answer_match = re.search(r'<answer>\s*([A-E])\s*</answer>', response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).upper()
    else:
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
            letters = re.findall(r'\b([A-E])\b', response.upper())
            if letters:
                answer = letters[-1]

    return reasoning, answer


class ReasoningClient:
    """Client for reasoning generation using SGLang server."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        if OpenAI is None:
            raise ImportError("OpenAI package required: pip install openai")
        self.client = OpenAI(
            base_url=config.server_url,
            api_key="EMPTY",
        )
        self._validate_connection()

    def _validate_connection(self) -> bool:
        try:
            models = self.client.models.list()
            print(f"Connected to server. Available models: {[m.id for m in models.data]}")
            return True
        except Exception as e:
            print(f"Warning: Could not connect to server at {self.config.server_url}: {e}")
            return False

    def _encode_image(self, image) -> str:
        """Encode image to base64."""
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
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def generate_reasoning(self, image, question: str, choices: List[str],
                          correct_answer: str, hint: str = "") -> Tuple[str, str, float]:
        """Generate reasoning for an item."""
        image_base64 = self._encode_image(image)
        prompt = build_prompt(question, choices, correct_answer, hint)

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
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
                    top_p=0.95,
                )
                elapsed = time.time() - start_time
                content = response.choices[0].message.content or ""
                reasoning, answer = extract_reasoning_and_answer(content)
                return reasoning, answer, elapsed
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                else:
                    return f"ERROR: {str(e)}", "", 0.0


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint(checkpoint_path: str) -> Dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"processed_ids": [], "results": {}}


def save_checkpoint(checkpoint_path: str, checkpoint_data: Dict):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temp_path = checkpoint_path + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f)
    os.rename(temp_path, checkpoint_path)


# ============================================================================
# Data Loading
# ============================================================================

def load_manipulation_vqa(config: PipelineConfig) -> pd.DataFrame:
    """Load and sample ManipulationVQA-60k from HuggingFace."""
    print(f"\n{'='*60}")
    print("Loading ManipulationVQA-60k from HuggingFace...")
    print(f"{'='*60}")

    api = HfApi()
    files = api.list_repo_files(config.hf_source_repo, repo_type='dataset')
    parquet_files = [f for f in files if f.endswith('.parquet')]

    dfs = []
    for pf in tqdm(parquet_files, desc="Downloading parquet files"):
        local_path = hf_hub_download(repo_id=config.hf_source_repo, filename=pf, repo_type='dataset')
        df = pd.read_parquet(local_path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} rows from ManipulationVQA-60k")

    # Sample if needed
    if len(combined) > config.manipulation_samples:
        print(f"Sampling {config.manipulation_samples} from {len(combined)} rows...")
        combined = combined.sample(n=config.manipulation_samples, random_state=config.seed).reset_index(drop=True)

    combined['_source'] = 'manipulationvqa'
    combined['hint'] = ''  # ManipulationVQA doesn't have hints

    print(f"ManipulationVQA samples: {len(combined)}")
    return combined


def load_agibot_vqa(config: PipelineConfig) -> pd.DataFrame:
    """Load and sample local Agibot VQA data."""
    print(f"\n{'='*60}")
    print("Loading local Agibot VQA data...")
    print(f"{'='*60}")

    json_path = os.path.join(config.local_vqa_path, "vqa_data.json")
    images_path = os.path.join(config.local_vqa_path, "images")

    with open(json_path, 'r') as f:
        data = json.load(f)

    vqa_items = data['vqa_items']
    print(f"Found {len(vqa_items)} VQA items in local dataset")

    # Sample if needed
    if len(vqa_items) > config.agibot_samples:
        print(f"Sampling {config.agibot_samples} from {len(vqa_items)} items...")
        random.seed(config.seed)
        vqa_items = random.sample(vqa_items, config.agibot_samples)

    def process_item(args):
        idx, item = args
        try:
            image_ids = item['question']['image_ids']
            if not image_ids:
                return None

            image_id = image_ids[0]
            # image_id already contains extension (e.g., xxx.jpg)
            image_path = os.path.join(images_path, image_id)

            if not os.path.exists(image_path):
                return None

            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()

            question = item['question']['text']
            choices = []
            correct_idx = None
            for i, choice in enumerate(item['choices']):
                choices.append(choice['text'])
                if choice.get('is_correct', False):
                    correct_idx = i

            if correct_idx is None:
                return None
            correct_answer = chr(ord('A') + correct_idx)

            tag = item.get('metadata', {}).get('tag', 'unknown')
            unique_id = f"agibot_{tag}_{idx}"
            hint = item.get('hint', '')

            return {
                'id': unique_id,
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'image': {'bytes': image_bytes},
                'hint': hint,
                '_source': 'agibot'
            }
        except Exception:
            return None

    print(f"Processing {len(vqa_items)} items with 128 threads...")
    args_list = [(idx, item) for idx, item in enumerate(vqa_items)]

    results = []
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = {executor.submit(process_item, args): args[0] for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Agibot items"):
            result = future.result()
            if result is not None:
                results.append(result)

    df = pd.DataFrame(results)
    print(f"Agibot samples: {len(df)}")
    return df


# ============================================================================
# Reasoning Generation for DataFrames
# ============================================================================

def generate_reasoning_for_df(df: pd.DataFrame, config: PipelineConfig,
                              source_name: str) -> pd.DataFrame:
    """Generate reasoning for a DataFrame."""
    print(f"\n{'='*60}")
    print(f"Generating reasoning for {source_name} ({len(df)} samples)...")
    print(f"{'='*60}")

    client = ReasoningClient(config)

    checkpoint_path = os.path.join(config.checkpoint_dir, f"{source_name}_reasoning.json")
    checkpoint = load_checkpoint(checkpoint_path)
    processed_ids = set(checkpoint["processed_ids"])
    results = checkpoint["results"]

    print(f"Already processed: {len(processed_ids)} samples")

    # Collect items to process
    items_to_process = []
    for idx, row in df.iterrows():
        item_id = row.get('id', str(idx))
        if item_id not in processed_ids:
            items_to_process.append((idx, item_id, row))

    print(f"Remaining to process: {len(items_to_process)} samples")

    if items_to_process:
        correct_count = 0
        total_processed = 0

        def process_item(args):
            idx, item_id, row = args
            image = row['image']
            correct_answer = row.get('correct_answer', '')
            hint = row.get('hint', '')

            reasoning, answer, elapsed = client.generate_reasoning(
                image=image,
                question=row['question'],
                choices=row['choices'],
                correct_answer=correct_answer,
                hint=hint,
            )

            return {
                "id": item_id,
                "idx": idx,
                "reasoning": reasoning,
                "generated_answer": answer,
                "response_time": elapsed,
                "is_correct": answer.upper() == correct_answer.upper() if answer and correct_answer else False,
            }

        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {executor.submit(process_item, args): args[1] for args in items_to_process}

            pbar = tqdm(total=len(futures), desc=f"Reasoning {source_name}")
            for future in as_completed(futures):
                item_id = futures[future]
                try:
                    result = future.result()
                    results[item_id] = result
                    processed_ids.add(item_id)
                    total_processed += 1

                    if result.get("is_correct"):
                        correct_count += 1

                    accuracy = correct_count / total_processed * 100 if total_processed > 0 else 0
                    pbar.set_postfix({"accuracy": f"{accuracy:.1f}%"})
                    pbar.update(1)

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

    # Add reasoning columns to DataFrame
    reasoning_list = []
    generated_answer_list = []

    for idx, row in df.iterrows():
        item_id = row.get('id', str(idx))
        result = results.get(item_id, {})
        reasoning_list.append(result.get('reasoning', ''))
        generated_answer_list.append(result.get('generated_answer', ''))

    df = df.copy()
    df['reasoning'] = reasoning_list
    df['generated_answer'] = generated_answer_list

    # Filter out incorrect answers
    original_len = len(df)
    df['is_correct'] = df.apply(
        lambda row: row['generated_answer'].upper() == row['correct_answer'].upper()
        if row['generated_answer'] and row['correct_answer'] else False,
        axis=1
    )
    df = df[df['is_correct']].drop(columns=['is_correct']).reset_index(drop=True)

    print(f"  Filtered: {original_len} -> {len(df)} (removed {original_len - len(df)} incorrect)")

    return df


# ============================================================================
# Merging and Saving
# ============================================================================

def merge_and_save(manipulation_df: pd.DataFrame, agibot_df: pd.DataFrame,
                   config: PipelineConfig):
    """Merge datasets and save to local path."""
    print(f"\n{'='*60}")
    print("Merging datasets...")
    print(f"{'='*60}")

    # Ensure both DataFrames have the same columns
    required_cols = ['id', 'question', 'choices', 'correct_answer', 'image',
                     'hint', 'reasoning', 'generated_answer']

    for col in required_cols:
        if col not in manipulation_df.columns:
            manipulation_df[col] = ''
        if col not in agibot_df.columns:
            agibot_df[col] = ''

    # Keep only required columns
    manipulation_df = manipulation_df[required_cols].copy()
    agibot_df = agibot_df[required_cols].copy()

    # Merge
    merged_df = pd.concat([manipulation_df, agibot_df], ignore_index=True)
    print(f"Total merged rows: {len(merged_df)}")

    # Shuffle
    print("Shuffling data...")
    merged_df = merged_df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    # Calculate split sizes
    total_samples = len(merged_df)
    test_size = 4000
    val_size = 4000
    train_size = total_samples - val_size - test_size

    print(f"\nSplit sizes:")
    print(f"  Train: {train_size}")
    print(f"  Validation: {val_size}")
    print(f"  Test: {test_size}")

    # Split
    train_df = merged_df.iloc[:train_size].reset_index(drop=True)
    val_df = merged_df.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_df = merged_df.iloc[train_size + val_size:].reset_index(drop=True)

    # Convert to HuggingFace Dataset
    print("\nConverting to HuggingFace Dataset format...")

    features = Features({
        'id': Value('string'),
        'image': HFImage(),
        'question': Value('string'),
        'choices': Sequence(Value('string')),
        'correct_answer': Value('string'),
        'hint': Value('string'),
        'reasoning': Value('string'),
        'generated_answer': Value('string'),
    })

    def convert_to_dataset(df, name):
        print(f"  Converting {name}...")
        ds = Dataset.from_pandas(df, features=features, preserve_index=False)
        print(f"  {name} done: {len(ds)} rows")
        return ds

    with ThreadPoolExecutor(max_workers=3) as executor:
        train_future = executor.submit(convert_to_dataset, train_df, "train")
        val_future = executor.submit(convert_to_dataset, val_df, "validation")
        test_future = executor.submit(convert_to_dataset, test_df, "test")
        train_dataset = train_future.result()
        val_dataset = val_future.result()
        test_dataset = test_future.result()

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset,
    })

    print(f"\nDataset ready:")
    print(dataset_dict)

    # Save locally
    print(f"\nSaving dataset to {config.output_path}...")
    os.makedirs(config.output_path, exist_ok=True)
    # num_proc must not exceed smallest split size
    min_split_size = min(len(train_dataset), len(val_dataset), len(test_dataset))
    num_proc = min(128, min_split_size) if min_split_size > 0 else 1
    dataset_dict.save_to_disk(config.output_path, num_proc=num_proc)
    print(f"Dataset saved!")

    return dataset_dict


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Refactored pipeline: Sample -> Reasoning -> Merge -> Save"
    )

    # Data source arguments
    parser.add_argument("--local-vqa-path", type=str,
                        default="/home/syx/robo2VLM/vqa_output14000new3",
                        help="Path to local Agibot VQA output")
    parser.add_argument("--hf-source-repo", type=str,
                        default="keplerccc/ManipulationVQA-60k",
                        help="HuggingFace source repo for ManipulationVQA")

    # Sampling arguments
    parser.add_argument("--manipulation-samples", type=int, default=30000,
                        help="Number of samples from ManipulationVQA-60k")
    parser.add_argument("--agibot-samples", type=int, default=50000,
                        help="Number of samples from local Agibot VQA")

    # Output arguments
    parser.add_argument("--output", type=str,
                        default="/home/syx/robo2VLM/merged_dataset",
                        help="Output path for merged dataset")

    # Reasoning arguments
    parser.add_argument("--skip-reasoning", action="store_true",
                        help="Skip reasoning generation")
    parser.add_argument("--server-url", type=str,
                        default="http://localhost:30000/v1",
                        help="SGLang server URL for reasoning")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-VL-72B-Instruct",
                        help="Model name")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of concurrent workers for reasoning")

    # Split arguments
    parser.add_argument("--val-size", type=int, default=5000,
                        help="Validation set size")
    parser.add_argument("--test-size", type=int, default=5000,
                        help="Test set size")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/home/syx/robo2VLM/reasoning_checkpoints",
                        help="Directory for reasoning checkpoints")

    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        local_vqa_path=args.local_vqa_path,
        hf_source_repo=args.hf_source_repo,
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        manipulation_samples=args.manipulation_samples,
        agibot_samples=args.agibot_samples,
        val_size=args.val_size,
        test_size=args.test_size,
        server_url=args.server_url,
        model_name=args.model,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print("=" * 60)
    print("Refactored Pipeline: Sample -> Reasoning -> Merge -> Save")
    print("=" * 60)
    print(f"ManipulationVQA source: {config.hf_source_repo}")
    print(f"  Samples: {config.manipulation_samples}")
    print(f"Agibot VQA source: {config.local_vqa_path}")
    print(f"  Samples: {config.agibot_samples}")
    print(f"Output: {config.output_path}")
    print(f"Skip reasoning: {args.skip_reasoning}")
    print(f"Seed: {config.seed}")
    print("=" * 60)

    random.seed(config.seed)

    # Step 1: Load and sample ManipulationVQA
    manipulation_df = load_manipulation_vqa(config)

    # Step 2: Load and sample Agibot VQA
    agibot_df = load_agibot_vqa(config)

    # Step 3: Generate reasoning (if not skipped)
    if not args.skip_reasoning:
        manipulation_df = generate_reasoning_for_df(
            manipulation_df, config, "manipulation"
        )
        agibot_df = generate_reasoning_for_df(
            agibot_df, config, "agibot"
        )
    else:
        print("\nSkipping reasoning generation...")
        manipulation_df['reasoning'] = ''
        manipulation_df['generated_answer'] = ''
        agibot_df['reasoning'] = ''
        agibot_df['generated_answer'] = ''

    # Step 4: Merge and save
    dataset_dict = merge_and_save(manipulation_df, agibot_df, config)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Output saved to: {config.output_path}")
    print(f"\nDataset summary:")
    for split in dataset_dict:
        ds = dataset_dict[split]
        if 'reasoning' in ds.column_names:
            non_empty = sum(1 for r in ds['reasoning'] if r and not r.startswith("ERROR"))
            print(f"  {split}: {len(ds)} samples, {non_empty} with reasoning")
        else:
            print(f"  {split}: {len(ds)} samples")


if __name__ == "__main__":
    main()

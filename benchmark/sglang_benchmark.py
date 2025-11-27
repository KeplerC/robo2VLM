#!/usr/bin/env python3
"""
SGLang OpenAI Server Benchmark for robo2vlm-2 Dataset

Benchmarks VLM models using sglang's OpenAI-compatible server interface.
Supports multiple models, prompt types (zero-shot, chain-of-thought), and
comprehensive result saving.

Usage:
    1. Start sglang server:
       python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct \
           --host 0.0.0.0 --port 30000 --tp 4

    2. Run benchmark:
       python sglang_benchmark.py --server-url http://localhost:30000/v1

    3. With specific model and prompt type:
       python sglang_benchmark.py --model Qwen/Qwen2.5-VL-7B-Instruct --prompt-type cot
"""

import os
import sys
import json
import base64
import argparse
import re
import time
import random
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Server settings
    server_url: str = "http://localhost:30000/v1"
    api_key: str = "EMPTY"

    # Dataset settings - local path or HuggingFace repo
    dataset_name: str = "/home/syx/robo2VLM/merged_dataset"
    split: str = "test"
    max_samples: Optional[int] = None

    # Model settings - these are for metadata only, actual model is on server
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Inference settings
    temperature: float = 0.0
    max_tokens: int = 10240

    # Prompt type: "zero_shot", "cot" (chain-of-thought), "direct"
    prompt_type: str = "zero_shot"

    # Batch and concurrency settings
    batch_size: int = 1  # For API calls, typically 1
    num_workers: int = 8  # Concurrent requests

    # Output settings
    output_dir: str = "results"
    save_responses: bool = True

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


# ============================================================================
# Prompt Templates
# ============================================================================

class PromptTemplates:
    """Prompt templates for different inference strategies."""

    @staticmethod
    def zero_shot(question: str, choices: List[str]) -> str:
        """Zero-shot prompt: direct answer without reasoning."""
        formatted_choices = ""
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            formatted_choices += f" {letter}. {choice}"

        return f"""Answer the following multiple choice question by selecting the letter (A, B, C, D, or E).
ONLY output the correct option letter, i.e., A, B, C, D, E.

Question: {question}{formatted_choices}

Answer:"""

    @staticmethod
    def chain_of_thought(question: str, choices: List[str]) -> str:
        """Chain-of-thought prompt: reasoning before answer."""
        formatted_choices = ""
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            formatted_choices += f" {letter}. {choice}"

        return f"""Answer the following multiple choice question. First, analyze the image carefully and reason through the question step by step. Then provide your final answer.

Question: {question}{formatted_choices}

Let me analyze this step by step:
1. First, I'll examine what's shown in the image...
2. Then, I'll consider each answer option...
3. Based on my analysis...

Reasoning:"""

    @staticmethod
    def direct(question: str, choices: List[str]) -> str:
        """Direct prompt: minimal instructions."""
        formatted_choices = ""
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            formatted_choices += f" {letter}. {choice}"

        return f"""{question}{formatted_choices}

Select the correct answer (A, B, C, D, or E):"""

    @staticmethod
    def cot_extract_answer(response: str) -> str:
        """Extract final answer from chain-of-thought response."""
        # Look for common patterns indicating final answer
        patterns = [
            r"(?:the answer is|my answer is|final answer is|correct answer is)[:\s]*\(?([A-E])\)?",
            r"(?:answer|option)[:\s]*\(?([A-E])\)?[.\s]*$",
            r"\b([A-E])\b[.\s]*$",  # Last standalone letter
            r"\*\*([A-E])\*\*",  # Bold letter
        ]

        response_lower = response.lower()
        for pattern in patterns:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Fallback: find any letter in response
        letters = re.findall(r'\b([A-E])\b', response.upper())
        if letters:
            return letters[-1]  # Return last found letter

        return ""


# ============================================================================
# VLM Client
# ============================================================================

class SGLangVLMClient:
    """Client for interacting with sglang OpenAI-compatible server."""

    def __init__(self, config: BenchmarkConfig):
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

    def _encode_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_image(
        self,
        image: Image.Image,
        prompt: str,
        max_retries: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Analyze an image with the given prompt.

        Returns:
            Tuple of (response_text, response_time)
        """
        max_retries = max_retries or self.config.max_retries
        image_base64 = self._encode_image(image)

        messages = [
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

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                elapsed = time.time() - start_time

                content = response.choices[0].message.content
                if content is None:
                    content = ""

                return content, elapsed

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return f"ERROR: {str(e)}", 0.0


# ============================================================================
# Answer Extraction
# ============================================================================

def extract_letter_answer(response: str, prompt_type: str = "zero_shot") -> str:
    """Extract letter answer from model response."""
    if not response or response.startswith("ERROR:"):
        return ""

    response = response.strip()

    # For chain-of-thought, use specialized extraction
    if prompt_type == "cot":
        return PromptTemplates.cot_extract_answer(response)

    # Clean and normalize response
    response_clean = response.strip().upper()

    # Direct single letter response
    if len(response_clean) == 1 and response_clean in "ABCDE":
        return response_clean

    # Look for patterns
    patterns = [
        r"^([A-E])[\.\s]",  # Starts with letter and dot/space
        r"(?:answer is|answer:|option)\s*\(?([A-E])\)?",
        r"\b([A-E])\b[.\s]*$",  # Letter at end
        r"^\(?([A-E])\)?",  # Letter at start with optional parens
    ]

    for pattern in patterns:
        match = re.search(pattern, response_clean)
        if match:
            return match.group(1)

    # Fallback: any letter found
    letters = re.findall(r'\b([A-E])\b', response_clean)
    if letters:
        return letters[0]

    return ""


# ============================================================================
# Dataset Handling
# ============================================================================

class VQADataset:
    """Dataset wrapper for VQA evaluation."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        print(f"Loading dataset: {config.dataset_name} (split: {config.split})")

        # Check if dataset_name is a local path or HuggingFace repo
        if os.path.exists(config.dataset_name):
            # Load from local disk
            print(f"Loading from local path: {config.dataset_name}")
            dataset_dict = load_from_disk(config.dataset_name)
            self.dataset = dataset_dict[config.split]
        else:
            # Load from HuggingFace
            self.dataset = load_dataset(config.dataset_name, split=config.split)

        # Create indices
        self.indices = list(range(len(self.dataset)))

        # Limit samples if specified
        if config.max_samples is not None and config.max_samples < len(self.indices):
            random.seed(42)
            random.shuffle(self.indices)
            self.indices = self.indices[:config.max_samples]

        print(f"Loaded {len(self.indices)} samples for evaluation")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[self.indices[idx]]
        return {
            "id": item["id"],
            "question": item["question"],
            "choices": item["choices"],
            "correct_answer": item["correct_answer"],
            "image": item["image"],
        }

    def format_prompt(self, item: Dict[str, Any], prompt_type: str) -> str:
        """Format prompt based on prompt type."""
        question = item["question"]
        choices = item["choices"]

        if prompt_type == "zero_shot":
            return PromptTemplates.zero_shot(question, choices)
        elif prompt_type == "cot":
            return PromptTemplates.chain_of_thought(question, choices)
        elif prompt_type == "direct":
            return PromptTemplates.direct(question, choices)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


# ============================================================================
# Benchmark Runner
# ============================================================================

@dataclass
class InferenceResult:
    """Result of a single inference."""
    question_id: str
    question: str
    choices: List[str]
    correct_answer: str
    predicted_response: str
    predicted_letter: str
    is_correct: bool
    response_time: float
    prompt_type: str
    error: Optional[str] = None


class BenchmarkRunner:
    """Runs the benchmark evaluation."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = SGLangVLMClient(config)
        self.dataset = VQADataset(config)
        self.results: List[InferenceResult] = []

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _process_single_item(self, idx: int) -> InferenceResult:
        """Process a single dataset item."""
        item = self.dataset[idx]
        prompt = self.dataset.format_prompt(item, self.config.prompt_type)

        # Get image
        image = item["image"]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(BytesIO(image["bytes"]))
        elif not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image))

        # Run inference
        response, elapsed = self.client.analyze_image(image, prompt)

        # Extract answer
        predicted_letter = extract_letter_answer(response, self.config.prompt_type)
        is_correct = predicted_letter.upper() == item["correct_answer"].upper()

        error = None
        if response.startswith("ERROR:"):
            error = response

        return InferenceResult(
            question_id=item["id"],
            question=item["question"],
            choices=item["choices"],
            correct_answer=item["correct_answer"],
            predicted_response=response,
            predicted_letter=predicted_letter,
            is_correct=is_correct,
            response_time=elapsed,
            prompt_type=self.config.prompt_type,
            error=error,
        )

    def run(self) -> Dict[str, Any]:
        """Run the benchmark."""
        print(f"\nStarting benchmark:")
        print(f"  Model: {self.config.model_name}")
        print(f"  Prompt type: {self.config.prompt_type}")
        print(f"  Samples: {len(self.dataset)}")
        print(f"  Workers: {self.config.num_workers}")
        print()

        start_time = time.time()

        # Process samples with concurrent workers
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {
                executor.submit(self._process_single_item, idx): idx
                for idx in range(len(self.dataset))
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                self.results.append(result)

        total_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics(total_time)

        # Save results
        self._save_results(metrics)

        return metrics

    def _calculate_metrics(self, total_time: float) -> Dict[str, Any]:
        """Calculate benchmark metrics."""
        correct = sum(1 for r in self.results if r.is_correct)
        total = len(self.results)
        errors = sum(1 for r in self.results if r.error is not None)

        accuracy = correct / total * 100 if total > 0 else 0
        avg_response_time = sum(r.response_time for r in self.results) / total if total > 0 else 0

        return {
            "model": self.config.model_name,
            "prompt_type": self.config.prompt_type,
            "dataset": self.config.dataset_name,
            "split": self.config.split,
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "errors": errors,
            "avg_response_time": avg_response_time,
            "total_time": total_time,
            "samples_per_second": total / total_time if total_time > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
        }

    def _save_results(self, metrics: Dict[str, Any]) -> str:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.model_name.replace("/", "_")
        base_filename = f"{model_name}_{self.config.prompt_type}_{timestamp}"

        # Save metrics summary
        metrics_file = os.path.join(self.config.output_dir, f"{base_filename}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save detailed results
        if self.config.save_responses:
            results_file = os.path.join(self.config.output_dir, f"{base_filename}_results.json")
            results_data = {
                "metrics": metrics,
                "results": [asdict(r) for r in self.results],
            }
            with open(results_file, "w") as f:
                json.dump(results_data, f, indent=2)

        print(f"\nResults saved to: {metrics_file}")
        return metrics_file

    def print_summary(self, metrics: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Model:           {metrics['model']}")
        print(f"Prompt Type:     {metrics['prompt_type']}")
        print(f"Dataset:         {metrics['dataset']} ({metrics['split']})")
        print("-" * 60)
        print(f"Total Samples:   {metrics['total_samples']}")
        print(f"Correct:         {metrics['correct']}")
        print(f"Accuracy:        {metrics['accuracy']:.2f}%")
        print(f"Errors:          {metrics['errors']}")
        print("-" * 60)
        print(f"Avg Response:    {metrics['avg_response_time']:.3f}s")
        print(f"Total Time:      {metrics['total_time']:.2f}s")
        print(f"Throughput:      {metrics['samples_per_second']:.2f} samples/s")
        print("=" * 60)


# ============================================================================
# Multi-Model Benchmark
# ============================================================================

class MultiModelBenchmark:
    """Run benchmarks across multiple models and prompt types."""

    DEFAULT_MODELS = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-34b-hf",
    ]

    DEFAULT_PROMPT_TYPES = ["zero_shot", "cot", "direct"]

    def __init__(
        self,
        base_config: BenchmarkConfig,
        models: Optional[List[str]] = None,
        prompt_types: Optional[List[str]] = None,
    ):
        self.base_config = base_config
        self.models = models or self.DEFAULT_MODELS
        self.prompt_types = prompt_types or self.DEFAULT_PROMPT_TYPES
        self.all_results: List[Dict[str, Any]] = []

    def run_all(self) -> List[Dict[str, Any]]:
        """Run benchmark for all model/prompt combinations."""
        print("\n" + "=" * 70)
        print("MULTI-MODEL BENCHMARK")
        print("=" * 70)
        print(f"Models: {self.models}")
        print(f"Prompt types: {self.prompt_types}")
        print(f"Total configurations: {len(self.models) * len(self.prompt_types)}")
        print("=" * 70 + "\n")

        for model in self.models:
            for prompt_type in self.prompt_types:
                print(f"\n{'='*70}")
                print(f"RUNNING: {model} with {prompt_type}")
                print(f"{'='*70}")

                # Create config for this run
                config = BenchmarkConfig(
                    server_url=self.base_config.server_url,
                    api_key=self.base_config.api_key,
                    dataset_name=self.base_config.dataset_name,
                    split=self.base_config.split,
                    max_samples=self.base_config.max_samples,
                    model_name=model,
                    temperature=self.base_config.temperature,
                    max_tokens=self.base_config.max_tokens,
                    prompt_type=prompt_type,
                    num_workers=self.base_config.num_workers,
                    output_dir=self.base_config.output_dir,
                    save_responses=self.base_config.save_responses,
                )

                try:
                    runner = BenchmarkRunner(config)
                    metrics = runner.run()
                    runner.print_summary(metrics)
                    self.all_results.append(metrics)
                except Exception as e:
                    print(f"ERROR running {model} with {prompt_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.all_results.append({
                        "model": model,
                        "prompt_type": prompt_type,
                        "error": str(e),
                        "accuracy": 0,
                    })

        # Save combined results
        self._save_combined_results()
        self._print_comparison_table()

        return self.all_results

    def _save_combined_results(self):
        """Save combined results from all runs."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.base_config.output_dir,
            f"benchmark_comparison_{timestamp}.json"
        )

        with open(filename, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "models": self.models,
                "prompt_types": self.prompt_types,
                "results": self.all_results,
            }, f, indent=2)

        print(f"\nCombined results saved to: {filename}")

    def _print_comparison_table(self):
        """Print comparison table of all results."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)

        # Header
        header = f"{'Model':<40} | {'Prompt':<12} | {'Accuracy':<10} | {'Avg Time':<10}"
        print(header)
        print("-" * 80)

        # Sort by accuracy
        sorted_results = sorted(
            self.all_results,
            key=lambda x: x.get("accuracy", 0),
            reverse=True
        )

        for result in sorted_results:
            model = result.get("model", "N/A")[:38]
            prompt = result.get("prompt_type", "N/A")[:12]
            accuracy = result.get("accuracy", 0)
            avg_time = result.get("avg_response_time", 0)

            if "error" in result:
                print(f"{model:<40} | {prompt:<12} | {'ERROR':<10} | {'N/A':<10}")
            else:
                print(f"{model:<40} | {prompt:<12} | {accuracy:>8.2f}% | {avg_time:>8.3f}s")

        print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SGLang OpenAI Server Benchmark for VQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, single prompt type
  python sglang_benchmark.py --model Qwen/Qwen2.5-VL-7B-Instruct --prompt-type zero_shot

  # Multiple models with all prompt types
  python sglang_benchmark.py --multi-model

  # Custom model list
  python sglang_benchmark.py --models Qwen/Qwen2.5-VL-7B-Instruct llava-hf/llava-v1.6-34b-hf

  # Limit samples for quick testing
  python sglang_benchmark.py --max-samples 100
        """
    )

    # Server settings
    parser.add_argument(
        "--server-url",
        default="http://localhost:30000/v1",
        help="SGLang server URL (default: http://localhost:30000/v1)"
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for server (default: EMPTY)"
    )

    # Model settings
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name for single-model mode"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models for multi-model mode"
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Run benchmark with default model list"
    )

    # Prompt settings
    parser.add_argument(
        "--prompt-type",
        choices=["zero_shot", "cot", "direct"],
        default="zero_shot",
        help="Prompt type (default: zero_shot)"
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=["zero_shot", "cot", "direct"],
        help="List of prompt types to test"
    )
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        help="Test all prompt types"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        default="keplerccc/robo2vlm-2",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: all)"
    )

    # Inference settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of concurrent workers (default: 8)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--no-save-responses",
        action="store_true",
        help="Don't save individual responses"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create base config
    config = BenchmarkConfig(
        server_url=args.server_url,
        api_key=args.api_key,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_type=args.prompt_type,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        save_responses=not args.no_save_responses,
    )

    # Determine run mode
    if args.multi_model or args.models:
        # Multi-model mode
        models = args.models if args.models else MultiModelBenchmark.DEFAULT_MODELS
        prompt_types = args.prompt_types if args.prompt_types else (
            MultiModelBenchmark.DEFAULT_PROMPT_TYPES if args.all_prompts else [args.prompt_type]
        )

        benchmark = MultiModelBenchmark(config, models=models, prompt_types=prompt_types)
        benchmark.run_all()

    elif args.all_prompts or args.prompt_types:
        # Single model, multiple prompt types
        prompt_types = args.prompt_types if args.prompt_types else MultiModelBenchmark.DEFAULT_PROMPT_TYPES

        benchmark = MultiModelBenchmark(config, models=[args.model], prompt_types=prompt_types)
        benchmark.run_all()

    else:
        # Single model, single prompt type
        runner = BenchmarkRunner(config)
        metrics = runner.run()
        runner.print_summary(metrics)


if __name__ == "__main__":
    main()

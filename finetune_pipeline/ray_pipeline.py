#!/usr/bin/env python3
"""
Ray-based Parallel Training and Evaluation Pipeline

Efficiently parallelizes training and evaluation jobs across multiple GPUs
using Ray for resource management and task scheduling.

Usage:
    # Run full pipeline (train then eval)
    python ray_pipeline.py

    # Training only
    python ray_pipeline.py --train-only

    # Evaluation only
    python ray_pipeline.py --eval-only

    # Quick test mode
    python ray_pipeline.py --quick

    # Custom configuration
    python ray_pipeline.py --models "Qwen/Qwen2.5-VL-3B-Instruct:1" --max-samples 10000
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime

import ray


#######################
# CONFIGURATION
#######################

@dataclass
class PipelineConfig:
    """Configuration for the training/evaluation pipeline."""

    # Models: (model_name, num_gpus_required)
    models: List[Tuple[str, int]] = field(default_factory=lambda: [
        ("Qwen/Qwen2.5-VL-3B-Instruct", 1),
        ("Qwen/Qwen2.5-VL-7B-Instruct", 1),
        ("google/gemma-3-4b-it", 1),
        # # ("google/gemma-3-12b-it", 1),
        # ("google/gemma-3-4b-it", 1),
        # ("Qwen/Qwen3-VL-4B-Instruct", 1),
        # ("Qwen/Qwen3-VL-8B-Instruct", 1),
    ])

    # Training config
    checkpoint_intervals: List[int] = field(default_factory=lambda: [2500, 5000, 7500, 10000, 15000, 20000])
    max_samples: int = 20000
    mode: str = "zero_shot"  # "zero_shot" or "cot"
    use_wandb: bool = True

    # Evaluation config
    eval_sample_sizes: List[int] = field(default_factory=lambda: [0, 2500, 5000, 7500, 10000, 15000, 20000])
    datasets: List[str] = field(default_factory=lambda: ["merged_dataset", "ERQA"])
    max_eval_samples: int = 2000

    # Directories
    output_dir: str = "outputs"
    results_dir: str = "results"
    log_dir: str = "logs"

    # Resource config
    num_gpus: int = 8
    conda_env: str = "unsloth_env"


def get_model_slug(model_name: str, mode: str = "zero_shot") -> str:
    """Get model slug for directory names."""
    slug = model_name.replace("/", "_")
    if mode == "cot":
        slug = f"{slug}_cot"
    return slug


def get_dataset_slug(dataset: str) -> str:
    """Get dataset slug for file names."""
    return dataset.replace("/", "_").replace("-", "_")


#######################
# RAY TASKS
#######################

@ray.remote
class GPUWorker:
    """Worker that executes training/eval jobs on assigned GPUs."""

    def __init__(self, gpu_ids: List[int], conda_env: str):
        self.gpu_ids = gpu_ids
        self.cuda_visible_devices = ",".join(map(str, gpu_ids))
        self.conda_env = conda_env

    def run_training(
        self,
        model_name: str,
        max_samples: int,
        checkpoint_intervals: List[int],
        output_dir: str,
        mode: str,
        use_wandb: bool,
        log_dir: str,
    ) -> Dict:
        """Run training for a single model."""
        slug = get_model_slug(model_name, mode)
        log_file = Path(log_dir) / "training" / f"{slug}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        intervals_str = ",".join(map(str, checkpoint_intervals))

        # Build the Python command
        python_cmd = (
            f"python train_single.py "
            f"--model {model_name} "
            f"--max-samples {max_samples} "
            f"--checkpoint-intervals {intervals_str} "
            f"--output-dir {output_dir} "
            f"--mode {mode}"
        )

        if not use_wandb:
            python_cmd += " --no-wandb"

        # Wrap in conda activation
        cmd = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices} {python_cmd}"
        ]

        start_time = time.time()

        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent,
            )

        elapsed = time.time() - start_time

        return {
            "model": model_name,
            "mode": mode,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "log_file": str(log_file),
            "gpu_ids": self.gpu_ids,
        }

    def run_evaluation(
        self,
        model_name: Optional[str],
        checkpoint_path: Optional[str],
        num_samples: int,
        dataset: str,
        output_file: str,
        mode: str,
        max_eval_samples: int,
        log_dir: str,
    ) -> Dict:
        """Run evaluation for a single checkpoint on a single dataset."""
        # Determine log file name
        if checkpoint_path:
            slug = Path(checkpoint_path).parent.parent.name
        else:
            slug = get_model_slug(model_name, mode)

        ds_slug = get_dataset_slug(dataset)
        log_file = Path(log_dir) / "eval" / f"{slug}_{num_samples}samples_{ds_slug}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Build the Python command
        python_cmd = (
            f"python eval_single.py "
            f"--dataset {dataset} "
            f"--output-file {output_file} "
            f"--mode {mode} "
            f"--max-samples {max_eval_samples}"
        )

        if checkpoint_path:
            python_cmd += f" --checkpoint {checkpoint_path}"
        else:
            python_cmd += f" --model {model_name}"

        # Wrap in conda activation
        cmd = [
            "bash", "-c",
            f"source ~/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices} {python_cmd}"
        ]

        start_time = time.time()

        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent,
            )

        elapsed = time.time() - start_time

        # Try to read accuracy from result file
        accuracy = None
        if result.returncode == 0 and Path(output_file).exists():
            try:
                with open(output_file) as f:
                    data = json.load(f)
                    accuracy = data.get("accuracy")
            except Exception:
                pass

        return {
            "model": model_name,
            "checkpoint": checkpoint_path,
            "num_samples": num_samples,
            "dataset": dataset,
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "accuracy": accuracy,
            "elapsed_seconds": elapsed,
            "log_file": str(log_file),
            "output_file": output_file,
            "gpu_ids": self.gpu_ids,
        }


#######################
# PIPELINE ORCHESTRATION
#######################

class Pipeline:
    """Orchestrates training and evaluation jobs using Ray."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.script_dir = Path(__file__).parent

        # Ensure directories exist
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def get_training_jobs(self) -> List[Dict]:
        """Get list of training jobs to run."""
        jobs = []

        for model_name, num_gpus in self.config.models:
            slug = get_model_slug(model_name, self.config.mode)

            # Check if all checkpoints already exist
            all_exist = True
            for interval in self.config.checkpoint_intervals:
                if interval > self.config.max_samples:
                    continue
                checkpoint_path = Path(self.config.output_dir) / slug / f"{interval}samples" / "checkpoint-final"
                if not checkpoint_path.exists():
                    all_exist = False
                    break

            if all_exist:
                self.log(f"Skipping training (all checkpoints exist): {model_name}")
                continue

            jobs.append({
                "model_name": model_name,
                "num_gpus": num_gpus,
            })

        return jobs

    def get_eval_jobs(self) -> List[Dict]:
        """Get list of evaluation jobs to run."""
        jobs = []

        for model_name, num_gpus in self.config.models:
            slug = get_model_slug(model_name, self.config.mode)

            for num_samples in self.config.eval_sample_sizes:
                # For non-zero samples, check if checkpoint exists
                if num_samples > 0:
                    checkpoint_path = Path(self.config.output_dir) / slug / f"{num_samples}samples" / "checkpoint-final"
                    if not checkpoint_path.exists():
                        self.log(f"Skipping eval (no checkpoint): {model_name} {num_samples} samples")
                        continue
                else:
                    checkpoint_path = None

                for dataset in self.config.datasets:
                    ds_slug = get_dataset_slug(dataset)
                    result_file = Path(self.config.results_dir) / f"{slug}_{num_samples}samples_{ds_slug}.json"

                    # Skip if result already exists
                    if result_file.exists():
                        self.log(f"Skipping eval (exists): {model_name} {num_samples}samples on {dataset}")
                        continue

                    jobs.append({
                        "model_name": model_name if num_samples == 0 else None,
                        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                        "num_samples": num_samples,
                        "dataset": dataset,
                        "output_file": str(result_file),
                    })

        return jobs

    def allocate_gpus_for_jobs(self, jobs: List[Dict], key: str = "num_gpus") -> List[Tuple[List[int], Dict]]:
        """
        Allocate GPUs for jobs using bin-packing.
        Returns list of (gpu_ids, job) tuples.
        """
        # Sort jobs by GPU requirement (descending) for better packing
        sorted_jobs = sorted(jobs, key=lambda x: x.get(key, 1), reverse=True)

        # Track GPU availability
        gpu_available = [True] * self.config.num_gpus
        allocations = []
        pending_jobs = list(sorted_jobs)

        while pending_jobs:
            # Try to allocate as many jobs as possible
            allocated_this_round = []

            for job in pending_jobs[:]:
                needed = job.get(key, 1)

                # Find contiguous available GPUs
                gpu_ids = []
                for i in range(self.config.num_gpus):
                    if gpu_available[i]:
                        gpu_ids.append(i)
                        if len(gpu_ids) >= needed:
                            break
                    else:
                        gpu_ids = []

                if len(gpu_ids) >= needed:
                    # Allocate
                    for gid in gpu_ids[:needed]:
                        gpu_available[gid] = False
                    allocations.append((gpu_ids[:needed], job))
                    pending_jobs.remove(job)
                    allocated_this_round.append(job)

            if not allocated_this_round and pending_jobs:
                # No more jobs can be allocated in this round
                # Return current allocations and remaining jobs will be handled in next batch
                break

        return allocations, pending_jobs

    def run_training(self):
        """Run all training jobs in parallel."""
        self.log("=" * 60)
        self.log("TRAINING PHASE")
        self.log("=" * 60)

        jobs = self.get_training_jobs()
        if not jobs:
            self.log("No training jobs to run.")
            return []

        self.log(f"Training jobs to run: {len(jobs)}")
        for job in jobs:
            self.log(f"  - {job['model_name']} ({job['num_gpus']} GPUs)")

        results = []
        pending_jobs = jobs[:]

        while pending_jobs:
            allocations, pending_jobs = self.allocate_gpus_for_jobs(pending_jobs)

            if not allocations:
                self.log("Error: Could not allocate GPUs for remaining jobs")
                break

            # Create workers and submit tasks
            futures = []
            for gpu_ids, job in allocations:
                self.log(f"Starting: {job['model_name']} on GPUs {gpu_ids}")
                worker = GPUWorker.remote(gpu_ids, self.config.conda_env)
                future = worker.run_training.remote(
                    model_name=job["model_name"],
                    max_samples=self.config.max_samples,
                    checkpoint_intervals=self.config.checkpoint_intervals,
                    output_dir=self.config.output_dir,
                    mode=self.config.mode,
                    use_wandb=self.config.use_wandb,
                    log_dir=self.config.log_dir,
                )
                futures.append(future)

            # Wait for this batch to complete
            batch_results = ray.get(futures)

            for result in batch_results:
                status = "SUCCESS" if result["success"] else "FAILED"
                self.log(f"Completed: {result['model']} [{status}] ({result['elapsed_seconds']:.1f}s)")
                results.append(result)

        return results

    def run_evaluation(self):
        """Run all evaluation jobs in parallel."""
        self.log("=" * 60)
        self.log("EVALUATION PHASE")
        self.log("=" * 60)

        jobs = self.get_eval_jobs()
        if not jobs:
            self.log("No evaluation jobs to run.")
            return []

        self.log(f"Evaluation jobs to run: {len(jobs)}")

        results = []

        # Eval jobs all need 1 GPU, so we can run num_gpus jobs in parallel
        batch_size = self.config.num_gpus

        for batch_start in range(0, len(jobs), batch_size):
            batch_jobs = jobs[batch_start:batch_start + batch_size]

            futures = []
            for i, job in enumerate(batch_jobs):
                gpu_id = i % self.config.num_gpus
                model_desc = job["model_name"] or Path(job["checkpoint_path"]).parent.parent.name
                self.log(f"Starting: {model_desc} ({job['num_samples']} samples) on {job['dataset']} [GPU {gpu_id}]")

                worker = GPUWorker.remote([gpu_id], self.config.conda_env)
                future = worker.run_evaluation.remote(
                    model_name=job["model_name"],
                    checkpoint_path=job["checkpoint_path"],
                    num_samples=job["num_samples"],
                    dataset=job["dataset"],
                    output_file=job["output_file"],
                    mode=self.config.mode,
                    max_eval_samples=self.config.max_eval_samples,
                    log_dir=self.config.log_dir,
                )
                futures.append(future)

            # Wait for batch to complete
            batch_results = ray.get(futures)

            for result in batch_results:
                status = "SUCCESS" if result["success"] else "FAILED"
                acc_str = f" ({result['accuracy']*100:.1f}%)" if result['accuracy'] is not None else ""
                model_desc = result["model"] or Path(result["checkpoint"]).parent.parent.name
                self.log(f"Completed: {model_desc} ({result['num_samples']}s) on {result['dataset']} [{status}]{acc_str}")
                results.append(result)

        return results

    def run(self, train_only: bool = False, eval_only: bool = False):
        """Run the full pipeline."""
        self.log("=" * 60)
        self.log("Ray-based Training/Evaluation Pipeline")
        self.log("=" * 60)
        self.log(f"Models: {len(self.config.models)}")
        self.log(f"Mode: {self.config.mode}")
        self.log(f"Max training samples: {self.config.max_samples}")
        self.log(f"Checkpoint intervals: {self.config.checkpoint_intervals}")
        self.log(f"Eval sample sizes: {self.config.eval_sample_sizes}")
        self.log(f"Datasets: {self.config.datasets}")
        self.log(f"GPUs: {self.config.num_gpus}")
        self.log("=" * 60)

        training_results = []
        eval_results = []

        if not eval_only:
            training_results = self.run_training()

        if not train_only:
            eval_results = self.run_evaluation()

        # Summary
        self.log("")
        self.log("=" * 60)
        self.log("PIPELINE COMPLETE")
        self.log("=" * 60)

        if training_results:
            success = sum(1 for r in training_results if r["success"])
            self.log(f"Training: {success}/{len(training_results)} succeeded")

        if eval_results:
            success = sum(1 for r in eval_results if r["success"])
            self.log(f"Evaluation: {success}/{len(eval_results)} succeeded")

            # Print accuracy summary
            self.log("")
            self.log("Accuracy Summary:")
            for result in eval_results:
                if result["success"] and result["accuracy"] is not None:
                    model_desc = result["model"] or Path(result["checkpoint"]).parent.parent.name
                    self.log(f"  {model_desc} ({result['num_samples']}s) on {result['dataset']}: {result['accuracy']*100:.2f}%")

        self.log("")
        self.log(f"Checkpoints: {self.config.output_dir}")
        self.log(f"Results: {self.config.results_dir}")
        self.log(f"Logs: {self.config.log_dir}")

        return training_results, eval_results


#######################
# CLI
#######################

def parse_models(models_str: str) -> List[Tuple[str, int]]:
    """Parse model string like 'Qwen/Qwen2.5-VL-3B-Instruct:1 google/gemma-3-4b-it:1'."""
    models = []
    for item in models_str.split():
        parts = item.rsplit(":", 1)
        model_name = parts[0]
        num_gpus = int(parts[1]) if len(parts) > 1 else 1
        models.append((model_name, num_gpus))
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Ray-based Parallel Training and Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ray_pipeline.py                          # Full pipeline
  python ray_pipeline.py --train-only             # Training only
  python ray_pipeline.py --eval-only              # Evaluation only
  python ray_pipeline.py --quick                  # Quick test (5k samples)
  python ray_pipeline.py --mode zero_shot         # Zero-shot mode
  python ray_pipeline.py --models "Qwen/Qwen2.5-VL-3B-Instruct:1"
        """
    )

    # Mode selection
    parser.add_argument("--train-only", action="store_true",
                        help="Run training only (skip evaluation)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only (skip training)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode (5k samples only)")

    # Configuration
    parser.add_argument("--models", type=str,
                        help="Space-separated list of models (format: 'model:gpus')")
    parser.add_argument("--mode", type=str, choices=["zero_shot", "cot"], default="cot",
                        help="Training/eval mode (default: cot)")
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="Max training samples (default: 20000)")
    parser.add_argument("--intervals", type=str,
                        help="Comma-separated checkpoint intervals")
    parser.add_argument("--eval-sizes", type=str,
                        help="Comma-separated eval sample sizes (0 = base model)")
    parser.add_argument("--datasets", type=str,
                        help="Comma-separated datasets to evaluate on")
    parser.add_argument("--max-eval-samples", type=int, default=500,
                        help="Max samples per evaluation (default: 500)")

    # Resource config
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs available (default: 8)")

    # Other
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Results directory for evaluation outputs")

    args = parser.parse_args()

    # Build config
    config = PipelineConfig()

    if args.models:
        config.models = parse_models(args.models)

    config.mode = args.mode
    config.max_samples = args.max_samples
    config.num_gpus = args.num_gpus
    config.use_wandb = not args.no_wandb
    config.output_dir = args.output_dir
    config.results_dir = args.results_dir
    config.max_eval_samples = args.max_eval_samples

    if args.intervals:
        config.checkpoint_intervals = [int(x) for x in args.intervals.split(",")]

    if args.eval_sizes:
        config.eval_sample_sizes = [int(x) for x in args.eval_sizes.split(",")]

    if args.datasets:
        config.datasets = args.datasets.split(",")

    if args.quick:
        config.max_samples = 5000
        config.checkpoint_intervals = [5000]
        config.eval_sample_sizes = [0, 5000]
        config.max_eval_samples = 100

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    try:
        pipeline = Pipeline(config)
        pipeline.run(
            train_only=args.train_only,
            eval_only=args.eval_only,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Aggregate evaluation results into summary tables.

Usage:
    python aggregate_results.py
    python aggregate_results.py --results-dir results --output summary.csv
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


# Models (short names for display)
MODEL_NAMES = {
    "Qwen_Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
    "Qwen_Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "meta-llama_Llama-3.2-11B-Vision-Instruct": "Llama-3.2-11B",
    "OpenGVLab_InternVL3_5-4B": "InternVL3.5-4B",
    "OpenGVLab_InternVL3_5-8B": "InternVL3.5-8B",
}

# Datasets (short names for display)
DATASET_NAMES = {
    "merged_dataset": "MergedDS",
    "ERQA": "ERQA",
    "CV_Bench": "CV-Bench",
}

# Sample sizes
SAMPLE_SIZES = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]


def load_results(results_dir: str) -> dict:
    """Load all result JSON files."""
    results = defaultdict(lambda: defaultdict(dict))

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Parse filename: MODEL_SAMPLESsamples_DATASET.json
            name = json_file.stem
            parts = name.rsplit("_", 2)

            if len(parts) >= 3:
                # e.g., Qwen_Qwen2.5-VL-3B-Instruct_10000samples_merged_dataset
                dataset = parts[-1]
                samples_str = parts[-2]
                model = "_".join(parts[:-2])

                # Extract sample number
                samples = int(samples_str.replace("samples", ""))

                results[model][samples][dataset] = {
                    "accuracy": data.get("accuracy", 0),
                    "correct": data.get("correct", 0),
                    "total": data.get("total_samples", 0),
                }

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return results


def print_table(results: dict):
    """Print results as a formatted table."""
    datasets = list(DATASET_NAMES.keys())

    # Header
    print("\n" + "=" * 100)
    print("EVALUATION RESULTS")
    print("=" * 100)

    for model_slug, model_results in sorted(results.items()):
        model_name = MODEL_NAMES.get(model_slug, model_slug)
        print(f"\n{model_name}")
        print("-" * 80)

        # Table header
        header = f"{'Samples':>10}"
        for ds in datasets:
            ds_name = DATASET_NAMES.get(ds, ds)
            header += f" | {ds_name:>12}"
        print(header)
        print("-" * 80)

        # Rows
        for samples in SAMPLE_SIZES:
            if samples not in model_results:
                continue

            row = f"{samples:>10}" if samples > 0 else f"{'base':>10}"

            for ds in datasets:
                if ds in model_results[samples]:
                    acc = model_results[samples][ds]["accuracy"] * 100
                    row += f" | {acc:>11.2f}%"
                else:
                    row += f" | {'--':>12}"

            print(row)

    print("\n" + "=" * 100)


def save_csv(results: dict, output_file: str):
    """Save results as CSV."""
    datasets = list(DATASET_NAMES.keys())

    with open(output_file, "w") as f:
        # Header
        header = ["Model", "Samples"] + [DATASET_NAMES.get(ds, ds) for ds in datasets]
        f.write(",".join(header) + "\n")

        # Data
        for model_slug in sorted(results.keys()):
            model_name = MODEL_NAMES.get(model_slug, model_slug)
            model_results = results[model_slug]

            for samples in SAMPLE_SIZES:
                if samples not in model_results:
                    continue

                row = [model_name, str(samples) if samples > 0 else "base"]

                for ds in datasets:
                    if ds in model_results[samples]:
                        acc = model_results[samples][ds]["accuracy"] * 100
                        row.append(f"{acc:.2f}")
                    else:
                        row.append("")

                f.write(",".join(row) + "\n")

    print(f"\nCSV saved to: {output_file}")


def save_json_summary(results: dict, output_file: str):
    """Save results as JSON summary."""
    summary = {}

    for model_slug, model_results in results.items():
        model_name = MODEL_NAMES.get(model_slug, model_slug)
        summary[model_name] = {}

        for samples, ds_results in model_results.items():
            samples_key = str(samples) if samples > 0 else "base"
            summary[model_name][samples_key] = {}

            for ds, metrics in ds_results.items():
                ds_name = DATASET_NAMES.get(ds, ds)
                summary[model_name][samples_key][ds_name] = {
                    "accuracy": round(metrics["accuracy"] * 100, 2),
                    "correct": metrics["correct"],
                    "total": metrics["total"],
                }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"JSON summary saved to: {output_file}")


def print_best_results(results: dict):
    """Print best results per model and dataset."""
    print("\n" + "=" * 60)
    print("BEST RESULTS BY MODEL")
    print("=" * 60)

    for model_slug, model_results in sorted(results.items()):
        model_name = MODEL_NAMES.get(model_slug, model_slug)
        print(f"\n{model_name}:")

        for ds in DATASET_NAMES.keys():
            best_acc = 0
            best_samples = None

            for samples, ds_results in model_results.items():
                if ds in ds_results:
                    acc = ds_results[ds]["accuracy"]
                    if acc > best_acc:
                        best_acc = acc
                        best_samples = samples

            if best_samples is not None:
                ds_name = DATASET_NAMES.get(ds, ds)
                samples_str = str(best_samples) if best_samples > 0 else "base"
                print(f"  {ds_name}: {best_acc*100:.2f}% ({samples_str} samples)")


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, default="summary.csv",
                        help="Output CSV file")
    parser.add_argument("--json-output", type=str, default="summary.json",
                        help="Output JSON file")

    args = parser.parse_args()

    # Load results
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        print(f"Run ./run_training.sh and ./run_eval.sh first.")
        return

    # Print table
    print_table(results)

    # Print best results
    print_best_results(results)

    # Save files
    save_csv(results, args.output)
    save_json_summary(results, args.json_output)


if __name__ == "__main__":
    main()

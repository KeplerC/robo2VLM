#!/usr/bin/env python3
"""
Merge local VQA dataset with HuggingFace ManipulationVQA-60k dataset
and push to keplerccc/robo2vlm-2
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image, load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
LOCAL_VQA_PATH = "/shared/projects/agibot/vqa_output14000new"
LOCAL_JSON_PATH = os.path.join(LOCAL_VQA_PATH, "vqa_data.json")
LOCAL_IMAGES_PATH = os.path.join(LOCAL_VQA_PATH, "images")

# Local save path
LOCAL_DATASET_PATH = "/home/syx/robo2VLM/merged_dataset"

HF_SOURCE_REPO = "keplerccc/ManipulationVQA-60k"
HF_TARGET_REPO = "keplerccc/robo2vlm-2"


def load_hf_parquet_files():
    """Load all parquet files from HuggingFace dataset."""
    print("Loading HuggingFace dataset parquet files...")
    api = HfApi()
    files = api.list_repo_files(HF_SOURCE_REPO, repo_type='dataset')
    parquet_files = [f for f in files if f.endswith('.parquet')]

    dfs = []
    for pf in tqdm(parquet_files, desc="Downloading parquet files"):
        local_path = hf_hub_download(repo_id=HF_SOURCE_REPO, filename=pf, repo_type='dataset')
        df = pd.read_parquet(local_path)
        # Add split info
        if 'train' in pf:
            df['_split'] = 'train'
        elif 'test' in pf:
            df['_split'] = 'test'
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} rows from HuggingFace dataset")
    return combined


def process_single_item(args):
    """Process a single VQA item - optimized for thread pool (I/O bound)."""
    idx, item, images_path = args
    try:
        # Get the image path (use first image_id)
        image_ids = item['question']['image_ids']
        if not image_ids:
            return None

        image_id = image_ids[0]
        image_path = os.path.join(images_path, f"{image_id}.png")

        if not os.path.exists(image_path):
            return None

        # Read image bytes
        with open(image_path, 'rb') as img_file:
            image_bytes = img_file.read()

        # Extract question
        question = item['question']['text']

        # Extract choices and find correct answer
        choices = []
        correct_idx = None
        for i, choice in enumerate(item['choices']):
            choices.append(choice['text'])
            if choice.get('is_correct', False):
                correct_idx = i

        # Convert index to letter (A, B, C, D, E)
        if correct_idx is None:
            return None
        correct_answer = chr(ord('A') + correct_idx)

        # Create unique ID
        tag = item.get('metadata', {}).get('tag', 'unknown')
        unique_id = f"agibot_{tag}_{idx}"

        # Extract hint if available
        hint = item.get('hint', '')

        return {
            'id': unique_id,
            'question': question,
            'choices': choices,
            'correct_answer': correct_answer,
            'image': {'bytes': image_bytes},
            'hint': hint,
            '_split': 'train',
            '_source': 'agibot'
        }
    except Exception as e:
        return None


def convert_local_vqa_to_hf_format():
    """Convert local VQA JSON format to HuggingFace format using ThreadPoolExecutor.

    Using threads instead of Ray because this is I/O-bound (reading images from disk).
    Threads are more efficient for I/O-bound tasks - no serialization overhead.
    """
    print("Loading local VQA data...")
    with open(LOCAL_JSON_PATH, 'r') as f:
        data = json.load(f)

    vqa_items = data['vqa_items']
    print(f"Found {len(vqa_items)} VQA items in local dataset")

    # Prepare arguments for parallel processing
    args_list = [(idx, item, LOCAL_IMAGES_PATH) for idx, item in enumerate(vqa_items)]

    # Use ThreadPoolExecutor for I/O-bound work
    # More workers = more concurrent disk reads (up to I/O bandwidth limit)
    num_workers = 128  # High concurrency for I/O-bound tasks

    print(f"Processing {len(vqa_items)} items with {num_workers} threads...")

    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_item, args): args[0] for args in args_list}

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            if result is not None:
                all_results.append(result)

    print(f"Converted {len(all_results)} items")
    return pd.DataFrame(all_results)


def merge_and_push():
    """Merge both datasets and push to HuggingFace."""

    # Load HuggingFace dataset
    hf_df = load_hf_parquet_files()
    hf_df['_source'] = 'manipulationvqa'
    # Add empty hints column for HuggingFace data (doesn't have hints)
    hf_df['hint'] = ''

    # Convert local dataset
    local_df = convert_local_vqa_to_hf_format()

    # Merge all data together (ignore original train/test split)
    print("Merging datasets...")
    merged_df = pd.concat([hf_df, local_df], ignore_index=True)

    # Drop split and source columns - we'll resplit everything
    merged_df = merged_df.drop(columns=['_split', '_source'])
    print(f"Total merged rows: {len(merged_df)}")

    # Print statistics
    print("\nDataset statistics:")
    print(f"  From ManipulationVQA-60k: {len(hf_df)}")
    print(f"  From Agibot: {len(local_df)}")
    print(f"  Total: {len(merged_df)}")

    # Shuffle all data
    print("\nShuffling all data...")
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train, validation, test (100k train, 5k val, 5k test)
    train_size = 100000
    val_size = 5000
    test_size = 5000
    total_required = train_size + val_size + test_size

    if len(merged_df) < total_required:
        print(f"\nWarning: Only {len(merged_df)} samples available, need {total_required}")
        print("Adjusting splits proportionally...")
        ratio = len(merged_df) / total_required
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = len(merged_df) - train_size - val_size

    train_df = merged_df.iloc[:train_size]
    val_df = merged_df.iloc[train_size:train_size + val_size]
    test_df = merged_df.iloc[train_size + val_size:train_size + val_size + test_size]

    print(f"\nTrain split: {len(train_df)} rows")
    print(f"Validation split: {len(val_df)} rows")
    print(f"Test split: {len(test_df)} rows")

    # Convert to HuggingFace Dataset
    print("\nConverting to HuggingFace Dataset format (parallel)...")

    # Reset index to avoid __index_level_0__ column
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Define features explicitly
    features = Features({
        'id': Value('string'),
        'image': Image(),
        'question': Value('string'),
        'choices': Sequence(Value('string')),
        'correct_answer': Value('string'),
        'hint': Value('string'),
    })

    # Convert train, val, and test in parallel
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
        'test': test_dataset,
        'validation': val_dataset,
        'train': train_dataset
    })

    print(f"\nDataset ready:")
    print(dataset_dict)

    # Save locally first (so we don't lose data if push fails)
    # Use num_proc for parallel saving
    print(f"\nSaving dataset locally to {LOCAL_DATASET_PATH}...")
    os.makedirs(LOCAL_DATASET_PATH, exist_ok=True)
    dataset_dict.save_to_disk(LOCAL_DATASET_PATH, num_proc=128)
    print(f"Dataset saved locally!")

    # Push to HuggingFace
    print(f"\nPushing to {HF_TARGET_REPO}...")
    dataset_dict.push_to_hub(HF_TARGET_REPO, private=False)

    print(f"\nDone! Dataset pushed to https://huggingface.co/datasets/{HF_TARGET_REPO}")


def push_from_local():
    """Push previously saved local dataset to HuggingFace."""
    print(f"Loading dataset from {LOCAL_DATASET_PATH}...")
    dataset_dict = load_from_disk(LOCAL_DATASET_PATH)
    print(dataset_dict)

    print(f"\nPushing to {HF_TARGET_REPO}...")
    dataset_dict.push_to_hub(HF_TARGET_REPO, private=False)
    print(f"\nDone! Dataset pushed to https://huggingface.co/datasets/{HF_TARGET_REPO}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge and push VQA datasets")
    parser.add_argument("--push-only", action="store_true",
                        help="Only push previously saved local dataset (skip merging)")
    args = parser.parse_args()

    if args.push_only:
        push_from_local()
    else:
        merge_and_push()

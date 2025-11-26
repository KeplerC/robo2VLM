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
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image
import ray

# Paths
LOCAL_VQA_PATH = "/shared/projects/agibot/vqa_output14000new"
LOCAL_JSON_PATH = os.path.join(LOCAL_VQA_PATH, "vqa_data.json")
LOCAL_IMAGES_PATH = os.path.join(LOCAL_VQA_PATH, "images")

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


@ray.remote
def process_batch(batch_items, images_path):
    """Process a batch of VQA items using Ray."""
    results = []
    for idx, item in batch_items:
        try:
            # Get the image path (use first image_id)
            image_ids = item['question']['image_ids']
            if not image_ids:
                continue

            image_id = image_ids[0]
            image_path = os.path.join(images_path, f"{image_id}.png")

            if not os.path.exists(image_path):
                continue

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
                continue
            correct_answer = chr(ord('A') + correct_idx)

            # Create unique ID
            tag = item.get('metadata', {}).get('tag', 'unknown')
            unique_id = f"agibot_{tag}_{idx}"

            results.append({
                'id': unique_id,
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'image': {'bytes': image_bytes},
                '_split': 'train',
                '_source': 'agibot'
            })
        except Exception as e:
            continue
    return results


def convert_local_vqa_to_hf_format():
    """Convert local VQA JSON format to HuggingFace format using Ray."""
    print("Loading local VQA data...")
    with open(LOCAL_JSON_PATH, 'r') as f:
        data = json.load(f)

    vqa_items = data['vqa_items']
    print(f"Found {len(vqa_items)} VQA items in local dataset")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=32)

    # Create batches - smaller batches for better parallelism
    batch_size = 1000
    batches = []
    for i in range(0, len(vqa_items), batch_size):
        batch = [(j, vqa_items[j]) for j in range(i, min(i + batch_size, len(vqa_items)))]
        batches.append(batch)

    print(f"Processing {len(batches)} batches with Ray (batch_size={batch_size})...")

    # Submit all batches to Ray
    futures = [process_batch.remote(batch, LOCAL_IMAGES_PATH) for batch in batches]

    # Collect results using ray.wait for better progress tracking
    all_results = []
    remaining = futures
    pbar = tqdm(total=len(futures), desc="Processing batches")

    while remaining:
        done, remaining = ray.wait(remaining, num_returns=min(8, len(remaining)))
        batch_results_list = ray.get(done)
        for batch_results in batch_results_list:
            all_results.extend(batch_results)
        pbar.update(len(done))

    pbar.close()

    print(f"Converted {len(all_results)} items")
    return pd.DataFrame(all_results)


def merge_and_push():
    """Merge both datasets and push to HuggingFace."""

    # Load HuggingFace dataset
    hf_df = load_hf_parquet_files()
    hf_df['_source'] = 'manipulationvqa'

    # Convert local dataset
    local_df = convert_local_vqa_to_hf_format()

    # Merge
    print("Merging datasets...")
    merged_df = pd.concat([hf_df, local_df], ignore_index=True)
    print(f"Total merged rows: {len(merged_df)}")

    # Print statistics
    print("\nDataset statistics:")
    print(f"  From ManipulationVQA-60k: {len(hf_df)}")
    print(f"  From Agibot: {len(local_df)}")
    print(f"  Total: {len(merged_df)}")

    # Split by train/test
    train_df = merged_df[merged_df['_split'] == 'train'].drop(columns=['_split', '_source'])
    test_df = merged_df[merged_df['_split'] == 'test'].drop(columns=['_split', '_source'])

    print(f"\nTrain split: {len(train_df)} rows")
    print(f"Test split: {len(test_df)} rows")

    # Convert to HuggingFace Dataset
    print("\nConverting to HuggingFace Dataset format...")

    # Define features explicitly
    features = Features({
        'id': Value('string'),
        'question': Value('string'),
        'choices': Sequence(Value('string')),
        'correct_answer': Value('string'),
        'image': Image()
    })

    train_dataset = Dataset.from_pandas(train_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    print(f"\nDataset ready:")
    print(dataset_dict)

    # Push to HuggingFace
    print(f"\nPushing to {HF_TARGET_REPO}...")
    dataset_dict.push_to_hub(HF_TARGET_REPO, private=False)

    print(f"\nDone! Dataset pushed to https://huggingface.co/datasets/{HF_TARGET_REPO}")


if __name__ == "__main__":
    merge_and_push()

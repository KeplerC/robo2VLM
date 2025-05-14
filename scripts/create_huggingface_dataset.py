#!/usr/bin/env python3
import os
import json
import argparse
import random
from pathlib import Path
import re
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel, Image
from PIL import Image as PILImage
import numpy as np
import time
import pyarrow as pa
from tqdm import tqdm
import ray
from ray.data import Dataset as RayDataset
import psutil
import glob
import hashlib
import copy
import pandas as pd


def extract_vqa_data_from_episode(episode_path):
    """
    Process a single episode directory to extract VQA data.
    
    Args:
        episode_path: Path to the episode directory
        
    Returns:
        List of processed VQA items from this episode
    """
    episode_vqa_data = []
    episode_dir = os.path.basename(episode_path)
    vqa_data_path = os.path.join(episode_path, "vqa_data.json")
    
    if not os.path.exists(vqa_data_path):
        return episode_vqa_data
        
    try:
        # Load VQA data
        with open(vqa_data_path, 'r') as f:
            episode_data = json.load(f)
            
        # Extract trajectory ID (use episode directory name as a fallback)
        traj_id = episode_dir
        
        # Get trajectory metadata if available
        # metadata_path = os.path.join(episode_path, "metadata.json")
        # if os.path.exists(metadata_path):
        #     with open(metadata_path, 'r') as f:
        #         traj_metadata = json.load(f)
        #         if "episode_id" in traj_metadata:
        #             traj_id = traj_metadata["episode_id"]
        
        # Clean trajectory ID to make it filename-friendly
        traj_id = re.sub(r'[^\w\-_]', '_', traj_id)
        
        # Process VQA items
        vqa_items = episode_data.get('vqa_items', [])
        for idx, item in enumerate(vqa_items):
            # Create unique ID for this VQA item
            unique_id = f"{traj_id}_q{idx+1}"
            
            # Add unique ID and trajectory info to the item
            item['unique_id'] = unique_id
            item['trajectory_id'] = traj_id
            item['episode_dir'] = episode_dir
            item['source_file'] = episode_path
            
            # Add to the collection
            episode_vqa_data.append(item)
    except Exception as e:
        print(f"Error processing VQA data from {vqa_data_path}: {e}")
    
    return episode_vqa_data


def extract_vqa_data_batch(batch):
    """
    Process a batch of episode directories to extract VQA data.
    
    Args:
        batch: Batch of episode paths
        
    Returns:
        Dictionary containing the extracted VQA items
    """
    all_items = []
    for episode_path in batch["episode_path"]:
        items = extract_vqa_data_from_episode(episode_path)
        all_items.extend([{"vqa_item": item} for item in items])
    return {"results": all_items}


def get_episode_directories(base_dir):
    """Get all episode directories from the base directory"""
    episode_paths = []
    for directory in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, directory)
        if os.path.isdir(dir_path):
            episode_paths.append(dir_path)
    return episode_paths


def load_and_resize_image(image_path: str, max_size: int = 1280) -> Optional[PILImage.Image]:
    """
    Load an image from a file path and resize if necessary.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum size for any dimension
        
    Returns:
        PIL Image object or None if loading fails
    """
    if not image_path or not os.path.exists(image_path):
        return None
    
    try:
        img = PILImage.open(image_path).convert("RGB")
        
        # Resize if any dimension is larger than max_size
        if img.width > max_size or img.height > max_size:
            # Calculate new dimensions while preserving aspect ratio
            if img.width > img.height:
                new_width = max_size
                new_height = int(img.height * (max_size / img.width))
            else:
                new_height = max_size
                new_width = int(img.width * (max_size / img.height))
            
            img = img.resize((new_width, new_height), PILImage.LANCZOS)
        
        return img
    except Exception as e:
        print(f"Error loading/resizing image {image_path}: {e}")
        return None


def process_vqa_item(vqa_item):
    """Process a single VQA item, including loading images."""
    # Create a copy to avoid modifying the original
    item = copy.deepcopy(vqa_item)
        
    # Extract the question and options
    question_text = item["question"]["text"]
    choices = item["choices"]
    
    # Find the correct answer
    correct_answer = None
    for idx, choice in enumerate(choices):
        if choice.get("is_correct", False):
            correct_answer = idx
            break
    
    # Process and store image data
    pil_image = None
    encoded_image_data = None
    if "question" in item and "image_ids" in item["question"] and item["question"]["image_ids"] is not None and len(item["question"]["image_ids"]) > 0:
        image_id = item["question"]["image_ids"][0]  # Use the first image
        # Get the parent directory of the episode
        episode_dir = item["source_file"]
        
        # Find the image file
        image_dir = os.path.join(episode_dir, "images")
        image_file = os.path.join(image_dir, f"{image_id}.png")
        
        if os.path.exists(image_file):
            try:
                # Load and resize the image
                pil_image = load_and_resize_image(image_file)
                if pil_image:
                    # Encode the PIL image for Hugging Face datasets
                    # Image is already imported from datasets
                    image_encoder = Image() 
                    encoded_image_data = image_encoder.encode_example(pil_image)
            except Exception as e:
                print(f"Error loading or encoding image {image_file}: {e}")
    
    # Create the processed item
    processed_item = {
        "id": item["unique_id"],
        "question": question_text,
        "choices": [choice["text"] for choice in choices],
        "correct_answer": correct_answer,
        "image": encoded_image_data, # Store the encoded image data
        # Include original data for filtering in train/test split
        "trajectory_id": item["trajectory_id"],
        "source_file": item["source_file"]
    }
    
    return processed_item


def process_vqa_batch(batch):
    """Process a batch of VQA items"""
    results = []
    for item in batch["vqa_item"]:
        results.append(process_vqa_item(item))
    return {"results": results}


def should_include_in_split(item, split_ratios, random_val):
    """Determine if an item should be included in training or test split based on source"""
    # Extract dataset name from source path
    source_file = str(item.get("source_file", "unknown"))
    
    if "droid" in source_file.lower():
        dataset_type = "droid"
    elif "fractal20220817_data" in source_file:
        dataset_type = "fractal"
    elif "stanford_kuka_multimodal_dataset_converted_externally_to_rlds" in source_file:
        dataset_type = "stanford_kuka"
    elif "austin_sirius_dataset_converted_externally_to_rlds" in source_file:
        dataset_type = "austin_sirius"
    elif "berkeley_fanuc_manipulation" in source_file:
        dataset_type = "berkeley_fanuc"
    elif "taco_play" in source_file:
        dataset_type = "taco_play"
    elif "berkeley_autolab_ur5" in source_file:
        dataset_type = "berkeley_autolab"
    else:
        dataset_type = "other"
    
    # Get the threshold for this dataset type
    threshold = split_ratios.get(dataset_type, 0.9)  # Default to 90% train, 10% test
    
    # Return True for train set if random value is less than threshold
    return random_val < threshold


def apply_split_batch(batch, split_ratios):
    """Apply train/test split to a batch of items"""
    results = []
    for item in batch:
        random_val = random.random()
        item = item.copy()  # Create a copy to avoid modifying the original
        item['random_val'] = random_val
        item['split'] = 'train' if should_include_in_split(
            item, split_ratios, random_val) else 'test'
        results.append(item)
    return {"results": results}


def clean_for_hf_batch(batch):
    """Clean batch for HuggingFace format"""
    results = []
    for row in batch:
        results.append({
            "id": row["id"],
            "question": row["question"],
            "choices": row["choices"],
            "correct_answer": row["correct_answer"],
            "image": row["image"]
        })
    return {"results": results}


def main():
    parser = argparse.ArgumentParser(description="Create a HuggingFace dataset from VQA data using Ray Data")
    parser.add_argument("--data-dir", type=str, default="extracted_data",
                        help="Directory containing trajectory subdirectories with VQA data")
    parser.add_argument("--output-dir", type=str, default="hf_dataset",
                        help="Directory to save the HuggingFace dataset")
    parser.add_argument("--train-size", type=int, default=1000000,
                        help="Target number of questions for training set (default: 1000000)")
    parser.add_argument("--test-size", type=int, default=10000,
                        help="Target number of questions for testing set (default: 10000)")
    parser.add_argument("--droid-ratio", type=float, default=0.5,
                        help="Minimum ratio of data to come from the 'droid' dataset (default: 0.5)")
    parser.add_argument("--fractal-ratio", type=float, default=0.25,
                        help="Target ratio of data to come from the 'fractal20220817_data' dataset (default: 0.25)")
    parser.add_argument("--stanford-kuka-ratio", type=float, default=0.02,
                        help="Target ratio of data from 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds' (default: 0.02)")
    parser.add_argument("--austin-sirius-ratio", type=float, default=0.02,
                        help="Target ratio of data from 'austin_sirius_dataset_converted_externally_to_rlds' (default: 0.02)")
    parser.add_argument("--berkeley-fanuc-ratio", type=float, default=0.02,
                        help="Target ratio of data from 'berkeley_fanuc_manipulation' (default: 0.02)")
    parser.add_argument("--taco-play-ratio", type=float, default=0.02,
                        help="Target ratio of data from 'taco_play' (default: 0.02)")
    parser.add_argument("--berkeley-autolab-ratio", type=float, default=0.05,
                        help="Target ratio of data from 'berkeley_autolab_ur5' (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push the dataset to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="keplerccc/ehehee-vqa-100k",
                        help="HuggingFace Hub repository name (username/dataset_name)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes to use for parallel processing")
    parser.add_argument("--ray-address", type=str, default=None,
                        help="Ray cluster address to connect to (default: None, use local Ray)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for processing (default: 128)")
    parser.add_argument("--num-partitions", type=int, default=4096,
                        help="Number of partitions for dataset (default: 4096)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    arrow_dir = os.path.join(args.output_dir, "arrow")
    os.makedirs(arrow_dir, exist_ok=True)
    
    # Track overall time
    start_time = time.time()
    
    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
        print(f"Connected to Ray cluster at {args.ray_address}")
    else:
        num_cpus = args.num_workers if args.num_workers else max(1, psutil.cpu_count())
        ray.init(num_cpus=num_cpus)
        print(f"Ray initialized with {num_cpus} CPUs")
    
    print("Step 1: Collecting episode directories...")
    all_episode_paths = []
    for directory in os.listdir(args.data_dir):
        dir_path = os.path.join(args.data_dir, directory)
        if os.path.isdir(dir_path):
            all_episode_paths.extend(get_episode_directories(dir_path))
    
    print(f"Found {len(all_episode_paths)} episode directories")
    
    # Create Ray Dataset from episode directories
    print("Step 2: Creating Ray Dataset from episodes...")
    episodes_ds = ray.data.from_items([{"episode_path": path} for path in all_episode_paths])
    
    # Repartition to increase parallelism
    episodes_ds = episodes_ds.repartition(args.num_partitions)
    
    # Extract VQA data from each episode using batch processing
    print("Step 3: Extracting VQA data from episodes...")
    def extract_vqa_data_batch_wrapper(batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of episode paths, extract all VQA items, 
        and return a single flat DataFrame containing all items.
        Input batch_df is a DataFrame with an "episode_path" column.
        """
        all_extracted_items = []
        for episode_path in batch_df["episode_path"]:
            # extract_vqa_data_from_episode returns a list of VQA item dictionaries
            items_from_episode = extract_vqa_data_from_episode(episode_path)
            all_extracted_items.extend(items_from_episode)
        
        # Convert the list of all extracted VQA item dictionaries into a DataFrame
        # If all_extracted_items is empty, an empty DataFrame will be returned,
        # which is handled by downstream processes.
        return pd.DataFrame(all_extracted_items)
    
    vqa_items_ds = episodes_ds.map_batches(
        extract_vqa_data_batch_wrapper,
        batch_size=args.batch_size,  # This batch_size applies to the number of episode_paths
        batch_format="pandas",    # Input to wrapper is pandas DataFrame
        num_cpus=0.25
    )
    
    # Get the vqa_items as a flat dataset
    # vqa_items_ds = vqa_items_ds.flat_map(lambda x: x["results"]) # REMOVED - vqa_items_ds is now already flat
    
    # # Count total VQA items
    # total_items = vqa_items_ds.count() # This might be a good place to count if needed
    # print(f"Total VQA items collected: {total_items}")
    
    # if total_items == 0:
    #     print("No VQA items found. Exiting.")
    #     ray.shutdown()
    #     return
    
    # Process VQA items with loading images
    print("Step 4: Processing VQA items (loading images and encoding)...")
    def process_vqa_batch_wrapper(batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of VQA items (DataFrame) and return a DataFrame of processed items.
        Input batch_df has columns corresponding to keys in VQA item dicts 
        (e.g., 'unique_id', 'question', 'choices', 'source_file').
        """
        processed_item_list = []
        # batch_df is a DataFrame where each row represents a VQA item's fields.
        for _, row_series in batch_df.iterrows():
            # Reconstruct the original VQA item dictionary from the row
            raw_vqa_item_dict = row_series.to_dict()
            # process_vqa_item expects the full item dictionary
            processed_item_list.append(process_vqa_item(raw_vqa_item_dict))
        
        # Convert list of processed item dicts to a DataFrame
        expected_columns = ["id", "question", "choices", "correct_answer", "image", "trajectory_id", "source_file"]
        if not processed_item_list:
            return pd.DataFrame(columns=expected_columns)
        return pd.DataFrame(processed_item_list, columns=expected_columns) # Ensure column order/presence
    
    processed_ds = vqa_items_ds.map_batches(
        process_vqa_batch_wrapper,
        batch_size=args.batch_size,
        batch_format="pandas",
        num_cpus=0.25
    )
    
    # Get the processed items as a flat dataset
    # processed_ds = processed_ds.flat_map(lambda x: x["results"])
    
    # Calculate train/test split ratios based on source datasets
    total_target = args.train_size + args.test_size
    train_ratio = args.train_size / total_target if total_target > 0 else 0.8
    
    # Adjust ratios for different dataset sources to match requested distribution
    dataset_split_ratios = {
        "droid": train_ratio,
        "fractal": train_ratio,
        "stanford_kuka": train_ratio,
        "austin_sirius": train_ratio,
        "berkeley_fanuc": train_ratio,
        "taco_play": train_ratio,
        "berkeley_autolab": train_ratio,
        "other": train_ratio
    }
    
    # Add random value and split label
    print("Step 5: Applying train/test split...")
    def apply_split_batch_wrapper(batch: pd.DataFrame, split_ratios: Dict[str, float]) -> pd.DataFrame:
        """Apply train/test split to a batch of items (DataFrame)."""
        # batch is a DataFrame with columns like "id", "question", "image" (encoded), "source_file", etc.
        
        num_rows = len(batch)
        if num_rows == 0:
            # Add expected columns for empty DataFrame to ensure schema consistency
            batch['random_val'] = pd.Series(dtype='float64')
            batch['split'] = pd.Series(dtype='object')
            return batch

        random_vals = [random.random() for _ in range(num_rows)]
        splits = []

        for i in range(num_rows):
            # should_include_in_split expects a dictionary representation of the item
            item_dict_for_split = batch.iloc[i].to_dict()
            splits.append('train' if should_include_in_split(
                item_dict_for_split, split_ratios, random_vals[i]
            ) else 'test')
        
        # Assign new columns to the DataFrame
        # Use .assign to return a new DataFrame if preferred, or modify in place
        batch_copy = batch.copy() # Work on a copy to avoid SettingWithCopyWarning if batch is a slice
        batch_copy['random_val'] = random_vals
        batch_copy['split'] = splits
        return batch_copy
    
    processed_ds = processed_ds.map_batches(
        apply_split_batch_wrapper,
        batch_size=args.batch_size,
        batch_format="pandas",
        num_cpus=0.25,
        fn_kwargs={"split_ratios": dataset_split_ratios} # Pass split_ratios here
    )
    
    # Get the split items as a flat dataset
    # processed_ds = processed_ds.flat_map(lambda x: x["results"]) # REMOVE: processed_ds is still flat
    
    # Split into train and test
    # Each `row` in processed_ds is now a flat dictionary (item)
    # So we access row["split"] directly
    print("Filtering train and test sets...")
    train_ds = processed_ds.filter(lambda row: row["split"] == "train")
    test_ds = processed_ds.filter(lambda row: row["split"] == "test")
    
    # Clean up column and remove the random val
    def clean_for_hf_batch_wrapper(batch: pd.DataFrame) -> pd.DataFrame:
        """Clean a batch of items for HuggingFace format, input is a DataFrame."""
        # Columns in batch: "id", "question", "choices", "correct_answer", "image" (already encoded),
        # "trajectory_id", "source_file", "random_val", "split".
        # Desired columns for HF dataset: "id", "question", "choices", "correct_answer", "image".
        
        # Check if batch is empty
        if batch.empty:
            return pd.DataFrame(columns=["id", "question", "choices", "correct_answer", "image"])
            
        # Select the required columns
        # The image column already contains the encoded image dictionary.
        hf_columns = ["id", "question", "choices", "correct_answer", "image"]
        
        # Ensure all hf_columns exist in batch to avoid KeyError, though they should by this point.
        # For safety, can filter batch.columns for existing ones, but direct selection is fine if schema is stable.
        cleaned_batch = batch[hf_columns].copy() # Use .copy() to ensure it's a new DataFrame
        
        return cleaned_batch
    
    train_ds = train_ds.map_batches(
        clean_for_hf_batch_wrapper,
        batch_size=args.batch_size,
        batch_format="pandas", # Input is pandas DataFrame
        num_cpus=0.25
    )
    
    # Get the clean items as a flat dataset
    # train_ds = train_ds.flat_map(lambda x: x["results"]) # REMOVE: train_ds is already flat
    
    test_ds = test_ds.map_batches(
        clean_for_hf_batch_wrapper,
        batch_size=args.batch_size,
        batch_format="pandas", # Input is pandas DataFrame
        num_cpus=0.25
    )
    
    # Get the clean items as a flat dataset
    # test_ds = test_ds.flat_map(lambda x: x["results"]) # REMOVE: test_ds is already flat
    
    # If we want specific sizes, sample from the datasets
    train_count = train_ds.count()
    test_count = test_ds.count()
    
    print(f"Initial split: {train_count} train items, {test_count} test items")
    
    if train_count > args.train_size:
        train_ds = train_ds.random_sample(args.train_size / train_count)
        print(f"Sampled {args.train_size} items from train set")
    
    if test_count > args.test_size:
        test_ds = test_ds.random_sample(args.test_size / test_count)
        print(f"Sampled {args.test_size} items from test set")
    
    # Write to Arrow format
    print("Step 6: Writing to Arrow format...")
    train_arrow_path = os.path.join(arrow_dir, "train")
    test_arrow_path = os.path.join(arrow_dir, "test")
    
    # Define the PyArrow schema to ensure correct type handling during Parquet writing
    # This schema should match the structure of the DataFrames being written
    # and the features expected by HuggingFace datasets.
    parquet_schema = pa.schema([
        ('id', pa.string()),
        ('question', pa.string()),
        ('choices', pa.list_(pa.string())),
        ('correct_answer', pa.int64()),
        # datasets.Image() encodes to a struct with 'bytes' and 'path'
        ('image', pa.struct([
            ('bytes', pa.binary()),
            ('path', pa.string()) 
            # Mode is not typically part of the minimal struct datasets.Image uses for parquet.
            # If it were, it would be: ('mode', pa.string()) 
        ]))
    ])

    train_ds.write_parquet(train_arrow_path, pyarrow_options={'schema': parquet_schema})
    test_ds.write_parquet(test_arrow_path, pyarrow_options={'schema': parquet_schema})
    
    print(f"Written Arrow files to {arrow_dir}")
    
    # Create HuggingFace dataset from saved Parquet files
    print("Step 7: Creating HuggingFace dataset from saved Parquet files...")
    
    # Define features for the HuggingFace dataset
    # This schema should match the data written to Parquet by Ray
    # The 'image' column contains PIL.Image objects saved by Ray,
    # which datasets.Image() can typically load from Parquet.
    hf_features = Features({
        "id": Value("string"),
        "question": Value("string"),
        "choices": Sequence(Value("string")),
        "correct_answer": Value("int64"),  #  Ensuring this matches integer index
        "image": Image()
    })

    # Load the Parquet files (directories) into HuggingFace Datasets
    train_hf = Dataset.from_parquet(train_arrow_path, features=hf_features)
    test_hf = Dataset.from_parquet(test_arrow_path, features=hf_features)
    
    hf_dataset = DatasetDict({
        "train": train_hf,
        "test": test_hf
    })
    
    # Save the dataset locally
    print("Step 8: Saving HuggingFace dataset to disk...")
    hf_dataset.save_to_disk(args.output_dir)
    print(f"Saved HuggingFace dataset to {args.output_dir}")
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        print(f"Step 9: Pushing dataset to HuggingFace Hub: {args.hub_repo}")
        hf_dataset.push_to_hub(args.hub_repo)
        print(f"Successfully pushed dataset to {args.hub_repo}")
    
    # Shutdown Ray
    ray.shutdown()
    print("Ray has been shut down")
    
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 
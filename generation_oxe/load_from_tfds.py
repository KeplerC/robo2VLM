import tensorflow_datasets as tfds
import os
import subprocess
import tempfile
import argparse
from pathlib import Path
import sys
import re
import json
import ray
import time
import numpy as np
import tensorflow as tf
from trajectory_factory import create_trajectory
from oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from oxe_dataset_configs import OXE_DATASET_CONFIGS
from utils import clean_up_language_instruction
from collections import defaultdict

DEFAULT_SAMPLED_EPISODES = 300
# Hardcoded episode counts for each dataset
OPENX_DATASET_EPISODES = {
    "droid": 70000, 
    "berkeley_autolab_ur5": 600, 
    "fractal20220817_data": 6000, 
    "taco_play": DEFAULT_SAMPLED_EPISODES,
    "viola": DEFAULT_SAMPLED_EPISODES,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": DEFAULT_SAMPLED_EPISODES,
    "austin_buds_dataset_converted_externally_to_rlds": DEFAULT_SAMPLED_EPISODES,
    "austin_sirius_dataset_converted_externally_to_rlds": DEFAULT_SAMPLED_EPISODES,
    "berkeley_mvp_converted_externally_to_rlds": DEFAULT_SAMPLED_EPISODES,
    "asu_table_top_converted_externally_to_rlds": DEFAULT_SAMPLED_EPISODES,
    "nyu_rot_dataset_converted_externally_to_rlds": DEFAULT_SAMPLED_EPISODES,
    "berkeley_fanuc_manipulation": DEFAULT_SAMPLED_EPISODES,
    "nyu_door_opening_surprising_effectiveness": 200,
}

def dataset2path(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}'


def recursively_extract_data(data):
    """
    Recursively extract data from TensorFlow datasets, tensors, or numpy arrays
    into serializable Python native data structures.
    
    Args:
        data: Any data structure (dict, list, tensor, etc.)
        
    Returns:
        Data converted to Python native types (dicts, lists, etc.)
    """
    # Handle dictionaries recursively
    if isinstance(data, dict):
        return {k: recursively_extract_data(v) for k, v in data.items()}
    
    # Handle lists and tuples recursively
    elif isinstance(data, (list, tuple)):
        return [recursively_extract_data(item) for item in data]
    
    # Handle TensorFlow tensors by converting to numpy
    elif isinstance(data, tf.Tensor):
        numpy_data = data.numpy()
        # Handle direct bytes objects
        if isinstance(numpy_data, bytes):
            try:
                return numpy_data.decode('utf-8')
            except:
                return str(numpy_data)
        # Decode byte strings
        if numpy_data.dtype.type is np.bytes_:
            try:
                return numpy_data.decode('utf-8')
            except:
                return str(numpy_data)
        # Convert numpy arrays to lists or scalar values
        if np.isscalar(numpy_data) or numpy_data.size == 1:
            return numpy_data.item() if hasattr(numpy_data, 'item') else numpy_data
        return numpy_data.tolist() if hasattr(numpy_data, 'tolist') else numpy_data
    
    # Handle numpy arrays directly
    elif isinstance(data, np.ndarray):
        # Decode byte strings
        if data.dtype.type is np.bytes_:
            try:
                return data.decode('utf-8')
            except:
                return str(data)
        # Convert numpy arrays to lists or scalar values
        if np.isscalar(data) or data.size == 1:
            return data.item() if hasattr(data, 'item') else data
        return data.tolist() if hasattr(data, 'tolist') else data
    
    # Handle TensorFlow dataset by converting to list and recursively processing
    elif isinstance(data, tf.data.Dataset):
        return [recursively_extract_data(item) for item in data.as_numpy_iterator()]
    
    # Return other types as-is
    return data

@ray.remote(num_cpus = 4, memory = 10 * 1024 * 1024 * 1024, num_gpus = 0.3, scheduling_strategy="SPREAD")
def process_droid_episode(episode_data, i, temp_base_dir, output_base_dir):
    """Processes a DROID dataset episode: downloads data and extracts fields."""
    try:
        # Extract language instruction
        language_instruction = episode_data["language_instruction"]
        dataset_name = "droid"
        
        sanitized_instruction = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in language_instruction).rstrip()
        
        # Create output directory for this episode
        episode_output_dir = os.path.join(output_base_dir, f"droid_{sanitized_instruction}_{i+1}")
        os.makedirs(episode_output_dir, exist_ok=True)
        
        # Create trajectory from episode data using the factory
        trajectory = create_trajectory(
            episode_data=episode_data, 
            output_dir=episode_output_dir,
            download_raw=True,
            temp_base_dir=os.path.join(temp_base_dir, "droid")
        )
        
        if trajectory is None:
            return (i, False, f"episode_{i+1}", f"Failed to create trajectory for {dataset_name}")
        
        # Get episode ID
        episode_id = trajectory.get_episode_id()
        print(f"\nProcessing DROID episode {i+1} ({episode_id})")
        
        # Write metadata file
        metadata = {
            "language_instruction": language_instruction,
            "episode_id": episode_id,
            "dataset": dataset_name,
            "dataset_type": "droid"
        }
        
        # Add any extra metadata that might be available
        if "episode_metadata" in episode_data and isinstance(episode_data["episode_metadata"], dict):
            metadata.update({
                k: v for k, v in episode_data["episode_metadata"].items() 
                if k not in metadata and k != "file_path"
            })
            
        metadata_file = os.path.join(episode_output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Process dataset with parameters
        max_steps = trajectory.get_trajectory_length()

        # Process dataset
        try:
            trajectory.process_dataset(
                max_steps=max_steps,
            )
            
            # Close trajectory to release resources
            trajectory.close()
            
            print(f"Successfully processed DROID episode {i+1} ({episode_id})")
            return (i, True, episode_id, f"Success ({dataset_name})")
            
        except Exception as e:
            import traceback
            print(f"Error processing DROID dataset for episode {i+1} ({episode_id}): {e}")
            traceback.print_exc()
            
            # Close trajectory to release resources
            trajectory.close()
            
            return (i, False, episode_id, f"Processing error ({dataset_name}): {e}")
            
    except Exception as e:
        import traceback
        print(f"Error processing DROID episode {i+1}: {e}")
        traceback.print_exc()
        episode_id = f"episode_{i+1}" if 'episode_id' not in locals() else episode_id
        return (i, False, episode_id, f"Unhandled exception (droid_100): {e}")

@ray.remote(num_cpus = 2, memory=3 * 1024 * 1024 * 1024)
def process_openx_episode(episode_data, i, temp_base_dir, output_base_dir):
    """Processes an Open-X dataset episode: downloads data and extracts fields."""
    # try:
    # Extract language instruction
    language_instruction = episode_data["language_instruction"]
    language_instruction = clean_up_language_instruction(language_instruction)
    dataset_name = episode_data.get("dataset_name", "unknown")
    
    sanitized_instruction = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in language_instruction).rstrip()
    
    # Create output directory for this episode with dataset prefix for clarity
    episode_output_dir = os.path.join(output_base_dir, f"{dataset_name}_{sanitized_instruction}_{i+1}")
    os.makedirs(episode_output_dir, exist_ok=True)
    
    # Create trajectory from episode data using the factory for Open-X dataset
    trajectory = create_trajectory(
        episode_data=episode_data, 
        output_dir=episode_output_dir,
        download_raw=False,
        temp_base_dir=os.path.join(temp_base_dir, dataset_name)  # Use dataset-specific temp dir
    )
    
    if trajectory is None:
        return (i, False, f"episode_{i+1}", f"Failed to create trajectory for Open-X {dataset_name}")
    
    # Get episode ID
    episode_id = trajectory.get_episode_id()
    print(f"\nProcessing Open-X {dataset_name} episode {i+1} ({episode_id})")
    
    
    # Process dataset with parameters - Open-X specific handling
    max_steps = trajectory.get_trajectory_length()
    
    # Process dataset
    try:
        trajectory.process_dataset(
            max_steps=max_steps,
        )
        
        # Close trajectory to release resources
        trajectory.close()
            
        print(f"Successfully processed Open-X {dataset_name} episode {i+1} ({episode_id})")
        return (i, True, episode_id, f"Success ({dataset_name})")
        
    except Exception as e:
        import traceback
        print(f"Error processing Open-X {dataset_name} dataset for episode {i+1} ({episode_id}): {e}")
        traceback.print_exc()
            
        # Close trajectory to release resources
        trajectory.close()
        
        return (i, False, episode_id, f"Processing error ({dataset_name}): {e}")
        
    # except Exception as e:
    #     import traceback
    #     print(f"Error processing Open-X {dataset_name} episode {i+1}: {e}")
    #     traceback.print_exc()
    #     episode_id = f"episode_{i+1}" if 'episode_id' not in locals() else episode_id
    #     return (i, False, episode_id, f"Unhandled exception ({dataset_name}): {e}")

def add_batch_dimension(data):
    """
    Recursively add a batch dimension to all numpy arrays or TensorFlow tensors in a nested structure.
    Converts (shape) -> (1, shape)
    """
    if isinstance(data, dict):
        return {k: add_batch_dimension(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [add_batch_dimension(item) for item in data]
    elif isinstance(data, np.ndarray):
        return np.expand_dims(data, axis=0)
    elif isinstance(data, tf.Tensor):
        return tf.expand_dims(data, axis=0)
    else:
        return data
        
def remove_batch_dimension(data):
    """
    Recursively remove the batch dimension from all numpy arrays or TensorFlow tensors in a nested structure.
    Converts (1, shape) -> (shape)
    """
    if isinstance(data, dict):
        return {k: remove_batch_dimension(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [remove_batch_dimension(item) for item in data]
    elif isinstance(data, np.ndarray) and data.shape and data.shape[0] == 1:
        return np.squeeze(data, axis=0)
    elif isinstance(data, tf.Tensor):
        # Check if tensor has dimensions before accessing shape[0]
        if len(data.shape) > 0 and data.shape[0] == 1:
            return tf.squeeze(data, axis=0)
        return data
    else:
        return data

def default_datasets():
    return ["droid", 
            "berkeley_autolab_ur5", 
            "fractal20220817_data", 
            "taco_play",
            "viola",
            "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
            "austin_buds_dataset_converted_externally_to_rlds",
            "austin_sirius_dataset_converted_externally_to_rlds",
            "berkeley_mvp_converted_externally_to_rlds",
            "asu_table_top_converted_externally_to_rlds",
            "nyu_rot_dataset_converted_externally_to_rlds",
            "berkeley_fanuc_manipulation",
            ]
    
# def default_datasets():
#     return [ "taco_play" ]
def default_local_datasets():
    # return ["droid", "fractal20220817_data", "berkeley_autolab_ur5"]
    return ["droid"]

@ray.remote(memory = 5 * 1024 * 1024 * 1024)
def process_droid_dataset(dataset_name, num_to_process, temp_base_dir, output_base_dir):
    """Loads and processes DROID dataset episodes, returning task IDs for individual processing."""
    results_ids = []
    
    # Set dataset-specific output directory
    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    print(f"Loading {dataset_name} dataset definition...")
    # ds = tfds.load("droid_100", data_dir=".", split="train")
    ds = tfds.load("droid", data_dir="gs://gresearch/robotics", split="train")
    
    # ds = ds.shuffle(buffer_size=100)
    
    for i, episode in enumerate(ds.take(num_to_process)):
        # Extract only necessary data to pass to the remote function
        try:
            # Extract the first step to get language instruction
            first_step = next(episode["steps"].as_numpy_iterator())
            language_instruction = first_step["language_instruction"].decode("utf-8")
            
            language_instruction = clean_up_language_instruction(language_instruction)
            
            # Skip if no language instruction
            if language_instruction == "":
                print(f"Skipping episode {i+1} because language instruction is empty")
                continue
            
            # Create episode data dictionary with serializable content
            episode_data = {
                "episode_metadata": {
                    "file_path": episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
                },
                "language_instruction": language_instruction,
                "dataset_name": dataset_name  # Add dataset name to episode data
            }
            
        except Exception as e:
            print(f"Error extracting data for episode index {i} in dataset {dataset_name}. Skipping. Error: {e}")
            continue  # Skip this episode if metadata/steps are malformed

        results_ids.append(process_droid_episode.remote(
            episode_data, i, temp_base_dir, dataset_output_dir, 
        ))
        print(f"Launched task for episode {i+1} from {dataset_name}...")
    
    return results_ids

@ray.remote(memory = 5 * 1024 * 1024 * 1024)
def process_openx_dataset(dataset_name, num_to_process, temp_base_dir, output_base_dir):
    """Loads and processes OpenX dataset episodes, returning task IDs for individual processing."""
    results_ids = []
    
    # Set dataset-specific output directory
    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    print(f"Loading {dataset_name} dataset definition...")
    if os.path.exists("/data/" + dataset_name):
        # Use local path if available
        ds = tfds.load(dataset_name, data_dir="/data", split="train")
        print(f"Dataset {dataset_name} loaded from local path.")
    else:
        builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset_name))
        ds = builder.as_dataset(split="train")
        print(f"Dataset {dataset_name} loaded.")

    # shuffle 
    # ds = ds.shuffle(buffer_size=100)
    
    # Apply the standardization transform if available for this dataset
    transform_name = dataset_name if dataset_name != "bridge" else "bridge_dataset"
    
    if transform_name in OXE_STANDARDIZATION_TRANSFORMS:
        print(f"Applying standardization transform for {dataset_name}")
        transform_fn = OXE_STANDARDIZATION_TRANSFORMS[transform_name]
    else:
        print(f"No standardization transform found for {dataset_name}, skipping")
        return []
    
    for i, episode in enumerate(ds.take(num_to_process)):
        step_data = []
        # Create episode data dictionary with serializable content
        episode_data = {
            "language_instruction": "",
            "dataset_name": dataset_name  # Add dataset name to episode data
        }
        

        for step in episode["steps"].as_numpy_iterator():

            # Apply transform to get standardized data format
            batched_standardized_data = transform_fn(step)
            batched_standardized_data = recursively_extract_data(batched_standardized_data)
                        
            language_instruction = batched_standardized_data["language_instruction"].decode("utf-8")
            language_instruction = clean_up_language_instruction(language_instruction)
            episode_data["language_instruction"] = language_instruction

            # Convert all data to serializable format using recursively_extract_data
            step_data.append(batched_standardized_data)

        episode_data["steps_data"] = step_data
        print(f"Extracted {len(step_data)} steps for episode {i+1}")

        # Add dataset config if available
        if dataset_name in OXE_DATASET_CONFIGS or f"{dataset_name}_dataset" in OXE_DATASET_CONFIGS:
            config_key = dataset_name if dataset_name in OXE_DATASET_CONFIGS else f"{dataset_name}_dataset"
            episode_data["dataset_config"] = {
                "image_obs_keys": OXE_DATASET_CONFIGS[config_key]["image_obs_keys"],
                "depth_obs_keys": OXE_DATASET_CONFIGS[config_key]["depth_obs_keys"],
                "proprio_encoding": int(OXE_DATASET_CONFIGS[config_key]["proprio_encoding"]),
                "action_encoding": int(OXE_DATASET_CONFIGS[config_key]["action_encoding"])
            }

        results_ids.append(process_openx_episode.remote(
            episode_data, i, temp_base_dir, dataset_output_dir, 
        ))
        print(f"Launched task for episode {i+1} from {dataset_name}...")
    
    return results_ids

def main():
    parser = argparse.ArgumentParser(description="Download and extract robot datasets in parallel using Ray")
    parser.add_argument("--output_dir", type=str, default="./extracted_data",
                       help="Directory to save extracted data")
    parser.add_argument("--datasets", type=str, nargs="+", default = default_datasets(),
                       help="Datasets to process (default: droid_100). Available: droid_100 and Open-X datasets")
    parser.add_argument("--num_episodes", type=int, default=70000,
                       help="Number of episodes to process per dataset")
    parser.add_argument("--ray_address", type=str, default=None,
                        help="Optional Ray cluster address to connect to (e.g., 'auto' or 'ray://<head_node>:10001')")
    args = parser.parse_args()

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    try:
        if args.ray_address:
            print(f"Connecting to Ray cluster at: {args.ray_address}")
            ray.init(address=args.ray_address, ignore_reinit_error=True, dashboard_host="0.0.0.0")
        else:
            print("Initializing Ray locally...")
            ray.init(ignore_reinit_error=True, dashboard_host="0.0.0.0")
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        return 1

    # Create base output directory
    temp_base_dir = "./droid_100_raw"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(temp_base_dir, exist_ok=True)

    # Process each requested dataset in parallel
    dataset_task_ids = []
    
    for dataset_name in args.datasets:
        max_available_episodes = OPENX_DATASET_EPISODES[dataset_name]
        num_to_process = min(args.num_episodes, max_available_episodes)
        print(f"\nLaunching processing for dataset: {dataset_name} ({num_to_process}/{max_available_episodes} episodes)")
        
        if dataset_name == "droid":
            dataset_task_ids.append(process_droid_dataset.remote(
                dataset_name, num_to_process, temp_base_dir, args.output_dir
            ))
        else:
            dataset_task_ids.append(process_openx_dataset.remote(
                dataset_name, num_to_process, temp_base_dir, args.output_dir
            ))
    
    # Collect all episode task IDs from dataset tasks
    print(f"Waiting for dataset loading tasks to complete...")
    results_ids = []
    for dataset_task_id in dataset_task_ids:
        try:
            dataset_results_ids = ray.get(dataset_task_id)
            results_ids.extend(dataset_results_ids)
            print(f"Dataset task completed with {len(dataset_results_ids)} episode tasks.")
        except Exception as e:
            print(f"Error processing dataset: {e}")
    
    if not results_ids:
        print("No tasks were launched. Check dataset availability and episode counts.")
        ray.shutdown()
        return 1

    # Wait for tasks to complete and collect results
    print(f"\nWaiting for {len(results_ids)} episode tasks to complete...")
    results = []
    
    # Process tasks one by one as they complete, instead of waiting for all
    remaining_ids = results_ids.copy()
    while remaining_ids:
        # Wait for at least one task to complete, with a timeout
        done_ids, remaining_ids = ray.wait(remaining_ids, num_returns=1, timeout=30.0)
        
        # Process completed tasks
        for result_id in done_ids:
            try:
                result = ray.get(result_id)
                results.append(result)
                if isinstance(result, tuple) and len(result) == 4:
                    idx, success, ep_id, reason = result
                    if success:
                        print(f"✓ Episode {idx+1} ({ep_id}) completed successfully")
                    else:
                        print(f"✗ Episode {idx+1} ({ep_id}) failed: {reason}")
                else:
                    print(f"Warning: Unexpected result format: {result}")
            except ray.exceptions.RayTaskError as e:
                # Handle task failure without breaking the loop
                print(f"Task error: {e}")
                error_info = str(e)
                # Try to extract the episode index from the error
                idx_match = re.search(r'episode (\d+)', error_info)
                idx = int(idx_match.group(1))-1 if idx_match else -1
                # Extract episode ID if possible
                id_match = re.search(r'\(([^)]+)\)', error_info)
                ep_id = id_match.group(1) if id_match else "unknown"
                results.append((idx, False, ep_id, f"Task error: {e}"))
            except Exception as e:
                print(f"Error processing task result: {e}")
                results.append((-1, False, "unknown", f"Result processing error: {e}"))
        
        # Show progress
        if remaining_ids:
            print(f"{len(results)}/{len(results_ids)} tasks completed, {len(remaining_ids)} remaining...")
            time.sleep(2)  # Brief pause to avoid spamming the console

    print("All tasks processed.")

    # Process results
    successful_episodes = {}
    failed_episodes = {}
    for result in results:
        if isinstance(result, tuple) and len(result) == 4:
             idx, success, ep_id, reason = result
             
             # Extract dataset name from reason message
             dataset_match = re.search(r'\(([^)]+)\)', reason)
             dataset_name = dataset_match.group(1) if dataset_match else "unknown"
             
             # Initialize counter for this dataset if not already present
             if dataset_name not in successful_episodes:
                 successful_episodes[dataset_name] = 0
             if dataset_name not in failed_episodes:
                 failed_episodes[dataset_name] = []
                 
             if success:
                 successful_episodes[dataset_name] += 1
             else:
                 failed_episodes[dataset_name].append({'index': idx + 1, 'id': ep_id, 'reason': reason})
        else:
            # Handle unexpected result format
            print(f"Warning: Received unexpected result format: {result}")

    print("\n--- Processing Summary ---")
    print(f"Total datasets attempted: {len(args.datasets)}")
    print(f"Total episodes attempted: {len(results_ids)}")
    
    total_successful = sum(successful_episodes.values())
    total_failed = sum(len(failures) for failures in failed_episodes.values())
    
    print(f"Total successfully processed: {total_successful}")
    print(f"Total failed: {total_failed}")
    
    print("\n--- Results by Dataset ---")
    for dataset in sorted(set(successful_episodes.keys()).union(failed_episodes.keys())):
        success_count = successful_episodes.get(dataset, 0)
        fail_count = len(failed_episodes.get(dataset, []))
        total_count = success_count + fail_count
        
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            print(f"Dataset: {dataset}")
            print(f"  Success: {success_count}/{total_count} ({success_rate:.1f}%)")
            
            if fail_count > 0:
                print(f"  Failed episodes:")
                for failure in failed_episodes.get(dataset, []):
                    print(f"    - Episode Index: {failure['index']}, ID: {failure['id']}, Reason: {failure['reason']}")
            print("")

    # Shutdown Ray
    print("Shutting down Ray...")
    ray.shutdown()

    print("Script finished.")
    return 0

if __name__ == "__main__":
    return_code = main()
    sys.exit(return_code)
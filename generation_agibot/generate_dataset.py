#!/usr/bin/env python3
"""
Main script for generating VQA dataset from AgiBotWorld.

Uses Ray for parallel processing to speed up generation.

Usage:
    python generate_dataset.py --output-dir ./output --num-episodes 10
    python generate_dataset.py --task-ids 327 352 --num-episodes 5
    python generate_dataset.py --all-tasks --num-episodes 100 --num-workers 8
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("Warning: Ray not installed. Running in sequential mode.")

from trajectory import (
    AgiBotTrajectory,
    load_trajectory,
    get_all_episodes,
    get_all_task_ids
)
from vqa import (
    VQA,
    VQA_GENERATORS,
    generate_all_vqas,
    save_vqa_dataset
)


def sample_episodes(task_ids: List[int],
                   data_root: str,
                   num_episodes: int,
                   seed: int = 42) -> List[Tuple[int, int]]:
    """
    Sample episodes from specified tasks.

    Args:
        task_ids: List of task IDs to sample from
        data_root: Dataset root directory
        num_episodes: Total number of episodes to sample
        seed: Random seed

    Returns:
        List of (task_id, episode_id) tuples
    """
    random.seed(seed)

    all_episodes = []
    for task_id in task_ids:
        try:
            episodes = get_all_episodes(task_id, data_root)
            for ep_id in episodes:
                all_episodes.append((task_id, ep_id))
        except Exception as e:
            print(f"Warning: Could not load episodes for task {task_id}: {e}")

    if len(all_episodes) == 0:
        raise ValueError("No episodes found")

    # Sample
    if num_episodes < len(all_episodes):
        sampled = random.sample(all_episodes, num_episodes)
    else:
        sampled = all_episodes

    return sampled


def generate_vqas_for_episode(task_id: int,
                              episode_id: int,
                              data_root: str,
                              output_dir: str,
                              state_samples: int = 1,
                              segment_samples: int = 5) -> List[Dict]:
    """
    Generate VQAs for a single episode.

    Args:
        task_id: Task identifier
        episode_id: Episode identifier
        data_root: Dataset root directory
        output_dir: Output directory for saving images
        state_samples: Samples for robot state VQAs
        segment_samples: Samples for segment/trajectory VQAs

    Returns:
        List of VQA dictionaries (serialized)
    """
    try:
        trajectory = load_trajectory(task_id, episode_id, data_root)
        vqas = generate_all_vqas(
            trajectory,
            state_samples=state_samples,
            segment_samples=segment_samples
        )
        trajectory.close()

        # Save images to output directory
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for vqa in vqas:
            vqa.save_images(images_dir)

        # Convert to dictionaries for serialization
        return [vqa.to_dict() for vqa in vqas]
    except Exception as e:
        print(f"Error processing episode {task_id}/{episode_id}: {e}")
        return []


# Ray remote function for parallel processing
if HAS_RAY:
    @ray.remote
    def generate_vqas_for_episode_ray(task_id: int,
                                      episode_id: int,
                                      data_root: str,
                                      output_dir: str,
                                      state_samples: int = 1,
                                      segment_samples: int = 5) -> Tuple[int, int, List[Dict]]:
        """
        Ray remote function for generating VQAs.

        Returns:
            Tuple of (task_id, episode_id, list of VQA dicts)
        """
        vqa_dicts = generate_vqas_for_episode(
            task_id, episode_id, data_root, output_dir, state_samples, segment_samples
        )
        return (task_id, episode_id, vqa_dicts)


def generate_vqas_parallel(episodes: List[Tuple[int, int]],
                          data_root: str,
                          output_dir: str,
                          state_samples: int,
                          segment_samples: int,
                          num_workers: int) -> List[Dict]:
    """
    Generate VQAs in parallel using Ray.

    Args:
        episodes: List of (task_id, episode_id) tuples
        data_root: Dataset root directory
        output_dir: Output directory for saving images
        state_samples: Samples for robot state VQAs
        segment_samples: Samples for segment/trajectory VQAs
        num_workers: Number of parallel workers

    Returns:
        List of all VQA dictionaries
    """
    if not HAS_RAY:
        raise RuntimeError("Ray is not installed")

    # Create output directory for images
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_workers, ignore_reinit_error=True)

    # Submit all tasks
    futures = []
    for task_id, episode_id in episodes:
        future = generate_vqas_for_episode_ray.remote(
            task_id, episode_id, data_root, output_dir, state_samples, segment_samples
        )
        futures.append(future)

    # Collect results with progress bar
    all_vqa_dicts = []
    successful = 0

    with tqdm(total=len(futures), desc="Generating VQAs (parallel)") as pbar:
        while futures:
            # Wait for any task to complete
            done, futures = ray.wait(futures, num_returns=1)

            for future in done:
                try:
                    task_id, episode_id, vqa_dicts = ray.get(future)
                    if vqa_dicts:
                        all_vqa_dicts.extend(vqa_dicts)
                        successful += 1
                except Exception as e:
                    print(f"Error getting result: {e}")

                pbar.update(1)

    print(f"Successfully processed {successful}/{len(episodes)} episodes")
    return all_vqa_dicts


def generate_vqas_sequential(episodes: List[Tuple[int, int]],
                            data_root: str,
                            output_dir: str,
                            state_samples: int,
                            segment_samples: int) -> List[Dict]:
    """
    Generate VQAs sequentially (fallback when Ray not available).

    Args:
        episodes: List of (task_id, episode_id) tuples
        data_root: Dataset root directory
        output_dir: Output directory for saving images
        state_samples: Samples for robot state VQAs
        segment_samples: Samples for segment/trajectory VQAs

    Returns:
        List of all VQA dictionaries
    """
    # Create output directory for images
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    all_vqa_dicts = []
    successful = 0

    for task_id, episode_id in tqdm(episodes, desc="Generating VQAs"):
        vqa_dicts = generate_vqas_for_episode(
            task_id, episode_id, data_root, output_dir, state_samples, segment_samples
        )
        if vqa_dicts:
            all_vqa_dicts.extend(vqa_dicts)
            successful += 1

    print(f"Successfully processed {successful}/{len(episodes)} episodes")
    return all_vqa_dicts


def save_vqa_dicts(vqa_dicts: List[Dict],
                   output_dir: str,
                   metadata: Optional[Dict] = None) -> None:
    """
    Save VQA dictionaries to output directory.

    Note: Images are already saved by each worker, this just saves the JSON.

    Args:
        vqa_dicts: List of VQA dictionaries
        output_dir: Output directory
        metadata: Optional metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset = {
        "vqa_items": vqa_dicts,
        "metadata": metadata or {}
    }

    with open(os.path.join(output_dir, "vqa_data.json"), "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(vqa_dicts)} VQAs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VQA dataset from AgiBotWorld (with Ray parallelization)"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/shared/projects/agibot/agibot_alpha_full",
        help="Root directory of AgiBotWorld dataset"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./vqa_output",
        help="Output directory for VQA dataset"
    )

    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific task IDs to process (default: sample from all)"
    )

    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Process all available tasks"
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to process"
    )

    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=3,
        help="Number of VQA samples per type per episode (legacy, use --state-samples and --segment-samples)"
    )

    parser.add_argument(
        "--state-samples",
        type=int,
        default=1,
        help="Samples for robot state VQAs (gripper, arm, success) - be selective"
    )

    parser.add_argument(
        "--segment-samples",
        type=int,
        default=5,
        help="Samples for segment/trajectory VQAs (action, skill, transition, progress) - generate more"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (only used with Ray)"
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (run sequentially)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Get task IDs
    if args.task_ids:
        task_ids = args.task_ids
    elif args.all_tasks:
        task_ids = get_all_task_ids(args.data_root)
    else:
        # Default: sample a few common tasks
        task_ids = [327, 352, 356, 362, 377, 380]

    print(f"Processing {len(task_ids)} tasks: {task_ids[:10]}{'...' if len(task_ids) > 10 else ''}")

    # Sample episodes
    episodes = sample_episodes(
        task_ids,
        args.data_root,
        args.num_episodes,
        args.seed
    )

    print(f"Sampled {len(episodes)} episodes")

    # Generate VQAs
    use_parallel = HAS_RAY and not args.no_parallel

    print(f"Sampling: state_samples={args.state_samples}, segment_samples={args.segment_samples}")

    if use_parallel:
        print(f"Using Ray with {args.num_workers} workers")
        all_vqa_dicts = generate_vqas_parallel(
            episodes,
            args.data_root,
            args.output_dir,
            args.state_samples,
            args.segment_samples,
            args.num_workers
        )
    else:
        if not args.no_parallel and not HAS_RAY:
            print("Ray not available, falling back to sequential processing")
        all_vqa_dicts = generate_vqas_sequential(
            episodes,
            args.data_root,
            args.output_dir,
            args.state_samples,
            args.segment_samples
        )

    # Count by type
    type_counts = {}
    for vqa_dict in all_vqa_dicts:
        tag = vqa_dict.get("metadata", {}).get("tag", "unknown")
        type_counts[tag] = type_counts.get(tag, 0) + 1

    # Build metadata
    metadata = {
        "data_root": args.data_root,
        "task_ids": task_ids,
        "num_episodes": len(episodes),
        "num_workers": args.num_workers if use_parallel else 1,
        "parallel": use_parallel,
        "stats": {
            "total_episodes": len(episodes),
            "total_vqas": len(all_vqa_dicts),
            "vqa_counts": type_counts
        }
    }

    # Save dataset
    save_vqa_dicts(all_vqa_dicts, args.output_dir, metadata)

    # Print summary
    print("\n" + "=" * 50)
    print("VQA Generation Summary")
    print("=" * 50)
    print(f"Processing mode: {'Parallel (Ray)' if use_parallel else 'Sequential'}")
    print(f"Total episodes processed: {len(episodes)}")
    print(f"Total VQAs generated: {len(all_vqa_dicts)}")
    print("\nVQAs by type:")
    for tag, count in sorted(type_counts.items()):
        print(f"  {tag}: {count}")
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

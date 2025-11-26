"""
VQA Generation for AgiBotWorld Dataset

This module contains:
1. VQA base class for representing Visual Question Answering items
2. VQA generation functions with configurable frame sampling (1F/3F/5F)

================================================================================
VQA BASELINE FRAMEWORK
================================================================================

Frame Configurations:
- 1F: Single frame - for instantaneous state queries
- 3F: [start, middle, end] of segment - for action/motion queries
- 5F: Keyframes across trajectory - for task-level queries
- 3F×N: 3 frames per segment - for action counting

Problem Taxonomy:
--------------------------------------------------------------------------------
Problem ID          | Type          | Frames | Scope      | Description
--------------------------------------------------------------------------------
STATE QUERIES (Instantaneous):
gripper_state       | Binary        | 1F     | Instant    | Is gripper open/closed
task_complete       | Binary        | 1F     | Endpoint   | Has task been completed
scene_description   | Choice        | 1F     | Static     | Match environment description

ACTION QUERIES (Segment-level):
current_action      | Choice        | 3F     | Segment    | What action is being performed
next_action         | Choice        | 3F     | Segment    | Predict the next action
current_skill       | Choice        | 3F     | Segment    | Identify skill from taxonomy
active_arm          | Binary        | 3F     | Segment    | Which arm is moving
target_object       | Choice        | 3F     | Segment    | Object being manipulated
is_handover         | Binary        | 3F     | Segment    | Bimanual transfer detection
action_progress     | Choice        | 3F     | Segment    | Estimate % completion

TRAJECTORY QUERIES (Task-level):
goal_config         | Choice        | 5F     | Trajectory | Which frame shows goal state
task_instruction    | Choice        | 5F     | Trajectory | Match task description
action_count        | Choice        | 3F×N   | Trajectory | Count distinct actions

SEGMENTATION QUERIES (Boundary):
transition_frame    | Choice        | 5F+    | Boundary   | Identify transition point
--------------------------------------------------------------------------------

Variation Notes:
- Binary: Yes/No with 4 choices (Yes, No, Cannot determine, Partial)
- Choice: 5 options (1 correct + 4 distractors)
================================================================================
"""

import numpy as np
import random
import cv2
import json
import os
import hashlib
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass

from trajectory import AgiBotTrajectory, ActionSegment
from utils import (
    extract_target_object,
    extract_objects_from_instruction,
    extract_locations_from_instruction,
    detect_active_arm,
    get_skill_description,
    create_image_grid,
    create_temporal_montage,
    generate_distractor_instructions,
    resize_image,
    AGIBOT_SKILLS,
    GRASP_SKILLS,
    RELEASE_SKILLS,
    BIMANUAL_SKILLS
)


def _hash_image(img: np.ndarray) -> Optional[str]:
    """Generate a SHA-256 hash for an image.

    Uses raw pixel data for fast hashing instead of slow PNG encoding.
    """
    if img is None:
        return None
    # Hash raw bytes directly - much faster than PNG encoding
    img_bytes = img.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()


def _sample_segment_frames(trajectory: 'AgiBotTrajectory',
                           segment: 'ActionSegment',
                           num_frames: int = 3) -> Tuple[List[np.ndarray], List[int]]:
    """
    Sample frames from a segment with consistent spacing.

    Args:
        trajectory: AgiBotTrajectory instance
        segment: ActionSegment to sample from
        num_frames: Number of frames to sample (1, 3, or 5)

    Returns:
        Tuple of (frames, frame_indices)

    Frame sampling strategy:
    - 1F: middle frame only
    - 3F: [start, middle, end] - shows action progression
    - 5F: [start, 25%, 50%, 75%, end] - dense sampling
    """
    start = segment.start_frame
    end = segment.end_frame - 1  # end_frame is exclusive
    length = end - start

    if length < num_frames:
        # Segment too short, sample what we can
        indices = list(range(start, end + 1))
        while len(indices) < num_frames:
            indices.append(indices[-1])  # Pad with last frame
    elif num_frames == 1:
        indices = [start + length // 2]  # Middle frame
    elif num_frames == 3:
        indices = [start, start + length // 2, end]  # Start, middle, end
    elif num_frames == 5:
        indices = [
            start,
            start + length // 4,
            start + length // 2,
            start + 3 * length // 4,
            end
        ]
    else:
        # Generic even spacing
        indices = [start + int(i * length / (num_frames - 1)) for i in range(num_frames)]

    frames = []
    valid_indices = []
    for idx in indices:
        frame = trajectory.get_frame(idx)
        if frame is not None:
            frames.append(frame)
            valid_indices.append(idx)

    return frames, valid_indices


def _calculate_grid_layout(num_images: int) -> Tuple[int, int]:
    """
    Calculate optimal grid layout (rows, cols) for a given number of images.

    Prefers layouts closer to square, with slightly more columns than rows.

    Args:
        num_images: Number of images to arrange

    Returns:
        Tuple of (rows, cols)
    """
    import math

    if num_images <= 0:
        return (1, 1)
    elif num_images == 1:
        return (1, 1)
    elif num_images == 2:
        return (1, 2)
    elif num_images == 3:
        return (1, 3)  # 1x3 grid - all in same row
    elif num_images == 4:
        return (2, 2)
    elif num_images == 5:
        return (2, 3)  # 2x3 grid with one black cell
    elif num_images == 6:
        return (2, 3)
    elif num_images <= 8:
        return (2, 4)
    elif num_images <= 9:
        return (3, 3)
    elif num_images <= 12:
        return (3, 4)
    else:
        # For larger numbers, calculate near-square layout
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        return (rows, cols)


def _create_frame_montage(frames: List[np.ndarray],
                          num_frames: int) -> Optional[np.ndarray]:
    """
    Create appropriate montage based on frame count.

    Uses grid layouts (e.g., 2x2, 2x3) instead of single rows for better visualization.

    Args:
        frames: List of frames
        num_frames: Expected number of frames (1, 3, or 5)

    Returns:
        Montage image or single frame
    """
    if not frames:
        return None

    if num_frames == 1:
        return resize_image(frames[0], max_size=320)
    elif num_frames == 3:
        labels = ["Start", "Middle", "End"]
        rows, cols = _calculate_grid_layout(3)  # Returns (2, 2)
        return create_image_grid(frames[:3], rows=rows, cols=cols, labels=labels[:len(frames)])
    elif num_frames == 5:
        labels = ["1", "2", "3", "4", "5"]
        rows, cols = _calculate_grid_layout(5)  # Returns (2, 3)
        return create_image_grid(frames[:5], rows=rows, cols=cols, labels=labels[:len(frames)])
    else:
        # Generic grid using optimal layout
        rows, cols = _calculate_grid_layout(len(frames))
        return create_image_grid(frames, rows=rows, cols=cols)


def _generate_question_id(tag: str,
                          metadata: Dict,
                          task_id: Optional[str] = None,
                          traj_id: Optional[str] = None) -> str:
    """
    Generate informative question ID.

    Format: agibot_{task_id}_{traj_id}_{tag}_{unique_hash}_{num_frames}F

    Args:
        tag: VQA type tag (e.g., "current_action", "gripper_state")
        metadata: VQA metadata containing num_frames and other info
        task_id: Task identifier
        traj_id: Trajectory identifier

    Returns:
        Formatted question ID string
    """
    import time

    task_id = task_id or metadata.get("task_id", "unknown")
    traj_id = traj_id or metadata.get("traj_id", "unknown")
    num_frames = metadata.get("num_frames", 1)

    # Generate unique hash from timestamp + random
    unique_hash = hashlib.md5(f"{time.time()}_{random.random()}".encode()).hexdigest()[:6]

    return f"agibot_{task_id}_{traj_id}_{tag}_{unique_hash}_{num_frames}F"


class VQA:
    """
    Represents a Visual Question Answering item with multiple choices.

    Supports:
    - 4 choices for binary (yes/no) questions
    - 5 choices for multi-choice questions
    - Optional images for question and choices
    - Metadata for tracking question type and context

    Question ID Format:
        agibot_{task_id}_{traj_id}_{tag}_{unique_hash}_{num_frames}F
    """

    def __init__(self,
                 question_text: str,
                 choices: List[str],
                 correct_idx: int,
                 question_images: Optional[List[np.ndarray]] = None,
                 choice_images: Optional[List[np.ndarray]] = None,
                 metadata: Optional[Dict] = None,
                 question_image_ids: Optional[List[str]] = None,
                 choice_image_ids: Optional[List[str]] = None):
        """
        Initialize a VQA instance.

        Args:
            question_text: The question text
            choices: List of choice texts (4 for binary, 5 for multi-choice)
            correct_idx: Index of the correct answer
            question_images: Optional images for the question
            choice_images: Optional images for each choice
            metadata: Additional metadata (should include tag, num_frames, task_id, traj_id)
            question_image_ids: Pre-computed image IDs for question images
            choice_image_ids: Pre-computed image IDs for choice images
        """
        self.question_text = question_text
        self.choices = choices
        self.correct_idx = correct_idx
        self.question_images = question_images if question_images is not None else []
        self.choice_images = choice_images if choice_images is not None else [None] * len(choices)
        self.metadata = metadata or {}

        # Generate question ID
        tag = self.metadata.get("tag", "unknown")
        self.question_id = _generate_question_id(tag, self.metadata)
        self.metadata["question_id"] = self.question_id

        # Generate image IDs
        self.question_image_ids = question_image_ids or [_hash_image(img) for img in self.question_images]
        self.choice_image_ids = choice_image_ids or [
            _hash_image(img) if img is not None else None for img in self.choice_images
        ]

        self._validate()
        self._shuffle_choices()

    def _validate(self):
        """Validate VQA structure."""
        # Check for binary question (yes/no)
        is_binary = False
        if len(self.choices) == 4:
            first_two = [c.strip().lower() for c in self.choices[:2]]
            if set(first_two) == {"yes", "no"}:
                is_binary = True

        if is_binary and len(self.choices) != 4:
            raise ValueError(f"Binary VQA must have 4 choices, got {len(self.choices)}")
        elif not is_binary and len(self.choices) != 5:
            raise ValueError(f"Non-binary VQA must have 5 choices, got {len(self.choices)}")

        if not (0 <= self.correct_idx < len(self.choices)):
            raise ValueError(f"Invalid correct_idx: {self.correct_idx}")

        # Check for NaN in images
        for img in self.question_images:
            if img is not None and np.issubdtype(img.dtype, np.floating):
                if np.isnan(img).any():
                    raise ValueError("Question image contains NaN")

    def _shuffle_choices(self):
        """Shuffle choices while tracking correct index."""
        n = len(self.choices)
        indices = list(range(n))
        random.shuffle(indices)

        # Shuffle everything
        self.choices = [self.choices[i] for i in indices]
        self.choice_images = [self.choice_images[i] for i in indices]
        self.choice_image_ids = [self.choice_image_ids[i] for i in indices]

        # Update correct index
        self.correct_idx = indices.index(self.correct_idx)

        # 20% chance to add "None of the above" for non-binary
        if len(self.choices) == 5 and random.random() < 0.2:
            old_correct_idx = self.correct_idx
            self.choices[old_correct_idx] = "None of the above"
            self.choice_images[old_correct_idx] = None
            self.choice_image_ids[old_correct_idx] = None

    def to_dict(self, image_ext: str = ".jpg") -> Dict:
        """Convert VQA to dictionary.

        Args:
            image_ext: Image file extension (default ".jpg")
        """
        # Convert metadata values to JSON-serializable types
        clean_metadata = {}
        for k, v in self.metadata.items():
            if isinstance(v, (np.bool_, np.integer)):
                clean_metadata[k] = int(v) if isinstance(v, np.integer) else bool(v)
            elif isinstance(v, np.floating):
                clean_metadata[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_metadata[k] = v.tolist()
            else:
                clean_metadata[k] = v

        # Add extension to image IDs for easier loading
        def add_ext(img_id):
            return f"{img_id}{image_ext}" if img_id else None

        return {
            "question_id": self.question_id,
            "question": {
                "text": self.question_text,
                "image_ids": [add_ext(img_id) for img_id in self.question_image_ids]
            },
            "choices": [
                {
                    "text": choice,
                    "image_id": add_ext(img_id),
                    "is_correct": bool(i == self.correct_idx)
                }
                for i, (choice, img_id) in enumerate(zip(self.choices, self.choice_image_ids))
            ],
            "metadata": clean_metadata
        }

    def to_json(self) -> str:
        """Convert VQA to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_images(self, images_dir: str, use_jpeg: bool = True) -> None:
        """Save all images to directory.

        Args:
            images_dir: Directory to save images
            use_jpeg: Use JPEG format (faster) instead of PNG (default True)
        """
        os.makedirs(images_dir, exist_ok=True)

        ext = ".jpg" if use_jpeg else ".png"
        # JPEG quality 95 is visually lossless but much faster than PNG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95] if use_jpeg else []

        for img, img_id in zip(self.question_images, self.question_image_ids):
            if img is not None and img_id is not None:
                cv2.imwrite(os.path.join(images_dir, f"{img_id}{ext}"), img, encode_params)

        for img, img_id in zip(self.choice_images, self.choice_image_ids):
            if img is not None and img_id is not None:
                cv2.imwrite(os.path.join(images_dir, f"{img_id}{ext}"), img, encode_params)

    def __repr__(self):
        return f"VQA(question='{self.question_text[:50]}...', tag={self.metadata.get('tag', 'unknown')})"


# ==================== STATE QUERIES (1F) ====================

def vqa_gripper_state(trajectory: AgiBotTrajectory,
                      step_idx: int,
                      arm: int = 0) -> Optional[VQA]:
    """
    [gripper_state] Is the robot's gripper open/closed?

    Type: Binary | Frames: 1F | Scope: Instant

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query
        arm: Which arm to query (0=left, 1=right)

    Returns:
        VQA instance or None if generation fails
    """
    try:
        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        is_open = trajectory.is_gripper_open(step_idx, arm)
        arm_name = "left" if arm == 0 else "right"

        question_text = f"Is the robot's {arm_name} gripper open?"
        correct_answer = "Yes" if is_open else "No"
        incorrect_answer = "No" if is_open else "Yes"
        distractors = ["Cannot be determined", "Partially open"]

        return VQA(
            question_text=question_text,
            choices=[correct_answer, incorrect_answer] + distractors,
            correct_idx=0,
            question_images=[resize_image(frame, max_size=320)],
            metadata={
                "tag": "gripper_state",
                "num_frames": 1,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "arm": arm_name,
                "is_open": is_open,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_gripper_state: {e}")
        return None


def vqa_task_complete(trajectory: AgiBotTrajectory,
                      step_idx: int) -> Optional[VQA]:
    """
    [task_complete] Has the task been completed successfully?

    Type: Binary | Frames: 1F | Scope: Endpoint

    Note: Only generates for final 20% of trajectory to ensure
    the question is meaningful (asking about completion near the end).

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        traj_length = trajectory.get_trajectory_length()
        # Only ask about completion in final 20% of trajectory
        if step_idx < traj_length * 0.8:
            return None

        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        segments = trajectory.get_action_segments()
        if not segments:
            return None

        last_segment = segments[-1]
        is_in_last_segment = last_segment.contains(step_idx)
        is_success = is_in_last_segment and trajectory.is_task_successful()

        task_name = trajectory.task_name.lower()
        question_text = f"The robot is to {task_name}. Has the robot successfully completed the task?"

        correct_answer = "Yes" if is_success else "No"
        incorrect_answer = "No" if is_success else "Yes"
        distractors = ["Cannot be determined", "Task was not attempted"]

        return VQA(
            question_text=question_text,
            choices=[correct_answer, incorrect_answer] + distractors,
            correct_idx=0,
            question_images=[resize_image(frame, max_size=320)],
            metadata={
                "tag": "task_complete",
                "num_frames": 1,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "is_success": is_success,
                "is_in_last_segment": is_in_last_segment,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_task_complete: {e}")
        return None


# ==================== TRAJECTORY QUERIES (5F) ====================

def vqa_goal_config(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    [goal_config] Which frame shows the goal state?

    Type: Choice | Frames: 5F | Scope: Trajectory

    Shows 5 frames: the last frame (goal) and 4 from other segments.

    Args:
        trajectory: AgiBotTrajectory instance

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segments = trajectory.get_action_segments()
        if not segments:
            return None

        traj_length = trajectory.get_trajectory_length()
        last_step_idx = traj_length - 1

        goal_frame = trajectory.get_frame(last_step_idx)
        if goal_frame is None:
            return None

        # Sample 4 other frames from different segments
        sampled_frames = [goal_frame]
        sampled_indices = [last_step_idx]

        other_frames = []
        for seg in segments[:-1]:
            mid_frame = trajectory.get_frame(seg.get_middle_frame())
            if mid_frame is not None:
                other_frames.append((seg.get_middle_frame(), mid_frame))

        random.shuffle(other_frames)
        for idx, frame in other_frames[:4]:
            sampled_frames.append(frame)
            sampled_indices.append(idx)

        while len(sampled_frames) < 5:
            if len(other_frames) > 0:
                idx, frame = random.choice(other_frames)
                sampled_frames.append(frame.copy())
                sampled_indices.append(idx)
            else:
                rand_idx = random.randint(0, traj_length - 2)
                rand_frame = trajectory.get_frame(rand_idx)
                if rand_frame is not None:
                    sampled_frames.append(rand_frame)
                    sampled_indices.append(rand_idx)

        if len(sampled_frames) != 5:
            return None

        labels = ["A", "B", "C", "D", "E"]
        grid_image = create_image_grid(sampled_frames, rows=2, cols=3, labels=labels)
        if grid_image is None:
            return None

        question_text = f"The robot's task is to {trajectory.task_name.lower()}. " \
                       f"Which configuration shows the goal state that the robot should achieve?"

        choices = [f"Configuration {label}" for label in labels]

        return VQA(
            question_text=question_text,
            choices=choices,
            correct_idx=0,
            question_images=[grid_image],
            metadata={
                "tag": "goal_config",
                "num_frames": 5,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "task_name": trajectory.task_name,
                "sampled_indices": sampled_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_goal_config: {e}")
        return None


# ==================== ACTION QUERIES (3F) ====================

def vqa_current_action(trajectory: AgiBotTrajectory,
                       step_idx: int) -> Optional[VQA]:
    """
    [current_action] What action is the robot currently performing?

    Type: Choice | Frames: 3F | Scope: Segment

    Shows [start, middle, end] frames from the current action segment.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segment = trajectory.get_segment_for_step(step_idx)
        if segment is None:
            return None

        current_skill = segment.skill
        current_action_text = segment.action_text

        # Sample 3 frames: start, middle, end
        frames, frame_indices = _sample_segment_frames(trajectory, segment, num_frames=3)
        if len(frames) < 3:
            return None

        montage = _create_frame_montage(frames, num_frames=3)
        if montage is None:
            return None

        target_object = extract_target_object(current_action_text)
        if not target_object:
            target_object = "object"

        correct_description = get_skill_description(current_skill, target_object)

        all_skills = list(AGIBOT_SKILLS.keys())
        other_skills = [s for s in all_skills if s != current_skill]
        random.shuffle(other_skills)
        incorrect_descriptions = [get_skill_description(s, target_object) for s in other_skills[:4]]

        question_text = f"The robot is tasked to {trajectory.task_name.lower()}. " \
                       f"Based on the sequence of images, which action is the robot currently performing?"

        return VQA(
            question_text=question_text,
            choices=[correct_description] + incorrect_descriptions,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "current_action",
                "num_frames": 3,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "current_skill": current_skill,
                "action_text": current_action_text,
                "target_object": target_object,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_current_action: {e}")
        return None


def vqa_next_action(trajectory: AgiBotTrajectory,
                    step_idx: int) -> Optional[VQA]:
    """
    [next_action] What will be the robot's next action?

    Type: Choice | Frames: 3F | Scope: Segment

    Shows [start, middle, end] of current segment to predict next action.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segments = trajectory.get_action_segments()
        current_segment = trajectory.get_segment_for_step(step_idx)

        if current_segment is None:
            return None

        current_idx = current_segment.segment_id
        if current_idx >= len(segments) - 1:
            return None  # Last segment - no next action

        next_segment = segments[current_idx + 1]

        # Sample 3 frames: start, middle, end
        frames, frame_indices = _sample_segment_frames(trajectory, current_segment, num_frames=3)
        if len(frames) < 3:
            return None

        montage = _create_frame_montage(frames, num_frames=3)
        if montage is None:
            return None

        current_object = extract_target_object(current_segment.action_text) or "object"
        next_object = extract_target_object(next_segment.action_text) or "object"

        current_description = get_skill_description(current_segment.skill, current_object)
        next_description = get_skill_description(next_segment.skill, next_object)

        other_segments = [s for s in segments if s.segment_id != next_segment.segment_id]
        random.shuffle(other_segments)

        incorrect_descriptions = []
        for seg in other_segments[:4]:
            obj = extract_target_object(seg.action_text) or "object"
            desc = get_skill_description(seg.skill, obj)
            if desc != next_description:
                incorrect_descriptions.append(desc)

        generic_actions = [
            "Waiting for next instruction",
            "Moving to initial position",
            "Stopping all motion",
            "Returning to home position"
        ]
        while len(incorrect_descriptions) < 4:
            action = random.choice(generic_actions)
            if action not in incorrect_descriptions:
                incorrect_descriptions.append(action)

        question_text = f"Based on the sequence of images showing the robot {current_description.lower()}, " \
                       f"what will be the robot's NEXT action?"

        return VQA(
            question_text=question_text,
            choices=[next_description] + incorrect_descriptions[:4],
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "next_action",
                "num_frames": 3,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "current_segment": current_segment.segment_id,
                "next_segment": next_segment.segment_id,
                "current_skill": current_segment.skill,
                "next_skill": next_segment.skill,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_next_action: {e}")
        return None


def vqa_task_instruction(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    [task_instruction] Which task description matches the trajectory?

    Type: Choice | Frames: 5F | Scope: Trajectory

    Shows keyframes from segment boundaries.

    Args:
        trajectory: AgiBotTrajectory instance

    Returns:
        VQA instance or None if generation fails
    """
    try:
        correct_instruction = trajectory.task_name
        if not correct_instruction:
            return None

        keyframes = trajectory.get_segment_keyframes()
        if len(keyframes) < 2:
            keyframes = trajectory.get_keyframes(5)
        if not keyframes:
            return None

        frames = [kf[1] for kf in keyframes[:5]]

        # Don't include skill labels - they leak the answer
        montage = create_temporal_montage(frames, labels=None)
        if montage is None:
            rows, cols = _calculate_grid_layout(len(frames))
            montage = create_image_grid(frames, rows=rows, cols=cols)
        if montage is None:
            return None

        incorrect_instructions = generate_distractor_instructions(correct_instruction, 4)

        question_text = "Which task description best matches the robot's actions shown in the images?"

        return VQA(
            question_text=question_text,
            choices=[correct_instruction] + incorrect_instructions,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "task_instruction",
                "num_frames": 5,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "task_name": correct_instruction,
                "num_keyframes": len(keyframes)
            }
        )
    except Exception as e:
        print(f"Error in vqa_task_instruction: {e}")
        return None


def vqa_current_skill(trajectory: AgiBotTrajectory,
                      step_idx: int) -> Optional[VQA]:
    """
    [current_skill] What skill is the robot executing?

    Type: Choice | Frames: 3F | Scope: Segment

    Uses AgiBotWorld's 31 skill taxonomy.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segment = trajectory.get_segment_for_step(step_idx)
        if segment is None:
            return None

        correct_skill = segment.skill

        # Sample 3 frames: start, middle, end
        frames, frame_indices = _sample_segment_frames(trajectory, segment, num_frames=3)
        if len(frames) < 3:
            return None

        montage = _create_frame_montage(frames, num_frames=3)
        if montage is None:
            return None

        all_skills = list(AGIBOT_SKILLS.keys())
        other_skills = [s for s in all_skills if s != correct_skill]
        random.shuffle(other_skills)
        distractor_skills = other_skills[:4]

        question_text = "Based on the sequence of images, what skill is the robot currently executing?"

        return VQA(
            question_text=question_text,
            choices=[correct_skill] + distractor_skills,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "current_skill",
                "num_frames": 3,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "skill": correct_skill,
                "action_text": segment.action_text,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_current_skill: {e}")
        return None


def vqa_active_arm(trajectory: AgiBotTrajectory,
                   step_idx: int) -> Optional[VQA]:
    """
    [active_arm] Which arm is the robot using?

    Type: Binary | Frames: 3F | Scope: Segment

    Shows [start, middle, end] to observe arm motion.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segment = trajectory.get_segment_for_step(step_idx)
        if segment is None:
            return None

        active_arm = detect_active_arm(segment.action_text)
        if active_arm is None:
            return None

        # Sample 3 frames to show arm motion
        frames, frame_indices = _sample_segment_frames(trajectory, segment, num_frames=3)
        if len(frames) < 3:
            return None

        montage = _create_frame_montage(frames, num_frames=3)
        if montage is None:
            return None

        if active_arm == "left":
            question_text = "Is the robot using its LEFT arm for this action?"
        elif active_arm == "right":
            question_text = "Is the robot using its RIGHT arm for this action?"
        else:  # both
            question_text = "Is the robot using BOTH arms for this action?"

        correct_answer = "Yes"
        incorrect_answer = "No"
        distractors = ["Cannot be determined", "Partially using"]

        return VQA(
            question_text=question_text,
            choices=[correct_answer, incorrect_answer] + distractors,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "active_arm",
                "num_frames": 3,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "active_arm": active_arm,
                "action_text": segment.action_text,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_active_arm: {e}")
        return None


def vqa_scene_description(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    [scene_description] What environment is the robot in?

    Type: Choice | Frames: 1F | Scope: Static

    Uses first frame and init_scene_text.

    Args:
        trajectory: AgiBotTrajectory instance

    Returns:
        VQA instance or None if generation fails
    """
    try:
        init_scene = trajectory.init_scene_text
        if not init_scene or len(init_scene) < 20:
            return None

        frame = trajectory.get_frame(0)
        if frame is None:
            return None

        distractor_scenes = [
            "The robot is in a kitchen preparing food on a counter.",
            "The robot is in a warehouse sorting packages on shelves.",
            "The robot is in an office organizing documents on a desk.",
            "The robot is in a laboratory handling scientific equipment."
        ]

        distractors = [d for d in distractor_scenes if d.lower() not in init_scene.lower()][:4]

        question_text = "Which description best matches the robot's environment shown in the image?"

        return VQA(
            question_text=question_text,
            choices=[init_scene] + distractors,
            correct_idx=0,
            question_images=[resize_image(frame, max_size=320)],
            metadata={
                "tag": "scene_description",
                "num_frames": 1,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "init_scene_text": init_scene
            }
        )
    except Exception as e:
        print(f"Error in vqa_scene_description: {e}")
        return None


def vqa_target_object(trajectory: AgiBotTrajectory,
                      step_idx: int) -> Optional[VQA]:
    """
    [target_object] What object is the robot interacting with?

    Type: Choice | Frames: 3F | Scope: Segment

    Shows [start, middle, end] to observe object interaction.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segment = trajectory.get_segment_for_step(step_idx)
        if segment is None:
            return None

        target_object = extract_target_object(segment.action_text)
        if not target_object:
            return None

        # Sample 3 frames to show object interaction
        frames, frame_indices = _sample_segment_frames(trajectory, segment, num_frames=3)
        if len(frames) < 3:
            return None

        montage = _create_frame_montage(frames, num_frames=3)
        if montage is None:
            return None

        common_objects = [
            "cup", "box", "bottle", "plate", "book", "pen", "toy",
            "bag", "cloth", "tool", "container", "package", "bowl",
            "spoon", "fork", "tray", "basket", "cart"
        ]

        distractors = [obj for obj in common_objects if obj.lower() not in target_object.lower()]
        random.shuffle(distractors)
        distractor_objects = distractors[:4]

        question_text = "What object is the robot interacting with in this action?"

        return VQA(
            question_text=question_text,
            choices=[target_object] + distractor_objects,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "target_object",
                "num_frames": 3,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "target_object": target_object,
                "action_text": segment.action_text,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_target_object: {e}")
        return None


def vqa_action_count(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    [action_count] How many distinct actions in the task?

    Type: Choice | Frames: 1F×N | Scope: Trajectory

    Shows 1 representative frame per action segment, shuffled randomly.
    Model must identify action boundaries from visual differences.

    Args:
        trajectory: AgiBotTrajectory instance

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segments = trajectory.get_action_segments()
        num_segments = len(segments)
        if num_segments < 2 or num_segments > 8:  # Limit to reasonable range
            return None

        # Sample 1 frame per segment (middle frame) - requires model to identify actions
        all_frames = []
        all_indices = []

        for segment in segments:
            mid_idx = segment.get_middle_frame()
            frame = trajectory.get_frame(mid_idx)
            if frame is None:
                continue
            all_frames.append(frame)
            all_indices.append(mid_idx)

        if len(all_frames) < num_segments:
            return None

        # Shuffle frames so temporal order doesn't reveal structure
        combined = list(zip(all_frames, all_indices))
        random.shuffle(combined)
        all_frames, all_indices = zip(*combined)
        all_frames = list(all_frames)
        all_indices = list(all_indices)

        # Use simple numeric labels (no action numbering)
        labels = [str(i + 1) for i in range(len(all_frames))]

        # Calculate grid layout
        rows, cols = _calculate_grid_layout(len(all_frames))
        montage = create_image_grid(all_frames, rows=rows, cols=cols, labels=labels)
        if montage is None:
            return None

        correct_answer = str(num_segments)

        distractors = []
        for delta in [-2, -1, 1, 2]:
            count = num_segments + delta
            if count > 0:
                distractors.append(str(count))

        while len(distractors) < 4:
            rand_count = random.randint(1, 15)
            if str(rand_count) != correct_answer and str(rand_count) not in distractors:
                distractors.append(str(rand_count))

        distractors = distractors[:4]

        question_text = f"These images show different moments from a robot performing: {trajectory.task_name}. " \
                       f"How many distinct actions can you identify?"

        return VQA(
            question_text=question_text,
            choices=[correct_answer] + distractors,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "action_count",
                "num_frames": num_segments,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "num_actions": num_segments,
                "task_name": trajectory.task_name,
                "frame_indices": all_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_action_count: {e}")
        return None


def vqa_is_handover(trajectory: AgiBotTrajectory,
                    step_idx: int) -> Optional[VQA]:
    """
    [is_handover] Is the robot performing a handover?

    Type: Binary | Frames: 3F | Scope: Segment

    Shows [start, middle, end] to observe bimanual transfer.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segment = trajectory.get_segment_for_step(step_idx)
        if segment is None:
            return None

        is_handover = segment.skill == "HandOver"

        # Sample 3 frames to show handover motion
        frames, frame_indices = _sample_segment_frames(trajectory, segment, num_frames=3)
        if len(frames) < 3:
            return None

        montage = _create_frame_montage(frames, num_frames=3)
        if montage is None:
            return None

        question_text = "Is the robot performing a handover (transferring an object between its arms)?"

        correct_answer = "Yes" if is_handover else "No"
        incorrect_answer = "No" if is_handover else "Yes"
        distractors = ["Cannot be determined", "Partially transferring"]

        return VQA(
            question_text=question_text,
            choices=[correct_answer, incorrect_answer] + distractors,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "is_handover",
                "num_frames": 3,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "is_handover": is_handover,
                "skill": segment.skill,
                "action_text": segment.action_text,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_is_handover: {e}")
        return None


# ==================== SEGMENTATION VQA GENERATORS ====================

def vqa_transition_frame(trajectory: AgiBotTrajectory, step_idx: int) -> Optional[VQA]:
    """
    Generate VQA asking which frame best represents the transition between actions.

    Shows a dense sequence of 8 frames (at 15-frame intervals) around a segment
    boundary, asking the user to identify the transition frame.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Current step index (used to find nearby transitions)

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segments = trajectory.get_action_segments()
        if len(segments) < 2:
            return None

        # Find the closest transition point to step_idx
        transitions = []
        for i in range(len(segments) - 1):
            transition_frame = segments[i].end_frame
            transitions.append({
                'frame': transition_frame,
                'from_skill': segments[i].skill,
                'to_skill': segments[i + 1].skill,
                'from_action': segments[i].action_text,
                'to_action': segments[i + 1].action_text
            })

        if not transitions:
            return None

        # Find closest transition to step_idx
        closest_transition = min(transitions, key=lambda t: abs(t['frame'] - step_idx))
        transition_frame = closest_transition['frame']

        # Sample 8 frames at 15-frame intervals: 4 before and 4 after the transition
        spacing = 15
        num_frames = 8
        half = num_frames // 2
        traj_length = trajectory.get_trajectory_length()

        frame_indices = []
        for i in range(-half, half):
            frame_idx = transition_frame + i * spacing
            frame_idx = max(0, min(frame_idx, traj_length - 1))
            frame_indices.append(frame_idx)

        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        frame_indices = unique_indices

        if len(frame_indices) < 4:
            return None

        # Load frames
        frames = []
        for idx in frame_indices:
            frame = trajectory.get_frame(idx)
            if frame is None:
                return None
            frames.append(frame)

        # Grid layout: 2 rows x 4 cols for 8 frames
        num_frames = len(frames)
        rows, cols = 2, 4

        # Pad frames if needed
        while len(frames) < rows * cols:
            frames.append(frames[-1])
            frame_indices.append(frame_indices[-1])

        # Create labels for each frame
        labels = [str(i + 1) for i in range(rows * cols)]

        # Create grid image
        montage = create_image_grid(frames[:rows * cols], rows=rows, cols=cols, labels=labels)

        # Find which frame label is closest to the actual transition
        correct_frame_idx = 0
        min_dist = float('inf')
        for i, idx in enumerate(frame_indices[:rows * cols]):
            dist = abs(idx - transition_frame)
            if dist < min_dist:
                min_dist = dist
                correct_frame_idx = i

        # Build question
        question_text = (
            f"The robot is transitioning from '{closest_transition['from_skill']}' to "
            f"'{closest_transition['to_skill']}'. Which frame best represents the "
            f"transition point between these two actions?"
        )

        # Create choices: 4 frame options + "No clear transition"
        # Always include the correct frame in the choices
        base_choices = [0, 2, 5, 7]  # Frames 1, 3, 6, 8

        # Ensure correct frame is in choices - replace closest one if needed
        if correct_frame_idx not in base_choices:
            # Find the closest choice to replace
            distances = [(abs(c - correct_frame_idx), i) for i, c in enumerate(base_choices)]
            distances.sort()
            replace_idx = distances[0][1]
            base_choices[replace_idx] = correct_frame_idx
            base_choices.sort()

        choices = [f"Frame {i + 1}" for i in base_choices]
        choices.append("No clear transition visible")

        # Find which choice is correct
        correct_choice_idx = base_choices.index(correct_frame_idx)

        return VQA(
            question_text=question_text,
            choices=choices,
            correct_idx=correct_choice_idx,
            question_images=[montage],
            metadata={
                "tag": "transition_frame",
                "num_frames": 8,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "transition_frame": transition_frame,
                "from_skill": closest_transition['from_skill'],
                "to_skill": closest_transition['to_skill'],
                "frame_indices": frame_indices[:rows * cols],
                "correct_frame_label": correct_frame_idx + 1,
                "spacing": spacing,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_transition_frame: {e}")
        return None


def vqa_action_progress(trajectory: AgiBotTrajectory, step_idx: int) -> Optional[VQA]:
    """
    [action_progress] What percentage of the action is completed?

    Type: Choice | Frames: 3F | Scope: Segment

    Shows [start (0%), query (?%), end (100%)] for reference.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Current step index

    Returns:
        VQA instance or None if generation fails
    """
    try:
        segment = trajectory.get_segment_for_step(step_idx)
        if segment is None:
            return None

        segment_length = segment.end_frame - segment.start_frame
        if segment_length <= 0:
            return None

        position_in_segment = step_idx - segment.start_frame
        progress_pct = (position_in_segment / segment_length) * 100

        # Get 3 frames: start (0%), query frame (?%), end (100%)
        start_frame = trajectory.get_frame(segment.start_frame)
        query_frame = trajectory.get_frame(step_idx)
        end_frame = trajectory.get_frame(segment.end_frame - 1)

        # Check for None (can't use `in` with numpy arrays)
        if start_frame is None or query_frame is None or end_frame is None:
            return None

        frames = [start_frame, query_frame, end_frame]
        frame_indices = [segment.start_frame, step_idx, segment.end_frame - 1]

        # Create montage with labels showing reference points
        labels = ["Start (0%)", "Current (?)", "End (100%)"]
        rows, cols = _calculate_grid_layout(len(frames))
        montage = create_image_grid(frames, rows=rows, cols=cols, labels=labels)
        if montage is None:
            return None

        progress_buckets = [
            ("0-20%", 0, 20),
            ("20-40%", 20, 40),
            ("40-60%", 40, 60),
            ("60-80%", 60, 80),
            ("80-100%", 80, 100)
        ]

        correct_bucket_idx = 0
        for i, (label, low, high) in enumerate(progress_buckets):
            if low <= progress_pct < high or (high == 100 and progress_pct >= 80):
                correct_bucket_idx = i
                break

        question_text = (
            f"The robot is performing: '{segment.action_text}'. "
            f"The first image shows the start (0%), the last shows the end (100%). "
            f"What percentage of this action has been completed in the middle image?"
        )

        choices = [bucket[0] for bucket in progress_buckets]

        return VQA(
            question_text=question_text,
            choices=choices,
            correct_idx=correct_bucket_idx,
            question_images=[montage],
            metadata={
                "tag": "action_progress",
                "num_frames": 3,
                "task_id": getattr(trajectory, 'task_id', 'unknown'),
                "traj_id": getattr(trajectory, 'traj_id', 'unknown'),
                "skill": segment.skill,
                "action_text": segment.action_text,
                "actual_progress": round(progress_pct, 1),
                "correct_bucket": choices[correct_bucket_idx],
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_action_progress: {e}")
        return None


# ==================== VQA GENERATION REGISTRY ====================

VQA_GENERATORS = {
    # State queries (1F)
    "gripper_state": vqa_gripper_state,
    "task_complete": vqa_task_complete,
    "scene_description": vqa_scene_description,

    # Action queries (3F)
    "current_action": vqa_current_action,
    "next_action": vqa_next_action,
    "current_skill": vqa_current_skill,
    "active_arm": vqa_active_arm,
    "target_object": vqa_target_object,
    "is_handover": vqa_is_handover,
    "action_progress": vqa_action_progress,

    # Trajectory queries (5F)
    "goal_config": vqa_goal_config,
    "task_instruction": vqa_task_instruction,
    "action_count": vqa_action_count,

    # Segmentation queries
    "transition_frame": vqa_transition_frame,
}


def generate_all_vqas(trajectory: AgiBotTrajectory,
                      num_samples_per_type: int = 3,
                      state_samples: int = 1,
                      segment_samples: int = 3) -> List[VQA]:
    """
    Generate all types of VQAs for a trajectory.

    Args:
        trajectory: AgiBotTrajectory instance
        num_samples_per_type: Default samples per VQA type (legacy, overridden by specific params)
        state_samples: Samples for robot state VQAs (gripper, scene) - be selective
        segment_samples: Samples for segment/trajectory VQAs (action, skill, transition) - generate more

    Returns:
        List of generated VQA instances
    """
    # Preload all frames sequentially (much faster than random access)
    trajectory.decode_all_frames()

    vqas = []
    traj_length = trajectory.get_trajectory_length()

    # Sample step indices (enough for all types)
    max_samples = max(state_samples, segment_samples)
    sample_steps = random.sample(
        range(traj_length),
        min(max_samples * 5, traj_length)
    )

    # State queries (1F) - instantaneous
    state_based = [
        "gripper_state",
        "task_complete",
    ]

    for vqa_type in state_based:
        generator = VQA_GENERATORS[vqa_type]
        count = 0
        for step_idx in sample_steps:
            if count >= state_samples:
                break

            if vqa_type == "gripper_state":
                vqa = generator(trajectory, step_idx, arm=random.choice([0, 1]))
            else:
                vqa = generator(trajectory, step_idx)

            if vqa is not None:
                vqas.append(vqa)
                count += 1

    # Action queries (3F) - segment-based
    action_based = [
        "current_action",
        "next_action",
        "current_skill",
        "active_arm",
        "target_object",
        "is_handover",
        "action_progress",
    ]

    for vqa_type in action_based:
        generator = VQA_GENERATORS[vqa_type]
        count = 0
        for step_idx in sample_steps:
            if count >= segment_samples:
                break

            vqa = generator(trajectory, step_idx)
            if vqa is not None:
                vqas.append(vqa)
                count += 1

    # Trajectory queries (5F / 3F×N) - one per trajectory
    trajectory_based = [
        "goal_config",
        "task_instruction",
        "scene_description",
        "action_count",
    ]

    for vqa_type in trajectory_based:
        generator = VQA_GENERATORS[vqa_type]
        vqa = generator(trajectory)
        if vqa is not None:
            vqas.append(vqa)

    # Segmentation queries - transition detection
    for step_idx in sample_steps[:segment_samples]:
        vqa = VQA_GENERATORS["transition_frame"](trajectory, step_idx)
        if vqa is not None:
            vqas.append(vqa)
            break  # Only need one transition frame VQA per trajectory

    return vqas


def save_vqa_dataset(vqas: List[VQA],
                     output_dir: str,
                     metadata: Optional[Dict] = None) -> None:
    """
    Save a list of VQAs to a dataset directory.

    Args:
        vqas: List of VQA instances
        output_dir: Output directory path
        metadata: Optional dataset metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save images
    for vqa in vqas:
        vqa.save_images(images_dir)

    # Save VQA data
    vqa_data = [vqa.to_dict() for vqa in vqas]
    dataset = {
        "vqa_items": vqa_data,
        "metadata": metadata or {}
    }

    with open(os.path.join(output_dir, "vqa_data.json"), "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(vqas)} VQAs to {output_dir}")

"""
VQA Generation for AgiBotWorld Dataset

This module contains:
1. VQA base class for representing Visual Question Answering items
2. Fully ported VQA generation functions from OXE
3. New AgiBotWorld-specific VQA generation functions

Question Categories:
- S1: Robot Gripper State (ported)
- I1: Task Success State (ported)
- I3: Goal Configuration (ported)
- I4: Action Understanding (ported with skill adaptation)
- I4b: Next Action (ported with segment adaptation)
- I6: Trajectory Understanding (ported)
- NEW: Skill Recognition
- NEW: Arm Coordination
- NEW: Scene Understanding
- NEW: Target Object
- NEW: Action Count
- NEW: Handover Detection
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
    AGIBOT_SKILLS,
    GRASP_SKILLS,
    RELEASE_SKILLS,
    BIMANUAL_SKILLS
)


def _hash_image(img: np.ndarray) -> Optional[str]:
    """Generate a SHA-256 hash for an image."""
    if img is None:
        return None
    _, buffer = cv2.imencode('.png', img)
    img_bytes = buffer.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()


class VQA:
    """
    Represents a Visual Question Answering item with multiple choices.

    Supports:
    - 4 choices for binary (yes/no) questions
    - 5 choices for multi-choice questions
    - Optional images for question and choices
    - Metadata for tracking question type and context
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
            metadata: Additional metadata
            question_image_ids: Pre-computed image IDs for question images
            choice_image_ids: Pre-computed image IDs for choice images
        """
        self.question_text = question_text
        self.choices = choices
        self.correct_idx = correct_idx
        self.question_images = question_images if question_images is not None else []
        self.choice_images = choice_images if choice_images is not None else [None] * len(choices)
        self.metadata = metadata or {}

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

    def to_dict(self) -> Dict:
        """Convert VQA to dictionary."""
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

        return {
            "question": {
                "text": self.question_text,
                "image_ids": self.question_image_ids
            },
            "choices": [
                {
                    "text": choice,
                    "image_id": img_id,
                    "is_correct": bool(i == self.correct_idx)
                }
                for i, (choice, img_id) in enumerate(zip(self.choices, self.choice_image_ids))
            ],
            "metadata": clean_metadata
        }

    def to_json(self) -> str:
        """Convert VQA to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_images(self, images_dir: str) -> None:
        """Save all images to directory."""
        os.makedirs(images_dir, exist_ok=True)

        for img, img_id in zip(self.question_images, self.question_image_ids):
            if img is not None and img_id is not None:
                cv2.imwrite(os.path.join(images_dir, f"{img_id}.png"), img)

        for img, img_id in zip(self.choice_images, self.choice_image_ids):
            if img is not None and img_id is not None:
                cv2.imwrite(os.path.join(images_dir, f"{img_id}.png"), img)

    def __repr__(self):
        return f"VQA(question='{self.question_text[:50]}...', tag={self.metadata.get('tag', 'unknown')})"


# ==================== PORTED VQA FUNCTIONS ====================

# S1: Robot Gripper State
def vqa_robot_gripper_open(trajectory: AgiBotTrajectory,
                           step_idx: int,
                           arm: int = 0) -> Optional[VQA]:
    """
    Generate a VQA asking if the robot's gripper is open.

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query
        arm: Which arm to query (0=left, 1=right)

    Returns:
        VQA instance or None if generation fails
    """
    try:
        # Get frame
        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        # Check gripper state
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
            question_images=[frame],
            metadata={
                "tag": "vqa_robot_gripper_open",
                "arm": arm_name,
                "is_open": is_open,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_robot_gripper_open: {e}")
        return None


# I1: Task Success State
def vqa_task_success_state(trajectory: AgiBotTrajectory,
                           step_idx: int) -> Optional[VQA]:
    """
    Generate a VQA asking if the task has been completed successfully.

    For AgiBotWorld, success is determined by:
    - Being in the last action segment
    - AgiBotWorld contains successful demonstrations

    Args:
        trajectory: AgiBotTrajectory instance
        step_idx: Step index to query

    Returns:
        VQA instance or None if generation fails
    """
    try:
        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        segments = trajectory.get_action_segments()
        if not segments:
            return None

        # Check if in last segment
        last_segment = segments[-1]
        is_in_last_segment = last_segment.contains(step_idx)

        # Task successful if in last segment (AgiBotWorld has successful demos)
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
            question_images=[frame],
            metadata={
                "tag": "vqa_task_success_state",
                "is_success": is_success,
                "is_in_last_segment": is_in_last_segment,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_task_success_state: {e}")
        return None


# I3: Goal Configuration
def vqa_goal_configuration(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    Generate a VQA asking which image shows the goal configuration.

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

        # Get last frame (goal configuration)
        goal_frame = trajectory.get_frame(last_step_idx)
        if goal_frame is None:
            return None

        # Sample 4 other frames from different segments
        sampled_frames = [goal_frame]
        sampled_indices = [last_step_idx]

        # Get frames from segment midpoints
        other_frames = []
        for seg in segments[:-1]:  # Exclude last segment
            mid_frame = trajectory.get_frame(seg.get_middle_frame())
            if mid_frame is not None:
                other_frames.append((seg.get_middle_frame(), mid_frame))

        # Shuffle and take up to 4
        random.shuffle(other_frames)
        for idx, frame in other_frames[:4]:
            sampled_frames.append(frame)
            sampled_indices.append(idx)

        # If not enough frames, duplicate from available
        while len(sampled_frames) < 5:
            if len(other_frames) > 0:
                idx, frame = random.choice(other_frames)
                sampled_frames.append(frame.copy())
                sampled_indices.append(idx)
            else:
                # Sample random frames
                rand_idx = random.randint(0, traj_length - 2)
                rand_frame = trajectory.get_frame(rand_idx)
                if rand_frame is not None:
                    sampled_frames.append(rand_frame)
                    sampled_indices.append(rand_idx)

        if len(sampled_frames) != 5:
            return None

        # Create grid image with labels
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
            correct_idx=0,  # First frame is the goal
            question_images=[grid_image],
            metadata={
                "tag": "vqa_goal_configuration",
                "task_name": trajectory.task_name,
                "sampled_indices": sampled_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_goal_configuration: {e}")
        return None


# I4: Action Understanding
def vqa_action_understanding(trajectory: AgiBotTrajectory,
                             step_idx: int) -> Optional[VQA]:
    """
    Generate a VQA asking about the current action/skill.

    Uses AgiBotWorld's skill annotations instead of gripper-based phases.
    Shows 4 frames from the current segment for better context.

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

        # Get 4 frames from the current segment
        segment_length = segment.end_frame - segment.start_frame
        if segment_length < 4:
            return None

        # Sample 4 evenly spaced frames from the segment
        frame_indices = [
            segment.start_frame + int(i * segment_length / 4)
            for i in range(4)
        ]

        frames = []
        for idx in frame_indices:
            frame = trajectory.get_frame(idx)
            if frame is None:
                return None
            frames.append(frame)

        # Create a 2x2 grid of frames
        montage = create_image_grid(frames, rows=2, cols=2, labels=["1", "2", "3", "4"])
        if montage is None:
            return None

        # Get target object
        target_object = extract_target_object(current_action_text)
        if not target_object:
            target_object = "object"

        # Build choices: correct skill description + 4 other skills
        correct_description = get_skill_description(current_skill, target_object)

        # Get other skills for distractors
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
                "tag": "vqa_action_understanding",
                "current_skill": current_skill,
                "action_text": current_action_text,
                "target_object": target_object,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_action_understanding: {e}")
        return None


# I4b: Next Action
def vqa_next_action(trajectory: AgiBotTrajectory,
                    step_idx: int) -> Optional[VQA]:
    """
    Generate a VQA asking about the next action in the sequence.

    Uses action segment sequence to predict next action.
    Shows 4 frames from the current segment for better context.

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

        # Find next segment
        current_idx = current_segment.segment_id
        if current_idx >= len(segments) - 1:
            # Last segment - no next action
            return None

        next_segment = segments[current_idx + 1]

        # Get 4 frames from the current segment
        segment_length = current_segment.end_frame - current_segment.start_frame
        if segment_length < 4:
            return None

        # Sample 4 evenly spaced frames from the segment
        frame_indices = [
            current_segment.start_frame + int(i * segment_length / 4)
            for i in range(4)
        ]

        frames = []
        for idx in frame_indices:
            frame = trajectory.get_frame(idx)
            if frame is None:
                return None
            frames.append(frame)

        # Create a 2x2 grid of frames
        montage = create_image_grid(frames, rows=2, cols=2, labels=["1", "2", "3", "4"])
        if montage is None:
            return None

        # Get target objects
        current_object = extract_target_object(current_segment.action_text) or "object"
        next_object = extract_target_object(next_segment.action_text) or "object"

        current_description = get_skill_description(current_segment.skill, current_object)
        next_description = get_skill_description(next_segment.skill, next_object)

        # Build incorrect choices from other segments
        other_segments = [s for s in segments if s.segment_id != next_segment.segment_id]
        random.shuffle(other_segments)

        incorrect_descriptions = []
        for seg in other_segments[:4]:
            obj = extract_target_object(seg.action_text) or "object"
            desc = get_skill_description(seg.skill, obj)
            if desc != next_description:
                incorrect_descriptions.append(desc)

        # Add generic descriptions if needed
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
                "tag": "vqa_next_action",
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


# I6: Trajectory Understanding
def vqa_trajectory_understanding(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    Generate a VQA asking which instruction describes the trajectory.

    Shows keyframes and asks which task description matches.

    Args:
        trajectory: AgiBotTrajectory instance

    Returns:
        VQA instance or None if generation fails
    """
    try:
        correct_instruction = trajectory.task_name
        if not correct_instruction:
            return None

        # Get keyframes from segment boundaries
        keyframes = trajectory.get_segment_keyframes()

        if len(keyframes) < 2:
            # Fallback to evenly spaced frames
            keyframes = trajectory.get_keyframes(5)

        if not keyframes:
            return None

        # Create temporal montage
        frames = [kf[1] for kf in keyframes[:5]]  # Take up to 5 frames
        skills = []
        for kf in keyframes[:5]:
            if len(kf) > 2:  # Has segment info
                skills.append(kf[2].skill)
            else:
                skills.append("")

        montage = create_temporal_montage(frames, labels=skills)

        if montage is None:
            # Fallback to grid
            montage = create_image_grid(frames, rows=1, cols=len(frames))

        if montage is None:
            return None

        # Generate distractor instructions
        incorrect_instructions = generate_distractor_instructions(correct_instruction, 4)

        question_text = "Which task description best matches the robot's actions shown in the images?"

        return VQA(
            question_text=question_text,
            choices=[correct_instruction] + incorrect_instructions,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "vqa_trajectory_understanding",
                "task_name": correct_instruction,
                "num_keyframes": len(keyframes)
            }
        )
    except Exception as e:
        print(f"Error in vqa_trajectory_understanding: {e}")
        return None


# ==================== NEW AGIBOT-SPECIFIC VQA FUNCTIONS ====================

# NEW: Skill Recognition
def vqa_skill_recognition(trajectory: AgiBotTrajectory,
                          step_idx: int) -> Optional[VQA]:
    """
    Generate a VQA asking to identify the skill being executed.

    Uses AgiBotWorld's 31 skill taxonomy.
    Shows 4 frames from the current segment for better context.

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

        # Get 4 frames from the current segment
        segment_length = segment.end_frame - segment.start_frame
        if segment_length < 4:
            return None

        # Sample 4 evenly spaced frames from the segment
        frame_indices = [
            segment.start_frame + int(i * segment_length / 4)
            for i in range(4)
        ]

        frames = []
        for idx in frame_indices:
            frame = trajectory.get_frame(idx)
            if frame is None:
                return None
            frames.append(frame)

        # Create a 2x2 grid of frames
        montage = create_image_grid(frames, rows=2, cols=2, labels=["1", "2", "3", "4"])
        if montage is None:
            return None

        # Get 4 other skills as distractors
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
                "tag": "vqa_skill_recognition",
                "skill": correct_skill,
                "action_text": segment.action_text,
                "step_idx": step_idx,
                "frame_indices": frame_indices
            }
        )
    except Exception as e:
        print(f"Error in vqa_skill_recognition: {e}")
        return None


# NEW: Arm Coordination
def vqa_arm_coordination(trajectory: AgiBotTrajectory,
                         step_idx: int) -> Optional[VQA]:
    """
    Generate a VQA asking about which arm is active.

    Detects arm usage from action text.

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

        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        active_arm = detect_active_arm(segment.action_text)
        if active_arm is None:
            return None  # Cannot determine arm from text

        question_text = "Which arm is the robot primarily using for this action?"

        # Binary question with 4 choices (yes/no style)
        if active_arm == "left":
            correct_answer = "Yes"
            incorrect_answer = "No"
            question_text = "Is the robot using its LEFT arm for this action?"
        elif active_arm == "right":
            correct_answer = "Yes"
            incorrect_answer = "No"
            question_text = "Is the robot using its RIGHT arm for this action?"
        else:  # both
            correct_answer = "Yes"
            incorrect_answer = "No"
            question_text = "Is the robot using BOTH arms for this action?"

        distractors = ["Cannot be determined", "Partially using"]

        return VQA(
            question_text=question_text,
            choices=[correct_answer, incorrect_answer] + distractors,
            correct_idx=0,
            question_images=[frame],
            metadata={
                "tag": "vqa_arm_coordination",
                "active_arm": active_arm,
                "action_text": segment.action_text,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_arm_coordination: {e}")
        return None


# NEW: Scene Understanding
def vqa_scene_understanding(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    Generate a VQA about the scene/environment.

    Uses init_scene_text to ask about the scene.

    Args:
        trajectory: AgiBotTrajectory instance

    Returns:
        VQA instance or None if generation fails
    """
    try:
        init_scene = trajectory.init_scene_text
        if not init_scene or len(init_scene) < 20:
            return None

        # Get first frame
        frame = trajectory.get_frame(0)
        if frame is None:
            return None

        # Generate distractor scene descriptions
        distractor_scenes = [
            "The robot is in a kitchen preparing food on a counter.",
            "The robot is in a warehouse sorting packages on shelves.",
            "The robot is in an office organizing documents on a desk.",
            "The robot is in a laboratory handling scientific equipment."
        ]

        # Filter out similar distractors
        distractors = [d for d in distractor_scenes if d.lower() not in init_scene.lower()][:4]

        question_text = "Which description best matches the robot's environment shown in the image?"

        return VQA(
            question_text=question_text,
            choices=[init_scene] + distractors,
            correct_idx=0,
            question_images=[frame],
            metadata={
                "tag": "vqa_scene_understanding",
                "init_scene_text": init_scene
            }
        )
    except Exception as e:
        print(f"Error in vqa_scene_understanding: {e}")
        return None


# NEW: Target Object
def vqa_target_object(trajectory: AgiBotTrajectory,
                      step_idx: int) -> Optional[VQA]:
    """
    Generate a VQA asking about the target object.

    Extracts object from action text using NLP.

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

        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        target_object = extract_target_object(segment.action_text)
        if not target_object:
            return None

        # Generate distractor objects
        common_objects = [
            "cup", "box", "bottle", "plate", "book", "pen", "toy",
            "bag", "cloth", "tool", "container", "package", "bowl",
            "spoon", "fork", "tray", "basket", "cart"
        ]

        # Filter out the correct object
        distractors = [obj for obj in common_objects if obj.lower() not in target_object.lower()]
        random.shuffle(distractors)
        distractor_objects = distractors[:4]

        question_text = "What object is the robot interacting with in this action?"

        return VQA(
            question_text=question_text,
            choices=[target_object] + distractor_objects,
            correct_idx=0,
            question_images=[frame],
            metadata={
                "tag": "vqa_target_object",
                "target_object": target_object,
                "action_text": segment.action_text,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_target_object: {e}")
        return None


# NEW: Action Count
def vqa_action_count(trajectory: AgiBotTrajectory) -> Optional[VQA]:
    """
    Generate a VQA asking how many actions are in the task.

    Args:
        trajectory: AgiBotTrajectory instance

    Returns:
        VQA instance or None if generation fails
    """
    try:
        num_segments = trajectory.get_num_segments()
        if num_segments < 2:
            return None

        # Get keyframes for visualization
        keyframes = trajectory.get_keyframes(5)
        if not keyframes:
            return None

        frames = [kf[1] for kf in keyframes]
        montage = create_temporal_montage(frames)

        if montage is None:
            return None

        correct_answer = str(num_segments)

        # Generate distractor counts
        distractors = []
        for delta in [-2, -1, 1, 2]:
            count = num_segments + delta
            if count > 0:
                distractors.append(str(count))

        # Ensure we have 4 distractors
        while len(distractors) < 4:
            rand_count = random.randint(1, 15)
            if str(rand_count) != correct_answer and str(rand_count) not in distractors:
                distractors.append(str(rand_count))

        distractors = distractors[:4]

        question_text = f"How many distinct actions does the robot perform in this task ({trajectory.task_name})?"

        return VQA(
            question_text=question_text,
            choices=[correct_answer] + distractors,
            correct_idx=0,
            question_images=[montage],
            metadata={
                "tag": "vqa_action_count",
                "num_actions": num_segments,
                "task_name": trajectory.task_name
            }
        )
    except Exception as e:
        print(f"Error in vqa_action_count: {e}")
        return None


# NEW: Handover Detection
def vqa_handover_detection(trajectory: AgiBotTrajectory,
                           step_idx: int) -> Optional[VQA]:
    """
    Generate a VQA asking about handover between arms.

    Detects HandOver skill in action segments.

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

        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        is_handover = segment.skill == "HandOver"

        question_text = "Is the robot performing a handover (transferring an object between its arms)?"

        correct_answer = "Yes" if is_handover else "No"
        incorrect_answer = "No" if is_handover else "Yes"
        distractors = ["Cannot be determined", "Partially transferring"]

        return VQA(
            question_text=question_text,
            choices=[correct_answer, incorrect_answer] + distractors,
            correct_idx=0,
            question_images=[frame],
            metadata={
                "tag": "vqa_handover_detection",
                "is_handover": is_handover,
                "skill": segment.skill,
                "action_text": segment.action_text,
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_handover_detection: {e}")
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
                "tag": "vqa_transition_frame",
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


def vqa_progress_estimation(trajectory: AgiBotTrajectory, step_idx: int) -> Optional[VQA]:
    """
    Generate VQA asking what percentage of the current action is completed.

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

        frame = trajectory.get_frame(step_idx)
        if frame is None:
            return None

        # Define progress buckets
        progress_buckets = [
            ("0-20%", 0, 20),
            ("20-40%", 20, 40),
            ("40-60%", 40, 60),
            ("60-80%", 60, 80),
            ("80-100%", 80, 100)
        ]

        # Find correct bucket
        correct_idx = 0
        for i, (label, low, high) in enumerate(progress_buckets):
            if low <= progress_pct < high or (high == 100 and progress_pct >= 80):
                correct_idx = i
                break

        question_text = (
            f"The robot is performing: '{segment.action_text}'. "
            f"What percentage of this action has been completed?"
        )

        choices = [bucket[0] for bucket in progress_buckets]

        return VQA(
            question_text=question_text,
            choices=choices,
            correct_idx=correct_idx,
            question_images=[frame],
            metadata={
                "tag": "vqa_progress_estimation",
                "skill": segment.skill,
                "action_text": segment.action_text,
                "actual_progress": round(progress_pct, 1),
                "correct_bucket": choices[correct_idx],
                "step_idx": step_idx
            }
        )
    except Exception as e:
        print(f"Error in vqa_progress_estimation: {e}")
        return None


# ==================== VQA GENERATION REGISTRY ====================

VQA_GENERATORS = {
    # Ported from OXE
    "vqa_robot_gripper_open": vqa_robot_gripper_open,
    "vqa_task_success_state": vqa_task_success_state,
    "vqa_goal_configuration": vqa_goal_configuration,
    "vqa_action_understanding": vqa_action_understanding,
    "vqa_next_action": vqa_next_action,
    "vqa_trajectory_understanding": vqa_trajectory_understanding,

    # New AgiBotWorld-specific
    "vqa_skill_recognition": vqa_skill_recognition,
    "vqa_arm_coordination": vqa_arm_coordination,
    "vqa_scene_understanding": vqa_scene_understanding,
    "vqa_action_count": vqa_action_count,
    "vqa_handover_detection": vqa_handover_detection,

    # Segmentation VQAs
    "vqa_transition_frame": vqa_transition_frame,
    "vqa_progress_estimation": vqa_progress_estimation,
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
        state_samples: Samples for robot state VQAs (gripper, arm, success) - be selective
        segment_samples: Samples for segment/trajectory VQAs (action, skill, transition) - generate more

    Returns:
        List of generated VQA instances
    """
    vqas = []
    traj_length = trajectory.get_trajectory_length()

    # Sample step indices (enough for all types)
    max_samples = max(state_samples, segment_samples)
    sample_steps = random.sample(
        range(traj_length),
        min(max_samples * 5, traj_length)
    )

    # Robot state VQAs (single frame, instantaneous) - be selective
    state_based = [
        "vqa_robot_gripper_open",
        "vqa_task_success_state",
        "vqa_arm_coordination",
        "vqa_handover_detection",
    ]

    for vqa_type in state_based:
        generator = VQA_GENERATORS[vqa_type]
        count = 0
        for step_idx in sample_steps:
            if count >= state_samples:
                break

            if vqa_type == "vqa_robot_gripper_open":
                vqa = generator(trajectory, step_idx, arm=random.choice([0, 1]))
            else:
                vqa = generator(trajectory, step_idx)

            if vqa is not None:
                vqas.append(vqa)
                count += 1

    # Segment-based VQAs (action understanding, skill, transition, progress) - generate all
    segment_based = [
        "vqa_action_understanding",
        "vqa_next_action",
        "vqa_skill_recognition",
        "vqa_transition_frame",
        "vqa_progress_estimation",
    ]

    for vqa_type in segment_based:
        generator = VQA_GENERATORS[vqa_type]
        count = 0
        for step_idx in sample_steps:
            if count >= segment_samples:
                break

            vqa = generator(trajectory, step_idx)
            if vqa is not None:
                vqas.append(vqa)
                count += 1

    # Trajectory-level VQAs (one per trajectory)
    trajectory_based = [
        "vqa_goal_configuration",
        "vqa_trajectory_understanding",
        "vqa_scene_understanding",
        "vqa_action_count"
    ]

    for vqa_type in trajectory_based:
        generator = VQA_GENERATORS[vqa_type]
        vqa = generator(trajectory)
        if vqa is not None:
            vqas.append(vqa)

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

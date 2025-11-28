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


def _generate_hint_gripper_state(
    arm_name: str,
    is_open: bool,
    task_name: str = "",
    step_idx: int = 0,
    traj_length: int = 0,
    other_gripper_state: Optional[bool] = None,
    current_skill: str = "",
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for gripper state VQA.

    Includes: task context, robot state, scene info, temporal position, visual cues, reasoning.
    """
    state = "open" if is_open else "closed"
    opposite = "closed" if is_open else "open"

    # Temporal context
    progress_pct = (step_idx / traj_length * 100) if traj_length > 0 else 0
    temporal_phase = "beginning" if progress_pct < 33 else ("middle" if progress_pct < 66 else "end")

    # Other arm state
    other_arm = "right" if arm_name == "left" else "left"
    other_state_str = ""
    if other_gripper_state is not None:
        other_state_str = f"The {other_arm} gripper is {'open' if other_gripper_state else 'closed'}. "

    # Skill context for gripper expectation
    gripper_expectation = ""
    if current_skill:
        if current_skill in GRASP_SKILLS:
            gripper_expectation = f"During '{current_skill}', the gripper typically transitions from open to closed to grasp an object. "
        elif current_skill in RELEASE_SKILLS:
            gripper_expectation = f"During '{current_skill}', the gripper typically transitions from closed to open to release an object. "

    return (
        f"[TASK CONTEXT] The robot is performing the task: '{task_name}'. "
        f"This is a manipulation task requiring precise gripper control. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[ROBOT STATE] Querying the {arm_name} gripper state. {other_state_str}"
        f"The robot has two parallel-jaw grippers (left and right) that can open and close independently. "
        f"[TEMPORAL CONTEXT] Currently at step {step_idx}/{traj_length} ({progress_pct:.1f}% through trajectory), "
        f"in the {temporal_phase} phase of the task. {gripper_expectation}"
        f"[VISUAL CUES] To determine gripper state, observe: "
        f"1) Finger separation - open grippers show gap between fingers, closed grippers have fingers together; "
        f"2) Object contact - closed grippers often hold objects between fingers; "
        f"3) Gripper geometry - the {arm_name} gripper mechanism visible in the image. "
        f"[REASONING] The {arm_name} gripper is {state}. "
        f"Visual evidence: the gripper fingers are {'spread apart with visible gap' if is_open else 'together, likely grasping or ready to grasp'}. "
        f"A {opposite} gripper would show fingers {'pressed together' if is_open else 'separated with a gap'}. "
        f"The correct answer is '{'Yes' if is_open else 'No'}' because the gripper is observably {state}."
    )


def _generate_hint_task_complete(
    is_success: bool,
    task_name: str,
    is_in_last_segment: bool,
    step_idx: int = 0,
    traj_length: int = 0,
    num_segments: int = 0,
    current_segment_idx: int = 0,
    init_scene: str = "",
    final_action: str = ""
) -> str:
    """
    Generate comprehensive hint for task completion VQA.

    Includes: task goal, trajectory position, completion criteria, visual cues.
    """
    status = "completed successfully" if is_success else "not yet completed"
    progress_pct = (step_idx / traj_length * 100) if traj_length > 0 else 0

    # Completion criteria based on task type
    completion_criteria = (
        "For manipulation tasks, completion means: "
        "1) Target object is in goal position/orientation; "
        "2) Robot has released the object (if placing); "
        "3) Robot is returning to neutral or ready for next task."
    )

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. "
        f"The goal is to successfully complete this manipulation objective. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[TRAJECTORY STATE] At step {step_idx}/{traj_length} ({progress_pct:.1f}% progress). "
        f"Trajectory has {num_segments} action segments. Currently in segment {current_segment_idx + 1}/{num_segments}. "
        f"{'This is the FINAL segment of the trajectory.' if is_in_last_segment else 'Not yet in the final segment.'} "
        f"[COMPLETION CRITERIA] {completion_criteria} "
        f"[FINAL ACTION] {final_action if final_action else 'The final action brings objects to goal configuration.'} "
        f"[VISUAL CUES] To assess completion: "
        f"1) Check if objects are in their target locations; "
        f"2) Observe if the robot has finished active manipulation; "
        f"3) Look for goal state indicators (objects placed, assembled, arranged). "
        f"[REASONING] The task is {status}. "
        f"{'The robot has reached the final configuration and objects are in goal positions.' if is_success else 'The robot is still executing actions or has not achieved the goal configuration.'} "
        f"{'Being in the last segment with successful execution indicates completion.' if is_success else 'More actions are needed or the goal state has not been achieved.'} "
        f"The correct answer is '{'Yes' if is_success else 'No'}'."
    )


def _generate_hint_goal_config(
    task_name: str,
    num_segments: int = 0,
    init_scene: str = "",
    action_sequence: List[str] = None
) -> str:
    """
    Generate comprehensive hint for goal configuration VQA.

    Includes: task goal definition, action sequence, visual comparison guidance.
    """
    action_seq_str = ""
    if action_sequence:
        action_seq_str = f"The task involves these actions: {' -> '.join(action_sequence[:5])}. "

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. "
        f"The goal configuration is the final state after successful task completion. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[ACTION SEQUENCE] {action_seq_str}"
        f"The trajectory contains {num_segments} distinct action segments leading to the goal. "
        f"[GOAL STATE DEFINITION] The goal configuration should show: "
        f"1) All manipulated objects in their target positions; "
        f"2) Correct object orientations as required by the task; "
        f"3) Robot in a neutral/finished pose (not mid-action). "
        f"[VISUAL COMPARISON] When comparing configurations: "
        f"1) Identify the START state (objects in initial positions); "
        f"2) Identify INTERMEDIATE states (objects being manipulated); "
        f"3) Identify the GOAL state (task objective achieved). "
        f"Look for: object displacement from initial position, assembly completion, arrangement patterns. "
        f"[REASONING] For task '{task_name}', the goal configuration shows the successful outcome. "
        f"Configuration A is correct because it displays the final state after all {num_segments} actions "
        f"are completed, with objects in their target arrangement."
    )


def _generate_hint_current_action(
    correct_description: str,
    current_skill: str,
    target_object: str,
    task_name: str,
    segment_idx: int = 0,
    num_segments: int = 0,
    prev_skill: str = "",
    next_skill: str = "",
    active_arm: str = "",
    frame_indices: List[int] = None,
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for current action VQA.

    Includes: action context, skill taxonomy, motion patterns, visual cues.
    """
    # Skill category
    skill_category = ""
    if current_skill in GRASP_SKILLS:
        skill_category = "This is a GRASPING skill - the robot is acquiring an object. "
    elif current_skill in RELEASE_SKILLS:
        skill_category = "This is a RELEASING skill - the robot is placing/releasing an object. "
    elif current_skill in BIMANUAL_SKILLS:
        skill_category = "This is a BIMANUAL skill - both arms coordinate together. "

    # Sequence context
    seq_context = f"This is action {segment_idx + 1} of {num_segments} in the task. "
    if prev_skill:
        seq_context += f"Previous action was '{prev_skill}'. "
    if next_skill:
        seq_context += f"Next action will be '{next_skill}'. "

    frame_str = f"Frames shown: {frame_indices}" if frame_indices else "3 frames: start, middle, end of action"

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Currently executing action segment {segment_idx + 1}/{num_segments}. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[ACTION SEQUENCE] {seq_context}"
        f"[SKILL INFORMATION] Current skill: '{current_skill}'. {skill_category}"
        f"Target object: '{target_object}'. "
        f"{'Active arm: ' + active_arm + '. ' if active_arm else ''}"
        f"[FRAME ANALYSIS] {frame_str}. "
        f"The 3-frame sequence shows action progression: "
        f"Frame 1 (Start): Initial pose before action; "
        f"Frame 2 (Middle): Peak of action execution; "
        f"Frame 3 (End): Action completion state. "
        f"[VISUAL CUES] To identify '{current_skill}': "
        f"1) Motion trajectory - observe arm movement direction and path; "
        f"2) Gripper interaction - how the gripper engages with '{target_object}'; "
        f"3) Object state change - how '{target_object}' position/state changes; "
        f"4) Arm configuration - joint angles and end-effector orientation. "
        f"[SKILL TAXONOMY] AgiBotWorld skills include: Reach, Grasp, Lift, Move, Place, Push, Pull, "
        f"Insert, Pour, Open, Close, Rotate, HandOver, etc. "
        f"[REASONING] The correct action is '{correct_description}'. "
        f"Evidence: The motion pattern shows {current_skill} execution on {target_object}, "
        f"matching the visual characteristics of this skill type."
    )


def _generate_hint_next_action(
    current_description: str,
    next_description: str,
    task_name: str,
    current_skill: str = "",
    next_skill: str = "",
    current_segment_idx: int = 0,
    num_segments: int = 0,
    target_object: str = "",
    next_target_object: str = "",
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for next action prediction VQA.

    Includes: action sequence logic, task planning, prediction reasoning.
    """
    remaining_actions = num_segments - current_segment_idx - 1

    # Logical sequence patterns
    sequence_logic = ""
    if current_skill in GRASP_SKILLS:
        sequence_logic = "After grasping, typical next actions are: Lift, Move, or direct manipulation. "
    elif current_skill == "Lift":
        sequence_logic = "After lifting, typical next actions are: Move, Transport, or Place. "
    elif current_skill == "Move":
        sequence_logic = "After moving, typical next actions are: Place, Insert, Pour, or another Move. "
    elif current_skill in RELEASE_SKILLS:
        sequence_logic = "After releasing/placing, typical next actions are: Retract, Reach for next object, or task completion. "

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. "
        f"Predicting the next action in the manipulation sequence. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[CURRENT STATE] Currently executing: '{current_description}' (skill: {current_skill}). "
        f"On object: '{target_object}'. "
        f"This is segment {current_segment_idx + 1}/{num_segments}. {remaining_actions} actions remaining. "
        f"[SEQUENCE LOGIC] {sequence_logic}"
        f"Manipulation tasks follow logical patterns: approach -> grasp -> manipulate -> place. "
        f"[PREDICTION FACTORS] Consider: "
        f"1) Current action completion state - what naturally follows; "
        f"2) Task objective - what actions lead toward the goal; "
        f"3) Object states - what objects need manipulation next; "
        f"4) Spatial constraints - reachability and collision avoidance. "
        f"[NEXT ACTION] The next action will be: '{next_description}' (skill: {next_skill}). "
        f"{'Target object: ' + next_target_object + '. ' if next_target_object else ''}"
        f"[REASONING] After '{current_description}', the logical next step is '{next_description}'. "
        f"This follows the manipulation sequence pattern and progresses toward task completion. "
        f"The action sequence maintains physical feasibility and task coherence."
    )


def _generate_hint_current_skill(
    correct_skill: str,
    action_text: str,
    task_name: str = "",
    segment_idx: int = 0,
    num_segments: int = 0,
    target_object: str = "",
    active_arm: str = "",
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for current skill VQA.

    Includes: skill taxonomy, skill definitions, visual characteristics.
    """
    # Skill definition
    skill_def = AGIBOT_SKILLS.get(correct_skill, "A manipulation primitive skill.")

    # Skill category
    skill_category = "general manipulation"
    if correct_skill in GRASP_SKILLS:
        skill_category = "object acquisition (grasping)"
    elif correct_skill in RELEASE_SKILLS:
        skill_category = "object release (placing)"
    elif correct_skill in BIMANUAL_SKILLS:
        skill_category = "bimanual coordination"

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Identifying the manipulation skill being executed. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[ACTION CONTEXT] Full action: '{action_text}'. "
        f"Segment {segment_idx + 1}/{num_segments}. "
        f"{'Target: ' + target_object + '. ' if target_object else ''}"
        f"{'Active arm: ' + active_arm + '. ' if active_arm else ''}"
        f"[SKILL TAXONOMY] AgiBotWorld defines 31 manipulation skills: "
        f"Reach, Grasp, Lift, Lower, Move, Place, Push, Pull, Insert, Extract, "
        f"Pour, Scoop, Stir, Open, Close, Rotate, Flip, Fold, Unfold, Stack, "
        f"Unstack, Align, HandOver, Wipe, Press, Twist, Shake, Cut, Spread, Squeeze, Hold. "
        f"[SKILL DEFINITION] '{correct_skill}': {skill_def} "
        f"Category: {skill_category}. "
        f"[VISUAL CHARACTERISTICS] For skill '{correct_skill}', observe: "
        f"1) End-effector trajectory pattern; "
        f"2) Gripper state and changes; "
        f"3) Object motion and state changes; "
        f"4) Arm configuration evolution. "
        f"[REASONING] The skill is '{correct_skill}'. "
        f"The motion pattern, gripper behavior, and object interaction match the definition "
        f"of {correct_skill} in the AgiBotWorld skill taxonomy."
    )


def _generate_hint_active_arm(
    active_arm: str,
    action_text: str,
    task_name: str = "",
    current_skill: str = "",
    target_object: str = "",
    left_gripper_state: Optional[bool] = None,
    right_gripper_state: Optional[bool] = None,
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for active arm VQA.

    Includes: bimanual robot description, motion analysis, arm coordination.
    """
    arm_desc = "left arm" if active_arm == "left" else ("right arm" if active_arm == "right" else "both arms")

    # Gripper states
    gripper_info = ""
    if left_gripper_state is not None:
        gripper_info += f"Left gripper: {'open' if left_gripper_state else 'closed'}. "
    if right_gripper_state is not None:
        gripper_info += f"Right gripper: {'open' if right_gripper_state else 'closed'}. "

    # Bimanual context
    bimanual_context = ""
    if current_skill in BIMANUAL_SKILLS:
        bimanual_context = f"'{current_skill}' is a bimanual skill requiring coordination of both arms. "
    elif active_arm == "both":
        bimanual_context = "Both arms are actively engaged in this action. "
    else:
        bimanual_context = f"This is a single-arm action using the {active_arm} arm. "

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Determining which arm is active during the action. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[ROBOT DESCRIPTION] The robot is a bimanual manipulator with two independent arms "
        f"(left and right), each with a parallel-jaw gripper. Arms can operate independently "
        f"or coordinate for bimanual tasks. "
        f"[CURRENT ACTION] '{action_text}'. Skill: '{current_skill}'. "
        f"{'Target object: ' + target_object + '. ' if target_object else ''}"
        f"[GRIPPER STATES] {gripper_info}"
        f"[ARM COORDINATION] {bimanual_context}"
        f"[VISUAL CUES] To identify active arm(s): "
        f"1) Motion detection - compare arm positions across frames; "
        f"2) Active arm shows significant pose changes; "
        f"3) Inactive arm remains relatively stationary; "
        f"4) Object interaction indicates which arm is manipulating. "
        f"[REASONING] The {arm_desc} {'is' if active_arm != 'both' else 'are'} active. "
        f"Visual evidence shows {'this arm' if active_arm != 'both' else 'both arms'} "
        f"{'is' if active_arm != 'both' else 'are'} moving and interacting with the workspace. "
        f"The correct answer is 'Yes' for the question about {arm_desc} being active."
    )


def _generate_hint_scene_description(
    init_scene: str,
    task_name: str = "",
    objects_present: List[str] = None
) -> str:
    """
    Generate comprehensive hint for scene description VQA.

    Includes: environment analysis, object identification, spatial layout.
    """
    objects_str = ""
    if objects_present:
        objects_str = f"Objects visible in scene: {', '.join(objects_present)}. "

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Identifying the manipulation environment. "
        f"[SCENE ANALYSIS] The image shows a robot manipulation workspace. "
        f"Key elements to identify: "
        f"1) Workspace type (tabletop, kitchen counter, industrial setting, etc.); "
        f"2) Background elements (walls, equipment, furniture); "
        f"3) Lighting conditions and camera viewpoint. "
        f"[OBJECTS IN SCENE] {objects_str}"
        f"Identify all manipulable objects and fixtures. "
        f"[SPATIAL LAYOUT] Observe: "
        f"1) Robot position relative to workspace; "
        f"2) Object arrangement and spacing; "
        f"3) Workspace boundaries and constraints. "
        f"[ENVIRONMENT CATEGORIES] Common manipulation environments: "
        f"Kitchen (food prep, utensils), Office (documents, supplies), "
        f"Warehouse (packages, shelves), Laboratory (equipment, samples), "
        f"Assembly (parts, tools), Home (household items). "
        f"[CORRECT DESCRIPTION] '{init_scene}' "
        f"[REASONING] The scene matches this description based on: "
        f"visible objects, workspace layout, and environmental context. "
        f"Other descriptions don't match the visual evidence in the image."
    )


def _generate_hint_target_object(
    target_object: str,
    action_text: str,
    task_name: str = "",
    current_skill: str = "",
    active_arm: str = "",
    init_scene: str = "",
    other_objects: List[str] = None
) -> str:
    """
    Generate comprehensive hint for target object VQA.

    Includes: object identification, interaction analysis, visual features.
    """
    other_obj_str = ""
    if other_objects:
        other_obj_str = f"Other objects in scene: {', '.join(other_objects[:5])}. "

    # Interaction type based on skill
    interaction_type = "manipulating"
    if current_skill in GRASP_SKILLS:
        interaction_type = "grasping/acquiring"
    elif current_skill in RELEASE_SKILLS:
        interaction_type = "placing/releasing"
    elif current_skill in ["Push", "Pull"]:
        interaction_type = "pushing/pulling"
    elif current_skill == "Pour":
        interaction_type = "pouring from"

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Identifying the object being manipulated. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"{other_obj_str}"
        f"[CURRENT ACTION] '{action_text}'. "
        f"Skill: '{current_skill}' - the robot is {interaction_type} an object. "
        f"{'Active arm: ' + active_arm + '. ' if active_arm else ''}"
        f"[OBJECT IDENTIFICATION] To identify the target object: "
        f"1) Follow the gripper - what is it approaching/holding; "
        f"2) Track object motion - which object moves with the gripper; "
        f"3) Observe contact points - gripper-object interaction; "
        f"4) Consider task context - what object should be manipulated. "
        f"[OBJECT PROPERTIES] The target '{target_object}' can be identified by: "
        f"Shape, color, size, position relative to the gripper, and how it's being handled. "
        f"[DISTRACTORS] Other objects in the scene are not being directly manipulated "
        f"in this action segment. "
        f"[REASONING] The target object is '{target_object}'. "
        f"Visual evidence: the {'active' if active_arm else ''} gripper is directly "
        f"interacting with this object during the '{current_skill}' action."
    )


def _generate_hint_action_count(
    num_actions: int,
    task_name: str,
    action_sequence: List[str] = None,
    skill_sequence: List[str] = None,
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for action count VQA.

    Includes: segmentation criteria, action boundaries, counting strategy.
    """
    action_list = ""
    if action_sequence:
        action_list = "Action sequence: " + " -> ".join([f"{i+1}.{a}" for i, a in enumerate(action_sequence[:num_actions])]) + ". "

    skill_list = ""
    if skill_sequence:
        skill_list = f"Skills: {', '.join(skill_sequence[:num_actions])}. "

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Counting distinct actions in the trajectory. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[ACTION SEQUENCE] {action_list}{skill_list}"
        f"[SEGMENTATION CRITERIA] An 'action' is a continuous motion segment with: "
        f"1) Single manipulation primitive (skill); "
        f"2) Consistent motion direction/pattern; "
        f"3) Clear start and end boundaries; "
        f"4) Often marked by gripper state changes or direction reversals. "
        f"[BOUNDARY INDICATORS] Action boundaries occur at: "
        f"1) Gripper open/close transitions; "
        f"2) Motion direction changes; "
        f"3) Contact/release events; "
        f"4) Arm switching (in bimanual tasks). "
        f"[COUNTING STRATEGY] "
        f"1) Identify each distinct motion phase; "
        f"2) Count transitions between different skills; "
        f"3) Don't count pauses as separate actions; "
        f"4) Bimanual coordinated motions count as one action. "
        f"[CORRECT COUNT] {num_actions} distinct actions. "
        f"[REASONING] The trajectory contains {num_actions} action segments. "
        f"Each represents a distinct manipulation primitive with clear boundaries. "
        f"Common mistakes: over-counting (splitting continuous motions) or "
        f"under-counting (merging distinct actions)."
    )


def _generate_hint_is_handover(
    is_handover: bool,
    skill: str,
    action_text: str,
    task_name: str = "",
    init_scene: str = "",
    left_gripper_state: Optional[bool] = None,
    right_gripper_state: Optional[bool] = None
) -> str:
    """
    Generate comprehensive hint for handover detection VQA.

    Includes: handover definition, visual patterns, coordination analysis.
    """
    gripper_info = ""
    if left_gripper_state is not None and right_gripper_state is not None:
        gripper_info = (f"Left gripper: {'open' if left_gripper_state else 'closed'}. "
                       f"Right gripper: {'open' if right_gripper_state else 'closed'}. ")

    handover_pattern = ""
    if is_handover:
        handover_pattern = (
            "Handover pattern detected: One arm holds object, other arm approaches, "
            "object transfers between grippers, releasing arm opens. "
        )
    else:
        handover_pattern = (
            "No handover pattern: This action involves single-arm manipulation or "
            "non-transfer bimanual coordination. "
        )

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Detecting bimanual handover action. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[CURRENT ACTION] '{action_text}'. Skill: '{skill}'. "
        f"[GRIPPER STATES] {gripper_info}"
        f"[HANDOVER DEFINITION] A handover is object transfer between arms: "
        f"1) Source arm holds object; "
        f"2) Receiving arm approaches and grasps; "
        f"3) Source arm releases; "
        f"4) Object now held by receiving arm. "
        f"[VISUAL INDICATORS] Handover characteristics: "
        f"1) Both arms converging toward shared workspace; "
        f"2) Object visible between the two grippers; "
        f"3) Sequential gripper state changes (one closes as other opens); "
        f"4) Coordination pattern in arm motions. "
        f"[ANALYSIS] {handover_pattern}"
        f"[REASONING] {'This IS a handover action.' if is_handover else 'This is NOT a handover action.'} "
        f"The skill '{skill}' {'is' if is_handover else 'is not'} a HandOver skill. "
        f"Visual evidence {'confirms' if is_handover else 'shows no'} object transfer between arms. "
        f"The correct answer is '{'Yes' if is_handover else 'No'}'."
    )


def _generate_hint_transition_frame(
    from_skill: str,
    to_skill: str,
    correct_frame_label: int,
    frame_indices: List[int] = None,
    transition_frame_idx: int = 0,
    task_name: str = "",
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for transition frame VQA.

    Includes: boundary detection criteria, visual transition patterns.
    """
    frame_info = f"Frame indices shown: {frame_indices}" if frame_indices else "8 frames at regular intervals"

    # Transition characteristics based on skills
    from_char = ""
    to_char = ""
    if from_skill in GRASP_SKILLS:
        from_char = "ending: object secured in gripper"
    elif from_skill in RELEASE_SKILLS:
        from_char = "ending: object released, gripper opening"
    elif from_skill == "Move":
        from_char = "ending: arm reaching target position"

    if to_skill in GRASP_SKILLS:
        to_char = "starting: gripper approaching object"
    elif to_skill in RELEASE_SKILLS:
        to_char = "starting: arm descending toward placement location"
    elif to_skill == "Move":
        to_char = "starting: arm beginning new trajectory"

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Identifying action boundary between segments. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[TRANSITION] From skill '{from_skill}' to skill '{to_skill}'. "
        f"Actual transition occurs at frame index {transition_frame_idx}. "
        f"[FRAME SEQUENCE] {frame_info}. "
        f"Frames are sampled around the transition point to show context. "
        f"[TRANSITION INDICATORS] "
        f"'{from_skill}' {from_char}. "
        f"'{to_skill}' {to_char}. "
        f"[BOUNDARY DETECTION CRITERIA] "
        f"1) Motion direction change - arm reverses or redirects; "
        f"2) Gripper state change - opens or closes; "
        f"3) Contact event - makes or breaks object contact; "
        f"4) Velocity profile - deceleration then acceleration; "
        f"5) Pose discontinuity - significant configuration change. "
        f"[VISUAL ANALYSIS] Look for the frame where: "
        f"- Previous action is clearly complete; "
        f"- Next action is about to begin; "
        f"- There's a 'pause' or 'pivot' in the motion. "
        f"[REASONING] Frame {correct_frame_label} best shows the transition because "
        f"it captures the boundary moment between '{from_skill}' (ending) and '{to_skill}' (starting). "
        f"Adjacent frames show either the previous action or next action in progress."
    )


def _generate_hint_action_progress(
    action_text: str,
    actual_progress: float,
    correct_bucket: str,
    skill: str = "",
    task_name: str = "",
    segment_start: int = 0,
    segment_end: int = 0,
    current_step: int = 0,
    init_scene: str = ""
) -> str:
    """
    Generate comprehensive hint for action progress VQA.

    Includes: progress estimation method, visual reference points.
    """
    segment_length = segment_end - segment_start

    # Progress description
    if actual_progress < 20:
        progress_desc = "just beginning, near start pose"
    elif actual_progress < 40:
        progress_desc = "early phase, initial motion"
    elif actual_progress < 60:
        progress_desc = "middle phase, peak action"
    elif actual_progress < 80:
        progress_desc = "late phase, approaching completion"
    else:
        progress_desc = "near completion, final adjustments"

    return (
        f"[TASK CONTEXT] Task: '{task_name}'. Estimating progress within action segment. "
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[CURRENT ACTION] '{action_text}'. Skill: '{skill}'. "
        f"Segment spans frames {segment_start} to {segment_end} ({segment_length} frames). "
        f"Query frame is {current_step} (actual progress: {actual_progress:.1f}%). "
        f"[REFERENCE FRAMES] "
        f"Frame 1 (Start, 0%): Action beginning - initial pose before motion; "
        f"Frame 2 (Query, ?%): Current state - estimate progress here; "
        f"Frame 3 (End, 100%): Action complete - final pose after motion. "
        f"[PROGRESS ESTIMATION METHOD] "
        f"1) Compare query frame to start/end references; "
        f"2) Estimate position along motion trajectory; "
        f"3) Consider gripper state progression; "
        f"4) Evaluate object position relative to start/goal. "
        f"[PROGRESS CATEGORIES] "
        f"0-20%: Near start, motion just beginning; "
        f"20-40%: Early phase, noticeable motion from start; "
        f"40-60%: Middle phase, between start and end; "
        f"60-80%: Late phase, closer to end than start; "
        f"80-100%: Near end, almost complete. "
        f"[ANALYSIS] Current state: {progress_desc}. "
        f"[REASONING] The correct answer is '{correct_bucket}' (actual: {actual_progress:.1f}%). "
        f"The query frame shows the action is approximately {correct_bucket} complete based on "
        f"arm position and object state relative to the reference frames."
    )


def _generate_hint_task_instruction(
    task_name: str,
    num_keyframes: int,
    action_sequence: List[str] = None,
    skill_sequence: List[str] = None,
    init_scene: str = "",
    objects_involved: List[str] = None
) -> str:
    """
    Generate comprehensive hint for task instruction matching VQA.

    Includes: task decomposition, visual evidence mapping.
    """
    action_list = ""
    if action_sequence:
        action_list = "Actions shown: " + " -> ".join(action_sequence[:5]) + ". "

    skill_list = ""
    if skill_sequence:
        skill_list = f"Skills used: {', '.join(set(skill_sequence[:5]))}. "

    objects_str = ""
    if objects_involved:
        objects_str = f"Objects manipulated: {', '.join(objects_involved[:5])}. "

    return (
        f"[SCENE CONTEXT] {init_scene if init_scene else 'A robotic manipulation workspace.'} "
        f"[VISUAL SEQUENCE] {num_keyframes} keyframes showing the complete task execution. "
        f"[ACTION DECOMPOSITION] {action_list}{skill_list}"
        f"[OBJECTS] {objects_str}"
        f"[TASK MATCHING CRITERIA] To identify the correct task: "
        f"1) Identify all objects being manipulated; "
        f"2) Observe the sequence of actions performed; "
        f"3) Note the initial and final configurations; "
        f"4) Match to task description that fits all observations. "
        f"[TASK CATEGORIES] Common manipulation tasks: "
        f"Pick-and-place (move object A to location B); "
        f"Assembly (combine parts into structure); "
        f"Sorting (arrange objects by property); "
        f"Pouring (transfer contents between containers); "
        f"Tool use (use tool to manipulate other objects). "
        f"[CORRECT TASK] '{task_name}' "
        f"[REASONING] The visual sequence demonstrates '{task_name}' because: "
        f"1) The objects match those required for this task; "
        f"2) The action sequence follows the expected pattern; "
        f"3) The final configuration achieves the task goal. "
        f"Other task descriptions don't match all visual evidence."
    )


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
    - Hints for auxiliary information about task and decision reasoning

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
                 choice_image_ids: Optional[List[str]] = None,
                 hint: Optional[str] = None):
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
            hint: Optional hint providing task context and reasoning for the correct answer
        """
        self.question_text = question_text
        self.choices = choices
        self.correct_idx = correct_idx
        self.question_images = question_images if question_images is not None else []
        self.choice_images = choice_images if choice_images is not None else [None] * len(choices)
        self.metadata = metadata or {}
        self.hint = hint or ""

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
            "metadata": clean_metadata,
            "hint": self.hint
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
        other_arm = 1 if arm == 0 else 0
        other_gripper_state = trajectory.is_gripper_open(step_idx, other_arm)

        # Get current segment info for skill context
        current_skill = ""
        segment = trajectory.get_segment_for_step(step_idx)
        if segment:
            current_skill = segment.skill

        question_text = f"Is the robot's {arm_name} gripper open?"
        correct_answer = "Yes" if is_open else "No"
        incorrect_answer = "No" if is_open else "Yes"
        distractors = ["Cannot be determined", "Partially open"]

        hint = _generate_hint_gripper_state(
            arm_name=arm_name,
            is_open=is_open,
            task_name=trajectory.task_name,
            step_idx=step_idx,
            traj_length=trajectory.get_trajectory_length(),
            other_gripper_state=other_gripper_state,
            current_skill=current_skill,
            init_scene=getattr(trajectory, 'init_scene_text', '')
        )

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
            },
            hint=hint
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

        # Get current segment index
        current_segment_idx = 0
        for i, seg in enumerate(segments):
            if seg.contains(step_idx):
                current_segment_idx = i
                break

        task_name = trajectory.task_name.lower()
        question_text = f"The robot is to {task_name}. Has the robot successfully completed the task?"

        correct_answer = "Yes" if is_success else "No"
        incorrect_answer = "No" if is_success else "Yes"
        distractors = ["Cannot be determined", "Task was not attempted"]

        # Get final action description
        final_action = last_segment.action_text if last_segment else ""

        hint = _generate_hint_task_complete(
            is_success=is_success,
            task_name=task_name,
            is_in_last_segment=is_in_last_segment,
            step_idx=step_idx,
            traj_length=traj_length,
            num_segments=len(segments),
            current_segment_idx=current_segment_idx,
            init_scene=getattr(trajectory, 'init_scene_text', ''),
            final_action=final_action
        )

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
            },
            hint=hint
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

        # Extract action sequence for hint
        segments = trajectory.get_action_segments()
        action_sequence = [seg.action_text for seg in segments] if segments else None

        hint = _generate_hint_goal_config(
            task_name=trajectory.task_name,
            num_segments=len(segments) if segments else 0,
            init_scene=getattr(trajectory, 'init_scene_text', ''),
            action_sequence=action_sequence
        )

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
            },
            hint=hint
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

        # Get segment context for hint
        segments = trajectory.get_action_segments()
        segment_idx = 0
        prev_skill = ""
        next_skill = ""
        for i, seg in enumerate(segments):
            if seg.segment_id == segment.segment_id:
                segment_idx = i
                if i > 0:
                    prev_skill = segments[i-1].skill
                if i < len(segments) - 1:
                    next_skill = segments[i+1].skill
                break

        active_arm = detect_active_arm(segment.action_text) if segment.action_text else ""

        hint = _generate_hint_current_action(
            correct_description=correct_description,
            current_skill=current_skill,
            target_object=target_object,
            task_name=trajectory.task_name,
            segment_idx=segment_idx,
            num_segments=len(segments),
            prev_skill=prev_skill,
            next_skill=next_skill,
            active_arm=active_arm,
            frame_indices=frame_indices,
            init_scene=getattr(trajectory, 'init_scene_text', '')
        )

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
            },
            hint=hint
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

        hint = _generate_hint_next_action(
            current_description=current_description,
            next_description=next_description,
            task_name=trajectory.task_name,
            current_skill=current_segment.skill,
            next_skill=next_segment.skill,
            current_segment_idx=current_idx,
            num_segments=len(segments),
            target_object=current_object,
            next_target_object=next_object,
            init_scene=getattr(trajectory, 'init_scene_text', '')
        )

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
            },
            hint=hint
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

        # Extract more context for hint
        segments = trajectory.get_action_segments()
        action_sequence = [seg.action_text for seg in segments] if segments else None
        skill_sequence = [seg.skill for seg in segments] if segments else None

        # Extract objects from task name
        objects_involved = extract_objects_from_instruction(correct_instruction)

        hint = _generate_hint_task_instruction(
            task_name=correct_instruction,
            num_keyframes=len(keyframes),
            action_sequence=action_sequence,
            skill_sequence=skill_sequence,
            init_scene=getattr(trajectory, 'init_scene_text', ''),
            objects_involved=objects_involved
        )

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
            },
            hint=hint
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

        # Get context for hint
        segments = trajectory.get_action_segments()
        segment_idx = segment.segment_id if segment else 0
        target_object = extract_target_object(segment.action_text) if segment.action_text else ""
        active_arm = detect_active_arm(segment.action_text) if segment.action_text else ""

        hint = _generate_hint_current_skill(
            correct_skill=correct_skill,
            action_text=segment.action_text,
            task_name=trajectory.task_name,
            segment_idx=segment_idx,
            num_segments=len(segments) if segments else 0,
            target_object=target_object,
            active_arm=active_arm,
            init_scene=getattr(trajectory, 'init_scene_text', '')
        )

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
            },
            hint=hint
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

        hint = _generate_hint_active_arm(active_arm, segment.action_text)

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
            },
            hint=hint
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

        hint = _generate_hint_scene_description(init_scene)

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
            },
            hint=hint
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

        hint = _generate_hint_target_object(target_object, segment.action_text)

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
            },
            hint=hint
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

        hint = _generate_hint_action_count(num_segments, trajectory.task_name)

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
            },
            hint=hint
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

        hint = _generate_hint_is_handover(is_handover, segment.skill, segment.action_text)

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
            },
            hint=hint
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

        hint = _generate_hint_transition_frame(
            closest_transition['from_skill'],
            closest_transition['to_skill'],
            correct_frame_idx + 1
        )

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
            },
            hint=hint
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

        hint = _generate_hint_action_progress(segment.action_text, progress_pct, choices[correct_bucket_idx])

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
            },
            hint=hint
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

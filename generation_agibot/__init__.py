"""
AgiBotWorld VQA Generation Package

This package provides tools for generating Visual Question Answering (VQA)
data from the AgiBotWorld-Alpha dataset.

Modules:
    trajectory: AgiBotTrajectory class for loading and accessing episode data
    vqa: VQA class and generation functions
    utils: Utility functions for NLP, image processing, and skills
"""

from .trajectory import (
    AgiBotTrajectory,
    ActionSegment,
    load_trajectory,
    get_all_episodes,
    get_all_task_ids
)

from .vqa import (
    VQA,
    # Ported VQA generators
    vqa_robot_gripper_open,
    vqa_task_success_state,
    vqa_goal_configuration,
    vqa_action_understanding,
    vqa_next_action,
    vqa_trajectory_understanding,
    # New AgiBotWorld-specific generators
    vqa_skill_recognition,
    vqa_arm_coordination,
    vqa_scene_understanding,
    vqa_action_count,
    vqa_handover_detection,
    # Segmentation VQA generators
    vqa_transition_frame,
    vqa_progress_estimation,
    # Utilities
    VQA_GENERATORS,
    generate_all_vqas,
    save_vqa_dataset
)

from .utils import (
    AGIBOT_SKILLS,
    extract_target_object,
    extract_objects_from_instruction,
    extract_locations_from_instruction,
    detect_active_arm,
    get_skill_description
)

__all__ = [
    # Trajectory
    "AgiBotTrajectory",
    "ActionSegment",
    "load_trajectory",
    "get_all_episodes",
    "get_all_task_ids",
    # VQA
    "VQA",
    "vqa_robot_gripper_open",
    "vqa_task_success_state",
    "vqa_goal_configuration",
    "vqa_action_understanding",
    "vqa_next_action",
    "vqa_trajectory_understanding",
    "vqa_skill_recognition",
    "vqa_arm_coordination",
    "vqa_scene_understanding",
    "vqa_action_count",
    "vqa_handover_detection",
    "vqa_transition_frame",
    "vqa_progress_estimation",
    "VQA_GENERATORS",
    "generate_all_vqas",
    "save_vqa_dataset",
    # Utils
    "AGIBOT_SKILLS",
    "extract_target_object",
    "detect_active_arm",
    "get_skill_description",
]

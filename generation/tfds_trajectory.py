import tensorflow as tf
import numpy as np
import os
import re
import json
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from pathlib import Path

from trajectory import BaseTrajectory
from oxe_dataset_configs import OXE_DATASET_CONFIGS
from vqa import *  # Import VQA functions including enhance_vqa_with_metadata



class TFDSTrajectory(BaseTrajectory):
    """
    Implementation of BaseTrajectory for TFDS data only.
    This class handles trajectories from the TFDS dataset without
    downloading or processing the full raw data.
    """
    
    def __init__(self, episode_data: Dict, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize a trajectory from TFDS episode data.
        
        Args:
            episode_data: Dictionary containing TFDS episode metadata
            output_dir: Optional output directory for processed data
        """
        super().__init__(output_dir)
        
        self.episode_data = episode_data
        # Initialize steps data from pre-extracted data
        self.steps_data = []
        if "steps_data" in episode_data and episode_data["steps_data"]:
            self.steps_data = episode_data["steps_data"]
            self.trajectory_length = len(self.steps_data)
        
        self.language_instruction = episode_data["language_instruction"]
        self.target_object = get_target_object(self.language_instruction)
        self.metadata = dict()
        
        # Compute grasp phases to determine interested timesteps
        self.get_grasp_phases()
        
        # Determine interested timesteps
        self.interested_timesteps = self._get_interested_timesteps()
        
        # load only interested frames to cache
        self.frame_cache = CameraFrameCache()
        for camera_name in self.get_camera_names(): # primary, secondary, wrist
            for step_idx in range(self.trajectory_length):
                if step_idx in self.interested_timesteps:
                    image = self.episode_data["steps_data"][step_idx]["observation"][camera_name]
                    image = np.array(image)
                    # Convert BGR to RGB by swapping the channels
                    if len(image.shape) == 3 and image.shape[-1] == 3 and (camera_name != self.get_wrist_camera_name() or len(self.get_camera_names()) == 1):
                        image = image[..., ::-1]
                    self.frame_cache.add_frame(camera_name, step_idx, image, None, None)
                else:
                    # Store None for non-interested frames
                    self.frame_cache.add_frame(camera_name, step_idx, None, None, None)
                    
                    # Replace camera data in episode_data with None to save memory
                    self.episode_data["steps_data"][step_idx]["observation"][camera_name] = None

    def get_camera_names(self) -> List[str]:
        cameras = []
        if OXE_DATASET_CONFIGS[self.episode_data["dataset_name"]]["image_obs_keys"]["primary"] is not None: 
            cameras.append(
                OXE_DATASET_CONFIGS[self.episode_data["dataset_name"]]["image_obs_keys"]["primary"]
            )
        if OXE_DATASET_CONFIGS[self.episode_data["dataset_name"]]["image_obs_keys"]["secondary"] is not None:
            cameras.append(
                OXE_DATASET_CONFIGS[self.episode_data["dataset_name"]]["image_obs_keys"]["secondary"]
            )
        if OXE_DATASET_CONFIGS[self.episode_data["dataset_name"]]["image_obs_keys"]["wrist"] is not None:
            cameras.append(
                OXE_DATASET_CONFIGS[self.episode_data["dataset_name"]]["image_obs_keys"]["wrist"]
            )
        
        return cameras
    
    def get_wrist_camera_name(self) -> str:
        return OXE_DATASET_CONFIGS[self.episode_data["dataset_name"]]["image_obs_keys"]["wrist"]
    
    def get_language_instruction(self) -> str:
        """Get the natural language instruction for this trajectory"""
        return self.episode_data["language_instruction"]
    
    def get_episode_id(self) -> str:
        """Get the episode ID"""
        return self.episode_id
    
    def get_trajectory_length(self) -> int:
        """Get the length of the trajectory"""
        return self.trajectory_length
    
    def _extract_data_safely(self, step_data, path_list, default=None):
        """
        Safely extract data from a nested structure using a path list.
        
        Args:
            step_data: Nested data structure
            path_list: List of keys to follow in the structure
            default: Default value to return if path not found
            
        Returns:
            Extracted data or default value
        """
        current = step_data
        try:
            for key in path_list:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, (list, tuple)) and isinstance(key, int) and 0 <= key < len(current):
                    current = current[key]
                else:
                    return default
            return current
        except (KeyError, IndexError, TypeError):
            return default
    
    def get_gripper_position(self, step_idx: int) -> float:
        """
        Get the gripper position for a specific step.
        """
        # the gripper position is the last element of the action
        return float(self.steps_data[step_idx]["observation"]["gripper_position"])
        
        
    def get_non_image_data(self, step_idx: int) -> Dict:
        """
        Get non-image data for a specific step.
        For TFDS trajectories, this only includes limited information.
        
        Args:
            step_idx: The step index
            
        Returns:
            Dictionary with available step data
        """
        if step_idx >= self.trajectory_length:
            raise ValueError(f"Step index {step_idx} exceeds trajectory length {self.trajectory_length}")
            
        # For TFDS trajectories, we might only have limited data per step
        if self.steps_data and step_idx < len(self.steps_data):
            step_data = self.steps_data[step_idx]
            
            # Convert step data to a more usable format
            result = {
                "timestamp": {},
                "action": {},
                "robot_state": {},
                "cameras": {}
            }
            
            # Extract action data from any path that might have it
            action_data = self._extract_data_safely(step_data, ["action"])
            if action_data is not None:
                result["action"] = action_data
            
            # Extract robot state from any path that might have it
            robot_state = self._extract_data_safely(step_data, ["observation", "robot_state"])
            if robot_state is not None:
                result["robot_state"] = robot_state
            
            # Extract timestamp from any path that might have it
            timestamp = self._extract_data_safely(step_data, ["timestamp"])
            if timestamp is not None:
                result["timestamp"] = timestamp
            
            return result
        
        # If no step data available, return an empty structure
        return {
            "timestamp": {},
            "action": {},
            "robot_state": {},
            "cameras": {}
        }
        
    def is_task_successful(self) -> bool:
        """
        Check if the task is successful.
        """
        return True # open x datasets don't have a notion of task success
    
    def get_cartesian_state(self, step_idx: int) -> str:
        """
        Get the cartesian state of the robot.
        """
        # get this from proprio first three number 
        print(self.steps_data[step_idx]["observation"]["proprio"])
        return np.squeeze(self.steps_data[step_idx]["observation"]["proprio"])[:3]
    
    def get_image_data(self, camera_name: str, step_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get camera frames for a specific step and camera.
        For TFDS-only trajectories, this typically returns None as raw images 
        may not be directly available without downloading.
        
        Args:
            camera_name: Name of the camera
            step_idx: The step index
            
        Returns:
            Tuple of (left_image, right_image, depth_image) as numpy arrays, typically None
        """
        return self.frame_cache.get_frame(camera_name, step_idx)
    
    def close(self):
        """Clean up resources (minimal for TFDS trajectories)"""
        # TFDS trajectories don't typically need explicit cleanup
        pass 
    
    def save_frames_with_phase_info(self, output_dir: Union[str, Path]):
        """
        Save all image frames with phase information overlay to a directory.
        
        Args:
            output_dir: Directory to save the frames
        """
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure phases are computed
        if self._grasp_phases is None:
            self.get_grasp_phases()
        
        # Only save interested timesteps
        for step_idx in self.interested_timesteps:
            # Get the current phase for this step
            phase = self.get_phase_for_step(step_idx)
            
            # For each camera, get and save the image
            for camera_name in self.get_camera_names():
                # Get image data
                left_img, _, _ = self.get_image_data(camera_name, step_idx)
                
                if left_img is not None:
                    # Make a copy to avoid modifying the original
                    img_to_save = left_img.copy()
                    
                    # Ensure the image is in uint8 format and contiguous in memory
                    if img_to_save.dtype != np.uint8:
                        img_to_save = img_to_save.astype(np.uint8)
                    
                    # Ensure the image is contiguous in memory
                    if not img_to_save.flags.contiguous:
                        img_to_save = np.ascontiguousarray(img_to_save)
                    
                    # bgr to rgb
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB)
                    
                    # Add phase information as text overlay
                    # Parameters: image, text, position, font, scale, color, thickness
                    cv2.putText(
                        img_to_save, 
                        f"Phase: {phase}", 
                        (10, 30),  # position at top-left
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,  # font scale
                        (0, 255, 0),  # green color
                        2  # thickness
                    )
                    
                    # Add step index information
                    cv2.putText(
                        img_to_save, 
                        f"Step: {step_idx}", 
                        (10, 70),  # position below the phase text
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,  # font scale
                        (0, 255, 0),  # green color
                        2  # thickness
                    )
                    
                    # Add gripper position
                    gripper_pos = self.get_gripper_position(step_idx)
                    cv2.putText(
                        img_to_save, 
                        f"Gripper: {gripper_pos:.3f}", 
                        (10, 110),  # position below the step text
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1,  # font scale
                        (0, 255, 0),  # green color
                        2  # thickness
                    )
                    
                    # Save the image
                    file_name = f"step_{step_idx:04d}_{camera_name}_{phase}.png"
                    file_path = output_dir / file_name
                    
                    # Convert RGB to BGR for OpenCV imwrite
                    img_to_save_bgr = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(file_path), img_to_save_bgr)
                    
        print(f"Saved {len(self.interested_timesteps)} interested frames with phase information to {output_dir}")
    
    def process_dataset(self, max_steps=None):
        """
        Process the dataset and save frames with phase information
        
        Args:
            max_steps: Maximum number of steps to process
            
        Returns:
            Dictionary with processing results
        """
        # Call the parent class process_dataset method first
        results = super().process_dataset(max_steps)
        
        # Create a frames directory to store the annotated frames
        # frames_dir = Path(self.output_dir) / "frames"
        # # Save frames with phase information
        # self.save_frames_with_phase_info(frames_dir)
        
        return results
    
    def generate_vqa_data(self) -> List[VQA]:
        """
        Generate VQA data for this trajectory
            
        Returns:
            List of VQA instances
        """
        vqa_list = []
        
        # Extract VLM metadata if not already done
        # if not any(self.vlm_metadata.values()):
        #     self.extract_vlm_metadata()
        
        # Use the same interested timesteps that were identified during initialization
        for timestep in self.interested_timesteps:
            
            vqa = is_stable_grasp(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)
                    
            vqa = vqa_task_success_state(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)
                
            vqa = vqa_robot_gripper_open(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)
                
            vqa = vqa_object_reachable(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)


            vqa = vqa_goal_configuration(self)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)

            vqa = vqa_action_understanding(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)

            vqa = vqa_next_action(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)

        # Remove None entries and return
        vqa_list = [vqa for vqa in vqa_list if vqa is not None]
        return vqa_list
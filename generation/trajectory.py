from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from pathlib import Path
from utils import *  # Import camera and utility functions
from vqa import *

class BaseTrajectory(ABC):
    """
    Abstract base class for robot trajectories that provides a common interface
    for working with trajectory data from different sources.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the trajectory.
        
        Args:
            output_dir: Optional path to store processed data
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.language_instruction = ""
        self.episode_id = ""
        self.trajectory_length = 0
        
        self.frame_cache = CameraFrameCache()
        self.step_cache = StepDataCache()
        
        # Grasp phases
        self._grasp_phases = None  # Will store phase information once computed
        self._step_to_phase_map = None  # Will map step_idx -> phase_name for quick lookup
        
        self.grasp_threshold = 0.2
        self.contact_threshold = 0.9
        
        self.interested_timesteps = set()
        
        # VLM metadata extracted from frames
        self.vlm_metadata = {
            "physical_location": "",
            "task": "",
            "target_object": "",
            "location": ""
        }
        
    def extract_vlm_metadata(self, step_idx=None, camera_name=None):
        """
        Extract metadata using VLM (Vision Language Model) from a frame.
        
        Args:
            step_idx: Optional specific step to use for metadata extraction.
                     If None, a random step will be chosen from interested_timesteps.
            camera_name: Optional specific camera to use. 
                        If None, will try primary camera first, then others.
                        
        Returns:
            Dictionary with extracted metadata
        """
        try:
            # Ensure we have interested timesteps
            if not self.interested_timesteps and step_idx is None:
                self._get_interested_timesteps()
                
            # Select a step_idx if not provided
            if step_idx is None:
                if self.interested_timesteps:
                    step_idx = list(self.interested_timesteps)[0]  # Take first interested timestep
                else:
                    step_idx = self.trajectory_length // 2  # Default to middle of trajectory
            
            # Get available cameras
            cameras = self.get_camera_names()
            
            # Select camera_name if not provided
            if camera_name is None and cameras:
                # Try to find a non-wrist camera first
                for cam in cameras:
                    if "wrist" not in cam.lower():
                        camera_name = cam
                        break
                # If not found, use the first available camera
                if camera_name is None and cameras:
                    camera_name = cameras[0]
                    
            # Safety check - make sure we have a camera
            if not camera_name or not cameras:
                print(f"No cameras available for VLM metadata extraction")
                return self.vlm_metadata
                    
            # Get the frame
            left_img, _, _ = self.get_image_data(camera_name, step_idx)
            
            if left_img is None:
                print(f"No frame found for camera {camera_name} at step {step_idx}")
                return self.vlm_metadata
                
            # Extract metadata from the frame
            try:
                metadata = extract_metadata_from_frame(left_img, self.get_language_instruction())
                
                # Update and store the metadata
                self.vlm_metadata.update(metadata)
                
                # Also update metadata in VQA data if it's being used
                if hasattr(self, 'metadata') and isinstance(self.metadata, dict):
                    self.metadata.update({
                        "vlm_physical_location": metadata["physical_location"],
                        "vlm_task": metadata["task"],
                        "vlm_target_object": metadata["target_object"],
                        "vlm_location": metadata["location"]
                    })
            except Exception as e:
                print(f"Error in VLM metadata extraction: {e}")
                import traceback
                traceback.print_exc()
                
            return self.vlm_metadata
            
        except Exception as e:
            print(f"Error in extract_vlm_metadata: {e}")
            import traceback
            traceback.print_exc()
            return self.vlm_metadata
    
    @abstractmethod
    def get_language_instruction(self) -> str:
        """Get the natural language instruction for this trajectory."""
        pass
    
    @abstractmethod
    def get_episode_id(self) -> str:
        """Get the episode ID."""
        pass
    
    @abstractmethod
    def get_trajectory_length(self) -> int:
        """Get the length of the trajectory."""
        pass
    
    @abstractmethod
    def get_non_image_data(self, step_idx: int) -> Dict:
        """
        Get all non-image data for a specific step.
        
        Args:
            step_idx: The step index
            
        Returns:
            Dictionary with step data including robot state, action, etc.
        """
        pass
    
    @abstractmethod
    def get_image_data(self, camera_name: str, step_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get camera frames (left, right, depth) for a specific step and camera.
        
        Args:
            camera_name: Name of the camera
            step_idx: The step index
            
        Returns:
            Tuple of (left_image, right_image, depth_image) as numpy arrays
        """
        pass

    @abstractmethod
    def get_gripper_position(self, step_idx: int) -> float:
        """
        Get the gripper position for a specific step.
        """
        pass
    
    @abstractmethod
    def get_camera_names(self) -> List[str]:
        """
        Get the names of the cameras in the trajectory.
        """
        pass
    
    @abstractmethod
    def generate_vqa_data(self) -> List[VQA]:
        """
        Generate VQA data for this trajectory.
        """
        pass
    
    def process_dataset(self, max_steps=None):
        """
        Process the entire dataset with the standard pipeline
        
        Args:
            max_steps: Maximum number of steps to process
            
        Returns:
            Dictionary with processing results
        """
        if not self.output_dir:
            raise ValueError("Output directory must be provided for processing dataset")
            
        if max_steps is None:
            max_steps = self.trajectory_length
        else:
            max_steps = min(max_steps, self.trajectory_length)
        
        # Precompute grasp phases for all steps
        print("Precomputing grasp phases...")
        # this assume the grasp phases are already computed in the constructor
        # self.precompute_phases()
        grasp_phases = self._grasp_phases
        
        # Extract metadata using VLM
        # print("Extracting metadata using VLM...")
        # self.extract_vlm_metadata()
        
        # Generate VQA data
        vqa_list = self.generate_vqa_data()

        save_vqa_dataset(vqa_list, self.output_dir, self.metadata)

        return {
            "max_steps": max_steps, 
            "phases": grasp_phases,
            "phase_map": self._step_to_phase_map,
            "vqa_list": vqa_list,
            "vlm_metadata": self.vlm_metadata
        }
        
    @abstractmethod
    def is_task_successful(self) -> bool:
        """
        Check if the task is successful.
        """
        pass
    
    @abstractmethod
    def get_cartesian_state(self, step_idx: int) -> str:
        """
        Get the cartesian state of the robot.
        """
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass 

    def precompute_phases(self):
        """
        Precompute all grasp phases for faster access later.
        
        Args:
            grasp_threshold: Threshold close to 0 to determine when gripper is considered open
            contact_threshold: Threshold close to 1 to determine when gripper is considered closed
            
        Returns:
            Dictionary mapping each step to its phase
        """
        self.get_grasp_phases(recompute=True)
        return self._step_to_phase_map
    
    
    def get_grasp_phases(self, recompute=False) -> Dict:
        """
        Determine different phases of the grasp in this trajectory.
        Results are cached for faster future access.
        
        Args:
            recompute: If True, force recomputation even if already cached
            grasp_threshold: Threshold close to 0 to determine when gripper is considered open
            contact_threshold: Threshold close to 1 to determine when gripper is considered closed
            
        Returns:
            Dictionary with identified grasp phases and their ranges
        """
        if self._grasp_phases is None or recompute:
            # Compute phases using the utility function
            self._grasp_phases = determine_grasp_phases(self, self.grasp_threshold, self.contact_threshold)
            
            # Create a step_idx -> phase_name mapping for quick lookup
            self._step_to_phase_map = {}
            for phase_name, phase_ranges in self._grasp_phases["phase_ranges"].items():
                for start_idx, end_idx in phase_ranges:
                    for step_idx in range(start_idx, end_idx + 1):
                        self._step_to_phase_map[step_idx] = phase_name
        
        return self._grasp_phases
    
    
    def get_phase_for_step(self, step_idx: int) -> str:
        """
        Get the grasp phase name for a specific step.
        
        Args:
            step_idx: The step index to query
            
        Returns:
            The phase name as a string (e.g., "pre_grasp", "contact", etc.)
        """
        # Ensure phases are computed
        if self._grasp_phases is None:
            self.get_grasp_phases()
            
        # Return the phase for this step if available
        if self._step_to_phase_map and step_idx in self._step_to_phase_map:
            return self._step_to_phase_map[step_idx]
        
        # If step is out of range or not found in any phase
        if step_idx < 0 or step_idx >= self.trajectory_length:
            return "out_of_range"
        
        return "unknown"

    
    def _get_interested_timesteps(self) -> Set[int]:
        """
        Get the set of interested timesteps based on grasp phases.
        
        Returns:
            Set of interested timestep indices
        """
        # Check if interested_timesteps is already populated
        if self.interested_timesteps:
            return self.interested_timesteps
            
        interested_timesteps = set()
        
        # Ensure grasp phases are computed
        if not self._grasp_phases:
            self.get_grasp_phases(recompute=True)
            
        # Sample timesteps from each phase range
        for phase_name, phase_ranges in self._grasp_phases["phase_ranges"].items():
            for start_idx, end_idx in phase_ranges:
                l = sorted([start_idx, end_idx])
                if l[0] == l[1]:
                    interested_timesteps.add(l[0])
                else:
                    rd = np.random.randint(l[0], l[1])
                    interested_timesteps.add(rd)
    
        # return interested_timesteps
        
        
        # add beginning and end of trajectory
        interested_timesteps.add(0)
        interested_timesteps.add(self.trajectory_length - 1)
        
        self.interested_timesteps = interested_timesteps
        return interested_timesteps
        
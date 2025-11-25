import tensorflow as tf
import numpy as np
import os
import h5py
import json
import cv2
from pathlib import Path
import glob
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from scipy.spatial.transform import Rotation
import random
import requests
from trajectory import BaseTrajectory
from utils import *  # Import camera and utility functions
from vqa import *  # Import VQA functions including enhance_vqa_with_metadata
import shutil

# URLs to the camera extrinsics JSON files on Hugging Face
HF_JSON_URLS = {
    "cam2base_extrinsics": "https://huggingface.co/KarlP/droid/resolve/main/cam2base_extrinsics.json",
    "cam2cam_extrinsics": "https://huggingface.co/KarlP/droid/resolve/main/cam2cam_extrinsics.json",
    "cam2base_extrinsic_superset": "https://huggingface.co/KarlP/droid/resolve/main/cam2base_extrinsic_superset.json"
}

class DROIDTrajectory(BaseTrajectory):
    """
    Class for working with DROID trajectories that combines TFDS metadata and raw trajectory data.
    This implementation can download and process the full raw data for rich interactions.
    
    Interfaces:
    - get_language_instruction()
    - get_trajectory_length()

    - get_non_image_data(step_idx: int) -> Dict
    - get_image_data_from_all_cameras(step_idx: int) -> Dict[str, Dict[str, Optional[np.ndarray]]]
    - get_image_data(camera_name: str, step_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]

    - get_grasp_phases(recompute=False, grasp_threshold=0.05, contact_threshold=0.9) -> Dict
    - get_phase_for_step(step_idx: int) -> str

    - get_episode_id()
    - close() -> None
    """
    
    def __init__(self, episode_data: Optional[Dict] = None, 
                 scene_path: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None, 
                 save_original_image_to_disk: bool = False, 
                 gsutil_path: Optional[str] = None,
                 use_hf_camera_extrinsics: bool = True):
        """
        Initialize a DROIDTrajectory with either TFDS episode data and/or path to raw data.
        
        Args:
            episode_data: Dictionary containing TFDS episode metadata
            scene_path: Path to raw trajectory data
            output_dir: Optional output directory for extracted data
            save_original_image_to_disk: Whether to save original images to disk
            gsutil_path: Optional path to GCS storage
            use_hf_camera_extrinsics: Whether to use improved camera extrinsics from Hugging Face
        """
        # Initialize the base class
        super().__init__(output_dir)
        
        # Initialize data structures
        self.episode_data = episode_data
        self.scene_path = Path(scene_path) if scene_path else None
        self.save_original_image_to_disk = save_original_image_to_disk 
        self.gsutil_path = gsutil_path
        self.use_hf_camera_extrinsics = use_hf_camera_extrinsics

        # Trajectory data
        self.metadata = None
        self.trajectory = None
        self.action = None
        self.robot_state = None
        
        # Camera data
        self.cameras = {}
        self.camera_serials = {}
        self.camera_frames = {}
        
        # Camera extrinsics from Hugging Face
        self.hf_camera_extrinsics = {
            "cam2base_extrinsics": None,
            "cam2cam_extrinsics": None,
            "cam2base_extrinsic_superset": None
        }

        # Load data if paths are provided
        if episode_data:
            self._load_tfds_data()

        if scene_path:
            self._load_raw_data()
            
        # Load camera extrinsics from Hugging Face if requested
        if use_hf_camera_extrinsics:
            self._load_hf_camera_extrinsics()

    def get_language_instruction(self) -> str:
        """Get the natural language instruction for this trajectory"""
        return self.language_instruction.lower()
    
    def get_episode_id(self) -> str:
        """Get the episode ID"""
        return self.episode_id
    
    def get_trajectory_length(self) -> int:
        """Get the length of the trajectory"""
        return self.trajectory_length
    
    def get_camera_names(self) -> List[str]:
        """Get list of available camera names"""
        return list(self.cameras.keys())
    
    def _load_hf_camera_extrinsics(self):
        """
        Download and load camera extrinsics JSON files from Hugging Face.
        These contain more accurate camera calibration data.
        """
        # Create cache directory if it doesn't exist
        cache_dir = Path("./huggingface_cache")
        cache_dir.mkdir(exist_ok=True)
        
        for file_key, url in HF_JSON_URLS.items():
            cache_path = cache_dir / f"{file_key}.json"
            
            # Download the file if not already cached
            if not cache_path.exists():
                try:
                    print(f"Downloading {file_key} from Hugging Face...")
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(cache_path, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {file_key} successfully.")
                    else:
                        print(f"Failed to download {file_key}: {response.status_code}")
                        continue
                except Exception as e:
                    print(f"Error downloading {file_key}: {e}")
                    continue
            
            # Load the JSON file
            try:
                with open(cache_path, 'r') as f:
                    self.hf_camera_extrinsics[file_key] = json.load(f)
                print(f"Loaded {file_key} with {len(self.hf_camera_extrinsics[file_key])} entries.")
            except Exception as e:
                print(f"Error loading {file_key}: {e}")
    
    def get_hf_camera_extrinsics(self, camera_name: str, step_idx: int) -> Dict:
        """
        Get the camera extrinsics from Hugging Face JSON files for a specific camera and step.
        
        Args:
            camera_name: Name of the camera
            step_idx: Step index
            
        Returns:
            Dictionary with camera extrinsics or None if not found
        """
        if not self.use_hf_camera_extrinsics or self.episode_id == "":
            return None
        
        serial = self.camera_serials.get(camera_name)
        if not serial:
            return None
                
        # First try cam2base_extrinsic_superset.json
        if self.hf_camera_extrinsics["cam2base_extrinsic_superset"]:
            if self.episode_id in self.hf_camera_extrinsics["cam2base_extrinsic_superset"]:
                entry = self.hf_camera_extrinsics["cam2base_extrinsic_superset"][self.episode_id]
                if str(serial) in entry:
                    # The extrinsics are stored as a flat array [x, y, z, rx, ry, rz]
                    extrinsics = entry[str(serial)]
                    return  {
                        "left": extrinsics,
                        "right": extrinsics
                    }
        
        # Then try cam2base_extrinsics.json
        if self.hf_camera_extrinsics["cam2base_extrinsics"]:
            if self.episode_id in self.hf_camera_extrinsics["cam2base_extrinsics"]:
                entry = self.hf_camera_extrinsics["cam2base_extrinsics"][self.episode_id]
                if str(serial) in entry:
                    extrinsics = entry[str(serial)]
                    return {
                        "left": extrinsics,
                        "right": extrinsics
                    }
        
        # Finally try cam2cam_extrinsics.json
        if self.hf_camera_extrinsics["cam2cam_extrinsics"]:
            if self.episode_id in self.hf_camera_extrinsics["cam2cam_extrinsics"]:
                entry = self.hf_camera_extrinsics["cam2cam_extrinsics"][self.episode_id]
                if str(serial) in entry:
                    extrinsics = entry[str(serial)]
                    return {
                        "left": extrinsics,
                        "right": extrinsics
                    }
        
        return None
    
    def get_non_image_data(self, step_idx: int) -> Dict:
        """
        Get all data for a specific step (cached)
        
        Args:
            step_idx: The step index
            
        Returns:
            Dictionary with step data including robot state, action, etc.
        """
        if step_idx >= self.trajectory_length:
            raise ValueError(f"Step index {step_idx} exceeds trajectory length {self.trajectory_length}")
            
        # Check cache first
        cached_data = self.step_cache.get_step(step_idx)
        if cached_data:
            return cached_data
            
        # Extract step data if not in cache
        step_data = {
            "timestamp": {},
            "action": {},
            "robot_state": {},
            "cameras": {}
        }
        
        # Add camera intrinsics
        for camera_name, camera in self.cameras.items():
            if camera_name not in step_data["cameras"]:
                step_data["cameras"][camera_name] = {}
            
            step_data["cameras"][camera_name]["intrinsics"] = {
                "left": camera.left_intrinsic_mat.tolist() if camera.left_intrinsic_mat is not None else None,
                "right": camera.right_intrinsic_mat.tolist() if camera.right_intrinsic_mat is not None else None
            }
        
        # Extract timestamps
        if "timestamp" in self.trajectory["observation"]:
            ts = self.trajectory["observation"]["timestamp"]
            if "robot_state" in ts:
                step_data["timestamp"]["robot_timestamp_nanos"] = ts["robot_state"]["robot_timestamp_nanos"][step_idx]
                step_data["timestamp"]["robot_timestamp_seconds"] = ts["robot_state"]["robot_timestamp_seconds"][step_idx]
            
            # Camera timestamps
            if "cameras" in ts:
                step_data["timestamp"]["cameras"] = {}
                for camera_name, serial in self.camera_serials.items():
                    capture_key = f"{serial}_estimated_capture"
                    if capture_key in ts["cameras"]:
                        step_data["timestamp"]["cameras"][camera_name] = ts["cameras"][capture_key][step_idx]
        
        # Extract action data
        for key in self.action.keys():
            if isinstance(self.action[key], h5py.Dataset):
                try:
                    step_data["action"][key] = self.action[key][step_idx].tolist()
                except (IndexError, ValueError) as e:
                    print(f"Error extracting action/{key}: {e}")
        
        # Extract robot state
        for key in self.robot_state.keys():
            if isinstance(self.robot_state[key], h5py.Dataset):
                try:
                    step_data["robot_state"][key] = self.robot_state[key][step_idx].tolist()
                except (IndexError, ValueError) as e:
                    print(f"Error extracting robot_state/{key}: {e}")
        
        # Extract camera extrinsics
        if "camera_extrinsics" in self.trajectory["observation"]:
            step_data["cameras"]["extrinsics"] = {}
            extrinsics = self.trajectory["observation"]["camera_extrinsics"]
        
        for camera_name, serial in self.camera_serials.items():
            step_data["cameras"]["extrinsics"][camera_name] = {}
            
            # Try to get HF extrinsics first if enabled
            if self.use_hf_camera_extrinsics:
                hf_extrinsics = self.get_hf_camera_extrinsics(camera_name, step_idx)
                if type(hf_extrinsics) == dict:
                    if "left" in hf_extrinsics:
                        step_data["cameras"]["extrinsics"][camera_name]["left"] = hf_extrinsics["left"]
                    if "right" in hf_extrinsics:
                        step_data["cameras"]["extrinsics"][camera_name]["right"] = hf_extrinsics["right"]
                    
                    # If both left and right extrinsics are available from HF, continue to next camera
                    if "left" in hf_extrinsics and "right" in hf_extrinsics:
                        continue            
            
            # Fall back to trajectory file extrinsics if HF extrinsics not available
            if "camera_extrinsics" in self.trajectory["observation"]:
                extrinsics = self.trajectory["observation"]["camera_extrinsics"]
                
                left_key = f"{serial}_left"
                right_key = f"{serial}_right"
                
                if left_key in extrinsics and "left" not in step_data["cameras"]["extrinsics"][camera_name]:
                    step_data["cameras"]["extrinsics"][camera_name]["left"] = extrinsics[left_key][step_idx].tolist()
                
                if right_key in extrinsics and "right" not in step_data["cameras"]["extrinsics"][camera_name]:
                    step_data["cameras"]["extrinsics"][camera_name]["right"] = extrinsics[right_key][step_idx].tolist()

        # Add to cache
        self.step_cache.add_step(step_idx, step_data)
        return step_data
    
    def get_image_data_from_all_cameras(self, step_idx: int) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        Get image data for a specific step
        
        Args:
            step_idx: The step index
        """
        ret = {}
        for camera_name in self.cameras:
            left_img, right_img, depth_img = self.get_image_data(camera_name, step_idx)
            ret[camera_name] = {
                "left": left_img,
                "right": right_img,
                "depth": depth_img
            }
        return ret


    def get_image_data(self, camera_name: str, step_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get camera frames (left, right, depth) for a specific step and camera
        
        Args:
            camera_name: Name of the camera
            step_idx: The step index
            
        Returns:
            Tuple of (left_image, right_image, depth_image) as numpy arrays
        """
        # Check if frame is in cache
        cached_frame = self.frame_cache.get_frame(camera_name, step_idx)
        # if cached_frame[0] is not None:
        return cached_frame
            
        print(f"Frame not in cache for {camera_name} at step {step_idx} for {self.interested_timesteps}")

        # get it 
        camera = self.cameras[camera_name]
        
        # # Reset camera to beginning then seek to desired frame
        # self._reset_cameras()
        
        # # Skip frames until we reach the desired step
        # for current_step in range(step_idx + 1):
        #     frames = camera.get_next_frame()
        #     if current_step == step_idx and frames:
        #         left_img, right_img, depth_img = frames
                
        #         # Cache the retrieved frame
        #         self.frame_cache.add_frame(camera_name, step_idx, left_img, right_img, depth_img, None, None, None)
                
        #         return left_img, right_img, depth_img
        
        # print(f"Could not find frame for {camera_name} at step {step_idx}")
        return None, None, None
        
    def get_gripper_position(self, step_idx: int) -> float:
        """
        Get the gripper position for a specific step.
        """
        step_data = self.get_non_image_data(step_idx)
        gripper_pos = step_data["robot_state"]["gripper_position"]
        return gripper_pos

    
    def _reset_cameras(self):
        """Reset cameras to the beginning of the recording"""
        for camera_name, camera in self.cameras.items():
            if hasattr(camera, "zed"):
                camera.zed.close()
                
                import pyzed.sl as sl
                init_params = sl.InitParameters()
                svo_path = camera.recordings / "SVO" / f"{camera.serial}.svo"
                init_params.set_from_svo_file(str(svo_path))
                init_params.depth_mode = sl.DEPTH_MODE.QUALITY
                init_params.svo_real_time_mode = False
                init_params.coordinate_units = sl.UNIT.METER
                init_params.depth_minimum_distance = 0.2
                
                camera.zed = sl.Camera()
                camera.zed.open(init_params)
            elif hasattr(camera, "cap"):
                camera.cap.release()
                
                mp4_path = None
                if (camera.recordings / "MP4" / f'{camera.serial}-stereo.mp4').exists():
                    mp4_path = camera.recordings / "MP4" / f'{camera.serial}-stereo.mp4'
                elif (camera.recordings / "MP4" / f'{camera.serial}.mp4').exists():
                    mp4_path = camera.recordings / "MP4" / f'{camera.serial}.mp4'
                
                camera.cap = cv2.VideoCapture(str(mp4_path))
    
    def get_camera_names(self) -> List[str]:
        """
        Get the names of the cameras in the trajectory.
        """
        return list(self.cameras.keys())

    def is_task_successful(self) -> bool:
        """
        Check if the task is successful.
        """
        return 'success' in self.gsutil_path.lower() if self.gsutil_path else False

    def get_cartesian_state(self, step_idx: int) -> str:
        """
        Get the cartesian state of the robot.
        """
        step_data = self.get_non_image_data(step_idx)
        return step_data["robot_state"]["cartesian_position"]

    def _load_tfds_data(self):
        """Load data from TFDS episode_data"""
        if not self.episode_data:
            return
        
        # Extract language instruction if available
        if "language_instruction" in self.episode_data:
            self.language_instruction = self.episode_data["language_instruction"]
        
        # Extract file path and derive episode ID
        if "episode_metadata" in self.episode_data and "file_path" in self.episode_data["episode_metadata"]:
            file_path = self.episode_data["episode_metadata"]["file_path"]
            
            # Try to extract episode ID from file path
            import re
            episode_id_match = re.search(r'([^/]+)/trajectory\.h5$', file_path)
            if episode_id_match:
                self.episode_id = episode_id_match.group(1)
            else:
                # Fallback to using the last part of the path
                self.episode_id = file_path.split('/')[-2] if "/" in file_path else "unknown"
    
    def _load_raw_data(self):
        """Load raw trajectory data from scene_path"""
        if not self.scene_path or not self.scene_path.exists():
            return
        
        self._load_metadata()
        self._load_trajectory()
        self._load_cameras()
            
        # Compute grasp phases to determine interested timesteps
        self._grasp_phases = self.get_grasp_phases(recompute=True)
        
        # Determine interested timesteps
        self.interested_timesteps = self._get_interested_timesteps()
        
        # Extract all frames AFTER determining interested timesteps
        self.extract_all_frames(save_to_disk=self.save_original_image_to_disk)

    def _load_metadata(self):
        """Load metadata from JSON file"""
        json_file_paths = glob.glob(str(self.scene_path) + "/*.json")
        if len(json_file_paths) < 1:
            raise Exception(f"Unable to find metadata file at '{self.scene_path}'")

        with open(json_file_paths[0], "r") as metadata_file:
            self.metadata = json.load(metadata_file)
            
        # Get camera names and serials
        self.camera_serials = {}
        for camera_name in CAMERA_NAMES:
            serial_key = f"{camera_name}_cam_serial"
            if serial_key in self.metadata:
                self.camera_serials[camera_name] = self.metadata[serial_key]
        
        # Extract episode ID from metadata file name
        metadata_filename = os.path.basename(json_file_paths[0])
        if metadata_filename.startswith("metadata_"):
            # Extract the ID after the last underscore before the .json extension
            self.episode_id = metadata_filename[:-5].split("_")[-1]
            print(f"Extracted episode ID from metadata filename: {self.episode_id}")
    
    def _load_trajectory(self):
        """Load trajectory data from H5 file"""
        h5_file = self.scene_path / "trajectory.h5"
        if not h5_file.exists():
            raise Exception(f"Trajectory file not found: {h5_file}")
            
        self.trajectory = h5py.File(str(h5_file), "r")
        self.action = self.trajectory['action']
        self.robot_state = self.trajectory['observation']['robot_state']
        
        if "trajectory_length" in self.metadata:
            self.trajectory_length = self.metadata["trajectory_length"]
        else:
            # Try to determine from trajectory data
            for key in self.action.keys():
                if isinstance(self.action[key], h5py.Dataset):
                    self.trajectory_length = self.action[key].shape[0] 
                    break
    
    def _load_cameras(self):
        """Initialize camera readers"""
        if not self.camera_serials:
            return
            
        for camera_name, serial in self.camera_serials.items():
            try:
                self.cameras[camera_name] = StereoCamera(
                    self.scene_path / "recordings",
                    serial
                )
            except Exception as e:
                print(f"Error loading camera {camera_name}: {e}")
    

    def extract_all_frames(self, max_steps=None, save_to_disk=False):
        """
        Extract camera frames, but only for interested timesteps to save memory
        
        Args:
            max_steps: Maximum number of steps to extract (None for all)
            save_to_disk: Whether to save images to disk
            
        Returns:
            Dictionary with information about extracted frames
        """
        if max_steps is None:
            max_steps = self.trajectory_length
        else:
            max_steps = min(max_steps, self.trajectory_length)
                        
        # Reset cameras
        self._reset_cameras()
        
        # Create output directory if saving to disk
        if save_to_disk and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
        min_step_idx = 0
        for step_idx in range(max_steps):
            # Only process interested timesteps for memory efficiency
            is_interested = step_idx in self.interested_timesteps
            
            for camera_name, camera in self.cameras.items():
                camera_frames = camera.get_next_frame()
                
                if camera_frames:
                    left, right, depth = camera_frames
                    
                    # Paths to save images (if needed)
                    left_path = None
                    right_path = None
                    depth_path = None
                    
                    if is_interested and save_to_disk and self.output_dir:
                        camera_dir = os.path.join(self.output_dir, "images", f"step_{step_idx:04d}", camera_name)
                        os.makedirs(camera_dir, exist_ok=True)

                        if left is not None:
                            left_path = os.path.join(camera_dir, "left.png")
                            cv2.imwrite(left_path, left)
                            left_path = os.path.relpath(left_path, self.output_dir)

                        if right is not None:
                            right_path = os.path.join(camera_dir, "right.png")
                            cv2.imwrite(right_path, right)
                            right_path = os.path.relpath(right_path, self.output_dir)
                            
                        if depth is not None:
                            depth_path = os.path.join(camera_dir, "depth.png")
                            # Normalize depth for visualization
                            if depth.dtype != np.uint16:
                                depth_vis = np.clip(depth, 0, 65535).astype(np.uint16)
                                cv2.imwrite(depth_path, depth_vis)
                            else:
                                cv2.imwrite(depth_path, depth)
                            depth_path = os.path.relpath(depth_path, self.output_dir)
                    
                    # Add to cache - only store actual frames for interested timesteps
                    # save also first frame and last frame and half of the trajectory
                    self.frame_cache.add_frame(camera_name, step_idx, left, right, depth, left_path, right_path, depth_path)
                    
                    min_step_idx = step_idx
        
        if min_step_idx < max_steps:
            print(f"Warning: Only {min_step_idx + 1} frames were extracted out of {max_steps}, using {min_step_idx + 1} as trajectory length")
            self.trajectory_length = min_step_idx + 1
            
        print(f"Cached {(self.interested_timesteps)} interested frames out of {self.trajectory_length} total frames")
        return {"max_steps": max_steps}
    
    def visualize_action(self, step_idx, save_to_disk=True, num_skip_frames=10):
        """
        Visualize robot action at the given step index
        
        Args:
            step_idx: The step index to visualize
            save_to_disk: Whether to save visualizations to disk
            num_skip_frames: Number of frames to skip for displacement arrow
            
        Returns:
            Dictionary with visualization results
        """
        # Implementation similar to RawDatasetExtractor.visualize_action
        # But using the new class structure
        future_step_idx = min(step_idx + num_skip_frames, self.trajectory_length - 1)
        
        # Get step data
        step_data = self.get_non_image_data(step_idx)
        future_step_data = self.get_non_image_data(future_step_idx)

        # Get positions
        current_pos = np.array(step_data["robot_state"]["cartesian_position"])
        target_pos = np.array(step_data["action"]["target_cartesian_position"])
        future_pos = np.array(future_step_data["robot_state"]["cartesian_position"])
        
        results = {}
        
        # For each camera, project points and draw on images
        for camera_name in self.cameras.keys():
            # Get frames
            left_img, right_img, _ = self.get_image_data(camera_name, step_idx)
            
            if left_img is None or right_img is None:
                results[f"{camera_name}_status"] = "Images not available"
                continue
                
            # Get camera parameters
            try:
                left_intrinsic = np.array(step_data["cameras"][camera_name]["intrinsics"]["left"])
                right_intrinsic = np.array(step_data["cameras"][camera_name]["intrinsics"]["right"])
                
                left_extrinsic = np.array(step_data["cameras"]["extrinsics"][camera_name]["left"])
                right_extrinsic = np.array(step_data["cameras"]["extrinsics"][camera_name]["right"])
                
                # Project positions
                left_current_px = project_point_to_image(current_pos, left_extrinsic, left_intrinsic)
                left_target_px = project_point_to_image(target_pos, left_extrinsic, left_intrinsic)
                left_future_px = project_point_to_image(future_pos, left_extrinsic, left_intrinsic)
                
                right_current_px = project_point_to_image(current_pos, right_extrinsic, right_intrinsic)
                right_target_px = project_point_to_image(target_pos, right_extrinsic, right_intrinsic)
                right_future_px = project_point_to_image(future_pos, right_extrinsic, right_intrinsic)

                # Draw visualizations
                left_vis_img = left_img.copy()
                right_vis_img = right_img.copy()
                
                draw_projected_points(left_vis_img, left_current_px, left_target_px, left_future_px, 
                                      current_pos, target_pos, left_extrinsic, left_intrinsic)
                draw_projected_points(right_vis_img, right_current_px, right_target_px, right_future_px,
                                      current_pos, target_pos, right_extrinsic, right_intrinsic)
                
                # Save visualizations
                if save_to_disk and self.output_dir:
                    viz_dir = os.path.join(self.output_dir, "visualizations", f"step_{step_idx:04d}", camera_name)
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    left_viz_path = os.path.join(viz_dir, "left_viz.png")
                    right_viz_path = os.path.join(viz_dir, "right_viz.png")
                    
                    cv2.imwrite(left_viz_path, left_vis_img)
                    cv2.imwrite(right_viz_path, right_vis_img)
                    
                    results[f"{camera_name}_left_viz_path"] = os.path.relpath(left_viz_path, self.output_dir)
                    results[f"{camera_name}_right_viz_path"] = os.path.relpath(right_viz_path, self.output_dir)
                else:
                    results[f"{camera_name}_left_viz"] = left_vis_img
                    results[f"{camera_name}_right_viz"] = right_vis_img
                    
            except (KeyError, ValueError) as e:
                results[f"{camera_name}_status"] = f"Error projecting points: {e}"
                continue
        
        return results
    
    def visualize_trajectory(self, start_idx=0, end_idx=None, image_idx=None, step_interval=1, save_to_disk=True):
        """
        Visualize robot trajectory on camera images
        
        Args:
            start_idx: Starting step index
            end_idx: Ending step index (None for all)
            image_idx: Index of frame to draw trajectory on (None for middle of range)
            step_interval: Process every Nth step
            save_to_disk: Whether to save visualizations to disk
            
        Returns:
            Dictionary with visualization results
        """
        if end_idx is None:
            end_idx = self.trajectory_length - 1
        else:
            end_idx = min(end_idx, self.trajectory_length - 1)
        
        # Ensure indices are within valid range
        start_idx = max(0, start_idx)
        end_idx = max(start_idx, end_idx)
        
        if image_idx is None:
            # Use middle frame as background by default
            image_idx = (start_idx + end_idx) // 2
        else:
            image_idx = max(0, min(image_idx, self.trajectory_length - 1))
        
        # Collect trajectory positions
        trajectory_positions = []
        step_indices = list(range(start_idx, end_idx + 1, step_interval))
        for i in step_indices:
            step_data = self.get_non_image_data(i)
            trajectory_positions.append(np.array(step_data["robot_state"]["cartesian_position"]))
        
        # Get image_idx step data for camera parameters
        image_step_data = self.get_non_image_data(image_idx)
        
        # Create color gradient
        num_points = len(trajectory_positions)
        colors = []
        if num_points > 1:
            for i in range(num_points):
                r = int(255 * i / (num_points - 1))
                b = int(255 * (num_points - 1 - i) / (num_points - 1))
                g = 0
                colors.append((b, g, r))  # BGR format
        elif num_points == 1:
            colors.append((255, 0, 0))  # Single blue point
        
        results = {}
        
        # For each camera, project points and draw on images
        for camera_name in self.cameras.keys():
            # Get frames from the specified image_idx
            left_img, right_img, _ = self.get_image_data(camera_name, image_idx)
            
            if left_img is None or right_img is None:
                results[f"{camera_name}_status"] = f"Images not available for step {image_idx}"
                continue
            
            try:
                # Get camera parameters
                left_intrinsic = np.array(image_step_data["cameras"][camera_name]["intrinsics"]["left"])
                right_intrinsic = np.array(image_step_data["cameras"][camera_name]["intrinsics"]["right"])
                
                left_extrinsic = np.array(image_step_data["cameras"]["extrinsics"][camera_name]["left"])
                right_extrinsic = np.array(image_step_data["cameras"]["extrinsics"][camera_name]["right"])
                
                # Project all trajectory positions
                left_trajectory_px = []
                right_trajectory_px = []
                
                for pos in trajectory_positions:
                    left_px = project_point_to_image(pos, left_extrinsic, left_intrinsic)
                    right_px = project_point_to_image(pos, right_extrinsic, right_intrinsic)
                    
                    left_trajectory_px.append(left_px)
                    right_trajectory_px.append(right_px)
                
                # Draw trajectory on images
                left_vis_img = left_img.copy()
                right_vis_img = right_img.copy()
                draw_trajectory(left_vis_img, left_trajectory_px, colors)
                draw_trajectory(right_vis_img, right_trajectory_px, colors)
                
                # Save or store visualized images
                if save_to_disk and self.output_dir:
                    viz_dir = os.path.join(self.output_dir, "visualizations", 
                                         f"trajectory_{start_idx}_to_{end_idx}_on_{image_idx}", camera_name)
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    left_viz_path = os.path.join(viz_dir, "left_trajectory.png")
                    right_viz_path = os.path.join(viz_dir, "right_trajectory.png")
                    
                    cv2.imwrite(left_viz_path, left_vis_img)
                    cv2.imwrite(right_viz_path, right_vis_img)
                    
                    results[f"{camera_name}_left_trajectory_path"] = os.path.relpath(left_viz_path, self.output_dir)
                    results[f"{camera_name}_right_trajectory_path"] = os.path.relpath(right_viz_path, self.output_dir)
                else:
                    results[f"{camera_name}_left_trajectory"] = left_vis_img
                    results[f"{camera_name}_right_trajectory"] = right_vis_img
                    
            except (KeyError, ValueError) as e:
                results[f"{camera_name}_status"] = f"Error projecting points: {e}"
                continue
        
        return results
    
    
    def close(self):
        """Clean up resources"""
        # Close trajectory file
        if self.trajectory:
            self.trajectory.close()
        
        # Close camera resources
        for camera_name, camera in self.cameras.items():
            if hasattr(camera, "zed"):
                camera.zed.close()
            elif hasattr(camera, "cap"):
                camera.cap.release()

        # remove the temporary directory if it exists
        if self.scene_path and self.scene_path.exists():
            shutil.rmtree(self.scene_path, ignore_errors=True)
        

    def generate_vqa_data(self) -> List[VQA]:
        """
        Generate VQA data for this trajectory
            
        Returns:
            List of VQA instances
        """
        # Use the predefined interested_timesteps computed during initialization
        vqa_list = []
        
        # Extract VLM metadata if not already done
        # if not any(self.vlm_metadata.values()):
        #     self.extract_vlm_metadata()
            
        for camera_name in self.get_camera_names():
            if camera_name == "wrist":
                continue
            vqa = vqa_trajectory_understanding(self, camera_name)
            if vqa:
                # Enhance VQA with metadata
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)

        for timestep in self.interested_timesteps:
            
            # vqa = is_stable_grasp(self, timestep)
            # if vqa:
            #         vqa_list.append(vqa)
            vqa = vqa_task_success_state(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)

            vqa = vqa_multi_view_correspondence(self, timestep)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)
            
            vqa = vqa_goal_configuration(self)
            if vqa:
                vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                vqa_list.append(vqa)
                
            for camera_name in self.get_camera_names():

                # vqa = vqa_robot_gripper_open(self, camera_name, timestep)
                # if vqa:
                #     vqa_list.append(vqa)
                
                # vqa = vqa_object_reachable(self, camera_name, timestep)
                # if vqa:
                #     vqa_list.append(vqa)

                # vqa = vqa_relative_direction(self, camera_name, timestep)
                # if vqa:
                #     vqa_list.append(vqa)

                if camera_name != "wrist":
                    vqa = vqa_relative_depth(self, camera_name, timestep)
                    if vqa:
                        vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                        vqa_list.append(vqa)
                        

                    vqa = generate_action_direction_selection_vqa(self, timestep, camera_name)
                    if vqa:
                        vqa = enhance_vqa_with_metadata(vqa, self.vlm_metadata)
                        vqa_list.append(vqa)
                    
                # vqa = vqa_action_understanding(self, camera_name, timestep)
                # if vqa:
                #     vqa_list.append(vqa)

                # vqa = vqa_next_action(self, camera_name, timestep)
                # if vqa:
                #     vqa_list.append(vqa)

        # Remove None entries and return
        vqa_list = [vqa for vqa in vqa_list if vqa is not None]
        return vqa_list
    
# A helper function to create a DROIDTrajectory from TFDS episode data
def create_from_tfds_episode(episode_data, output_dir=None):
    """
    Create a DROIDTrajectory instance from TFDS episode data.
    This will download the raw data if needed.
    
    Args:
        episode_data: Dictionary containing TFDS episode data
        output_dir: Output directory for processed data
        
    Returns:
        DROIDTrajectory instance with VQA data generation capability
    """
    # Extract file path
    file_path = episode_data["episode_metadata"]["file_path"]
    
    # Extract episode ID
    import re
    episode_id_match = re.search(r'([^/]+)/trajectory\.h5$', file_path)
    if episode_id_match:
        episode_id = episode_id_match.group(1)
    else:
        episode_id = f"episode_{random.randint(1000, 9999)}"
    
    # Determine data paths
    temp_dir = f"./droid_raw/{episode_id}"
    episode_output_dir = output_dir
    
    # Extract GS path for download
    path_parts = file_path.replace("/trajectory.h5", "").split('/')
    try:
        base_index = path_parts.index("droid_raw")
        if path_parts[base_index+1] != '1.0.1':
            raise ValueError("Found 'droid_raw' but not '1.0.1' following it.")
        episode_folder = "/".join(path_parts[base_index+2:])
    except (ValueError, IndexError):
        if len(path_parts) < 4:
            raise ValueError(f"Path too short for fallback: {file_path}")
        episode_folder = "/".join(path_parts[-4:])

    gs_parent_dir = f"gs://gresearch/robotics/droid_raw/1.0.1/{episode_folder}/"
    
    # Download raw data
    import subprocess
    import os
    
    os.makedirs(temp_dir, exist_ok=True)
    try:
        result = subprocess.run(["gsutil", "-m", "cp", "-r", gs_parent_dir, temp_dir], 
                                capture_output=True)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    # Construct expected local path to trajectory file
    downloaded_content_dir_name = gs_parent_dir.strip('/').split('/')[-1]
    local_scene_path = os.path.join(temp_dir, downloaded_content_dir_name)
    
    # Create instance with both TFDS and raw data
    trajectory = DROIDTrajectory(
        episode_data=episode_data,
        scene_path=local_scene_path,
        output_dir=episode_output_dir,
        gsutil_path=gs_parent_dir
    )
    
    return trajectory


# For JSON serialization of VQA objects
def vqa_json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable") 
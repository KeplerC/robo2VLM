import cv2
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional, Dict
from pathlib import Path
import math
import random
from scipy.spatial.transform import Rotation
import requests
import json
import spacy
from typing import List
import base64
# Removing spaCy dependency and replacing with Ollama service

# Load spaCy model as fallback
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model isn't installed, download it
    import subprocess
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Assuming these values are defined in common.py
# If not available, we'll use reasonable defaults
CAMERA_NAMES = ["ext1", "ext2", "wrist"]
POS_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]

class CameraFrameCache:
    """Cache for storing camera frames in memory"""
    
    def __init__(self):
        self.frames = defaultdict(dict)  # camera_name -> step_idx -> (left_img, right_img, depth_img)
        self.paths = defaultdict(dict)   # camera_name -> step_idx -> (left_path, right_path, depth_path)
        
    def add_frame(self, camera_name, step_idx, left_img, right_img, depth_img=None, left_path=None, right_path=None, depth_path=None):
        """Add camera frames to the cache"""
        # remove alpha channel
        if left_img is not None:
            left_img = left_img[:, :, :3]
        if right_img is not None:
            right_img = right_img[:, :, :3]

        # Store copies of images to prevent modification of originals
        self.frames[camera_name][step_idx] = (
            left_img.copy() if left_img is not None else None,
            right_img.copy() if right_img is not None else None,
            depth_img.copy() if depth_img is not None else None
        )
        if left_path or right_path or depth_path:
            self.paths[camera_name][step_idx] = (left_path, right_path, depth_path)
    
    def get_frame(self, camera_name, step_idx):
        """Get camera frames from the cache"""
        if camera_name in self.frames and step_idx in self.frames[camera_name]:
            # Return copies to prevent modification of cached images
            left_img, right_img, depth_img = self.frames[camera_name][step_idx]
            return (
                left_img.copy() if left_img is not None else None,
                right_img.copy() if right_img is not None else None,
                depth_img.copy() if depth_img is not None else None
            )
        return None, None, None
    
    def get_paths(self, camera_name, step_idx):
        """Get camera frame paths from the cache"""
        if camera_name in self.paths and step_idx in self.paths[camera_name]:
            return self.paths[camera_name][step_idx]
        return None, None, None
    
    def has_frame(self, camera_name, step_idx):
        """Check if camera frames exist in the cache"""
        return camera_name in self.frames and step_idx in self.frames[camera_name]

class StepDataCache:
    """Cache for storing step data in memory"""
    
    def __init__(self):
        self.steps = {}  # step_idx -> step_data
        
    def add_step(self, step_idx, step_data):
        """Add step data to the cache
        
        Args:
            step_idx: The step index
            step_data: The step data dictionary
        """
        # Store a deep copy of step_data to avoid modification
        self.steps[step_idx] = step_data
    
    def get_step(self, step_idx):
        """Get step data from the cache
        
        Args:
            step_idx: The step index
            
        Returns:
            The step data dictionary or None if not in cache
        """
        if step_idx in self.steps:
            # Return a deep copy to prevent modification of cached data
            return self.steps[step_idx]
        return None
    
    def has_step(self, step_idx):
        """Check if step data exists in the cache
        
        Args:
            step_idx: The step index
            
        Returns:
            True if step data exists in cache, False otherwise
        """
        return step_idx in self.steps

class StereoCamera:
    """Handles reading camera data from SVO or MP4 files"""
    
    def __init__(self, recordings: Path, serial: int):
        self.serial = serial
        self.recordings = recordings
        self.left_intrinsic_mat = None
        self.right_intrinsic_mat = None
        self.frame_count = 0
        
        try:
            import pyzed.sl as sl
            init_params = sl.InitParameters()
            svo_path = recordings / "SVO" / f"{serial}.svo"
            init_params.set_from_svo_file(str(svo_path))
            init_params.depth_mode = sl.DEPTH_MODE.QUALITY
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = 0.2

            zed = sl.Camera()
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                raise Exception(f"Error reading camera data: {err}")

            params = zed.get_camera_information().camera_configuration.calibration_parameters
            
            self.left_intrinsic_mat = np.array([
                [params.left_cam.fx, 0, params.left_cam.cx],
                [0, params.left_cam.fy, params.left_cam.cy],
                [0, 0, 1],
            ])
            self.right_intrinsic_mat = np.array([
                [params.right_cam.fx, 0, params.right_cam.cx],
                [0, params.right_cam.fy, params.right_cam.cy],
                [0, 0, 1],
            ])
            self.zed = zed
            
            # Get frame count
            self.frame_count = zed.get_svo_number_of_frames()
            
        except (ModuleNotFoundError, Exception) as e:
            print(f"ZED SDK not available or error: {e}")
            # Default intrinsic parameters if ZED SDK not available
            self.left_intrinsic_mat = np.array([
                [733.37261963,   0.,         625.26251221],
                [  0.,         733.37261963,  361.92279053],
                [  0.,           0.,           1.,        ]
            ])
            self.right_intrinsic_mat = self.left_intrinsic_mat
            
            # Try to load from MP4 file
            mp4_path = None
            if (recordings / "MP4" / f'{serial}-stereo.mp4').exists():
                mp4_path = recordings / "MP4" / f'{serial}-stereo.mp4'
            elif (recordings / "MP4" / f'{serial}.mp4').exists():
                mp4_path = recordings / "MP4" / f'{serial}.mp4'
            else:
                raise Exception(f"Unable to find video file for camera {serial}")

            self.cap = cv2.VideoCapture(str(mp4_path))
            if not self.cap.isOpened():
                raise Exception(f"Failed to open {mp4_path}")
                
            # Get frame count
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Opened {mp4_path} with {self.frame_count} frames")
        
    def get_next_frame(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Gets the next frame from both cameras and possibly computes the depth."""
        if hasattr(self, "zed"):
            import pyzed.sl as sl
            left_image = sl.Mat()
            right_image = sl.Mat()
            depth_image = sl.Mat()

            rt_param = sl.RuntimeParameters()
            err = self.zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(left_image, sl.VIEW.LEFT)
                left_image = np.array(left_image.numpy())

                self.zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                right_image = np.array(right_image.numpy())

                self.zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
                depth_image = np.array(depth_image.numpy())

                return (left_image, right_image, depth_image)
            else:
                return None
            
        return None



def project_point_to_image(point_3d, extrinsic, intrinsic):
    """Project a 3D point to 2D image coordinates using camera parameters.
    
    Args:
        point_3d: 3D point in world coordinates [x, y, z, roll, pitch, yaw]
        extrinsic: Camera extrinsic parameters [x, y, z, roll, pitch, yaw]
        intrinsic: Camera intrinsic matrix (3x3)
        
    Returns:
        Tuple (x, y) of pixel coordinates
    """
    # Extract position components
    position_3d = point_3d[:3]  # [x, y, z]
    cam_position = extrinsic[:3]  # Camera position [x, y, z]
    
    # Create rotation matrix from camera Euler angles
    cam_rotation = Rotation.from_euler('xyz', extrinsic[3:6]).as_matrix()
    
    # Create camera-to-world transformation matrix (similar to link_to_world_transform)
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = cam_rotation
    cam_to_world[:3, 3] = cam_position
    
    # Create world-to-camera transformation matrix by inverting cam_to_world
    world_to_cam = np.linalg.inv(cam_to_world)
    
    # Convert point to homogeneous coordinates [x, y, z, 1]
    point_homogeneous = np.ones(4)
    point_homogeneous[:3] = position_3d
    
    # Transform point from world to camera coordinates
    point_in_cam_homogeneous = world_to_cam @ point_homogeneous
    point_in_cam = point_in_cam_homogeneous[:3]
    
    # Check if point is in front of camera (Z > 0)
    if point_in_cam[2] <= 0:
        return (-1, -1)  # Point is behind camera
    
    # Project to image plane using intrinsic matrix
    point_2d = intrinsic @ (point_in_cam / point_in_cam[2])
    
    # Return pixel coordinates
    return (int(point_2d[0]), int(point_2d[1]))

def draw_projected_points(image, current_px, target_px, future_px, current_pos, target_pos, extrinsic, intrinsic):
    """Draw alternative displacement arrows based on current and future points.
    Original points, text, axes, and displacement arrow are commented out.
    
    Args:
        image: Image to draw on
        current_px: Current position in pixel coordinates (x, y)
        target_px: Target position in pixel coordinates (x, y) - unused currently
        future_px: Position after num_skip_frames in pixel coordinates (x, y)
        current_pos: Current position in 3D [x, y, z, roll, pitch, yaw] - unused currently
        target_pos: Target position in 3D [x, y, z, roll, pitch, yaw] - unused currently
        extrinsic: Camera extrinsic parameters [x, y, z, roll, pitch, yaw] - unused currently
        intrinsic: Camera intrinsic matrix (3x3) - unused currently
    """
    # draw displacement arrow from current to future
    if current_px[0] >= 0 and current_px[1] >= 0 and future_px[0] >= 0 and future_px[1] >= 0:
        # Use a fixed arrow length for consistency
        arrow_length = 80  # Fixed arrow length in pixels
        
        # Calculate direction vector
        dx = future_px[0] - current_px[0]
        dy = future_px[1] - current_px[1]
        
        # Calculate magnitude of the vector
        mag = math.sqrt(dx*dx + dy*dy)
        
        if mag > 1e-6:  # Avoid division by zero
            # Normalize and scale to fixed length
            dx = dx / mag * arrow_length
            dy = dy / mag * arrow_length
            
            # Calculate end point of fixed-length arrow
            end_px = (int(current_px[0] + dx), int(current_px[1] + dy))
            
            # Draw the fixed-length arrow
            cv2.arrowedLine(image, current_px, end_px, (255, 255, 0), 3, tipLength=0.3)  # Cyan arrow

    # Draw only the alternative arrows
    draw_alternative_arrows(image, current_px, future_px)

def draw_alternative_arrows(image, current_px, future_px):
    """Draw three plausible alternative displacement arrows with fixed length."""
    if not (current_px[0] >= 0 and current_px[1] >= 0 and future_px[0] >= 0 and future_px[1] >= 0):
        return # Cannot draw if points are invalid

    # Fixed arrow length for all arrows
    arrow_length = 80  # Fixed arrow length in pixels
    
    disp_orig_x = future_px[0] - current_px[0]
    disp_orig_y = future_px[1] - current_px[1]
    
    mag = math.hypot(disp_orig_x, disp_orig_y)
    angle_orig = math.atan2(disp_orig_y, disp_orig_x)
    
    if mag < 1e-6: # Avoid issues with zero magnitude displacement
        return 
        
    min_angle_diff = math.radians(15)
    alternative_angles = []
    alternative_colors = [(0, 165, 255), (0, 255, 255), (0, 100, 255)] # Orange, Yellow, Darker Orange BGR

    attempts = 0
    max_attempts = 100 # Prevent infinite loops

    while len(alternative_angles) < 3 and attempts < max_attempts:
        attempts += 1
        # Generate random angle, ensuring it's somewhat different from original
        angle_new = angle_orig + random.uniform(math.radians(20), math.radians(340)) # Bias away from original
        angle_new = (angle_new + math.pi) % (2 * math.pi) - math.pi # Normalize to [-pi, pi]

        # Check difference from original angle (handle wrap-around)
        diff_orig = abs(angle_new - angle_orig)
        diff_orig = min(diff_orig, 2 * math.pi - diff_orig) # Check shortest angular distance
        if diff_orig < min_angle_diff:
            continue

        # Check difference from other alternative angles
        valid_new_angle = True
        for angle_alt in alternative_angles:
            diff_alt = abs(angle_new - angle_alt)
            diff_alt = min(diff_alt, 2 * math.pi - diff_alt)
            if diff_alt < min_angle_diff:
                valid_new_angle = False
                break
        
        if valid_new_angle:
            alternative_angles.append(angle_new)

    if len(alternative_angles) < 3:
            print(f"Warning: Could only generate {len(alternative_angles)} distinct alternative arrows.")

    # Draw the alternative arrows with fixed length
    for i, angle_alt in enumerate(alternative_angles):
        # Use fixed arrow length
        end_px_alt_x = int(current_px[0] + arrow_length * math.cos(angle_alt))
        end_px_alt_y = int(current_px[1] + arrow_length * math.sin(angle_alt))
        end_px_alt = (end_px_alt_x, end_px_alt_y)
        color = alternative_colors[i % len(alternative_colors)]
        cv2.arrowedLine(image, current_px, end_px_alt, color, 3, tipLength=0.3)

def draw_trajectory(image, trajectory_px, colors):
    """Draw trajectory points and connecting lines on an image.
    
    Args:
        image: Image to draw on
        trajectory_px: List of pixel coordinates for trajectory points
        colors: List of colors for each trajectory point
    """
    # Draw points and connecting lines
    prev_point = None
    prev_idx = None  # Track the index of the previous point
    
    for i, point in enumerate(trajectory_px):
        # Skip points that are not visible in the camera
        if point[0] < 0 or point[1] < 0:
            prev_point = None
            prev_idx = None
            continue
            
        # Draw current point using the color corresponding to its index in the original list
        # Use i as the index into the colors list, assuming colors has the same length as trajectory_px
        if i < len(colors):
            cv2.circle(image, point, 3, colors[i], -1)
        else:
            cv2.circle(image, point, 3, (255, 255, 255), -1) # Fallback color: white

        # Draw line connecting to previous point if it exists
        if prev_point is not None and prev_idx is not None:
            # Use the color of the previous point for the line
            if prev_idx < len(colors):
                line_color = colors[prev_idx]
                cv2.line(image, prev_point, point, line_color, 2)
            else:
                cv2.line(image, prev_point, point, (255, 255, 255), 2) # Fallback color: white

        prev_point = point
        prev_idx = i  # Store the current index for the next iteration
    
    # Mark start and end of trajectory
    start_point = next((p for p in trajectory_px if p[0] >= 0 and p[1] >= 0), None)
    end_point = next((p for p in reversed(trajectory_px) if p[0] >= 0 and p[1] >= 0), None)
    
    if start_point:
        cv2.circle(image, start_point, 5, (255, 0, 0), -1)  # Blue start point
        cv2.putText(image, "Start", (start_point[0] + 10, start_point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    if end_point and end_point != start_point:
        cv2.circle(image, end_point, 5, (0, 0, 255), -1)  # Red end point
        cv2.putText(image, "End", (end_point[0] + 10, end_point[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


def determine_grasp_phases(trajectory, grasp_threshold=0.4, contact_threshold=0.9):
    """Determine interested frames for visualizing actions.
    pre-grasp, immobization, contact, detach, post-grasp, transition based on grasp_threshold and contact_threshold
    
    Args:
        trajectory: DROIDTrajectory or RawDatasetExtractor instance
        grasp_threshold: Threshold close to 0 to determine when gripper is considered open
        contact_threshold: Threshold close to 1 to determine when gripper is considered closed
        
    Returns:
        Dictionary with phase information and frame ranges
    """

    traj_length = trajectory.get_trajectory_length()

    normalized_gripper_actions = []
    # normalize gripper action
    max_gripper_position = -1
    min_gripper_position = 1
    for step_idx in range(traj_length):
        gripper_position = trajectory.get_gripper_position(step_idx)
        max_gripper_position = max(max_gripper_position, gripper_position)
        min_gripper_position = min(min_gripper_position, gripper_position)
    for step_idx in range(traj_length):
        gripper_position = trajectory.get_gripper_position(step_idx)
        gripper_position = (gripper_position - min_gripper_position) / (max_gripper_position - min_gripper_position + 1e-6)
        normalized_gripper_actions.append(gripper_position)

    # Initialize phase tracking
    phases = []  # Will store (step_idx, phase_name) tuples
    phase_ranges = {
        "pre_grasp": [],      # Open gripper before approaching object (gripper ≤ grasp_threshold)
        "immobilization": [], # Gripper closing but not yet in firm contact (grasp_threshold < gripper < contact_threshold)
        "contact": [],        # Gripper in firm contact with object (gripper ≥ contact_threshold)
        "detach": [],         # Gripper opening but not fully open (grasp_threshold < gripper < contact_threshold)
        "post_grasp": [],     # Gripper fully open after grasp (gripper ≤ grasp_threshold)
        "transition": []      # Unexpected or temporary transitions
    }
    
    # Track current phase and its start index
    current_phase = "pre_grasp"
    phase_start_idx = 0
    
    # Previous gripper position for detecting direction changes
    prev_gripper_position = None
    
    # Process all steps to determine phases
    for step_idx in range(traj_length):
        # Note: gripper=0 means open, gripper=1 means closed
        gripper_position = normalized_gripper_actions[step_idx]
        
        # First step initialization
        if prev_gripper_position is None:
            prev_gripper_position = gripper_position
            
        # Calculate direction of gripper movement
        gripper_closing = gripper_position > prev_gripper_position
        gripper_opening = gripper_position < prev_gripper_position
        
        # Determine the current phase based on gripper position
        if current_phase == "pre_grasp":
            # In pre-grasp phase, gripper is open (≤ grasp_threshold)
            if gripper_position > grasp_threshold and gripper_closing:
                # Transition to immobilization when gripper starts closing
                if step_idx > phase_start_idx:  # Ensure phase has duration
                    phase_ranges["pre_grasp"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "pre_grasp"))
                
                current_phase = "immobilization"
                phase_start_idx = step_idx
                print(f"Step {step_idx}: Transitioning pre_grasp -> immobilization (gripper: {gripper_position:.3f})")
        
        elif current_phase == "immobilization":
            # In immobilization phase, gripper is closing (grasp_threshold < gripper < contact_threshold)
            if gripper_position >= contact_threshold:
                # Transition to contact when gripper reaches contact threshold (nearly closed)
                if step_idx > phase_start_idx:
                    phase_ranges["immobilization"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "immobilization"))
                
                current_phase = "contact"
                phase_start_idx = step_idx
                print(f"Step {step_idx}: Transitioning immobilization -> contact (gripper: {gripper_position:.3f})")
            
            elif gripper_position <= grasp_threshold and gripper_opening:
                # If gripper opens without reaching contact, go back to pre_grasp
                if step_idx > phase_start_idx:
                    phase_ranges["immobilization"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "immobilization"))
                
                current_phase = "pre_grasp"
                phase_start_idx = step_idx
                print(f"Step {step_idx}: Transitioning immobilization -> pre_grasp (grasp aborted) (gripper: {gripper_position:.3f})")
        
        elif current_phase == "contact":
            # In contact phase, gripper is closed (≥ contact_threshold)
            if gripper_position < contact_threshold and gripper_opening:
                # Transition to detach when gripper starts opening
                if step_idx > phase_start_idx:
                    phase_ranges["contact"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "contact"))
                
                current_phase = "detach"
                phase_start_idx = step_idx
                print(f"Step {step_idx}: Transitioning contact -> detach (gripper: {gripper_position:.3f})")
            
            elif gripper_position < contact_threshold and gripper_closing:
                # Handle case where contact is momentarily lost but closing again
                phase_ranges["transition"].append((step_idx, step_idx))
                phases.append((step_idx, step_idx, "transition"))
                print(f"Step {step_idx}: Transition - brief contact loss (gripper: {gripper_position:.3f})")
        
        elif current_phase == "detach":
            # In detach phase, gripper is opening (grasp_threshold < gripper < contact_threshold)
            if gripper_position <= grasp_threshold:
                # Transition to post-grasp when gripper is fully open
                if step_idx > phase_start_idx:
                    phase_ranges["detach"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "detach"))
                
                current_phase = "post_grasp"
                phase_start_idx = step_idx
                print(f"Step {step_idx}: Transitioning detach -> post_grasp (gripper: {gripper_position:.3f})")
            
            elif gripper_position >= contact_threshold and gripper_closing:
                # If gripper closes again during detach, go back to contact
                if step_idx > phase_start_idx:
                    phase_ranges["detach"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "detach"))
                
                current_phase = "contact"
                phase_start_idx = step_idx
                print(f"Step {step_idx}: Transitioning detach -> contact (re-grasping) (gripper: {gripper_position:.3f})")
        
        elif current_phase == "post_grasp":
            # In post-grasp phase, gripper is open again (≤ grasp_threshold)
            if gripper_position > grasp_threshold and gripper_closing:
                # Starting a new grasp cycle
                if step_idx > phase_start_idx:
                    phase_ranges["post_grasp"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "post_grasp"))
                
                current_phase = "immobilization"
                phase_start_idx = step_idx
                print(f"Step {step_idx}: Transitioning post_grasp -> immobilization (new grasp cycle) (gripper: {gripper_position:.3f})")
        
        # Update previous gripper position
        prev_gripper_position = gripper_position
        

    # Add the final phase range
    if phase_start_idx < traj_length:
        phase_ranges[current_phase].append((phase_start_idx, traj_length - 1))
        phases.append((phase_start_idx, traj_length - 1, current_phase))
    
    # Print phase information summary
    # print("\nGrasp phases summary:")
    # for start_idx, end_idx, phase_name in sorted(phases, key=lambda x: x[0]):
    #     duration = end_idx - start_idx + 1
    #     print(f"  {phase_name}: steps {start_idx} to {end_idx} ({duration} steps)")
    
    return {
        "phases": phases,
        "phase_ranges": phase_ranges
    }

def get_target_object(language_instruction: str) -> str:
    """
    Extracts the target object from the language instruction using Ollama service.
    Falls back to spaCy if Ollama service fails.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": f"Extract the main target object from this instruction. Return only the object name, nothing else: '{language_instruction}'",
                "stream": False
            },
            timeout=50
        )
        response.raise_for_status()
        result = response.json()
        target_object = result["response"].strip().lower()
        return target_object
    except Exception as e:
        print(f"Error calling Ollama service: {e}")
        # Fallback to spaCy
        doc = nlp(language_instruction)

        for chunk in doc.noun_chunks:
            return chunk.text
        # Find the first noun in the instruction
        for token in doc:
            if token.pos_ == "NOUN":
                return token.text
        
        return None
    
def extract_objects_from_instruction(language_instruction: str) -> List[str]:
    """
    Extracts all objects from the language instruction using Ollama service.
    Falls back to spaCy if Ollama service fails.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": f"Extract all objects mentioned in this instruction. Return as a comma-separated list with no additional text: '{language_instruction}'",
                "stream": False
            },
            timeout=50
        )
        response.raise_for_status()
        result = response.json()
        objects_text = result["response"].strip().lower()
        return [obj.strip() for obj in objects_text.split(",") if obj.strip()]
    except Exception as e:
        print(f"Error calling Ollama service: {e}")
        # Fallback to spaCy
        doc = nlp(language_instruction)
        return [chunk.text for chunk in doc.noun_chunks]

def extract_locations_from_instruction(language_instruction: str) -> List[str]:
    """
    Extracts all locations from the language instruction using Ollama service.
    Falls back to spaCy if Ollama service fails.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": f"Extract all locations or spatial references mentioned in this instruction. Return as a comma-separated list with no additional text: '{language_instruction}'",
                "stream": False
            },
            timeout=50
        )
        response.raise_for_status()
        result = response.json()
        locations_text = result["response"].strip().lower()
        return [loc.strip() for loc in locations_text.split(",") if loc.strip()]
    except Exception as e:
        print(f"Error calling Ollama service: {e}")
        # Fallback to spaCy
        doc = nlp(language_instruction)
        return [chunk.text for chunk in doc.noun_chunks]


def clean_up_language_instruction(language_instruction: str) -> str:
    """
    Cleans up the language instruction by removing extra words and phrases.
    """
    return language_instruction.lower().strip(".").strip(",").strip()

def extract_metadata_from_frame(image: np.ndarray, language_instruction:str) -> Dict[str, str]:
    """
    Extracts metadata from a frame using Llama 3.2 via Ollama API.
    
    Args:
        image: Image as a numpy array in RGB format
        
    Returns:
        Dictionary with extracted metadata including:
        - physical_location: The environment/setting (office, kitchen, etc.)
        - task: Action being performed (pick, place, etc.)
        - target_object: Main object of interest
        - location: Spatial information about where objects are
    """
    try:
        # Make sure we have a valid image
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image: None or not a numpy array")

        # Check image dimensions
        if len(image.shape) < 2:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
            
        # Convert image to base64
        # Handle different image formats properly
        if len(image.shape) == 2:  # Grayscale
            # Convert grayscale to BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Check if image is already in BGR format (common in OpenCV)
            # This is a heuristic, not foolproof, but avoids unnecessary conversions
            image_bgr = image.copy()  # Assume it's already BGR (or RGB, doesn't matter for encoding)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # With alpha channel
            # Remove alpha channel
            image_bgr = image[:, :, :3]
        else:
            raise ValueError(f"Unsupported image format with shape {image.shape}")
        
        # Now encode the image
        success, buffer = cv2.imencode('.png', image_bgr)
        if not success:
            raise ValueError("Failed to encode image")
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        
        # Prepare the prompt for vision analysis
        prompt = f"""
        Analyze this image of a robotic manipulation scene and extract the following information:
        1. Physical location/environment (e.g., office, kitchen, lab)
        2. Task being performed (e.g., pick, place, push)
        3. Target object(s) visible in the scene
        4. Current locations of key objects (e.g., "table", "box")
        
        The langauge instruction given to the robot is {language_instruction}
        
        Your must format your response as JSON:
        {
            "physical_location": "...",
            "task": "...",
            "target_object": "...",
            "location": "..."
        }
        """
        
        # Make the API call to Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2-vision",
                "prompt": prompt,
                "stream": False,
                "images": [image_base64]
            },
            timeout=100
        )
        response.raise_for_status()
        result = response.json()
        
        # Parse the response
        # Since we requested JSON, try to extract it from the text response
        response_text = result["response"].strip()
        print(f"Response from Ollama: {response_text}")
        
        # Try to find JSON in the response using regex
        import re
        json_match = re.search(r'({[\s\S]*})', response_text)
        
        if json_match:
            try:
                metadata = json.loads(json_match.group(1))
                # Ensure all expected keys exist
                for key in ["physical_location", "task", "target_object", "location"]:
                    if key not in metadata:
                        metadata[key] = ""
                return metadata
            except json.JSONDecodeError:
                print("Failed to parse JSON from response")

        # Fallback: extract information manually
        metadata = {
            "physical_location": "",
            "task": "",
            "target_object": "",
            "location": ""
        }
        
        # Simple extraction based on keywords
        if "office" in response_text.lower():
            metadata["physical_location"] = "office"
        elif "kitchen" in response_text.lower():
            metadata["physical_location"] = "kitchen"
        elif "lab" in response_text.lower():
            metadata["physical_location"] = "lab"
        
        for task in ["pick", "place", "push", "pull", "grasp", "move"]:
            if task in response_text.lower():
                metadata["task"] = task
                break
        
        return metadata
        
    except Exception as e:
        print(f"Error extracting metadata from frame: {e}")
        import traceback
        traceback.print_exc()
        return {
            "physical_location": "",
            "task": "",
            "target_object": "",
            "location": ""
        }
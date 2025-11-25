import numpy as np
import random
from typing import List, Optional, Dict, Tuple, Any
from utils import *
import cv2
import math
import base64
import json
import os
import uuid
import hashlib
import traceback

def _hash_image(img: np.ndarray) -> str:
    """Generate a hash for an image.
    
    Args:
        img: Numpy image array
        
    Returns:
        String hash of the image
    """
    if img is None:
        return None
    
    # Convert image to bytes
    _, buffer = cv2.imencode('.png', img)
    img_bytes = buffer.tobytes()
    
    # Generate SHA-256 hash
    return hashlib.sha256(img_bytes).hexdigest()

class VQA:
    """Represents a Visual Question Answering item with four choices and one correct answer."""
    
    def __init__(self, question_text: str, choices: List[str], correct_idx: int, 
                 question_images: Optional[List[np.ndarray]] = None,
                 choice_images: Optional[List[np.ndarray]] = None,
                 metadata: Optional[Dict] = None,
                 question_image_ids: Optional[List[str]] = None,
                 choice_image_ids: Optional[List[str]] = None):
        """Initialize a VQA instance.
        
        Args:
            question_text: The text of the question
            choices: List of four or five text choices
            correct_idx: Index (0-4) of the correct answer
            question_images: Optional list of images related to the question
            choice_images: Optional list of images for choices (can be None for text-only choices)
            metadata: Optional dictionary with additional information about the VQA
            question_image_ids: Optional list of image IDs for question images
            choice_image_ids: Optional list of image IDs for choice images
        """
        self.question_text = question_text
        self.choices = choices
        self.correct_idx = correct_idx
        self.question_images = question_images if question_images is not None else []
        self.choice_images = choice_images if choice_images is not None else [None] * len(choices)
        self.metadata = metadata or {}
        
        # Initialize image IDs using hashes
        self.question_image_ids = question_image_ids or [_hash_image(img) for img in self.question_images]
        self.choice_image_ids = choice_image_ids or [_hash_image(img) if img is not None else None for img in self.choice_images]
        
        self._validate()

        self._shuffle_choices(self.choices[self.correct_idx], self.choices[:self.correct_idx] + self.choices[self.correct_idx+1:])
    
    def _validate(self):
        """Validate that the VQA is well-formed."""
        # Determine if this is a binary question (yes/no)
        is_binary = False
        if len(self.choices) == 4:
            yes_no_answers = ["Yes", "No"]
            first_two_answers = [choice.strip() for choice in self.choices[:2]]
            if set(first_two_answers) == set(yes_no_answers):
                is_binary = True
        
        if is_binary and len(self.choices) != 4:
            raise ValueError(f"Binary VQA must have exactly 4 choices, got {len(self.choices)}")
        elif not is_binary and len(self.choices) != 5:
            raise ValueError(f"Non-binary VQA must have exactly 5 choices, got {len(self.choices)}")
        
        if not (0 <= self.correct_idx < len(self.choices)):
            raise ValueError(f"Correct index must be between 0-{len(self.choices)-1}, got {self.correct_idx}")
        
        if self.choice_images and len(self.choice_images) != len(self.choices):
            raise ValueError(f"If choice images are provided, must have exactly {len(self.choices)}, got {len(self.choice_images)}")
        
        # images cannot contain nan 
        for img in self.question_images:
            if img is not None:
                # Use np.issubdtype to check for float type before calling isnan
                if np.issubdtype(img.dtype, np.floating) and np.isnan(img).any():
                    raise ValueError("Question image contains NaN")
        for img in self.choice_images:
            if img is not None:
                # Use np.issubdtype to check for float type before calling isnan
                if np.issubdtype(img.dtype, np.floating) and np.isnan(img).any():
                    raise ValueError("Choice image contains NaN")
    

    def get_question(self) -> Dict:
        """Get the question part of the VQA.
        
        Returns:
            Dictionary with question text and images
        """
        return {
            "text": self.question_text,
            "image_ids": self.question_image_ids
        }
    
    def get_choices(self) -> List[Dict]:
        """Get the choices for the question.
        
        Returns:
            List of dictionaries with text and optional image for each choice
        """
        result = []
        for i, (text, image_id) in enumerate(zip(self.choices, self.choice_image_ids)):
            result.append({
                "text": text,
                "image_id": image_id,
                "is_correct": (i == self.correct_idx)
            })
        return result
    
    def get_correct_choice(self) -> Dict:
        """Get only the correct choice.
        
        Returns:
            Dictionary with text and optional image for the correct choice
        """
        return {
            "text": self.choices[self.correct_idx],
            "image_id": self.choice_image_ids[self.correct_idx],
            "index": self.correct_idx
        }
    
    def _encode_image(self, img: np.ndarray) -> Optional[str]:
        """Convert numpy image array to base64 encoded string.
        
        Args:
            img: Numpy image array
            
        Returns:
            Base64 encoded string or None if input is None
        """
        if img is None:
            return None
        
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def to_dict(self) -> Dict:
        """Convert the VQA to a dictionary representation.
        
        Returns:
            Dictionary representation of the VQA with image IDs instead of actual images
        """
        data = {
            "question": {
                "text": self.question_text,
                "image_ids": self.question_image_ids
            },
            "choices": [
                {
                    "text": choice,
                    "image_id": image_id,
                    "is_correct": (i == self.correct_idx)
                }
                for i, (choice, image_id) in enumerate(zip(self.choices, self.choice_image_ids))
            ]
        }
        
        # Add metadata if present
        if self.metadata:
            data["metadata"] = self.metadata
            
        return data
    
    def to_json(self) -> str:
        """Convert the VQA to a JSON string.
        
        Returns:
            JSON string representation of the VQA
        """
        return json.dumps(self.to_dict())
    
    def save_images(self, images_dir: str) -> None:
        """Save all images to the specified directory.
        
        Args:
            images_dir: Directory path where images will be saved
        """
        os.makedirs(images_dir, exist_ok=True)
        
        # Save question images
        for img, img_id in zip(self.question_images, self.question_image_ids):
            if img is not None:
                img_path = os.path.join(images_dir, f"{img_id}.png")
                cv2.imwrite(img_path, img)
        # Save choice images
        for img, img_id in zip(self.choice_images, self.choice_image_ids):
            if img is not None and img_id is not None:
                img_path = os.path.join(images_dir, f"{img_id}.png")
                cv2.imwrite(img_path, img)
    
    def load_images(self, images_dir: str) -> None:
        """Load all images from the specified directory.
        
        Args:
            images_dir: Directory path where images are stored
        """
        # Load question images
        self.question_images = []
        for img_id in self.question_image_ids:
            img_path = os.path.join(images_dir, f"{img_id}.png")
            if os.path.exists(img_path):
                self.question_images.append(cv2.imread(img_path))
            else:
                self.question_images.append(None)
        
        # Load choice images
        self.choice_images = []
        for img_id in self.choice_image_ids:
            if img_id is not None:
                img_path = os.path.join(images_dir, f"{img_id}.png")
                if os.path.exists(img_path):
                    self.choice_images.append(cv2.imread(img_path))
                else:
                    self.choice_images.append(None)
            else:
                self.choice_images.append(None)
    
    def __str__(self):
        return f"VQA(question={self.question_text}, choices={self.choices}, correct_idx={self.correct_idx})"
    
    def __repr__(self):
        return self.__str__()
    
    def _shuffle_choices(self, correct_choice: str, incorrect_choices: List[str]) -> Tuple[List[str], int]:
        """Shuffles correct and incorrect choices and returns the shuffled list and new correct index."""
        all_choices = [correct_choice] + incorrect_choices
        original_indices = list(range(len(all_choices)))
        shuffled_indices = random.sample(original_indices, len(original_indices))
        
        shuffled_choices = [""] * len(all_choices)
        shuffled_choice_images = [None] * len(self.choice_images)
        shuffled_choice_image_ids = [None] * len(self.choice_image_ids)
        new_correct_idx = -1
        
        # Determine if this is a non-binary question (5 choices)
        is_non_binary = len(all_choices) == 5
        
        # For non-binary questions, have a 20% chance to use "None of the above"
        use_none_of_above = is_non_binary and random.random() < 0.2
        
        for i, original_idx in enumerate(shuffled_indices):
            shuffled_choices[i] = all_choices[original_idx]
            if self.choice_images and original_idx < len(self.choice_images):
                shuffled_choice_images[i] = self.choice_images[original_idx]
            if self.choice_image_ids and original_idx < len(self.choice_image_ids):
                shuffled_choice_image_ids[i] = self.choice_image_ids[original_idx]
            if original_idx == 0: # Original index 0 was the correct answer
                new_correct_idx = i
        
        # If we're using "None of the above" for a non-binary question
        if use_none_of_above:
            # Store the old correct choice
            old_correct_idx = new_correct_idx
            
            # Replace the correct choice with "None of the above"
            shuffled_choices[old_correct_idx] = "None of the above"
            
            # If the correct answer had an image, remove it
            if self.choice_images and old_correct_idx < len(shuffled_choice_images):
                shuffled_choice_images[old_correct_idx] = None
            if self.choice_image_ids and old_correct_idx < len(shuffled_choice_image_ids):
                shuffled_choice_image_ids[old_correct_idx] = None
                
        self.choices = shuffled_choices
        self.choice_images = shuffled_choice_images
        self.choice_image_ids = shuffled_choice_image_ids
        self.correct_idx = new_correct_idx
    
    @classmethod
    def from_dict(cls, data: Dict, images_dir: Optional[str] = None) -> 'VQA':
        """Create a VQA instance from a dictionary representation.
        
        Args:
            data: Dictionary representation of a VQA
            images_dir: Optional directory path to load images from
            
        Returns:
            VQA instance
        """
        question_text = data["question"]["text"]
        question_image_ids = data["question"].get("image_ids", [])
        
        # Initialize with empty images (will load later if images_dir is provided)
        question_images = [None] * len(question_image_ids)
        
        choices = []
        choice_image_ids = []
        correct_idx = -1
        
        for i, choice in enumerate(data["choices"]):
            choices.append(choice["text"])
            choice_image_ids.append(choice.get("image_id"))
                
            if choice.get("is_correct"):
                correct_idx = i
        
        # Extract metadata if present
        metadata = data.get("metadata", {})
        
        # Create VQA instance
        vqa = cls(
            question_text=question_text,
            choices=choices,
            correct_idx=correct_idx,
            question_images=question_images,
            choice_images=[None] * len(choices),
            metadata=metadata,
            question_image_ids=question_image_ids,
            choice_image_ids=choice_image_ids
        )
        
        # Load images if directory is provided
        if images_dir:
            vqa.load_images(images_dir)
            
        return vqa

# Make VQA class JSON serializable
def vqa_json_serializer(obj):
    """JSON serializer for VQA objects"""
    if isinstance(obj, VQA):
        return obj.to_dict()
    raise TypeError(f"Type {type(obj)} not serializable")


def save_vqa_dataset(vqas: List[VQA], output_dir: str, metadata: Optional[Dict] = None) -> None:
    """Save a list of VQA instances to a dataset directory.
    
    Args:
        vqas: List of VQA instances
        output_dir: Directory to save the dataset
        metadata: Optional metadata for the dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Save all images
    for vqa in vqas:
        vqa.save_images(images_dir)
    
    # Save all VQA data as JSON
    vqa_data = [vqa.to_dict() for vqa in vqas]
    dataset = {
        "vqa_items": vqa_data,
        "metadata": metadata or {}
    }
    
    with open(os.path.join(output_dir, "vqa_data.json"), "w") as f:
        json.dump(dataset, f, indent=2)


def load_vqa_dataset(dataset_dir: str) -> Tuple[List[VQA], Dict]:
    """Load a VQA dataset from a directory.
    
    Args:
        dataset_dir: Directory containing the dataset
        
    Returns:
        Tuple of (list of VQA instances, dataset metadata)
    """
    # Load the JSON data
    with open(os.path.join(dataset_dir, "vqa_data.json"), "r") as f:
        dataset = json.load(f)
    
    images_dir = os.path.join(dataset_dir, "images")
    
    # Create VQA instances
    vqas = []
    for vqa_data in dataset["vqa_items"]:
        vqa = VQA.from_dict(vqa_data, images_dir=images_dir)
        vqas.append(vqa)
    
    return vqas, dataset.get("metadata", {})


# S1 
def vqa_robot_gripper_open(trajectory, step_idx: int) -> Optional[VQA]:
    """
    Generates a VQA asking about a simple state predicate (e.g., gripper open/closed).
    
    Args:
        trajectory: DROIDTrajectory instance
        step_idx: Step index to use
        
    Returns:
        VQA instance or None if generation fails
    """

    traj_length = trajectory.get_trajectory_length()
    max_gripper_position = -1
    min_gripper_position = 1
    gripper_position = 0
    for i in range(traj_length):
        g = trajectory.get_gripper_position(i)
        if i == step_idx:
            gripper_position = g
        max_gripper_position = max(max_gripper_position, g)
        min_gripper_position = min(min_gripper_position, g)
        
    normalized_gripper_position = (gripper_position - min_gripper_position) / (max_gripper_position - min_gripper_position + 1e-6)
    # Simple predicate: Is the gripper open? (Using grasp_threshold)
    grasp_threshold = 0.8
    is_open = normalized_gripper_position <= grasp_threshold
    
    question_text = f"Is the robot's gripper open?"
    correct_answer = "Yes" if is_open else "No"
    incorrect_answer = "No" if is_open else "Yes"
    
    # Distractors
    distractors = ["Cannot be determined", "Partially open"] 

    # Get stitched image from all cameras
    combined_img = stitch_all_cameras(trajectory, step_idx)
    if combined_img is None:
        return None
    
    return VQA(
        question_text=question_text,
        choices=[correct_answer] + [incorrect_answer] + distractors,
        correct_idx=0,
        question_images=[combined_img],
        metadata={"tag": "vqa_robot_gripper_open"}
    )
        



# S3
def vqa_object_reachable(trajectory, step_idx: int) -> Optional[VQA]:
    """
    Generates a VQA asking about whether the object is reachable.
    
    Args:
        trajectory: DROIDTrajectory instance
        step_idx: Step index to use
        
    Returns:
        VQA instance or None if generation fails
    """
    try:
        # get target object from language instruction
        language_instruction = trajectory.language_instruction
        target_object = get_target_object(language_instruction)
        if not target_object:
            print(f"No target object found in language instruction: {language_instruction}")
            return None
        
        is_blocking_obstacle = False
        
        question_text = f"Is there any obstacle blocking the robot from reaching {target_object}?"
        correct_answer = "Yes" if is_blocking_obstacle else "No"
        incorrect_answer = "No" if is_blocking_obstacle else "Yes"
        
        # Distractors
        distractors = ["Cannot be determined", "Partially reachable"]
        
        # Get stitched image from all cameras
        combined_img = stitch_all_cameras(trajectory, step_idx)
        if combined_img is None:
            return None

        return VQA(
            question_text=question_text,
            choices=[correct_answer] + [incorrect_answer] + distractors,
            correct_idx=0,
            question_images=[combined_img],
            metadata={"target_object": target_object, "tag": "vqa_object_reachable"}
        )
            
    except Exception as e:
        print(f"Error generating object reachable VQA: {e}")
        return None


# S4
def vqa_relative_direction(trajectory, camera_name: str, step_idx: int) -> Optional[VQA]:
    """
    Generates a VQA asking about the relative direction of the robot's end effector to the target object.
    """
    
    current_phase = trajectory.get_phase_for_step(step_idx)
    
    # Get current and target positions
    current_pos = np.array(trajectory.get_cartesian_state(step_idx))
    
    if current_phase == "pre_grasp":
        # use the cartesian position of first step in contact phase
        phases = trajectory.get_grasp_phases()
        contact_phase_ranges = phases["phase_ranges"]["contact"]
        if not contact_phase_ranges:
            print(f"No contact phase found in trajectory")
            return None
        contact_step_idx = contact_phase_ranges[0][0]
        contact_data = trajectory.get_non_image_data(contact_step_idx)
        contact_pos = np.array(trajectory.get_cartesian_state(contact_step_idx))

        target_object = get_target_object(trajectory.language_instruction)
        if not target_object:
            print(f"No target object found in language instruction: {trajectory.language_instruction}")
            return None
        
        question_text = f"In the image from {camera_name} at step {step_idx}, which direction is the {target_object} relative to the robot's end effector?"
        
        pos_diff = current_pos - contact_pos
        abs_diff = (pos_diff)
        
        # Determine primary vertical component (Upper/Lower)
        vertical = "Upper" if pos_diff[2] > 0 else "Lower"
        
        # Determine horizontal component based on x and y values
        # Only consider horizontal if the vertical component isn't dominant
        horizontal = ""
        
        # Check if either x or y has a significant component
        # (at least 40% of the maximum absolute difference)
        max_abs_diff = np.max(abs_diff)
        if abs_diff[0] > 0.4 * max_abs_diff or abs_diff[1] > 0.4 * max_abs_diff:
            # Determine which horizontal dimension is more significant
            if abs_diff[0] > abs_diff[1]:
                # X-axis is more significant
                horizontal = "Right" if pos_diff[0] > 0 else "Left"
            else:
                # Y-axis is more significant
                horizontal = "Forward" if pos_diff[1] > 0 else "Backward"
        
        # Combine directions or use just vertical if horizontal isn't significant
        if horizontal:
            correct_direction = f"{vertical} {horizontal}"
        else:
            correct_direction = vertical
        
        # Create all possible combined directions for choices
        all_directions = []
        for vert in ["Upper", "Lower"]:
            all_directions.append(vert)  # Just vertical
            for horiz in ["Left", "Right", "Forward", "Backward"]:
                all_directions.append(f"{vert} {horiz}")
        
        # Ensure correct direction is first in the list
        choices = [correct_direction]
        
        # Add other directions as incorrect choices, prioritizing those with same vertical component
        other_directions = [d for d in all_directions if d != correct_direction]
        # First add directions with the same vertical component
        same_vertical = [d for d in other_directions if d.startswith(vertical)]
        random.shuffle(same_vertical)
        
        # Then add directions with different vertical component
        diff_vertical = [d for d in other_directions if not d.startswith(vertical)]
        random.shuffle(diff_vertical)
        
        # Add alternatives until we have 5 total choices (increased from 4)
        for d in same_vertical + diff_vertical:
            if len(choices) < 5:
                choices.append(d)
            else:
                break
        
        # Determine correct index (should be 0 before shuffling)
        correct_idx = 0

        img, _, _ = trajectory.get_image_data(camera_name, step_idx)
        if img is None:
            print(f"No image found for {camera_name} at step {step_idx}")
            return None
        
        return VQA(
            question_text=question_text,
            choices=choices,
            correct_idx=correct_idx,
            question_images=[img],
            metadata={"tag": "vqa_relative_direction"}
        )
    else:
        return None

# S6
def vqa_relative_depth(trajectory, camera_name: str, step_idx: int) -> Optional[VQA]:
    """
    Generates a VQA asking about the relative depth of points in the scene.
    Uses depth image to sample 5 points with distinct depth values and asks
    which colored point is closest/farthest from the camera.
    """
    try:
        # Get image data
        color_img, _, depth_img = trajectory.get_image_data(camera_name, step_idx)
        
        if color_img is None or depth_img is None:
            return None
        
        # Create a copy of the color image for visualization
        vis_img = color_img.copy()
        
        # Define colors for marking points (in BGR format)
        colors = [
            (0, 0, 255),    # Red
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255)   # Purple
        ]
        color_names = ["Red", "Green", "Blue", "Yellow", "Purple"]
        
        # Filter out invalid depth values (0 usually means no depth data)
        valid_mask = (depth_img > 0.1) & (depth_img < 5.0) & (~np.isnan(depth_img))  # Reasonable depth range
        
        if np.sum(valid_mask) < 100:  # Need enough valid pixels
            return None
        
        # Get indices of valid pixels
        valid_indices = np.where(valid_mask)
        
        # Sample 5 points with distinct depths
        sampled_points = []
        sampled_depths = []
        max_attempts = 100
        
        # Ensure minimum depth difference between points (in meters)
        min_depth_diff = 0.05
        
        height, width = depth_img.shape
        attempts = 0
        
        # Keep trying until we have 5 points or run out of attempts
        while len(sampled_points) < 5 and attempts < max_attempts:
            # Randomly select an index from valid pixels
            idx = random.randrange(len(valid_indices[0]))
            y, x = valid_indices[0][idx], valid_indices[1][idx]
            
            # Get depth value at this point
            depth_value = depth_img[y, x]
            
            # Make sure point is not too close to image borders
            border = 20
            if x < border or x >= width - border or y < border or y >= height - border:
                attempts += 1
                continue
            
            # Check if this depth is distinct from already sampled depths
            is_distinct = True
            for existing_depth in sampled_depths:
                if abs(depth_value - existing_depth) < min_depth_diff:
                    is_distinct = False
                    break
            
            # If distinct, add to our samples
            if is_distinct:
                sampled_points.append((x, y))
                sampled_depths.append(depth_value)
            
            attempts += 1
        
        # If we couldn't find 5 distinct points, try with a smaller depth difference
        if len(sampled_points) < 5:
            min_depth_diff = 0.02
            attempts = 0
            
            while len(sampled_points) < 5 and attempts < max_attempts:
                idx = random.randrange(len(valid_indices[0]))
                y, x = valid_indices[0][idx], valid_indices[1][idx]
                depth_value = depth_img[y, x]
                
                border = 20
                if x < border or x >= width - border or y < border or y >= height - border:
                    attempts += 1
                    continue
                
                is_distinct = True
                for existing_depth in sampled_depths:
                    if abs(depth_value - existing_depth) < min_depth_diff:
                        is_distinct = False
                        break
                
                if is_distinct:
                    sampled_points.append((x, y))
                    sampled_depths.append(depth_value)
                
                attempts += 1
        
        # Last attempt with even smaller depth difference if needed
        if len(sampled_points) < 5:
            min_depth_diff = 0.01
            attempts = 0
            
            while len(sampled_points) < 5 and attempts < max_attempts:
                idx = random.randrange(len(valid_indices[0]))
                y, x = valid_indices[0][idx], valid_indices[1][idx]
                depth_value = depth_img[y, x]
                
                border = 20
                if x < border or x >= width - border or y < border or y >= height - border:
                    attempts += 1
                    continue
                
                is_distinct = True
                for existing_depth in sampled_depths:
                    if abs(depth_value - existing_depth) < min_depth_diff:
                        is_distinct = False
                        break
                
                if is_distinct:
                    sampled_points.append((x, y))
                    sampled_depths.append(depth_value)
                
                attempts += 1
        
        # If we still don't have 5 points, return None
        if len(sampled_points) < 5:
            return None
        
        # Draw markers on the image for visualization
        marker_size = 15
        for i, (x, y) in enumerate(sampled_points):
            cv2.circle(vis_img, (x, y), marker_size, colors[i], -1)  # Filled circle
            cv2.circle(vis_img, (x, y), marker_size + 2, (255, 255, 255), 2)  # White outline
            # Add text label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img, color_names[i], (x + marker_size, y + marker_size), 
                       font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Decide whether to ask about closest or farthest
        is_asking_closest = random.choice([True, False])
        
        # Determine the correct answer
        if is_asking_closest:
            correct_idx = np.argmin(sampled_depths)
            question_text = f"In the image from {camera_name}, which colored point is CLOSEST to the camera?"
        else:
            correct_idx = np.argmax(sampled_depths)
            question_text = f"In the image from {camera_name}, which colored point is FARTHEST from the camera?"
        
        correct_color = color_names[correct_idx]
        
        # Create VQA object
        return VQA(
            question_text=question_text,
            choices=color_names,
            correct_idx=color_names.index(correct_color),
            question_images=[vis_img],
            metadata={
                "tag": "vqa_relative_depth",
                "is_asking_closest": is_asking_closest
            }
        )
        
    except Exception as e:
        return None


# S8 
def vqa_multi_view_correspondence(trajectory, step_idx: int) -> Optional[VQA]:
    """
    Generates a VQA asking about the correspondence between two views of the scene.
    Samples a future robot position and asks which point in the second view corresponds
    to the marked point in the first view.
    """
    # Get available cameras
    camera_names = [cam for cam in trajectory.get_camera_names() if cam != "wrist"]
    if len(camera_names) < 2:
        return None  # Need at least two cameras
    
    # Randomly select two different cameras
    selected_cameras = random.sample(camera_names, 2)
    camera1, camera2 = selected_cameras
    
    # Choose a future step (between 5-15 steps ahead if possible)
    trajectory_length = trajectory.get_trajectory_length()
    min_future_step = min(step_idx + 5, trajectory_length - 1)
    max_future_step = min(step_idx + 15, trajectory_length - 1)
    
    if min_future_step >= trajectory_length - 1:
        return None  # Not enough future steps
    
    future_step = step_idx
    
    # Get the robot position at the future step
    future_data = trajectory.get_non_image_data(future_step)
    robot_pos = np.array(future_data["robot_state"]["cartesian_position"])
    
    # Get current images from both cameras
    img1, _, _ = trajectory.get_image_data(camera1, step_idx)
    img2, _, _ = trajectory.get_image_data(camera2, step_idx)
    
    if img1 is None or img2 is None:
        return None  # Missing images
    
    # Get camera parameters for both cameras
    step_data = trajectory.get_non_image_data(step_idx)
    
    # Get camera intrinsics and extrinsics
    try:
        cam1_intrinsic = np.array(step_data["cameras"][camera1]["intrinsics"]["left"])
        cam1_extrinsic = np.array(step_data["cameras"]["extrinsics"][camera1]["left"])
        
        cam2_intrinsic = np.array(step_data["cameras"][camera2]["intrinsics"]["left"])
        cam2_extrinsic = np.array(step_data["cameras"]["extrinsics"][camera2]["left"])
    except (KeyError, ValueError):
        return None  # Missing camera parameters
    
    # Project robot position to both camera views
    cam1_point = project_point_to_image(robot_pos, cam1_extrinsic, cam1_intrinsic)
    cam2_point = project_point_to_image(robot_pos, cam2_extrinsic, cam2_intrinsic)
    
    # Check if points are within image bounds
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if (cam1_point[0] < 0 or cam1_point[0] >= w1 or cam1_point[1] < 0 or cam1_point[1] >= h1 or
        cam2_point[0] < 0 or cam2_point[0] >= w2 or cam2_point[1] < 0 or cam2_point[1] >= h2):
        return None  # Points outside image bounds
    
    # Create copies for visualization
    vis_img1 = img1.copy()
    vis_img2 = img2.copy()
    
    # Mark the point in the first camera view
    marker_size = 15
    point_color = (0, 0, 255)  # Red in BGR
    cv2.circle(vis_img1, cam1_point, marker_size, point_color, -1)  # Filled circle
    cv2.circle(vis_img1, cam1_point, marker_size + 2, (255, 255, 255), 2)  # White outline
    
    # Generate 4 distractors directly in image space (increased from 3)
    distractors = []
    
    # Define minimum distance between points (in pixels)
    min_dist = 50
    
    # Create distractors at different parts of the image to ensure good distribution
    # Divide the image into quadrants and place one distractor in each (except where the correct answer is)
    
    # Find which quadrant the correct answer is in
    x_mid = w2 // 2
    y_mid = h2 // 2
    
    correct_quadrant = 0
    if cam2_point[0] < x_mid and cam2_point[1] < y_mid:
        correct_quadrant = 0  # Top-left
    elif cam2_point[0] >= x_mid and cam2_point[1] < y_mid:
        correct_quadrant = 1  # Top-right
    elif cam2_point[0] < x_mid and cam2_point[1] >= y_mid:
        correct_quadrant = 2  # Bottom-left
    else:
        correct_quadrant = 3  # Bottom-right
    
    # Define quadrant boundaries with some margin
    margin = 40
    quadrants = [
        (margin, x_mid - margin, margin, y_mid - margin),  # Top-left: (xmin, xmax, ymin, ymax)
        (x_mid + margin, w2 - margin, margin, y_mid - margin),  # Top-right
        (margin, x_mid - margin, y_mid + margin, h2 - margin),  # Bottom-left
        (x_mid + margin, w2 - margin, y_mid + margin, h2 - margin)  # Bottom-right
    ]
    
    # Generate one distractor in each quadrant except the one with the correct answer
    available_quadrants = [i for i in range(4) if i != correct_quadrant]
    random.shuffle(available_quadrants)
    
    for i in range(min(4, len(available_quadrants))):
        quad = available_quadrants[i]
        x_min, x_max, y_min, y_max = quadrants[quad]
        
        # Ensure the quadrant is large enough
        if x_max - x_min < 20 or y_max - y_min < 20:
            continue
        
        # Generate a random point in this quadrant
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        
        distractors.append((x, y))
    
    # If we need more distractors, add random points that are far enough from existing points
    while len(distractors) < 4:
        # Generate a random point in the image
        x = random.randint(margin, w2 - margin)
        y = random.randint(margin, h2 - margin)
        
        # Check if it's far enough from the correct point and other distractors
        too_close = False
        for point in [cam2_point] + distractors:
            dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
            if dist < min_dist:
                too_close = True
                break
                
        if not too_close:
            distractors.append((x, y))
    
    # Combine correct point and distractors
    all_points = [cam2_point] + distractors
    random.shuffle(all_points)  # Shuffle to randomize position
    
    # Find index of correct point after shuffling
    correct_idx = all_points.index(cam2_point)
    
    # Mark points in second camera view with letters A, B, C, D, E
    choices = ["A", "B", "C", "D", "E"]
    for i, point in enumerate(all_points):
        # Draw circle
        color = (0, 255, 0) #if point == cam2_point else (255, 0, 0)  
        cv2.circle(vis_img2, point, marker_size, color, -1)
        cv2.circle(vis_img2, point, marker_size + 2, (255, 255, 255), 2)
        
        # Add letter label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(choices[i], font, 0.9, 2)
        text_x = point[0] - text_size[0] // 2
        text_y = point[1] + text_size[1] // 2
        cv2.putText(vis_img2, choices[i], (text_x, text_y), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Combine images side by side for a better visualization
    h_max = max(vis_img1.shape[0], vis_img2.shape[0])
    # Resize images to the same height if needed
    if vis_img1.shape[0] != h_max:
        scale = h_max / vis_img1.shape[0]
        vis_img1 = cv2.resize(vis_img1, (int(vis_img1.shape[1] * scale), h_max))
    if vis_img2.shape[0] != h_max:
        scale = h_max / vis_img2.shape[0]
        vis_img2 = cv2.resize(vis_img2, (int(vis_img2.shape[1] * scale), h_max))
    
    # Create the combined image
    combined_img = np.hstack((vis_img1, vis_img2))
    
    # Add camera labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_img, f"Camera: {camera1}", (10, 30), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(combined_img, f"Camera: {camera2}", (vis_img1.shape[1] + 10, 30), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Create the question
    question_text = f"In the left image ({camera1} camera), a red dot is marked. Which point is the closest point in the right image ({camera2} camera) corresponding to the same 3D location?"
    
    # Create VQA object
    return VQA(
        question_text=question_text,
        choices=choices,
        correct_idx=correct_idx,
        question_images=[combined_img],
        metadata={
            "tag": "vqa_multi_view_correspondence",
            "camera1": camera1,
            "camera2": camera2,
            "future_step": future_step,
            "robot_position": robot_pos.tolist()
        }
    )

# I1 
def vqa_task_success_state(trajectory, timestep: int) -> Optional[VQA]:
    """
    Generates a VQA asking about the state of the task.
    Uses the last frame of the task as the problem image and determines
    success/failure from the gsutil path and phase information.
    
    Args:
        trajectory: DROIDTrajectory instance
        timestep: Step index to use for the VQA
        
    Returns:
        VQA instance or None if generation fails
    """
    # Get the language instruction
    language_instruction = trajectory.language_instruction
    
    # Check the gsutil path for success/failure
    path_indicates_success = trajectory.is_task_successful()
    
    
    # last_phase = trajectory.get_phase_for_step(last_step_idx)
    last_phase_range = trajectory.get_grasp_phases()["phase_ranges"]
    is_in_last_phase = True
    for phase_name, ranges in last_phase_range.items():
        for r in ranges:
            if r[0] > timestep:
                is_in_last_phase = False
                break
    
    # Task is successful if BOTH conditions are met:
    # 1. Current phase equals the last phase of trajectory
    # 2. "success" appears in the trajectory path
    is_success = is_in_last_phase and path_indicates_success
    
    question_text = f"The robot is to {language_instruction.lower()}. Has the robot successfully completed the task?"
    correct_answer = "Yes" if is_success else "No"
    incorrect_answer = "No" if is_success else "Yes"
    
    # Distractors
    distractors = ["Cannot be determined", "Task was not attempted"] 

    # Get stitched image from all cameras
    combined_img = stitch_all_cameras(trajectory, timestep)
    if combined_img is None:
        return None
    # Get camera names
    return VQA(
        question_text=question_text,
        choices=[correct_answer] + [incorrect_answer] + distractors,
        correct_idx=0,
        question_images=[combined_img],
        metadata={
            "tag": "vqa_task_success_state",
            "is_success": is_success,
            "is_in_last_phase": is_in_last_phase,
            "path_indicates_success": path_indicates_success,
        }
    )


# I2 
def is_stable_grasp(trajectory, timestep: int) -> Optional[VQA]:
    """
    Generates a VQA asking about the stability of the grasp.
    
    Determines grasp stability based on the robot's phase:
    - 'contact' phase is considered a stable grasp
    - 'immobilization' and 'detach' phases are considered unstable grasps
    
    Args:
        trajectory: DROIDTrajectory instance
        timestep: Step index to use for the VQA
        
    Returns:
        VQA instance or None if generation fails
    """
    # Get the current phase
    current_phase = trajectory.get_phase_for_step(timestep)
    
    # Only generate this VQA for the specified phases
    if current_phase not in ["contact", "immobilization", "detach"]:
        return None
    
    # Determine grasp stability based on phase
    is_stable = current_phase == "contact"
    
    # Get target object from language instruction
    language_instruction = trajectory.language_instruction
    target_object = get_target_object(language_instruction)
    
    # Default object name if not found
    if not target_object:
        target_object = "object"
    
    question_text = f"Is the robot's grasp of the {target_object} stable?"
    correct_answer = "Yes" if is_stable else "No"
    incorrect_answer = "No" if is_stable else "Yes"
    
    # Distractors
    distractors = ["Cannot be determined", "Partially stable"]
    
    # Get wrist camera image if available, otherwise any available camera
    images = []
    # First try to get wrist camera
    for cam_name in trajectory.get_camera_names():
        if "wrist" in cam_name.lower():
            img, _, _ = trajectory.get_image_data(cam_name, timestep)
            if img is not None:
                images.append(img)
                break
    
    # If no wrist camera, get any available camera
    if not images:
        for cam_name in trajectory.get_camera_names():
            img, _, _ = trajectory.get_image_data(cam_name, timestep)
            if img is not None:
                images.append(img)
                break
    
    # If no images found, return None
    if not images:
        return None
    
    return VQA(
        question_text=question_text,
        choices=[correct_answer] + [incorrect_answer] + distractors,
        correct_idx=0,
        question_images=images,
        metadata={
            "tag": "is_stable_grasp",
            "is_stable": is_stable,
            "phase": current_phase,
            "target_object": target_object
        }
    )


# I3
def vqa_goal_configuration(trajectory) -> Optional[VQA]:
    """
    Generates a VQA asking about the goal configuration of the robot.
    Shows five images: one from the last frame (goal configuration) and
    four from other phases of the trajectory, asking which one should be 
    the goal of the task described in the language instruction.
    
    Args:
        trajectory: DROIDTrajectory instance
        
    Returns:
        VQA instance or None if generation fails
    """
    # Get language instruction
    language_instruction = trajectory.language_instruction
    
    # Get the last frame index (goal configuration)
    last_step_idx = trajectory.get_trajectory_length() - 1
    
    # Get all phases and their ranges
    phase_data = trajectory.get_grasp_phases()
    if not phase_data or "phase_ranges" not in phase_data:
        return None
        
    phase_ranges = phase_data["phase_ranges"]
    
    # Sample frames from different phases
    sampled_frames = []
    sampled_phases = []
    
    # First add the last frame (goal configuration)
    sampled_frames.append(last_step_idx)
    
    # Get phase of the last step
    last_phase = trajectory.get_phase_for_step(last_step_idx)
    sampled_phases.append(last_phase)
    
    # Filter interested_timesteps to those not equal to last_step_idx
    filtered_frames = [step for step in trajectory.interested_timesteps if step != last_step_idx]
    
    # If we have enough frames, randomly sample 4
    if len(filtered_frames) >= 4:
        # Randomly sample 4 frames without replacement
        sample_indices = random.sample(range(len(filtered_frames)), 4)
        for idx in sample_indices:
            step_idx = filtered_frames[idx]
            sampled_frames.append(step_idx)
            sampled_phases.append(trajectory.get_phase_for_step(step_idx))
    elif len(filtered_frames) > 0:
        # If not enough frames but at least one, duplicate frames as needed
        # Shuffle the available frames first
        random.shuffle(filtered_frames)
        # Create a list of frames to sample from (with duplicates)
        frames_to_sample = filtered_frames * ((4 // len(filtered_frames)) + 1)
        # Take the first 4 frames
        for i in range(4):
            step_idx = frames_to_sample[i % len(filtered_frames)]
            sampled_frames.append(step_idx)
            sampled_phases.append(trajectory.get_phase_for_step(step_idx))
    else:
        # If no frames available at all, return None
        return None
        
    # Make sure we have exactly 5 frames (1 last frame + 4 sampled frames)
    if len(sampled_frames) != 5:
        return None
        
    # Get only the first five frames if we have more
    sampled_frames = sampled_frames[:5]
    sampled_phases = sampled_phases[:5]
    # print(f"Sampled frames: {sampled_frames}, Sampled phases: {sampled_phases}")
    # Get images for each frame
    images = []
    for step_idx in sampled_frames:
        img, _, _ = trajectory.get_image_data(trajectory.get_camera_names()[0], step_idx)
        if img is None:
            return None  # Skip if any image is missing
        images.append(img)
    
    # Make sure we have exactly 5 images
    if len(images) != 5:
        return None
        
    # Create a grid image (3x2 with one empty spot)
    # Resize images to the same size if needed
    img_heights = [img.shape[0] for img in images]
    img_widths = [img.shape[1] for img in images]
    
    # Use the maximum height and width
    max_height = max(img_heights)
    max_width = max(img_widths)
    
    # Resize images
    resized_images = []
    for img in images:
        if img.shape[0] != max_height or img.shape[1] != max_width:
            resized_img = cv2.resize(img, (max_width, max_height))
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    # Create a grid image (3 in first row, 2 in second row)
    top_row = np.hstack((resized_images[0], resized_images[1], resized_images[2]))
    bottom_row = np.hstack((resized_images[3], resized_images[4], np.zeros_like(resized_images[0])))  # Add empty spot
    grid_image = np.vstack((top_row, bottom_row))
    
    # Ensure the image is in the correct format for OpenCV operations
    if grid_image.dtype != np.uint8:
        grid_image = grid_image.astype(np.uint8)
    
    # Make sure image has 3 channels (BGR)
    if len(grid_image.shape) == 2:  # If grayscale
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_GRAY2BGR)
    elif grid_image.shape[2] == 4:  # If RGBA
        grid_image = grid_image[:, :, :3]  # Keep only BGR channels
    
    # Add labels A, B, C, D, E to the five grid positions
    quadrant_labels = ["A", "B", "C", "D", "E"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    font_color = (0, 0, 255)  # Blue color
    
    # Calculate positions for labels (top-left corner of each grid spot)
    label_positions = [
        (30, 50),                    # Top-left (A)
        (max_width + 30, 50),        # Top-middle (B)
        (2 * max_width + 30, 50),    # Top-right (C)
        (30, max_height + 50),       # Bottom-left (D)
        (max_width + 30, max_height + 50) # Bottom-middle (E)
    ]
    
    # Add text labels to image
    for idx, (label, pos) in enumerate(zip(quadrant_labels, label_positions)):
        cv2.putText(grid_image, label, pos, font, font_scale, font_color, font_thickness)
    
    # The correct answer is the image from the last frame, which is at index 0 in our list
    correct_label = quadrant_labels[0]
    
    # Create choices as the labels
    choices = [f"Configuration {label}" for label in quadrant_labels]
    
    # Make question text
    question_text = f"The robot's task is to {language_instruction.lower()}. Which configuration shows the goal state that the robot should achieve?"
    
    # Create the VQA instance with the grid image as question image
    return VQA(
        question_text=question_text,
        choices=choices,
        correct_idx=0,  # The first choice is the correct one
        question_images=[grid_image],  # Grid image as question image
        choice_images=None,  # No choice images
        metadata={
            "tag": "vqa_goal_configuration",
            "language_instruction": language_instruction,
            "sampled_frames": sampled_frames,
            "sampled_phases": sampled_phases,
        }
    )


# I4
def vqa_action_understanding(trajectory, timestep: int) -> Optional[VQA]:
    """
    Generates a VQA asking about the current action phase of the robot.
    
    The robot's action phases are:
    - pre_grasp: Open gripper before approaching object
    - immobilization: Gripper closing but not yet in firm contact
    - contact: Gripper in firm contact with object
    - detach: Gripper opening but not fully open
    - post_grasp: Gripper fully open after grasp
    
    Args:
        trajectory: DROIDTrajectory instance
        timestep: Step index to use for the VQA
        
    Returns:
        VQA instance or None if generation fails
    """
    # Get the current phase
    current_phase = trajectory.get_phase_for_step(timestep)
    
    # Only proceed if we have a valid phase (not transition)
    if current_phase == "transition" or not current_phase:
        return None
        
    # Get language instruction and target object
    language_instruction = trajectory.language_instruction
    target_object = get_target_object(language_instruction)
    
    # Default object name if not found
    if not target_object:
        target_object = "object"
        
    # Define phase descriptions for the question
    phase_descriptions = {
        "pre_grasp": f"Approaching the {target_object} with open gripper",
        "immobilization": f"Closing gripper to grasp the {target_object}",
        "contact": f"Firmly grasping the {target_object}",
        "detach": f"Releasing the {target_object} by opening gripper",
        "post_grasp": f"Moving away with open gripper after releasing the {target_object}"
    }
    
    # Ensure we have all 5 phases for choices
    all_phases = list(phase_descriptions.keys())
    
    # Use the full description for the correct answer
    correct_answer = phase_descriptions[current_phase]
    
    # Create incorrect choices using the other phase descriptions
    # Use all other phases to get 4 incorrect choices
    other_phases = [phase for phase in all_phases if phase != current_phase]
    incorrect_choices = [phase_descriptions[phase] for phase in other_phases]
    
    # Create the question text
    question_text = f"The robot is tasked to {language_instruction}. The robot is interacting with the {target_object}. Which phase of the grasp action is shown in the image?"
    
    # Get stitched image from all cameras
    combined_img = stitch_all_cameras(trajectory, timestep)
    if combined_img is None:
        return None
        
    return VQA(
        question_text=question_text,
        choices=[correct_answer] + incorrect_choices,
        correct_idx=0,
        question_images=[combined_img],
        metadata={
            "tag": "vqa_action_understanding",
            "current_phase": current_phase,
            "target_object": target_object,
            "timestep": timestep
        }
    )
            

def vqa_next_action(trajectory, timestep: int) -> Optional[VQA]:
    """
    Generates a VQA asking about the next action phase of the robot.
    
    The robot's phase sequence is:
    pre_grasp → immobilization → contact → detach → post_grasp
    
    Args:
        trajectory: DROIDTrajectory instance
        timestep: Step index to use for the VQA
        
    Returns:
        VQA instance or None if generation fails
    """
    # Get the current phase
    current_phase = trajectory.get_phase_for_step(timestep)
    
    # Only proceed if we have a valid phase (not transition)
    if current_phase == "transition" or not current_phase:
        return None
        
    # Define phase sequence
    phase_sequence = ["pre_grasp", "immobilization", "contact", "detach", "post_grasp"]
    
    # Determine the next phase based on current phase
    try:
        current_index = phase_sequence.index(current_phase)
        next_index = (current_index + 1) % len(phase_sequence)
        next_phase = phase_sequence[next_index]
    except ValueError:
        # If current phase not in sequence, we can't predict next
        return None
        
    # Get language instruction and target object
    language_instruction = trajectory.language_instruction
    target_object = get_target_object(language_instruction)
    
    # Default object name if not found
    if not target_object:
        target_object = "object"
        
    # Define phase descriptions for the question (same as in action_understanding)
    phase_descriptions = {
        "pre_grasp": f"Approaching the {target_object} with open gripper",
        "immobilization": f"Closing gripper to grasp the {target_object}",
        "contact": f"Firmly grasping the {target_object}",
        "detach": f"Releasing the {target_object} by opening gripper",
        "post_grasp": f"Moving away with open gripper after releasing the {target_object}"
    }
    
    # Use the full description for the correct answer
    correct_answer = phase_descriptions[next_phase]
    
    # Create incorrect choices using the other phase descriptions
    # Use all other phases to get 4 incorrect choices
    other_phases = [phase for phase in phase_sequence if phase != next_phase]
    incorrect_choices = [phase_descriptions[phase] for phase in other_phases]
    
    # Create the question text
    question_text = f"The robot is tasked to {language_instruction}. After {phase_descriptions[current_phase]}, what will be the robot's NEXT action phase?"
    
    # Get stitched image from all cameras
    combined_img = stitch_all_cameras(trajectory, timestep)
    if combined_img is None:
        return None
        
    return VQA(
        question_text=question_text,
        choices=[correct_answer] + incorrect_choices,
        correct_idx=0,
        question_images=[combined_img],
        metadata={
            "tag": "vqa_next_action",
            "current_phase": current_phase,
            "next_phase": next_phase,
            "target_object": target_object,
            "timestep": timestep
        }
    )

# I6
def vqa_trajectory_understanding(trajectory, camera_name: str) -> Optional[VQA]:
    """
    Generates a VQA asking about the full trajectory of the robot.
    Shows visualized trajectory and asks which language instruction best describes it.
    
    Args:
        trajectory: DROIDTrajectory instance
        camera_name: Camera to use for visualization
        timestep: Step index to use as reference (middle point of visualization)
        
    Returns:
        VQA instance or None if generation fails
    """
    # Get the correct language instruction
    correct_instruction = trajectory.language_instruction
    
    if not correct_instruction:
        return None
        
    # Use the full trajectory for visualization
    trajectory_length = trajectory.get_trajectory_length()
    
    # Visualize the full trajectory
    visualization_results = trajectory.visualize_trajectory(
        start_idx=0,
        end_idx=trajectory_length - 1,
        image_idx = list(trajectory.interested_timesteps)[0],
        save_to_disk=False 
    )
    
    # Check if visualization was successful
    if not visualization_results:
        return None
        
    # Get visualized images
    left_trajectory_img = visualization_results.get(f"{camera_name}_left_trajectory")
    right_trajectory_img = visualization_results.get(f"{camera_name}_right_trajectory")
    
    if left_trajectory_img is None:
        return None
        
    # Create alternative incorrect language instructions
    # List of template instructions to generate alternatives
    instruction_templates = [
        "Pick up the {} from the {}",
        "Move the {} to the {}",
        "Place the {} on the {}",
        "Push the {} towards the {}",
        "Rotate the {} clockwise",
        "Slide the {} to the right",
        "Grab the {} with the gripper",
        "Lift the {} upward",
        "Drop the {} into the {}",
        "Align the {} with the {}"
    ]
    common_objects = extract_objects_from_instruction(correct_instruction)
    common_locations = extract_locations_from_instruction(correct_instruction)
    if len(common_objects) < 2:
        common_objects = ["cup", "box", "ball", "book", "pen", "toy", "block", "plate", "bottle", "container"]
    if len(common_locations) < 2:
        common_locations = ["table", "shelf", "drawer", "bin", "tray", "surface", "floor", "corner", "center", "platform"]
    
    # Generate alternative language instructions
    incorrect_instructions = []
    while len(incorrect_instructions) < 4:  # Increased from 3 to 4
        # Choose a random template
        template = random.choice(instruction_templates)
        
        # If template has two placeholders, fill with object and location
        if template.count("{}") == 2:
            obj = random.choice(common_objects)
            loc = random.choice(common_locations)
            instruction = template.format(obj, loc)
        # If template has one placeholder, fill with object only
        elif template.count("{}") == 1:
            obj = random.choice(common_objects)
            instruction = template.format(obj)
        else:
            instruction = template
            
        # Ensure the instruction is different from the correct one and not duplicated
        if (instruction != correct_instruction and 
            instruction not in incorrect_instructions):
            incorrect_instructions.append(instruction)
    
    # Create question text
    question_text = "Which language instruction best describes the robot's trajectory shown in the image?"
    
    # Create VQA instance
    return VQA(
        question_text=question_text,
        choices=[correct_instruction] + incorrect_instructions,
        correct_idx=0,
        question_images=[left_trajectory_img, right_trajectory_img],
        metadata={
            "tag": "vqa_trajectory_understanding",
            "language_instruction": correct_instruction,
            "camera_name": camera_name,
        }
    )


# S6
def generate_action_direction_selection_vqa(trajectory, step_idx: int, camera_name: str) -> Optional[VQA]:
    """
    Generates a VQA asking about which colored arrow represents the correct direction
    the robot will move based on its current action target.
    
    Args:
        trajectory: DROIDTrajectory instance
        step_idx: The step index to generate the VQA for
        camera_name: The camera to use for visualization
        
    Returns:
        VQA instance or None if generation fails
    """
    language_instruction = trajectory.language_instruction

    
    # Ensure we can get a future step for arrow visualization
    if step_idx + 1 >= trajectory.get_trajectory_length():
        return None  # Need at least one future step
        
    # Extract step data
    step_data = trajectory.get_non_image_data(step_idx)
    future_step_data = trajectory.get_non_image_data(step_idx + 1)
    
    # Get camera image
    left_img, _, _ = trajectory.get_image_data(camera_name, step_idx)
    if left_img is None:
        return None  # No image available
        
    # Get current and target positions
    current_pos = np.array(step_data["robot_state"]["cartesian_position"])
    if "target_cartesian_position" in step_data["action"]:
        target_pos = np.array(step_data["action"]["target_cartesian_position"])
    else:
        target_pos = np.array(future_step_data["robot_state"]["cartesian_position"])
    future_pos = np.array(future_step_data["robot_state"]["cartesian_position"])
    
    # Get camera parameters
    left_intrinsic = np.array(step_data["cameras"][camera_name]["intrinsics"]["left"])
    left_extrinsic = np.array(step_data["cameras"]["extrinsics"][camera_name]["left"])
    
    # Project points to image coordinates
    current_px = project_point_to_image(current_pos, left_extrinsic, left_intrinsic)
    target_px = project_point_to_image(target_pos, left_extrinsic, left_intrinsic)
    future_px = project_point_to_image(future_pos, left_extrinsic, left_intrinsic)
    
    # Check if points are within image bounds
    if (current_px[0] < 0 or current_px[1] < 0 or 
        target_px[0] < 0 or target_px[1] < 0 or
        future_px[0] < 0 or future_px[1] < 0):
        return None  # Points outside image bounds
        
    # Create visualization image
    vis_img = left_img.copy()
    
    # Calculate the actual direction vector (from current to future position)
    actual_dx = future_px[0] - current_px[0]
    actual_dy = future_px[1] - current_px[1]
    actual_angle = math.atan2(actual_dy, actual_dx)
    
    # Generate five different colored arrows in different directions
    arrow_length = 80  # Fixed arrow length in pixels
    arrow_colors = [
        (0, 0, 255),    # Red (BGR)
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255)   # Purple
    ]
    color_names = ["Red", "Green", "Blue", "Yellow", "Purple"]
    
    # Create 5 directions (one correct, four incorrect)
    # Ensure the correct direction is included
    directions = []
    
    # Add the actual direction first (this will be the correct answer)
    directions.append(actual_angle)
    
    # Generate four random alternative directions that are meaningfully different
    min_angle_diff = math.radians(30)  # Minimum difference between directions
    while len(directions) < 5:
        # Generate a random angle
        new_angle = random.uniform(0, 2 * math.pi)
        
        # Check if it's different enough from existing directions
        is_different = True
        for angle in directions:
            diff = abs(new_angle - angle)
            diff = min(diff, 2 * math.pi - diff)  # Handle circular distance
            if diff < min_angle_diff:
                is_different = False
                break
                
        if is_different:
            directions.append(new_angle)
            
    # Shuffle the directions with their colors
    direction_colors = list(zip(directions, arrow_colors, color_names))
    random.shuffle(direction_colors)
    
    # Find the index of the correct direction after shuffling
    correct_idx = -1
    for i, (angle, color, name) in enumerate(direction_colors):
        # Draw the arrow
        end_x = int(current_px[0] + arrow_length * math.cos(angle))
        end_y = int(current_px[1] + arrow_length * math.sin(angle))
        cv2.arrowedLine(vis_img, current_px, (end_x, end_y), color, 2, tipLength=0.3)
        
        # Check if this is the correct direction
        if abs(angle - actual_angle) < 0.01:  # Small threshold for floating point comparison
            correct_idx = i
            
    # Draw the current position marker
    cv2.circle(vis_img, current_px, 5, (255, 255, 255), -1)  # White circle for current position
    
    # Generate question text
    question_text = f"The robot task is to {language_instruction}. Which colored arrow correctly shows the direction the robot will move next?"
    
    # Generate choices as color names
    choices = [color_name for _, _, color_name in direction_colors]
    
    # Create the VQA
    return VQA(
        question_text=question_text,
        choices=choices,
        correct_idx=correct_idx,
        question_images=[vis_img],
        metadata={"tag": "vqa_action_direction_selection"}
    )
    

# I5
def vqa_temporal_sequence(trajectory, camera_name: str, timesteps: List[int]) -> Optional[VQA]:
    """
    Generates a VQA based on a temporal sequence of frames concatenated together.
    Shows the progression of the robot's actions over time and asks about the correct sequence.
    
    Args:
        trajectory: DROIDTrajectory instance
        camera_name: Camera name to use for visualization
        timesteps: List of timesteps to use for the sequence
        
    Returns:
        VQA instance or None if generation fails
    """
    if len(timesteps) < 3:
        return None  # Need at least 3 frames for a meaningful sequence
        
    # Extract the images for each timestep
    frames = []
    for step_idx in timesteps:
        img, _, _ = trajectory.get_image_data(camera_name, step_idx)
        if img is None:
            return None  # Skip if any image is missing
        frames.append(img)
    
    # Resize all frames to the same size
    heights = [frame.shape[0] for frame in frames]
    widths = [frame.shape[1] for frame in frames]
    
    target_height = int(np.median(heights))
    target_width = int(np.median(widths))
    
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, (target_width, target_height))
        resized_frames.append(resized)
    
    # Create a single image with all frames concatenated horizontally
    combined_img = np.hstack(resized_frames)
    
    # Add frame numbers or timestamps
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (0, 255, 255)  # Yellow
    
    # Add timestep labels
    for i, step_idx in enumerate(timesteps):
        label_pos = (i * target_width + 10, 30)
        cv2.putText(combined_img, f"t={step_idx}", label_pos, font, font_scale, font_color, font_thickness)
    
    # Create question about the sequence
    # Get the language instruction
    language_instruction = trajectory.language_instruction
    
    # Get the phases for each timestep
    phases = [trajectory.get_phase_for_step(step_idx) for step_idx in timesteps]
    
    # Create question based on the phases shown
    if all(phase is not None for phase in phases):
        # Option 1: Ask about the correct sequence of phases
        question_text = f"For the task '{language_instruction}', what is the correct sequence of action phases shown in the images from left to right?"
        
        # Create correct answer
        correct_answer = " → ".join([phase.capitalize() for phase in phases])
        
        # Create incorrect alternatives by shuffling the phases
        incorrect_choices = []
        while len(incorrect_choices) < 4:  # Increased from 3 to 4
            shuffled = phases.copy()
            random.shuffle(shuffled)
            shuffled_text = " → ".join([phase.capitalize() for phase in shuffled])
            if shuffled_text != correct_answer and shuffled_text not in incorrect_choices:
                incorrect_choices.append(shuffled_text)
    else:
        # Option 2: Ask about the overall action being performed
        question_text = f"What task is the robot performing in this sequence of images?"
        
        correct_answer = language_instruction
        
        # Create plausible wrong answers
        instruction_templates = [
            "Pick up the {} from the {}",
            "Move the {} to the {}",
            "Place the {} on the {}",
            "Push the {} towards the {}",
            "Flip the {} onto its {}",
            "Stack the {} on top of the {}",
            "Arrange the {} next to the {}"
        ]
        
        common_objects = ["cup", "box", "ball", "toy", "block", "bottle", "plate", "container"]
        common_locations = ["table", "shelf", "bin", "tray", "surface", "corner", "edge", "side"]
        
        incorrect_choices = []
        while len(incorrect_choices) < 4:  # Increased from 3 to 4
            template = random.choice(instruction_templates)
            obj = random.choice(common_objects)
            loc = random.choice(common_locations)
            alternative = template.format(obj, loc)
            if alternative != correct_answer and alternative not in incorrect_choices:
                incorrect_choices.append(alternative)
    
    # Create VQA instance
    return VQA(
        question_text=question_text,
        choices=[correct_answer] + incorrect_choices,
        correct_idx=0,
        question_images=[combined_img],
        metadata={
            "tag": "vqa_temporal_sequence",
            "timesteps": timesteps,
            "phases": phases,
            "camera_name": camera_name
        }
    )
    

def stitch_all_cameras(trajectory, step_idx: int, layout: Optional[Tuple[int, int]] = None,
                      label_config: Optional[Dict] = None) -> Optional[np.ndarray]:
    """
    Stitch together images from all available cameras into a single grid image.
    Uses highest resolution and pads smaller images with black space.
    
    Args:
        trajectory: DROIDTrajectory instance
        step_idx: Step index to use
        layout: Optional tuple (rows, cols) specifying the grid layout
        label_config: Optional dictionary with label configuration
        
    Returns:
        Combined image with all camera views or None if no images available
    """
    try:
        # Get all available cameras
        camera_names = trajectory.get_camera_names()
        if not camera_names:
            return None
            
        # Get images from all cameras
        images = []
        valid_camera_names = []
        for camera_name in camera_names:
            img, _, _ = trajectory.get_image_data(camera_name, step_idx)
            if img is not None:
                images.append(img)
                valid_camera_names.append(camera_name)
                
        if not images:
            return None
            
        # Default label configuration
        default_label_config = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'font_scale': 0.8,
            'font_thickness': 2,
            'font_color': (0, 255, 255),  # Yellow in BGR
            'position': 'top_left',
            'offset': (10, 30)
        }
        
        # Merge provided label config with defaults
        if label_config:
            default_label_config.update(label_config)
        label_config = default_label_config
        
        # Get maximum dimensions
        max_height = max(img.shape[0] for img in images)
        max_width = max(img.shape[1] for img in images)
        
        # Create padded images
        padded_images = []
        for img in images:
            # Create black canvas of max size
            padded_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            
            # Calculate position to paste original image (centered)
            y_offset = (max_height - img.shape[0]) // 2
            x_offset = (max_width - img.shape[1]) // 2
            
            # Ensure image has 3 channels
            if len(img.shape) == 2:  # If grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # If RGBA
                img = img[:, :, :3]  # Keep only BGR channels
                
            # Paste original image onto black canvas
            padded_img[y_offset:y_offset + img.shape[0], 
                      x_offset:x_offset + img.shape[1]] = img
            padded_images.append(padded_img)
        
        # Determine grid layout
        num_images = len(padded_images)
        if layout is None:
            # Auto-determine layout based on number of images
            if num_images <= 2:
                rows, cols = 1, num_images
            else:
                rows = (num_images + 1) // 2
                cols = min(2, num_images)
        else:
            rows, cols = layout
        
        # Create grid of images
        grid = []
        for i in range(rows):
            row_images = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(padded_images):
                    row_images.append(padded_images[idx])
                else:
                    # Create blank image for empty spots
                    blank_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                    row_images.append(blank_img)
            grid.append(np.hstack(row_images))
        
        # Combine rows
        combined_img = np.vstack(grid)
        
        # Add camera labels
        for i, cam_name in enumerate(valid_camera_names[:num_images]):
            # Calculate position in grid
            row = i // cols
            col = i % cols
            
            # Calculate base position for this image in the grid
            base_x = col * max_width
            base_y = row * max_height
            
            # Calculate label position based on configuration
            offset_x, offset_y = label_config['offset']
            if label_config['position'] == 'top_left':
                x = base_x + offset_x
                y = base_y + offset_y
            elif label_config['position'] == 'top_right':
                x = base_x + max_width - offset_x
                y = base_y + offset_y
            elif label_config['position'] == 'bottom_left':
                x = base_x + offset_x
                y = base_y + max_height - offset_y
            else:  # bottom_right
                x = base_x + max_width - offset_x
                y = base_y + max_height - offset_y
            
            # Add label
            cv2.putText(
                combined_img,
                cam_name,
                (x, y),
                label_config['font'],
                label_config['font_scale'],
                label_config['font_color'],
                label_config['font_thickness']
            )
        
        return combined_img
        
    except Exception as e:
        print(f"Error stitching camera images: {e}")
        print(traceback.format_exc())
        return None

def enhance_vqa_with_metadata(vqa: VQA, vlm_metadata: Dict) -> VQA:
    """
    Enhance a VQA instance with VLM metadata by adding it to the metadata 
    and potentially modifying the question to include context.
    
    Args:
        vqa: The VQA instance to enhance
        vlm_metadata: Dictionary with VLM-extracted metadata
        
    Returns:
        Enhanced VQA instance
    """
    # Add VLM metadata to the VQA metadata
    if "metadata" not in vqa.__dict__ or vqa.metadata is None:
        vqa.metadata = {}
    
    for key, value in vlm_metadata.items():
        if value:  # Only add non-empty values
            vqa.metadata[f"vlm_{key}"] = value
    
    return vqa

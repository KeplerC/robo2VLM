"""
Utility functions for AgiBotWorld VQA generation.

Includes:
- NLP utilities for extracting objects and actions from text
- Image processing utilities
- Skill taxonomy and descriptions
"""

import re
import random
from typing import List, Optional, Set, Tuple
import numpy as np
import cv2


# ==================== Skill Taxonomy ====================

AGIBOT_SKILLS = {
    # Manipulation
    "Pick": "Picking up an object",
    "Place": "Placing an object down",
    "Grasp": "Grasping an object",
    "Release": "Releasing a held object",
    "Drop": "Dropping an object",

    # Movement
    "Push": "Pushing an object",
    "Pull": "Pulling an object",
    "PullApart": "Pulling apart items",
    "Lift": "Lifting an object",
    "Lower": "Lowering an object",
    "Move": "Moving in a direction",

    # Object handling
    "Fold": "Folding material or clothing",
    "Unfold": "Unfolding material",
    "Stretch": "Stretching material",
    "Stack": "Stacking items on top of each other",
    "Insert": "Inserting object into a space",

    # Dual-arm coordination
    "HandOver": "Transferring object between arms",
    "Hold": "Holding position or object steady",

    # Tool use
    "Pour": "Pouring liquid from container",
    "Brush": "Brushing with a tool",
    "Wipe": "Wiping a surface",
    "Iron": "Ironing clothes",
    "Scan": "Scanning a barcode",
    "dip": "Dipping object in liquid",

    # Environment interaction
    "Open": "Opening a container or door",
    "Close": "Closing a container or door",
    "PressButton": "Pressing a button",
    "Press": "Applying pressure",
    "Tap": "Tapping a surface",
    "Shake": "Shaking an object",
    "Hang": "Hanging an item"
}

# Skills that indicate active arm usage
LEFT_ARM_KEYWORDS = ["left arm", "left hand", "with the left"]
RIGHT_ARM_KEYWORDS = ["right arm", "right hand", "with the right"]
BOTH_ARMS_KEYWORDS = ["both arms", "both hands", "coordinate arms", "with both"]

# Skills that typically involve grasping
GRASP_SKILLS = {"Pick", "Grasp", "Lift", "Hold"}
RELEASE_SKILLS = {"Place", "Release", "Drop"}
BIMANUAL_SKILLS = {"HandOver", "Fold", "Unfold", "Stretch"}


# ==================== NLP Utilities ====================

def extract_target_object(text: str) -> Optional[str]:
    """
    Extract the target object from an action text.

    Args:
        text: Action text (e.g., "Pick up the air column film outside the carton")

    Returns:
        Extracted object name or None
    """
    if not text:
        return None

    # Common patterns for object extraction
    patterns = [
        r"(?:pick up|grab|grasp|lift|hold|take|retrieve)\s+(?:the\s+)?(.+?)(?:\s+from|\s+on|\s+with|\s+in|\.|$)",
        r"(?:place|put|drop|release)\s+(?:the\s+)?(?:held\s+)?(.+?)(?:\s+into|\s+on|\s+in|\s+onto|\.|$)",
        r"(?:pour|brush|wipe|iron|scan)\s+(?:the\s+)?(.+?)(?:\s+with|\s+into|\s+on|\.|$)",
        r"(?:open|close|push|pull)\s+(?:the\s+)?(.+?)(?:\s+with|\s+door|\.|$)",
    ]

    text_lower = text.lower()

    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            # Clean up the object name
            obj = re.sub(r'\s+held\s+.*', '', obj)
            obj = re.sub(r'\s+with\s+.*', '', obj)
            if len(obj) > 2 and len(obj) < 50:
                return obj

    # Fallback: extract nouns after common verbs
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in ['the', 'a', 'an'] and i + 1 < len(words):
            # Get the next few words as the object
            obj_words = []
            for j in range(i + 1, min(i + 5, len(words))):
                if words[j].lower() in ['from', 'to', 'into', 'on', 'with', 'in', 'onto']:
                    break
                obj_words.append(words[j])
            if obj_words:
                return ' '.join(obj_words).rstrip('.,')

    return None


def extract_objects_from_instruction(text: str) -> List[str]:
    """
    Extract all objects mentioned in an instruction.

    Args:
        text: Instruction text

    Returns:
        List of object names
    """
    if not text:
        return []

    objects = []

    # Pattern to match "the <object>"
    pattern = r"the\s+([a-zA-Z\s]+?)(?:\s+(?:from|to|into|on|with|in|onto|held|and|or)|[.,]|$)"
    matches = re.findall(pattern, text.lower())

    for match in matches:
        obj = match.strip()
        if len(obj) > 2 and len(obj) < 40:
            # Filter out common non-objects
            if obj not in ['left', 'right', 'front', 'back', 'top', 'bottom', 'side']:
                objects.append(obj)

    return list(set(objects))


def extract_locations_from_instruction(text: str) -> List[str]:
    """
    Extract location references from an instruction.

    Args:
        text: Instruction text

    Returns:
        List of location names
    """
    if not text:
        return []

    locations = []

    # Patterns for locations
    patterns = [
        r"(?:from|on|in|into|onto|at)\s+(?:the\s+)?([a-zA-Z\s]+?)(?:\s+with|\s+and|[.,]|$)",
    ]

    text_lower = text.lower()

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            loc = match.strip()
            if len(loc) > 2 and len(loc) < 40:
                locations.append(loc)

    return list(set(locations))


def detect_active_arm(action_text: str) -> Optional[str]:
    """
    Detect which arm is active in an action based on text.

    Args:
        action_text: Action description text

    Returns:
        "left", "right", "both", or None if unclear
    """
    text_lower = action_text.lower()

    # Check for both arms first
    for keyword in BOTH_ARMS_KEYWORDS:
        if keyword in text_lower:
            return "both"

    # Check for specific arm mentions
    has_left = any(kw in text_lower for kw in LEFT_ARM_KEYWORDS)
    has_right = any(kw in text_lower for kw in RIGHT_ARM_KEYWORDS)

    if has_left and has_right:
        return "both"
    elif has_left:
        return "left"
    elif has_right:
        return "right"

    return None


def get_skill_description(skill: str, target_object: Optional[str] = None) -> str:
    """
    Get a natural language description of a skill.

    Args:
        skill: Skill name
        target_object: Optional target object for context

    Returns:
        Human-readable description
    """
    base_description = AGIBOT_SKILLS.get(skill, f"Performing {skill}")

    if target_object:
        # Customize based on skill type
        if skill == "Pick":
            return f"Picking up the {target_object}"
        elif skill == "Place":
            return f"Placing the {target_object}"
        elif skill == "Grasp":
            return f"Grasping the {target_object}"
        elif skill == "Push":
            return f"Pushing the {target_object}"
        elif skill == "Pull":
            return f"Pulling the {target_object}"
        elif skill == "Pour":
            return f"Pouring from the {target_object}"

    return base_description


# ==================== Image Processing Utilities ====================

def resize_image(img: np.ndarray, max_size: int = 320) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.

    Args:
        img: Input image
        max_size: Maximum dimension (default 320 for longest side)

    Returns:
        Resized image
    """
    h, w = img.shape[:2]

    if max(h, w) <= max_size:
        return img

    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_image_grid(images: List[np.ndarray], rows: int = 2, cols: int = 3,
                      labels: Optional[List[str]] = None) -> np.ndarray:
    """
    Create a grid of images.

    Args:
        images: List of images
        rows: Number of rows
        cols: Number of columns
        labels: Optional labels for each image

    Returns:
        Grid image
    """
    if not images:
        return None

    # Resize all images to same size
    target_h = max(img.shape[0] for img in images)
    target_w = max(img.shape[1] for img in images)

    resized = []
    for img in images:
        if img.shape[0] != target_h or img.shape[1] != target_w:
            resized_img = cv2.resize(img, (target_w, target_h))
        else:
            resized_img = img.copy()
        resized.append(resized_img)

    # Pad with black images if needed
    total_slots = rows * cols
    while len(resized) < total_slots:
        black = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized.append(black)

    # Add labels if provided
    if labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        font_color = (0, 0, 255)  # Red

        for i, (img, label) in enumerate(zip(resized, labels)):
            if label:
                cv2.putText(img, label, (30, 60), font, font_scale, font_color, font_thickness)

    # Build grid
    grid_rows = []
    for r in range(rows):
        start_idx = r * cols
        end_idx = start_idx + cols
        row_images = resized[start_idx:end_idx]
        grid_rows.append(np.hstack(row_images))

    return np.vstack(grid_rows)


def create_temporal_montage(frames: List[np.ndarray],
                            labels: Optional[List[str]] = None,
                            max_width: int = 1920) -> np.ndarray:
    """
    Create a horizontal montage of frames showing temporal progression.

    Args:
        frames: List of frames in temporal order
        labels: Optional labels for each frame
        max_width: Maximum width of result

    Returns:
        Montage image
    """
    if not frames:
        return None

    n_frames = len(frames)

    # Calculate target width for each frame
    sample_h, sample_w = frames[0].shape[:2]
    target_w = min(sample_w, max_width // n_frames)
    scale = target_w / sample_w
    target_h = int(sample_h * scale)

    # Resize and annotate frames
    processed = []
    for i, frame in enumerate(frames):
        resized = cv2.resize(frame, (target_w, target_h))

        # Add frame number
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized, f"Frame {i+1}", (10, 30), font, 0.7, (255, 255, 0), 2)

        # Add label if provided
        if labels and i < len(labels):
            cv2.putText(resized, labels[i], (10, target_h - 10), font, 0.5, (255, 255, 255), 1)

        processed.append(resized)

    return np.hstack(processed)


def add_text_overlay(img: np.ndarray, text: str,
                     position: Tuple[int, int] = (10, 30),
                     font_scale: float = 0.7,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to an image.

    Args:
        img: Input image
        text: Text to add
        position: Text position (x, y)
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness

    Returns:
        Image with text overlay
    """
    result = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return result


# ==================== Question Generation Utilities ====================

def generate_distractor_instructions(correct_instruction: str,
                                    num_distractors: int = 4) -> List[str]:
    """
    Generate distractor instructions that are different from the correct one.

    Args:
        correct_instruction: The correct instruction
        num_distractors: Number of distractors to generate

    Returns:
        List of distractor instructions
    """
    templates = [
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

    objects = ["cup", "box", "ball", "book", "pen", "toy", "block", "plate", "bottle", "container",
               "cloth", "bag", "tool", "item", "product", "package"]
    locations = ["table", "shelf", "drawer", "bin", "tray", "surface", "floor", "corner",
                 "center", "platform", "counter", "basket", "cart"]

    # Extract objects from correct instruction to avoid using same ones
    correct_objects = extract_objects_from_instruction(correct_instruction)

    distractors = []
    attempts = 0
    max_attempts = 50

    while len(distractors) < num_distractors and attempts < max_attempts:
        template = random.choice(templates)

        # Fill template
        if template.count("{}") == 2:
            obj = random.choice(objects)
            loc = random.choice(locations)
            instruction = template.format(obj, loc)
        elif template.count("{}") == 1:
            obj = random.choice(objects)
            instruction = template.format(obj)
        else:
            instruction = template

        # Check if different enough from correct
        if (instruction.lower() != correct_instruction.lower() and
            instruction not in distractors):
            distractors.append(instruction)

        attempts += 1

    return distractors


def shuffle_choices(correct_choice: str, incorrect_choices: List[str],
                    correct_images: Optional[List[np.ndarray]] = None,
                    incorrect_images: Optional[List[np.ndarray]] = None) -> Tuple[List[str], int, Optional[List[np.ndarray]]]:
    """
    Shuffle choices and track the new correct index.

    Args:
        correct_choice: The correct choice text
        incorrect_choices: List of incorrect choice texts
        correct_images: Optional images for correct choice
        incorrect_images: Optional images for incorrect choices

    Returns:
        Tuple of (shuffled_choices, new_correct_idx, shuffled_images)
    """
    all_choices = [correct_choice] + incorrect_choices

    if correct_images is not None and incorrect_images is not None:
        all_images = correct_images + incorrect_images
    else:
        all_images = None

    # Create indices and shuffle
    indices = list(range(len(all_choices)))
    random.shuffle(indices)

    shuffled_choices = [all_choices[i] for i in indices]
    new_correct_idx = indices.index(0)  # Original correct was at index 0

    if all_images:
        shuffled_images = [all_images[i] for i in indices]
    else:
        shuffled_images = None

    return shuffled_choices, new_correct_idx, shuffled_images

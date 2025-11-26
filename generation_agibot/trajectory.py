"""
AgiBotWorld Trajectory Class

Provides a unified interface for accessing AgiBotWorld episode data including:
- Proprioceptive data (joint positions, end-effector states, gripper positions)
- Video frames
- Action segment annotations
- Task metadata
"""

import json
import h5py
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple, Any

# Try to import Decord for fast video decoding (GPU-accelerated)
try:
    import decord
    from decord import VideoReader, gpu, cpu
    HAS_DECORD = True
    # Try to use GPU, fallback to CPU
    try:
        _test_ctx = gpu(0)
        DEFAULT_DECORD_CTX = gpu(0)
        print("Decord: Using GPU (NVDEC) for video decoding")
    except Exception:
        DEFAULT_DECORD_CTX = cpu(0)
        print("Decord: Using CPU for video decoding (GPU not available)")
except ImportError:
    HAS_DECORD = False
    DEFAULT_DECORD_CTX = None
    print("Warning: Decord not installed. Install with: pip install decord")

# Fallback to PyAV for AV1 video decoding
try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    if not HAS_DECORD:
        print("Warning: Neither Decord nor PyAV installed. Video decoding will fail.")


@dataclass
class ActionSegment:
    """Represents a single action segment within a trajectory."""
    start_frame: int
    end_frame: int
    action_text: str
    skill: str
    segment_id: int

    @property
    def duration(self) -> int:
        """Duration of the segment in frames."""
        return self.end_frame - self.start_frame

    def contains(self, frame_idx: int) -> bool:
        """Check if a frame index is within this segment."""
        return self.start_frame <= frame_idx < self.end_frame

    def get_middle_frame(self) -> int:
        """Get the middle frame index of this segment."""
        return (self.start_frame + self.end_frame) // 2


class AgiBotTrajectory:
    """
    Trajectory wrapper for AgiBotWorld dataset.

    Provides unified access to:
    - Episode metadata (task_name, init_scene_text, action segments)
    - Proprioceptive data (joint positions, end-effector poses, gripper states)
    - Video frames from head_color camera
    """

    def __init__(self, task_id: int, episode_id: int, data_root: str, max_frame_size: int = 320):
        """
        Initialize trajectory from AgiBotWorld dataset.

        Args:
            task_id: Task identifier (e.g., 327, 352, etc.)
            episode_id: Episode identifier within the task
            data_root: Root directory of the AgiBotWorld dataset
            max_frame_size: Maximum size for the longest dimension of frames (default 320)
        """
        self.task_id = task_id
        self.episode_id = episode_id
        self.data_root = Path(data_root)
        self.max_frame_size = max_frame_size

        # Load metadata
        self._load_metadata()

        # Load proprioceptive data
        self._load_proprio()

        # Video reader (lazy loaded) - prefer Decord for GPU acceleration
        self._video_reader = None  # Decord VideoReader
        self._video_container = None  # PyAV fallback
        self._video_fps = None
        self._video_frame_count = None
        self._cached_frames = {}  # Cache decoded frames
        self._all_frames = None  # All frames loaded sequentially

    def _load_metadata(self):
        """Load episode metadata from JSON file."""
        json_path = self.data_root / f"task_{self.task_id}" / "task_info" / f"task_{self.task_id}.json"

        with open(json_path, 'r') as f:
            episodes = json.load(f)

        # Find the specific episode
        self._episode_data = None
        for ep in episodes:
            if ep['episode_id'] == self.episode_id:
                self._episode_data = ep
                break

        if self._episode_data is None:
            raise ValueError(f"Episode {self.episode_id} not found in task {self.task_id}")

        # Parse action segments
        self._action_segments = []
        action_config = self._episode_data.get('label_info', {}).get('action_config', [])

        for i, action in enumerate(action_config):
            segment = ActionSegment(
                start_frame=action['start_frame'],
                end_frame=action['end_frame'],
                action_text=action['action_text'],
                skill=action.get('skill', 'Unknown'),
                segment_id=i
            )
            self._action_segments.append(segment)

    def _load_proprio(self):
        """Load proprioceptive data from HDF5 file."""
        h5_path = self.data_root / "proprio_stats" / str(self.task_id) / str(self.episode_id) / "proprio_stats.h5"

        with h5py.File(h5_path, 'r') as f:
            self._proprio = {
                # State data
                'joint_position': f['state/joint/position'][:],  # (T, 14)
                'end_position': f['state/end/position'][:],      # (T, 2, 3)
                'end_orientation': f['state/end/orientation'][:], # (T, 2, 4)
                'gripper_position': f['state/effector/position'][:],  # (T, 2)
                'robot_position': f['state/robot/position'][:],   # (T, 3)
                'robot_orientation': f['state/robot/orientation'][:],  # (T, 4)
                'head_position': f['state/head/position'][:],     # (T, 2)
                'waist_position': f['state/waist/position'][:],   # (T, 2)
                'timestamps': f['timestamp'][:]  # (T,)
            }

        self._proprio_length = len(self._proprio['timestamps'])

    def _init_video(self):
        """Initialize video reader using Decord (GPU) or PyAV fallback."""
        if self._video_reader is not None or self._video_container is not None:
            return

        video_path = self.get_video_path()

        # Try Decord first (faster, supports GPU)
        if HAS_DECORD:
            try:
                self._video_reader = VideoReader(str(video_path), ctx=DEFAULT_DECORD_CTX)
                self._video_fps = self._video_reader.get_avg_fps()
                self._video_frame_count = len(self._video_reader)
                return
            except Exception as e:
                print(f"Decord failed to open video: {e}, falling back to PyAV")
                self._video_reader = None

        # Fallback to PyAV
        if HAS_PYAV:
            try:
                self._video_container = av.open(str(video_path))
                stream = self._video_container.streams.video[0]
                # Get FPS
                if stream.average_rate:
                    self._video_fps = float(stream.average_rate)
                else:
                    self._video_fps = 30.0
                # Get frame count
                if stream.frames and stream.frames > 0:
                    self._video_frame_count = stream.frames
                else:
                    # Estimate from duration
                    self._video_frame_count = 1000  # Default
            except Exception as e:
                print(f"PyAV failed to open video: {e}")
                self._video_container = None
        else:
            # Fallback to OpenCV (may not work for AV1)
            cap = cv2.VideoCapture(str(video_path))
            self._video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
            cap.release()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame so longest side is max_frame_size."""
        if self.max_frame_size is None or self.max_frame_size <= 0:
            return frame

        h, w = frame.shape[:2]
        if max(h, w) <= self.max_frame_size:
            return frame

        if h > w:
            new_h = self.max_frame_size
            new_w = int(w * (self.max_frame_size / h))
        else:
            new_w = self.max_frame_size
            new_h = int(h * (self.max_frame_size / w))

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _decode_frame_decord(self, frame_idx: int) -> Optional[np.ndarray]:
        """Decode a specific frame using Decord (fast random access, GPU support)."""
        if not HAS_DECORD or self._video_reader is None:
            return None

        # Check cache first
        if frame_idx in self._cached_frames:
            return self._cached_frames[frame_idx].copy()

        try:
            # Decord supports efficient random access
            frame = self._video_reader[frame_idx].asnumpy()
            # Decord returns RGB, convert to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Resize frame
            frame = self._resize_frame(frame)
            # Cache the frame
            self._cached_frames[frame_idx] = frame
            return frame.copy()
        except Exception as e:
            print(f"Error decoding frame {frame_idx} with Decord: {e}")
            return None

    def _decode_frame_pyav(self, frame_idx: int) -> Optional[np.ndarray]:
        """Decode a specific frame using PyAV with timestamp-based seeking."""
        if not HAS_PYAV or self._video_container is None:
            return None

        # Check cache first
        if frame_idx in self._cached_frames:
            return self._cached_frames[frame_idx].copy()

        try:
            stream = self._video_container.streams.video[0]
            fps = self._video_fps or 30.0
            time_base = stream.time_base

            # Calculate target timestamp
            target_pts = int(frame_idx / fps / time_base)

            # Seek to nearest keyframe before target
            self._video_container.seek(target_pts, stream=stream, backward=True, any_frame=False)

            # Decode frames until we reach target
            for frame in self._video_container.decode(video=0):
                current_idx = int(float(frame.pts * time_base) * fps)
                if current_idx >= frame_idx:
                    # Convert to numpy array (BGR for OpenCV compatibility)
                    img = frame.to_ndarray(format='bgr24')
                    # Resize frame
                    img = self._resize_frame(img)
                    # Cache the frame
                    self._cached_frames[frame_idx] = img
                    return img.copy()

            return None
        except Exception as e:
            print(f"Error decoding frame {frame_idx}: {e}")
            return None

    def decode_all_frames(self) -> List[np.ndarray]:
        """
        Decode entire video sequentially (no seeking - very fast).

        Returns:
            List of BGR frames as numpy arrays (resized to max_frame_size)
        """
        if self._all_frames is not None:
            return self._all_frames

        video_path = self.get_video_path()
        frames = []

        # Try PyAV sequential decode (fastest for this use case)
        if HAS_PYAV:
            try:
                container = av.open(str(video_path))
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'  # Enable threading

                # Get video info
                if stream.average_rate:
                    self._video_fps = float(stream.average_rate)
                else:
                    self._video_fps = 30.0

                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format='bgr24')
                    img = self._resize_frame(img)
                    frames.append(img)

                container.close()
                self._all_frames = frames
                self._video_frame_count = len(frames)
                return frames
            except Exception as e:
                print(f"PyAV decode failed: {e}")

        # Fallback to Decord
        if HAS_DECORD:
            try:
                vr = VideoReader(str(video_path), ctx=DEFAULT_DECORD_CTX)
                self._video_fps = vr.get_avg_fps()
                self._video_frame_count = len(vr)

                for i in range(len(vr)):
                    frame = vr[i].asnumpy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = self._resize_frame(frame)
                    frames.append(frame)

                self._all_frames = frames
                return frames
            except Exception as e:
                print(f"Decord decode failed: {e}")

        return frames

    def get_frame_from_preloaded(self, step_idx: int) -> Optional[np.ndarray]:
        """
        Get frame from preloaded frames (must call decode_all_frames first).

        Args:
            step_idx: Proprioceptive step index

        Returns:
            BGR frame or None
        """
        if self._all_frames is None:
            return self.get_frame(step_idx)  # Fallback to old method

        proprio_length = self.get_trajectory_length()
        video_length = len(self._all_frames)

        if proprio_length == 0 or video_length == 0:
            return None

        video_idx = int((step_idx / proprio_length) * video_length)
        video_idx = min(video_idx, video_length - 1)

        return self._all_frames[video_idx].copy()

    def close(self):
        """Release video resources."""
        if self._video_reader is not None:
            # Decord VideoReader doesn't need explicit close
            self._video_reader = None
        if self._video_container is not None:
            if HAS_PYAV:
                try:
                    self._video_container.close()
                except:
                    pass
            self._video_container = None
        self._cached_frames.clear()
        self._all_frames = None  # Free memory

    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.close()
        except:
            pass

    # ==================== Core Properties ====================

    @property
    def task_name(self) -> str:
        """Get the task name/description."""
        return self._episode_data.get('task_name', '')

    @property
    def language_instruction(self) -> str:
        """Get the main language instruction (task_name)."""
        return self.task_name

    @property
    def init_scene_text(self) -> str:
        """Get the initial scene description."""
        return self._episode_data.get('init_scene_text', '')

    @property
    def trajectory_id(self) -> str:
        """Get unique trajectory identifier."""
        return f"{self.task_id}_{self.episode_id}"

    # ==================== Trajectory Length ====================

    def get_trajectory_length(self) -> int:
        """Get the length of the trajectory in proprioceptive frames."""
        return self._proprio_length

    def get_video_frame_count(self) -> int:
        """Get the total number of video frames."""
        self._init_video()
        return self._video_frame_count or 1000

    def get_video_fps(self) -> float:
        """Get the video frames per second."""
        self._init_video()
        return self._video_fps or 30.0

    # ==================== Action Segments ====================

    def get_action_segments(self) -> List[ActionSegment]:
        """Get all action segments in the trajectory."""
        return self._action_segments

    def get_segment_for_step(self, step_idx: int) -> Optional[ActionSegment]:
        """Get the action segment containing the given step index."""
        for segment in self._action_segments:
            if segment.contains(step_idx):
                return segment
        return None

    def get_current_skill(self, step_idx: int) -> Optional[str]:
        """Get the skill being executed at the given step."""
        segment = self.get_segment_for_step(step_idx)
        return segment.skill if segment else None

    def get_action_text(self, step_idx: int) -> Optional[str]:
        """Get the action text for the given step."""
        segment = self.get_segment_for_step(step_idx)
        return segment.action_text if segment else None

    def get_num_segments(self) -> int:
        """Get the number of action segments."""
        return len(self._action_segments)

    @property
    def interested_timesteps(self) -> Set[int]:
        """Get set of interesting timesteps (segment boundaries and midpoints)."""
        timesteps = set()
        for segment in self._action_segments:
            timesteps.add(segment.start_frame)
            timesteps.add(segment.get_middle_frame())
            # Don't add end_frame as it's the start of next segment
        # Add last frame
        if self._action_segments:
            timesteps.add(self._action_segments[-1].end_frame - 1)
        return timesteps

    # ==================== Gripper State ====================

    def get_gripper_position(self, step_idx: int, arm: int = 0) -> float:
        """
        Get the gripper position at a given step.

        Args:
            step_idx: Step index
            arm: Arm index (0=left, 1=right)

        Returns:
            Gripper position value
        """
        step_idx = min(step_idx, self._proprio_length - 1)
        return float(self._proprio['gripper_position'][step_idx, arm])

    def get_normalized_gripper_position(self, step_idx: int, arm: int = 0) -> float:
        """
        Get normalized gripper position (0-1) based on trajectory min/max.

        Args:
            step_idx: Step index
            arm: Arm index (0=left, 1=right)

        Returns:
            Normalized gripper position (0=closed, 1=open typically)
        """
        gripper_vals = self._proprio['gripper_position'][:, arm]
        min_val = gripper_vals.min()
        max_val = gripper_vals.max()

        current_val = self.get_gripper_position(step_idx, arm)

        if max_val - min_val < 1e-6:
            return 0.5  # No variation

        return (current_val - min_val) / (max_val - min_val)

    def is_gripper_open(self, step_idx: int, arm: int = 0, threshold: float = 0.8) -> bool:
        """
        Check if the gripper is open at the given step.

        Args:
            step_idx: Step index
            arm: Arm index (0=left, 1=right)
            threshold: Threshold for considering gripper open (default 0.8)

        Returns:
            True if gripper is open
        """
        normalized = self.get_normalized_gripper_position(step_idx, arm)
        return normalized <= threshold

    # ==================== End-Effector State ====================

    def get_end_effector_position(self, step_idx: int, arm: int = 0) -> np.ndarray:
        """
        Get end-effector position (x, y, z) at a given step.

        Args:
            step_idx: Step index
            arm: Arm index (0=left, 1=right)

        Returns:
            numpy array of shape (3,) with x, y, z coordinates
        """
        step_idx = min(step_idx, self._proprio_length - 1)
        return self._proprio['end_position'][step_idx, arm].copy()

    def get_end_effector_orientation(self, step_idx: int, arm: int = 0) -> np.ndarray:
        """
        Get end-effector orientation (quaternion) at a given step.

        Args:
            step_idx: Step index
            arm: Arm index (0=left, 1=right)

        Returns:
            numpy array of shape (4,) with quaternion (w, x, y, z)
        """
        step_idx = min(step_idx, self._proprio_length - 1)
        return self._proprio['end_orientation'][step_idx, arm].copy()

    # ==================== Joint State ====================

    def get_joint_positions(self, step_idx: int) -> np.ndarray:
        """
        Get all joint positions at a given step.

        Args:
            step_idx: Step index

        Returns:
            numpy array of shape (14,) with joint angles
        """
        step_idx = min(step_idx, self._proprio_length - 1)
        return self._proprio['joint_position'][step_idx].copy()

    # ==================== Robot Base State ====================

    def get_robot_position(self, step_idx: int) -> np.ndarray:
        """Get robot base position (x, y, z)."""
        step_idx = min(step_idx, self._proprio_length - 1)
        return self._proprio['robot_position'][step_idx].copy()

    def get_robot_orientation(self, step_idx: int) -> np.ndarray:
        """Get robot base orientation (quaternion)."""
        step_idx = min(step_idx, self._proprio_length - 1)
        return self._proprio['robot_orientation'][step_idx].copy()

    # ==================== Timestamps ====================

    def get_timestamp(self, step_idx: int) -> int:
        """Get timestamp (nanoseconds) at a given step."""
        step_idx = min(step_idx, self._proprio_length - 1)
        return int(self._proprio['timestamps'][step_idx])

    def get_time_seconds(self, step_idx: int) -> float:
        """Get time in seconds from trajectory start."""
        start_time = self._proprio['timestamps'][0]
        current_time = self.get_timestamp(step_idx)
        return (current_time - start_time) / 1e9

    # ==================== Video/Image Data ====================

    def get_video_path(self) -> Path:
        """Get path to the head_color video file."""
        return self.data_root / f"task_{self.task_id}" / "observations" / str(self.episode_id) / "videos" / "head_color.mp4"

    def get_frame(self, step_idx: int) -> Optional[np.ndarray]:
        """
        Get video frame at the given proprioceptive step index.

        Note: This maps proprioceptive step to video frame using relative position.
        If decode_all_frames() was called first, uses preloaded frames (fast).

        Args:
            step_idx: Proprioceptive step index

        Returns:
            BGR image as numpy array, or None if failed
        """
        # Use preloaded frames if available (fast path)
        if self._all_frames is not None:
            return self.get_frame_from_preloaded(step_idx)

        self._init_video()

        # Map proprio step to video frame
        # Use relative position in trajectory
        proprio_length = self.get_trajectory_length()
        video_length = self.get_video_frame_count()

        if proprio_length == 0:
            return None

        # Map step_idx to video frame
        video_frame_idx = int((step_idx / proprio_length) * video_length)
        video_frame_idx = min(video_frame_idx, video_length - 1)

        return self.get_frame_at_video_idx(video_frame_idx)

    def get_frame_at_video_idx(self, video_frame_idx: int) -> Optional[np.ndarray]:
        """
        Get video frame at the exact video frame index.

        Args:
            video_frame_idx: Video frame index

        Returns:
            BGR image as numpy array (resized to max_frame_size), or None if failed
        """
        self._init_video()

        # Try Decord first (faster, GPU support)
        if HAS_DECORD and self._video_reader is not None:
            return self._decode_frame_decord(video_frame_idx)
        # Fallback to PyAV
        elif HAS_PYAV and self._video_container is not None:
            return self._decode_frame_pyav(video_frame_idx)
        else:
            # No decoder available
            return None

    def get_keyframes(self, num_frames: int = 5) -> List[Tuple[int, np.ndarray]]:
        """
        Get evenly spaced keyframes from the trajectory.

        Args:
            num_frames: Number of keyframes to extract

        Returns:
            List of (step_idx, frame) tuples
        """
        trajectory_length = self.get_trajectory_length()
        step_indices = np.linspace(0, trajectory_length - 1, num_frames, dtype=int)

        keyframes = []
        for step_idx in step_indices:
            frame = self.get_frame(step_idx)
            if frame is not None:
                keyframes.append((int(step_idx), frame))

        return keyframes

    def get_segment_keyframes(self) -> List[Tuple[int, np.ndarray, ActionSegment]]:
        """
        Get keyframes from each action segment (middle of each segment).

        Returns:
            List of (step_idx, frame, segment) tuples
        """
        keyframes = []
        for segment in self._action_segments:
            step_idx = segment.get_middle_frame()
            frame = self.get_frame(step_idx)
            if frame is not None:
                keyframes.append((step_idx, frame, segment))

        return keyframes

    # ==================== Utility Methods ====================

    def is_task_successful(self) -> bool:
        """
        Heuristic to determine if task was successful.

        For AgiBotWorld, we assume trajectories are successful demonstrations.
        """
        return True  # AgiBotWorld contains successful demonstrations

    def get_all_skills(self) -> List[str]:
        """Get list of all skills in this trajectory."""
        return [seg.skill for seg in self._action_segments]

    def get_unique_skills(self) -> Set[str]:
        """Get set of unique skills in this trajectory."""
        return set(self.get_all_skills())

    def has_skill(self, skill: str) -> bool:
        """Check if trajectory contains a specific skill."""
        return skill in self.get_unique_skills()

    def get_segments_by_skill(self, skill: str) -> List[ActionSegment]:
        """Get all segments with a specific skill."""
        return [seg for seg in self._action_segments if seg.skill == skill]

    def __repr__(self) -> str:
        return f"AgiBotTrajectory(task_id={self.task_id}, episode_id={self.episode_id}, " \
               f"task_name='{self.task_name}', segments={self.get_num_segments()})"


def load_trajectory(task_id: int, episode_id: int,
                   data_root: str = "/shared/projects/agibot/agibot_alpha_full",
                   max_frame_size: int = 320) -> AgiBotTrajectory:
    """
    Convenience function to load a trajectory.

    Args:
        task_id: Task identifier
        episode_id: Episode identifier
        data_root: Root directory of dataset
        max_frame_size: Maximum size for the longest dimension of frames (default 320)

    Returns:
        AgiBotTrajectory instance
    """
    return AgiBotTrajectory(task_id, episode_id, data_root, max_frame_size=max_frame_size)


def get_all_episodes(task_id: int,
                    data_root: str = "/shared/projects/agibot/agibot_alpha_full") -> List[int]:
    """
    Get all episode IDs for a given task.

    Args:
        task_id: Task identifier
        data_root: Root directory of dataset

    Returns:
        List of episode IDs
    """
    json_path = Path(data_root) / f"task_{task_id}" / "task_info" / f"task_{task_id}.json"

    with open(json_path, 'r') as f:
        episodes = json.load(f)

    return [ep['episode_id'] for ep in episodes]


def get_all_task_ids(data_root: str = "/shared/projects/agibot/agibot_alpha_full") -> List[int]:
    """
    Get all available task IDs.

    Args:
        data_root: Root directory of dataset

    Returns:
        List of task IDs
    """
    data_path = Path(data_root)
    task_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("task_")]
    task_ids = [int(d.name.split("_")[1]) for d in task_dirs]
    return sorted(task_ids)

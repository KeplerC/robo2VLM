# Design Document: Porting VQA Generation to AgiBotWorld Dataset

## Executive Summary

This document outlines the design for porting the VQA (Visual Question Answering) generation system from OXE (Open X-Embodiment) dataset to the AgiBotWorld-Alpha dataset. It identifies which VQA question types can be directly ported, which require adaptation, and which cannot be generated due to data limitations.

---

## 1. Dataset Comparison

### 1.1 OXE Dataset (Source)
| Feature | Details |
|---------|---------|
| **Cameras** | Multiple cameras (exterior + wrist cameras) |
| **Depth Data** | Available for most cameras |
| **Camera Calibration** | Full intrinsics and extrinsics per camera |
| **Robot Type** | Single-arm manipulators |
| **Gripper Data** | Continuous gripper position (0-1) |
| **End-Effector** | 6-DOF Cartesian position (x, y, z, roll, pitch, yaw) |
| **Phase Detection** | Gripper-based (pre_grasp, immobilization, contact, detach, post_grasp) |
| **Task Annotations** | Single language instruction per trajectory |

### 1.2 AgiBotWorld Dataset (Target)
| Feature | Details |
|---------|---------|
| **Cameras** | Single camera (`head_color.mp4`) |
| **Depth Data** | **NOT AVAILABLE** in local dataset |
| **Camera Calibration** | **NOT AVAILABLE** (no intrinsics/extrinsics) |
| **Robot Type** | Dual-arm humanoid (14 DOF arms) |
| **Gripper Data** | `state/effector/position` shape (T, 2) for 2 grippers |
| **End-Effector** | Position (T, 2, 3) + Orientation (T, 2, 4) for 2 arms |
| **Phase Detection** | **Skill-based annotations** (Pick, Place, Fold, etc.) |
| **Task Annotations** | Multi-stage with `action_text` per segment |

### 1.3 AgiBotWorld Data Fields Summary

```
Proprioceptive Data (HDF5):
├── state/joint/position: (T, 14) - Joint angles
├── state/end/position: (T, 2, 3) - End-effector XYZ per arm
├── state/end/orientation: (T, 2, 4) - End-effector quaternion per arm
├── state/effector/position: (T, 2) - Gripper positions per arm
├── state/robot/position: (T, 3) - Robot base position
├── state/robot/orientation: (T, 4) - Robot base quaternion
├── state/head/position: (T, 2) - Head position
├── state/waist/position: (T, 2) - Waist position
└── timestamp: (T,) - Unix nanosecond timestamps

Metadata (JSON):
├── task_id: Integer task identifier
├── task_name: Task description
├── episode_id: Episode identifier
├── init_scene_text: Initial scene description
└── label_info/action_config: List of action segments
    ├── start_frame: Segment start
    ├── end_frame: Segment end
    ├── action_text: Natural language description
    └── skill: Skill category (Pick, Place, etc.)
```

---

## 2. VQA Question Categories Analysis

### 2.1 Fully Portable Questions ✅

These questions can be directly ported with minimal modifications:

#### **S1: Robot Gripper State (`vqa_robot_gripper_open`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Data Source | Single gripper position | `state/effector/position[:, arm_idx]` | Select arm (0 or 1) |
| Normalization | Min-max across trajectory | Same approach | Compatible |
| Image Source | Multi-camera stitch | Single `head_color` frame | Simpler |

**Implementation Notes:**
- Need to decide which arm to query (left=0, right=1) or ask about both
- Can enhance question: "Is the robot's left/right gripper open?"

#### **I1: Task Success State (`vqa_task_success_state`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Success Detection | Path-based + phase | Segment-based (in last segment?) | Use last action segment |
| Language Instruction | Single instruction | Use `task_name` or last `action_text` | Direct mapping |

**Implementation Notes:**
- Success can be inferred from being in the final action segment
- Can use `init_scene_text` + final frame comparison

#### **I3: Goal Configuration (`vqa_goal_configuration`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Goal Frame | Last trajectory frame | Last trajectory frame | Direct |
| Alternative Frames | From different phases | From different action segments | Use segment boundaries |
| Language | Single instruction | `task_name` | Direct |

**Implementation Notes:**
- Sample frames from different action segments (skills)
- Use frame at segment boundaries for clear visual distinction

#### **I4: Action Understanding (`vqa_action_understanding`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Phase Source | Gripper-based phases | **Skill annotations** | Different taxonomy |
| Phase Types | 5 gripper phases | 31 skill types | Map to descriptions |

**Implementation Notes:**
- AgiBotWorld has richer skill annotations (Pick, Place, Fold, Pour, etc.)
- Question: "Which action is the robot currently performing?"
- Choices: Use `action_text` from current segment + texts from other segments

#### **I4b: Next Action (`vqa_next_action`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Sequence | Fixed phase sequence | **Action segment sequence** | Use segment order |
| Next Prediction | Based on phase cycle | Based on segment index + 1 | Direct |

**Implementation Notes:**
- Look up current segment, return next segment's `action_text`
- For last segment, either skip or use "Task complete" as answer

#### **I6: Trajectory Understanding (`vqa_trajectory_understanding`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Correct Answer | Language instruction | `task_name` | Direct |
| Visualization | Trajectory overlay | Keyframe montage | New approach |

**Implementation Notes:**
- Create montage of keyframes from each action segment
- Alternative: Show first, middle, and last frames
- Question remains the same

---

### 2.2 Partially Portable Questions ⚠️

These questions require significant adaptation but are feasible:

#### **S3: Object Reachability (`vqa_object_reachable`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Detection | Hardcoded (always "No") | Same limitation | Stub implementation |
| Target Object | NLP extraction | NLP extraction from `action_text` | Same approach |

**Status:** Currently a stub in OXE (always returns "No obstacle"). Same limitation applies.

#### **I2: Stable Grasp (`is_stable_grasp`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Phase Required | contact, immobilization, detach | Pick, Grasp skills | Map skills to grasp |
| Stability | phase == "contact" | Mid-segment of Pick/Grasp | Heuristic |

**Implementation Notes:**
- Map Pick/Grasp skills to grasp-related questions
- Stability heuristic: frame in middle 50% of Pick segment = stable
- Beginning/end of segment = unstable (transitioning)

#### **I5: Temporal Sequence (`vqa_temporal_sequence`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Frame Source | 3+ timesteps | Segment boundary frames | Use action transitions |
| Question Mode A | Phase sequence | **Skill sequence** | Map skills |
| Question Mode B | Task description | `task_name` | Direct |

**Implementation Notes:**
- Extract frames at action segment boundaries
- Ask: "What is the sequence of actions shown?"
- Choices: Correct skill sequence vs shuffled/alternative sequences

#### **S4: Relative Direction (`vqa_relative_direction`)**
| Aspect | OXE | AgiBotWorld | Adaptation |
|--------|-----|-------------|------------|
| Position Data | Cartesian state | `state/end/position` | Available |
| Phase Required | pre_grasp only | Beginning of Pick segment | Map to skill |
| Contact Position | From contact phase | End of Pick segment position | Use segment end |

**Implementation Notes:**
- Can compute direction from end-effector positions
- Direction = (current_pos - target_pos) analysis
- **LIMITATION:** Cannot project to image without camera calibration
- **Alternative:** Ask about 3D direction without image projection markers

---

### 2.3 Questions That CANNOT Be Generated ❌

These questions require data not available in AgiBotWorld:

#### **S6: Relative Depth (`vqa_relative_depth`)**
| Requirement | Available in AgiBotWorld |
|-------------|-------------------------|
| Depth images | ❌ **NOT AVAILABLE** |
| Valid depth range | ❌ N/A |

**Reason:** Requires depth images to sample points with different depths. AgiBotWorld only provides RGB video.

**Possible Alternatives:**
1. Use depth estimation models (MiDaS, DPT) - adds inference overhead
2. Skip this question type entirely

#### **S8: Multi-View Correspondence (`vqa_multi_view_correspondence`)**
| Requirement | Available in AgiBotWorld |
|-------------|-------------------------|
| Multiple cameras | ❌ Single camera only |
| Camera intrinsics | ❌ **NOT AVAILABLE** |
| Camera extrinsics | ❌ **NOT AVAILABLE** |

**Reason:** Requires at least 2 camera views with calibration data to project 3D points.

**Cannot be implemented** without significant additional data collection.

#### **Action Direction Selection (`generate_action_direction_selection_vqa`)**
| Requirement | Available in AgiBotWorld |
|-------------|-------------------------|
| Camera intrinsics | ❌ **NOT AVAILABLE** |
| Camera extrinsics | ❌ **NOT AVAILABLE** |
| 3D-to-2D projection | ❌ Cannot compute |

**Reason:** Requires projecting 3D robot positions to 2D image coordinates for arrow visualization.

**Possible Alternatives:**
1. Ask about 3D movement direction without image arrows
2. Use heuristic image-space motion estimation (optical flow)

---

## 3. New Question Types for AgiBotWorld

AgiBotWorld's rich annotations enable NEW question types not possible with OXE:

### 3.1 Dual-Arm Coordination Questions (NEW)

```python
def vqa_arm_coordination(trajectory, step_idx: int) -> Optional[VQA]:
    """Which arm is currently active/primary in this task?"""
    # Uses action_text which often specifies "left arm" or "right arm"
    # Example: "Grasp the left arm with white plastic bag"
```

**Data Source:** `action_text` parsing + `state/end/position` for both arms

### 3.2 Skill Recognition Questions (NEW)

```python
def vqa_skill_recognition(trajectory, step_idx: int) -> Optional[VQA]:
    """What skill is the robot currently executing?"""
    # Uses skill annotation directly
    # Choices: ["Pick", "Place", "Fold", "Pour", "Push", ...]
```

**Data Source:** `label_info/action_config[i]/skill`

### 3.3 Scene Description Questions (NEW)

```python
def vqa_scene_understanding(trajectory) -> Optional[VQA]:
    """Which description best matches the robot's environment?"""
    # Uses init_scene_text
```

**Data Source:** `init_scene_text` field

### 3.4 Object Identification Questions (NEW)

```python
def vqa_target_object(trajectory, step_idx: int) -> Optional[VQA]:
    """What object is the robot interacting with?"""
    # Extract object from action_text using NLP
```

**Data Source:** `action_text` NLP parsing

### 3.5 Action Sequence Questions (NEW)

```python
def vqa_action_count(trajectory) -> Optional[VQA]:
    """How many distinct actions does the robot perform in this task?"""
    # Count action segments
```

**Data Source:** `len(label_info/action_config)`

### 3.6 HandOver/Bimanual Questions (NEW)

```python
def vqa_handover_detection(trajectory, step_idx: int) -> Optional[VQA]:
    """Is the robot performing a handover between arms?"""
    # Detect HandOver skill
```

**Data Source:** `skill == "HandOver"`

---

## 4. Implementation Architecture

### 4.1 Trajectory Interface for AgiBotWorld

```python
class AgiBotTrajectory:
    def __init__(self, task_id: int, episode_id: int, data_root: str):
        self.task_id = task_id
        self.episode_id = episode_id
        self.data_root = data_root
        self._load_metadata()
        self._load_proprio()

    # Core trajectory info
    def get_trajectory_length(self) -> int
    def get_language_instruction(self) -> str  # Returns task_name
    def get_init_scene_text(self) -> str

    # Action segments (replaces phases)
    def get_action_segments(self) -> List[ActionSegment]
    def get_segment_for_step(self, step_idx: int) -> ActionSegment
    def get_current_skill(self, step_idx: int) -> str
    def get_action_text(self, step_idx: int) -> str

    # Robot state
    def get_gripper_position(self, step_idx: int, arm: int = 0) -> float
    def get_end_effector_position(self, step_idx: int, arm: int = 0) -> np.ndarray
    def get_end_effector_orientation(self, step_idx: int, arm: int = 0) -> np.ndarray
    def get_joint_positions(self, step_idx: int) -> np.ndarray

    # Image data
    def get_frame(self, step_idx: int) -> np.ndarray  # Single camera
    def get_video_path(self) -> str

    # Timestamps
    def get_timestamp(self, step_idx: int) -> int

    # Interested timesteps (segment boundaries)
    @property
    def interested_timesteps(self) -> Set[int]
```

### 4.2 Data Classes

```python
@dataclass
class ActionSegment:
    start_frame: int
    end_frame: int
    action_text: str
    skill: str
    segment_id: int

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame

    def contains(self, frame_idx: int) -> bool:
        return self.start_frame <= frame_idx < self.end_frame
```

### 4.3 File Structure

```
generation_agibot/
├── __init__.py
├── trajectory.py          # AgiBotTrajectory class
├── vqa.py                 # VQA generation functions
├── utils.py               # Helper functions
├── skills.py              # Skill taxonomy and mappings
├── nlp_utils.py           # Object/action extraction from text
├── video_utils.py         # Video frame extraction
└── generate_dataset.py    # Main generation script
```

---

## 5. Question Generation Summary

### 5.1 Portability Matrix

| VQA Type | OXE Tag | Portable | Adaptation Required |
|----------|---------|----------|---------------------|
| S1 | `vqa_robot_gripper_open` | ✅ Yes | Dual-arm selection |
| S3 | `vqa_object_reachable` | ⚠️ Partial | Still stub |
| S4 | `vqa_relative_direction` | ⚠️ Partial | No image projection |
| S6 | `vqa_relative_depth` | ❌ No | No depth data |
| S8 | `vqa_multi_view_correspondence` | ❌ No | Single camera |
| I1 | `vqa_task_success_state` | ✅ Yes | Segment-based |
| I2 | `is_stable_grasp` | ⚠️ Partial | Skill-based heuristic |
| I3 | `vqa_goal_configuration` | ✅ Yes | Direct |
| I4 | `vqa_action_understanding` | ✅ Yes | Use skills |
| I4b | `vqa_next_action` | ✅ Yes | Use segments |
| I5 | `vqa_temporal_sequence` | ⚠️ Partial | Skill sequences |
| I6 | `vqa_trajectory_understanding` | ✅ Yes | Direct |
| - | `generate_action_direction_selection_vqa` | ❌ No | No camera calibration |

### 5.2 Coverage Summary

| Category | Count | Status |
|----------|-------|--------|
| **Fully Portable** | 6 | Ready to implement |
| **Partially Portable** | 4 | Need adaptation |
| **Cannot Generate** | 3 | Missing data |
| **New AgiBotWorld-Specific** | 6+ | Additional questions |

---

## 6. Skill Taxonomy Mapping

### 6.1 AgiBotWorld Skills (31 types)

```python
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
    "Fold": "Folding material/clothing",
    "Unfold": "Unfolding material",
    "Stretch": "Stretching material",
    "Stack": "Stacking items",
    "Insert": "Inserting into a space",

    # Dual-arm
    "HandOver": "Transferring between arms",
    "Hold": "Holding position",

    # Tool use
    "Pour": "Pouring liquid",
    "Brush": "Brushing with tool",
    "Wipe": "Wiping surface",
    "Iron": "Ironing clothes",
    "Scan": "Scanning barcode",
    "dip": "Dipping in liquid",

    # Environment interaction
    "Open": "Opening container/door",
    "Close": "Closing container/door",
    "PressButton": "Pressing a button",
    "Press": "Applying pressure",
    "Tap": "Tapping surface",
    "Shake": "Shaking object",
    "Hang": "Hanging item"
}
```

### 6.2 Mapping to OXE Phases (for compatible questions)

```python
# For questions requiring grasp phases, map skills:
SKILL_TO_PHASE_MAPPING = {
    # pre_grasp equivalent
    "approaching": ["Move"],

    # immobilization equivalent (gripper closing)
    "grasping": ["Grasp"],

    # contact equivalent (holding object)
    "holding": ["Pick", "Hold", "Lift"],

    # detach equivalent (releasing)
    "releasing": ["Release", "Drop", "Place"],

    # tool_use (new category)
    "tool_use": ["Pour", "Brush", "Wipe", "Iron", "Scan", "dip"],

    # bimanual (new category)
    "bimanual": ["HandOver", "Fold", "Unfold", "Stretch"]
}
```

---

## 7. Implementation Priority

### Phase 1: Core Questions (Week 1)
1. ✅ `vqa_robot_gripper_open` - Adapt for dual-arm
2. ✅ `vqa_action_understanding` - Use skill annotations
3. ✅ `vqa_next_action` - Use segment sequence
4. ✅ `vqa_goal_configuration` - Direct port

### Phase 2: Extended Questions (Week 2)
5. ✅ `vqa_task_success_state` - Segment-based detection
6. ✅ `vqa_trajectory_understanding` - Keyframe montage
7. ⚠️ `is_stable_grasp` - Skill-based heuristic
8. ⚠️ `vqa_temporal_sequence` - Skill sequences

### Phase 3: New AgiBotWorld Questions (Week 3)
9. 🆕 `vqa_skill_recognition` - Skill identification
10. 🆕 `vqa_arm_coordination` - Dual-arm questions
11. 🆕 `vqa_scene_understanding` - Scene descriptions
12. 🆕 `vqa_target_object` - Object identification

### Phase 4: Advanced Questions (Week 4)
13. ⚠️ `vqa_relative_direction` - 3D direction (no image overlay)
14. 🆕 `vqa_handover_detection` - Bimanual coordination
15. 🆕 `vqa_action_count` - Sequence understanding

---

## 8. Data Access Patterns

### 8.1 Loading Episode Data

```python
import json
import h5py
import cv2

def load_episode(task_id: int, episode_id: int, data_root: str):
    # Load metadata
    json_path = f"{data_root}/task_{task_id}/task_info/task_{task_id}.json"
    with open(json_path) as f:
        episodes = json.load(f)

    episode_data = next(e for e in episodes if e['episode_id'] == episode_id)

    # Load proprioceptive data
    h5_path = f"{data_root}/proprio_stats/{task_id}/{episode_id}/proprio_stats.h5"
    with h5py.File(h5_path, 'r') as f:
        proprio = {
            'joint_position': f['state/joint/position'][:],
            'end_position': f['state/end/position'][:],
            'end_orientation': f['state/end/orientation'][:],
            'gripper_position': f['state/effector/position'][:],
            'timestamps': f['timestamp'][:]
        }

    # Video path
    video_path = f"{data_root}/task_{task_id}/observations/{episode_id}/videos/head_color.mp4"

    return episode_data, proprio, video_path
```

### 8.2 Extracting Video Frames

```python
def extract_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
```

---

## 9. Open Questions

1. **Frame-Proprio Synchronization:** Video FPS vs proprioceptive frequency alignment?
2. **Dual-Arm Questions:** Should questions focus on specific arm or both?
3. **Skill Granularity:** Use fine-grained skills or group into categories?
4. **Missing Depth:** Explore depth estimation models as alternative?
5. **Camera Calibration:** Can we obtain calibration data from AGIBOT?

---

## 10. Appendix: Task Statistics

### 10.1 Tasks by Scenario
| Scenario | Tasks | Episodes |
|----------|-------|----------|
| Home | 14 | ~20,000 |
| Industry | 8 | ~8,000 |
| Supermarket | 8 | ~4,500 |
| Restaurant | 3 | ~2,000 |

### 10.2 Top Skills by Frequency
| Skill | Count | Percentage |
|-------|-------|------------|
| Pick | 55,087 | 29.8% |
| Place | 47,231 | 25.5% |
| Fold | 16,466 | 8.9% |
| Release | 6,114 | 3.3% |
| Push | 6,023 | 3.3% |
| Pour | 5,958 | 3.2% |
| Grasp | 5,725 | 3.1% |
| HandOver | 5,501 | 3.0% |

### 10.3 End-Effector Types
| Type | Tasks |
|------|-------|
| Grippers | 35 tasks |
| Dexterous Hands | 1 task (task_475: Ironing) |

---

## 11. Conclusion

The AgiBotWorld dataset can support **9-10 of the 13 original OXE VQA question types** with adaptations, and enables **6+ new question types** leveraging its rich skill annotations and dual-arm capabilities. The main limitations are:

1. **No depth images** → Cannot generate depth-based questions
2. **Single camera** → Cannot generate multi-view questions
3. **No camera calibration** → Cannot project 3D points to image space

However, the skill-based annotations, dual-arm data, and scene descriptions provide opportunities for new question types not possible with the OXE dataset.

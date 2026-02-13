# V6 Pipeline: Complete Technical Explanation

> **Author**: GitHub Copilot  
> **Date**: 2026-02-12  
> **Purpose**: Comprehensive ground-up explanation of V6 multi-camera QA generation pipeline

---

## Table of Contents

1. [Ground-Up Principles](#1-ground-up-principles)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Data Structures](#3-data-structures)
4. [Step-by-Step Flow](#4-step-by-step-flow)
5. [Low-Level Implementation Details](#5-low-level-implementation-details)
6. [Memory Management](#6-memory-management)
7. [Question Generation Strategies](#7-question-generation-strategies)
8. [Validation & Quality Control](#8-validation--quality-control)

---

## 1. Ground-Up Principles

### 1.1 The Core Problem

**Input**: Multi-camera surveillance video + activity annotations  
**Output**: Multiple-choice questions testing temporal/spatial/perceptual reasoning

**Why is this hard?**
1. **Multi-camera coordination**: Same person appears on different cameras at different times
2. **Entity ambiguity**: No unique person IDs across cameras (actor_id is camera-specific)
3. **Scale**: 929 slots × 5-10 cameras × 10-100 activities each = huge search space
4. **Memory**: GEOM files (bounding boxes) can be 50+ MB per camera with 100K+ bboxes
5. **Quality**: Need diverse, non-trivial questions that can't be guessed

### 1.2 Design Philosophy

**Modular architecture**: 8 separate steps, each with clear input/output contracts
- Easy to debug: run each step independently
- Easy to extend: swap out entity resolution or question generators
- Easy to test: validate intermediate outputs

**Entity-centric model**: Build a scene graph where entities (people) are first-class objects
- Events are attributes of entities, not standalone facts
- Cross-camera links connect entities, not events
- Questions ask about entity activities, distances, appearances

**Ground truth + heuristics**: Use MEVID ground truth where available, fall back to heuristics
- MEVID provides camera-level person coverage (person 4 appears on cameras A, B, C)
- Heuristics fill gaps (temporal handoff: exit camera A → enter camera B within 5 seconds)

**Streaming for scale**: Process GEOM files line-by-line, not load-all-to-memory
- Avoids OOM on large files (50+ MB)
- Fast regex parsing (~1.5s for 100K lines)

### 1.3 Key Assumptions

1. **Framerate**: 30 fps (constant across all MEVA videos)
2. **Coordination window**: Cameras are synchronized within ±1 second
3. **Actor persistence**: actor_id is stable within one camera, but NOT across cameras
4. **Activity taxonomy**: 37 standardized activity names (Kitware uses these, not NIST variants)
5. **KRTD coverage**: ~80% of cameras have calibration models, rest are indoor (no KRTD)

---

## 2. Pipeline Architecture

### 2.1 High-Level Flow

```
INPUT: slot name (e.g., "2018-03-11.11-25-00.school")
   ↓
STEP 1: Parse Annotations → List[Event]
   ↓
STEP 2: Build Scene Graph → SceneGraph (cameras, entities, events)
   ↓
STEP 3: Entity Resolution → ResolvedGraph (cross-camera links, clusters)
   ↓
STEP 4: Generate Temporal QA → List[Dict] (temporal reasoning questions)
   ↓
STEP 5: Generate Spatial QA → List[Dict] (spatial distance questions)
   ↓
STEP 6: Generate Perception QA → List[Dict] (perception questions)
   ↓
STEP 7: Deduplication + Validation → Filtered questions
   ↓
OUTPUT: JSON file with 9 questions (3 per category)
```

### 2.2 Module Structure

```
scripts/v6/
├── parse_annotations.py       # Step 1: YAML → Event objects
├── build_scene_graph.py       # Step 2: Events → Scene graph with entities
├── entity_resolution.py       # Step 3: Cross-camera entity linking
├── generate_temporal.py       # Step 4: Temporal reasoning questions
├── generate_spatial.py        # Step 5: Spatial distance questions
├── generate_perception.py     # Step 6: Perception questions
├── run_pipeline.py            # Orchestrator + validation
└── utils/
    ├── yaml_stream.py         # Memory-efficient GEOM parser
    ├── krtd.py                # Camera calibration + 3D projection
    ├── iou.py                 # Bounding box IoU calculation
    └── mevid.py               # MEVID ground truth lookup
```

### 2.3 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     KITWARE ANNOTATIONS                     │
│  activities.yml: Events with actors (camera-specific IDs)   │
│  geom.yml: Bounding boxes for each actor at each frame      │
│  types.yml: Actor type (person/vehicle)                     │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
                 ┌──────────────┐
                 │  STEP 1      │
                 │  Parse YAML  │  → List[Event]
                 └──────┬───────┘
                        ↓
           ┌───────────────────────────┐
           │  STEP 2                   │
           │  Build Scene Graph        │  → SceneGraph
           │  - Load GEOM bboxes       │     (entities, cameras, events)
           │  - Create Entity objects  │
           │  - Link events to entities│
           └───────────┬───────────────┘
                       ↓
        ┌──────────────────────────────────┐
        │  STEP 3                          │
        │  Entity Resolution               │  → ResolvedGraph
        │  - MEVID ground truth (if avail) │     (links, clusters)
        │  - Temporal handoff heuristic    │
        │  - Union-Find clustering         │
        └───────────┬──────────────────────┘
                    ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
┌────────────┐              ┌────────────┐
│  STEP 4    │              │  STEP 5    │
│  Temporal  │              │  Spatial   │
│  Questions │              │  Questions │
└─────┬──────┘              └──────┬─────┘
      ↓                            ↓
      │         ┌──────────┐       │
      └────────→│  STEP 6  │←──────┘
                │Perception│
                └────┬─────┘
                     ↓
            ┌────────────────┐
            │  STEP 7        │
            │  Dedup + Valid │  → Final JSON
            └────────────────┘
```

---

## 3. Data Structures

### 3.1 Event (Raw Activity)

**Purpose**: Represents one annotated activity instance on one camera

```python
@dataclass
class Event:
    event_id: str               # "{camera_id}_evt_{activity_id}"
    activity: str               # "person_opens_facility_door"
    camera_id: str              # "G328"
    site: str                   # "school"
    start_frame: int            # 6965
    end_frame: int              # 7020
    start_sec: float            # 232.17
    end_sec: float              # 234.00
    duration_sec: float         # 1.83
    actors: List[Dict]          # [{"actor_id": 143, "entity_type": "person"}]
    video_file: str             # "2018-03-11.11-25-00.11-30-01.school.G421.r13.avi"
    annotation_source: str      # "kitware"
```

**Key properties**:
- **event_id**: Unique within slot (camera + activity instance ID from YAML)
- **actors**: Can have multiple actors (e.g., "person_talks_to_person" has 2 actors)
- **Frames → seconds**: `start_sec = start_frame / 30.0` (assumes 30 fps)

### 3.2 Entity (Person or Vehicle)

**Purpose**: Tracked individual on ONE camera with time span and bounding boxes

```python
@dataclass
class Entity:
    entity_id: str                       # "{camera_id}_actor_{actor_id}"
    camera_id: str                       # "G328"
    actor_id: int                        # 143 (from Kitware geom.yml)
    entity_type: str                     # "person" or "vehicle"
    first_frame: int                     # 6800 (first appearance)
    last_frame: int                      # 8991 (last appearance)
    first_sec: float                     # 226.67
    last_sec: float                      # 299.70
    keyframe_bboxes: Dict[int, List[int]]  # {frame: [x1, y1, x2, y2], ...}
    events: List[str]                    # ["G328_evt_101", "G328_evt_102"]
```

**Key properties**:
- **entity_id**: Format ensures uniqueness (camera + actor ensures no collisions)
- **keyframe_bboxes**: In V6, sampled every 30 frames; in V7, every frame
- **events**: List of event_ids this entity participates in (enables event lookup)
- **Temporal span**: `first_frame` to `last_frame` from geom.yml (actor's full trajectory)

### 3.3 SceneGraph (Slot-Level Knowledge)

**Purpose**: Complete representation of one slot (5-minute time window at one site)

```python
@dataclass
class SceneGraph:
    slot: str                              # "2018-03-11.11-25-00.school"
    cameras: Dict[str, CameraNode]         # {camera_id: CameraNode}
    entities: Dict[str, Entity]            # {entity_id: Entity}
    events: List[Event]                    # All events across all cameras
    events_by_camera: Dict[str, List[Event]]  # {camera_id: [Event, ...]}
```

**Key properties**:
- **cameras**: Metadata for each camera (has KRTD? indoor? position)
- **entities**: All tracked individuals across all cameras (NOT yet linked cross-camera)
- **events_by_camera**: Fast lookup for "what happened on camera X?"

### 3.4 ResolvedGraph (Cross-Camera Links)

**Purpose**: Entity resolution results — which entities are the same person across cameras

```python
@dataclass
class CrossCameraLink:
    entity_a: str                    # "G328_actor_143"
    entity_b: str                    # "G421_actor_122"
    camera_a: str                    # "G328"
    camera_b: str                    # "G421"
    link_type: str                   # "temporal_handoff" or "mevid_ground_truth"
    confidence: float                # 0.85 (heuristic) or 1.0 (MEVID)
    time_gap_sec: Optional[float]    # 3.03 (for temporal handoff)

@dataclass
class EntityCluster:
    cluster_id: str                  # "cluster_001"
    entities: List[str]              # ["G328_actor_143", "G421_actor_122", ...]
    cameras: List[str]               # ["G328", "G421"]
    mevid_person_id: Optional[int]   # 4 (if MEVID ground truth available)

@dataclass
class ResolvedGraph:
    cross_camera_links: List[CrossCameraLink]
    entity_clusters: List[EntityCluster]
    mevid_persons_in_slot: int       # How many MEVID persons cover 2+ cameras
    heuristic_link_count: int        # How many links are heuristic
    mevid_link_count: int            # How many links are MEVID (currently 0, see Section 5.3)
```

### 3.5 Question Output Format

```json
{
  "question_id": "v6_temporal_001",
  "category": "temporal",
  "difficulty": "easy",
  "question_template": "In the surveillance footage, which event occurred first...",
  "options": ["Event A", "Event B", "Both at same time", "Cannot determine"],
  "correct_answer": 0,
  "distractor_reasoning": ["Plausible near-simultaneous", ...],
  "requires_cameras": ["G328", "G421"],
  "verification": {
    "event_a": {
      "camera": "G328",
      "activity": "person_opens_facility_door",
      "start_sec": 232.17,
      "end_sec": 234.00
    },
    "event_b": {...},
    "gap_sec": 3.03,
    "entity_link": "heuristic"
  }
}
```

---

## 4. Step-by-Step Flow

### STEP 1: Parse Annotations

**Input**: Slot name (`"2018-03-11.11-25-00.school"`)  
**Output**: `List[Event]` (raw activity instances)

**Process**:

1. **Lookup slot in index**: Read `/home/ah66742/data/slot_index.json`
   - Find all cameras that have Kitware annotations for this slot
   - Get paths to `.activities.yml` files

2. **For each camera**:
   - Load `{clip}.activities.yml` (YAML list of activity dicts)
   - Load `{clip}.types.yml` (actor_id → entity_type mapping)
   - Parse each activity:
     ```yaml
     - act:
         id2: 101
         act2: {person_opens_facility_door: {}}
         timespan:
           - tsr0: [6965, 7020]  # [start_frame, end_frame]
         actors:
           - id1: 143            # actor_id
     ```
   - Convert to Event object:
     - `event_id = f"{camera_id}_evt_{act_id2}"`
     - `start_sec = start_frame / 30.0`
     - Look up entity_type from types.yml

3. **Aggregate**: Collect all events across all cameras into one list

**Example output**:
```python
[
  Event(event_id="G328_evt_101", activity="person_opens_facility_door", 
        camera_id="G328", start_sec=232.17, end_sec=234.00, 
        actors=[{"actor_id": 143, "entity_type": "person"}]),
  Event(event_id="G421_evt_87", activity="person_exits_scene_through_structure",
        camera_id="G421", start_sec=237.20, end_sec=240.50,
        actors=[{"actor_id": 122, "entity_type": "person"}]),
  ...
]
```

---

### STEP 2: Build Scene Graph

**Input**: `List[Event]`, slot name  
**Output**: `SceneGraph` (entities with bboxes + cameras)

**Process**:

1. **Build camera nodes**:
   - For each unique camera in events:
     - Load KRTD model (camera calibration) if available
     - Check if camera is indoor (no KRTD for indoor cameras)
     - Extract camera position (ENU coordinates) from KRTD
   - Create `CameraNode` object

2. **Extract entity actors from events**:
   - Iterate through all events
   - For each actor in event.actors:
     - Record `(camera_id, actor_id)` pair
     - Record entity_type ("person" or "vehicle")
     - Track which events this actor participates in

3. **Load bounding boxes from geom.yml**:
   - For each camera, locate `{clip}.geom.yml`
   - Stream-parse using regex (see Section 6.2 for details):
     ```python
     for rec in stream_geom_records(geom_path):
         # rec = {"id1": actor_id, "ts0": frame, "g0": [x1, y1, x2, y2]}
         if rec["id1"] in actor_ids:
             entity_bboxes[cam_id][actor_id][frame] = rec["g0"]
     ```
   - Also get frame range: `(first_frame, last_frame)` for each actor

4. **Build Entity objects**:
   - For each `(camera_id, actor_id)` pair:
     - `entity_id = f"{camera_id}_actor_{actor_id}"`
     - Get frame range from geom.yml (or estimate from events if missing)
     - Convert frames to seconds: `first_sec = first_frame / 30.0`
     - Attach keyframe_bboxes and event list

5. **Group events by camera**:
   - Create `events_by_camera` dict for fast camera-specific queries

6. **Assemble SceneGraph**:
   - Package cameras, entities, events into SceneGraph dataclass

**Example output**:
```python
SceneGraph(
    slot="2018-03-11.11-25-00.school",
    cameras={
        "G328": CameraNode(camera_id="G328", is_indoor=False, has_krtd=True, 
                          position_enu=(123.4, 567.8, 2.3)),
        "G421": CameraNode(camera_id="G421", is_indoor=False, has_krtd=True,
                          position_enu=(145.2, 580.1, 2.5)),
        ...
    },
    entities={
        "G328_actor_143": Entity(entity_id="G328_actor_143", camera_id="G328",
                                actor_id=143, first_sec=226.67, last_sec=299.70,
                                keyframe_bboxes={6800: [100, 200, 150, 300], ...},
                                events=["G328_evt_101"]),
        "G421_actor_122": Entity(...),
        ...
    },
    events=[...],
    events_by_camera={...}
)
```

---

### STEP 3: Entity Resolution

**Input**: `SceneGraph`  
**Output**: `ResolvedGraph` (cross-camera links + clusters)

**Process**:

1. **MEVID Ground Truth (Camera-Level)**:
   - Load MEVID annotations for slot:
     ```python
     # Example MEVID data
     Person 4: cameras ['G299', 'G328', 'G336', 'G419', 'G420', 'G421', 'G423']
     Person 12: cameras ['G328', 'G421']
     ```
   - **Note**: Can't create entity-to-entity links yet (no MEVID person_id → Kitware actor_id mapping)
   - Store for validation: if both cameras in question are in same person's camera list → plausible

2. **Heuristic Temporal Handoff**:
   - For each entity with EXIT activity (e.g., "person_exits_scene_through_structure"):
     - Look for entities on OTHER cameras with ENTRY activity within 5 seconds
     - Compute confidence based on:
       - Time gap (smaller = higher confidence)
       - Activity match (exit → entry = higher)
       - Entity type match (person → person = required)
   - Create `CrossCameraLink` if confidence > 0.7
   - **Anti-mega-cluster safeguards** (fixes from V6 debugging):
     - `MAX_HANDOFF_GAP = 10 seconds` (prevent linking distant events)
     - `MAX_LINKS_PER_ENTITY = 2` (prevent one entity linking to many)
     - Only link ACTIVE entities (those with events, not background actors)

3. **Union-Find Clustering**:
   - Initialize UnionFind data structure
   - For each CrossCameraLink:
     - `union(entity_a, entity_b)` — merge into same cluster
   - Extract connected components: `clusters = uf.clusters()`
   - Create `EntityCluster` objects:
     ```python
     EntityCluster(
         cluster_id="cluster_001",
         entities=["G328_actor_143", "G421_actor_122"],
         cameras=["G328", "G421"],
         mevid_person_id=None,  # (can't determine without image matching)
         link_type="heuristic"
     )
     ```

4. **Package results**:
   - Count heuristic vs MEVID links (currently all heuristic)
   - Return ResolvedGraph

**Example output**:
```python
ResolvedGraph(
    cross_camera_links=[
        CrossCameraLink(entity_a="G328_actor_143", entity_b="G421_actor_122",
                       camera_a="G328", camera_b="G421",
                       link_type="temporal_handoff", confidence=0.85,
                       time_gap_sec=3.03),
        ...
    ],
    entity_clusters=[
        EntityCluster(cluster_id="cluster_001", 
                     entities=["G328_actor_143", "G421_actor_122"],
                     cameras=["G328", "G421"],
                     link_type="heuristic"),
        ...
    ],
    mevid_persons_in_slot=41,
    heuristic_link_count=1330,
    mevid_link_count=0
)
```

---

### STEP 4: Generate Temporal Questions

**Input**: `SceneGraph`, `ResolvedGraph`, `random.Random` (seeded)  
**Output**: `List[Dict]` (3 temporal questions)

**Goal**: Ask "Which event happened first?" across cameras

**Process**:

1. **Find candidate pairs**:
   - For each pair of events `(event_a, event_b)`:
     - **Must be on DIFFERENT cameras** (cross-camera requirement)
     - Compute time gap: `gap = event_b.start_sec - event_a.end_sec`
     - **Filter**: 3 ≤ gap ≤ 20 seconds (too close = ambiguous, too far = trivial)
     - Check if entities are in same cluster (stronger question if yes)

2. **Rank/select candidates**:
   - Sort by:
     - Same cluster (preferred)
     - Related activities (using activity hierarchy, if V7)
     - Spatial proximity (if KRTD available)
     - Time gap (moderate gaps preferred)
   - Select top 3 diverse pairs (one near 3s, one ~10s, one ~20s)

3. **Generate question**:
   ```python
   question = (
       f"In the surveillance footage, which event occurred first: "
       f"{event_a.activity} on camera {event_a.camera_id} or "
       f"{event_b.activity} on camera {event_b.camera_id}?"
   )
   ```

4. **Generate distractors**:
   - Option 0: Event A occurred first (CORRECT if gap > 0)
   - Option 1: Event B occurred first
   - Option 2: Both occurred at approximately the same time
   - Option 3: Cannot be determined from the footage

5. **Add verification block**:
   ```json
   "verification": {
     "event_a": {
       "camera": "G328",
       "activity": "person_opens_facility_door",
       "start_sec": 232.17,
       "end_sec": 234.00
     },
     "event_b": {...},
     "gap_sec": 3.03,
     "entity_link": "heuristic"
   }
   ```

---

### STEP 5: Generate Spatial Questions

**Input**: `SceneGraph`, `ResolvedGraph`, `random.Random`  
**Output**: `List[Dict]` (3 spatial questions)

**Goal**: Ask "How far apart are these two people?" using 3D projection

**Process**:

1. **Find entities with KRTD coverage**:
   - Filter entities:
     - Must be on camera with KRTD model
     - Must be type "person" (not vehicle)
     - Must have at least one bbox in keyframe_bboxes

2. **Project entities to 3D world coordinates**:
   - For each entity:
     - Choose representative bbox (at mid_frame = `(first_frame + last_frame) / 2`)
     - If no keyframe bbox at mid_frame, find closest frame OR stream from geom.yml
     - Project bbox foot point to 3D: `pos = camera_model.bbox_foot_to_world(bbox)`
       - Foot point = `(x_center, y_bottom)` in image coordinates
       - Ray-cast to ground plane (Z=0 in ENU coordinates)
       - Returns `[East, North, Up]` position in meters

3. **Compute pairwise distances**:
   - For all pairs of entities with 3D positions:
     - `distance = || pos_a - pos_b ||` (Euclidean distance)
     - Filter out unreasonable distances (> 500m = likely projection error)
     - Classify proximity:
       - `near`: distance ≤ 5m
       - `moderate`: 5m < distance ≤ 15m
       - `far`: distance > 15m

4. **Select diverse candidates**:
   - Pick one from each proximity bucket (near, moderate, far)
   - Shuffle for variety

5. **Generate question**:
   ```python
   if camera_a != camera_b:
       question = (
           f"In the scene, are {activity_a} (camera {camera_a}) and "
           f"{activity_b} (camera {camera_b}) close together or far apart?"
       )
   else:
       question = (
           f"How close are {activity_a} and {activity_b} "
           f"in the scene visible on camera {camera_a}?"
       )
   ```

6. **Options based on proximity**:
   - "They are near each other (within a few meters)"
   - "They are at a moderate distance (5-15 meters)"
   - "They are far apart (more than 15 meters)"
   - "They are at the same location"

7. **Add verification block**:
   ```json
   "verification": {
     "entity_a": "G328_actor_143",
     "entity_b": "G421_actor_122",
     "distance_meters": 8.06,
     "proximity": "moderate",
     "world_pos_enu": {
       "entity_a": [123.4, 567.8, 0.0],
       "entity_b": [131.2, 570.5, 0.0]
     }
   }
   ```

---

### STEP 6: Generate Perception Questions

**Input**: `SceneGraph`, `ResolvedGraph`, `random.Random`  
**Output**: `List[Dict]` (3 perception questions)

**Goal**: Test visual/perceptual reasoning (activity identification, camera selection)

**Process**:

**Type 1: "Which camera shows activity X?"**
1. Find activities that occur on multiple cameras
2. Select one instance as correct answer
3. Distractors: other cameras in slot

**Type 2: "What is the person doing on camera X?"**
1. Select a camera with distinctive activity
2. Correct answer: actual activity (cleaned for readability)
3. Distractors: other activities from distractor bank (contextually plausible)

**Type 3: "Is activity X visible on both cameras A and B?"**
1. Find activities visible on 2+ cameras
2. Correct: "Yes, on both cameras" 
3. Distractors: "Only on A", "Only on B", "Neither camera"

**Distractor Selection** (uses `distractor_bank.py`):
- Load distractor bank (37 MEVA activities)
- Filter by:
  - **Context appropriateness**: if site=school, prefer school-like activities
  - **Visual similarity**: don't use activities too similar to correct answer (IoU of semantic space)
  - **Entity type match**: person activities for person questions

**Example question**:
```json
{
  "question_id": "v6_perception_001",
  "category": "perception",
  "question_template": "On camera G328, what is the person doing around timestamp 232s?",
  "options": [
    "Opening a door",
    "Talking to another person",
    "Carrying a heavy object",
    "Entering through a structure"
  ],
  "correct_answer": 0,
  "verification": {
    "question_type": "activity_identification",
    "camera": "G328",
    "activity": "person_opens_facility_door",
    "timestamp": 233.0
  }
}
```

---

### STEP 7: Deduplication + Validation

**Input**: `List[Dict]` (9 questions from steps 4-6)  
**Output**: Filtered `List[Dict]` + validation errors

**Deduplication logic** (category-specific):

**Temporal**:
- If two questions ask about same `(activity_a, activity_b)` pair → duplicate
- Check: `event_a.activity == other_event_a.activity AND event_b.activity == other_event_b.activity`

**Spatial**:
- If two questions ask about same `(entity_a, entity_b)` pair → duplicate
- Check: `entity_a == other_entity_a AND entity_b == other_entity_b`

**Perception**:
- If two questions have same `(question_type, key)` → duplicate
  - Key for "which_camera": activity name
  - Key for "activity_identification": camera_id
  - Key for "multi_camera_confirmation": activity name

**Validation rules**:

**Temporal validation**:
- ✅ Must be cross-camera (camera_a ≠ camera_b)
- ✅ Gap must be 3-20 seconds
- ✅ Event A must start before Event B

**Spatial validation**:
- ✅ Distance must match proximity label:
  - "near": ≤ 5m
  - "moderate": 5-15m
  - "far": > 15m

**Perception validation**:
- ✅ Correct answer index must be valid (0-3)
- ✅ All options must be non-empty strings

---

## 5. Low-Level Implementation Details

### 5.1 YAML Parsing Strategy

**Why not use `yaml.safe_load()` directly?**
- Safe load on 50+ MB file → 5-10 seconds + 500 MB RAM
- Only need subset of data (specific actor IDs)

**Solution: CSafeLoader for small files, streaming for large files**

```python
# For activities.yml and types.yml (small, <1 MB)
import yaml
with open(path) as f:
    data = yaml.load(f, Loader=yaml.CSafeLoader)
```

**For geom.yml (large, 10-100 MB): Stream-parse with regex**

```python
def stream_geom_records(path: Path):
    inline_re = re.compile(
        r'id1:\s*(\d+).*?ts0:\s*(\d+).*?g0:\s*(\d+\s+\d+\s+\d+\s+\d+)'
    )
    with open(path) as f:
        for line in f:
            m = inline_re.search(line)
            if m:
                yield {
                    'id1': int(m.group(1)),    # actor_id
                    'ts0': int(m.group(2)),    # frame
                    'g0': [int(x) for x in m.group(3).split()],  # bbox
                }
```

**Why this works**:
- Kitware uses inline YAML: `- { geom: {id1: 143, ts0: 6800, g0: 100 200 150 300} }`
- One line = one bbox record
- Regex extracts 3 fields in one pass
- Yield = generator, no memory accumulation

**Performance**:
- 100K-line geom.yml: **1.5 seconds**, **<50 MB RAM**
- Compare to full load: 8 seconds, 500 MB RAM

### 5.2 KRTD 3D Projection

**What is KRTD?**
- Camera calibration format: `K` (intrinsic), `R` (rotation), `T` (translation), `D` (distortion)
- Stored in `/nas/mars/dataset/MEVA/meva-data-repo/metadata/camera-models/krtd/{camera_id}.krtd`

**Projection pipeline**:
```
Image coordinates (bbox) → Undistorted image plane → 3D ray → Ground plane (Z=0)
```

**Code flow**:
```python
def bbox_foot_to_world(self, bbox: List[int]) -> np.ndarray:
    # 1. Extract foot point (person's feet on ground)
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_bottom = bbox[3]  # Bottom of bbox = feet
    
    # 2. Undistort (remove lens distortion)
    pts = np.array([[x_center, y_bottom]], dtype=np.float32)
    pts_undist = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
    
    # 3. Convert to normalized camera coordinates
    x_norm = (pts_undist[0, 0, 0] - self.K[0, 2]) / self.K[0, 0]
    y_norm = (pts_undist[0, 0, 1] - self.K[1, 2]) / self.K[1, 1]
    
    # 4. Create 3D ray from camera center
    ray_camera = np.array([x_norm, y_norm, 1.0])
    ray_world = self.R.T @ ray_camera  # Rotate to world frame
    
    # 5. Intersect with ground plane (Z = 0)
    # Ray: P = camera_center + t * ray_world
    # Ground: P_z = 0
    # Solve: camera_center[2] + t * ray_world[2] = 0
    t = -self.camera_center[2] / ray_world[2]
    
    # 6. Compute world position
    world_pos = self.camera_center + t * ray_world
    
    return world_pos  # [East, North, Up] in meters
```

**Why feet, not bbox center?**
- Feet are reliably on ground plane (Z=0)
- Bbox center is at person's waist (~1m above ground) → Z ≠ 0 → projection error

**ENU coordinate system**:
- **E**ast: +X axis (meters east of site origin)
- **N**orth: +Y axis (meters north of site origin)
- **U**p: +Z axis (meters above ground, always 0 for ground-plane projection)

### 5.3 Why No MEVID Links in V6?

**MEVID provides**:
```
Person 4: cameras ['G299', 'G328', 'G336', 'G419', 'G420', 'G421', 'G423']
```
- Person 4 appears on 7 cameras in this slot
- Ground truth: these cameras contain the same person

**MEVID does NOT provide**:
- Which Kitware `actor_id` on G299 corresponds to Person 4
- Which Kitware `actor_id` on G328 corresponds to Person 4

**Why not?**
- MEVID uses global person IDs (1-158 across all slots)
- Kitware uses camera-local actor IDs (e.g., 143 on G328, 122 on G421)
- **No mapping file exists**

**To create mapping, we need**:
1. Extract MEVID tracklet images (500K-800K crops from `mevid_data/images/`)
2. Extract Kitware geom.yml bboxes (same, from videos using geom.yml coordinates)
3. Match via Person ReID model (cosine similarity of embeddings)
4. Spatial-temporal validation (frame overlap + IoU)
5. **Total compute**: ~10 hours for 47 VSet7 slots, ~8 days for 929 slots

**V6 decision**: Skip MEVID entity-level mapping, use camera-level for validation only

**V7 enhancement**: Add `mevid_validated: true` flag if both cameras in question are in same MEVID person's camera list (plausibility check, not entity-level match)

### 5.4 Entity Resolution Bug Fixes (V6 Debugging)

**Bug 1: Inline YAML format** (Session 14)
- **Symptom**: 0 bboxes found for cameras G299, G419, G423 (3 of 8 cameras)
- **Root cause**: Kitware uses inline YAML, not block YAML:
  ```yaml
  # Kitware format (inline):
  - { geom: {id1: 143, ts0: 6800, g0: 100 200 150 300} }
  
  # Block format (what we expected):
  - geom:
      id1: 143
      ts0: 6800
      g0: 100 200 150 300
  ```
- **Fix**: Change regex to inline-first:
  ```python
  inline_re = re.compile(r'id1:\s*(\d+).*?ts0:\s*(\d+).*?g0:\s*(\d+\s+\d+\s+\d+\s+\d+)')
  ```
- **Result**: 516 actors found for G330 (was 0)

**Bug 2: Mega-cluster** (Session 15)
- **Symptom**: 1 giant cluster with 1,258 entities (all entities in slot)
- **Root cause**: Too permissive heuristic linking:
  - No gap limit → links across 30-second gaps
  - No per-entity link limit → one entity links to 50+ others
  - Low confidence threshold (0.5) → links background actors
- **Fix**: Add safeguards:
  ```python
  MAX_HANDOFF_GAP = 10.0  # seconds
  MAX_LINKS_PER_ENTITY = 2
  MIN_CONFIDENCE = 0.7
  # Only link active entities (those with events)
  ```
- **Result**: 20 meaningful clusters (was 1 mega-cluster)

**Bug 3: Deduplication `None == None`** (Session 15)
- **Symptom**: Only 1 spatial question and 1 perception question (should be 3 each)
- **Root cause**: Checked `event_a`/`event_b` for ALL categories, but spatial/perception lack these fields:
  ```python
  if v_new.get("event_a") == v_old.get("event_a"):  # Both None for spatial → duplicate!
      return True
  ```
- **Fix**: Category-specific dedup logic:
  ```python
  if cat == "temporal":
      # Check event_a/event_b activities
  elif cat == "spatial":
      # Check entity_a/entity_b IDs
  elif cat == "perception":
      # Check question_type + key
  ```
- **Result**: Full 9 questions generated (3+3+3)

---

## 6. Memory Management

### 6.1 V6 Strategy: Sampled Keyframes

**Trade-off**: Memory vs completeness
- **V6**: `sample_every=30` → load every 30th frame
- Actor with 9000 frames → 300 keyframe bboxes
- 20 actors × 300 bboxes = 6000 bboxes per camera (~240 KB)

**Consequences**:
- Incomplete trajectories (miss activities between keyframes)
- Spatial questions use "closest frame" (could be 15 frames = 0.5s off)
- Fast: 1.5s to load + 50 MB RAM for 8 cameras

### 6.2 V7 Strategy: Full Loading

**Trade-off**: Better quality vs more memory
- **V7**: `sample_every=1` → load ALL frames
- Actor with 9000 frames → 9000 bboxes
- 20 actors × 9000 bboxes = 180K bboxes per camera (~7 MB)
- 8 cameras: ~56 MB total (still <0.1% of 128 GB RAM)

**Benefits**:
- Complete trajectories (every frame has bbox)
- Exact spatial (no "closest frame" approximation)
- Enables trajectory-based questions (future: "Was person moving toward/away?")

**Why feasible**:
- Only load bboxes for actors with events (not all 50-100 actors in geom.yml)
- Streaming parser: no peak memory spike
- Modern machines: 128 GB RAM standard

**Recommendation**: Use V7 full loading for all future work

### 6.3 HDF5 mmap Strategy (Future: 929-Slot Batch Processing)

**When needed**: Batch processing 100+ slots with query reuse

**Pipeline**:
1. **One-time indexing** (1-2 min per geom.yml):
   ```python
   # Convert geom.yml → HDF5
   h5py.File(output_path, 'w').create_dataset(
       'geom',
       data=np.array(all_bboxes),
       chunks=True,
       compression='gzip'  # 60-70% size reduction
   )
   ```

2. **Query via mmap** (10 ms per query):
   ```python
   h5 = h5py.File(h5_path, 'r')  # OS mmap's file
   geom = h5['geom'][...]         # Fetch matching rows (lazy load)
   match = geom[(geom['actor_id'] == aid) & (geom['frame'] == frame)]
   ```

**Pros**:
- Constant RAM (~100-200 MB mapped, OS handles paging)
- Fast random access (O(log n) with HDF5 indexing)

**Cons**:
- Upfront cost (2 min × 500 geom files = 16 hours one-time)
- Format lock-in (can't inspect with text editor)

**Decision**: Defer to post-V7 (after VSet7 evaluation)

---

## 7. Question Generation Strategies

### 7.1 Temporal Questions: Gap Selection

**Goal**: Questions should be neither trivial nor ambiguous

**Gap thresholds**:
- **< 3 seconds**: Too close → ambiguous (camera synchronization ±1s)
- **3-20 seconds**: Ideal range (clear temporal order, both events in short window)
- **> 20 seconds**: Too far apart → trivial (obvious answer)

**Gap distribution strategy**:
- Select one question per range: 3-7s, 8-14s, 15-20s
- Ensures diversity: not all questions have same difficulty

**Why 3-20 seconds?**
- MEVA clips are 5 minutes (300 seconds)
- Events are 2-60 seconds each
- Most handoffs happen within 10 seconds (person exits camera A → enters camera B)

### 7.2 Spatial Questions: Proximity Buckets

**Classification**:
```python
def classify_proximity(distance_m: float) -> str:
    if distance_m <= 5.0:
        return "near"
    elif distance_m <= 15.0:
        return "moderate"
    else:
        return "far"
```

**Rationale**:
- **Near (≤5m)**: Same immediate area (same room, same sidewalk section)
- **Moderate (5-15m)**: Nearby but distinct locations (different sides of plaza)
- **Far (>15m)**: Separate areas (different buildings, opposite ends of site)

**Selection strategy**:
- Pick one from each bucket (ensures diverse difficulty)
- Shuffle to avoid pattern (not always near → moderate → far)

**Error handling**:
- Distances > 500m: likely projection error (discard)
- Common causes: indoor cameras with outdoor KRTD (ground plane assumption invalid)

### 7.3 Perception Questions: Distractor Quality

**Distractor bank** (`distractor_bank.py`):
- 37 MEVA activities grouped by context:
  - **Facility**: open_door, close_trunk, enter_vehicle
  - **Object**: pick_up, put_down, carry_heavy
  - **Social**: talk_to_person, hand_gesture
  - **Transport**: vehicle_stops, vehicle_turns

**Selection criteria**:
1. **Context match**: If site=school, prefer facility/social activities
2. **Type match**: Person questions get person activities (not vehicle)
3. **Visual plausibility**: Don't use activities too similar (e.g., "open_door" ≠ "close_door" as distractor)
4. **Frequency balance**: Use rare activities as distractors for common activities

**Example**:
```python
# Correct answer: "person_opens_facility_door"
# Good distractors:
#   - "person_talks_to_person" (different action, same context)
#   - "person_picks_up_object" (different action, plausible)
#   - "person_enters_vehicle" (different action, transportation)
# Bad distractors:
#   - "person_closes_facility_door" (too similar)
#   - "vehicle_starts" (wrong entity type)
```

---

## 8. Validation & Quality Control

### 8.1 Validation Rules (Per Category)

**Temporal**:
```python
def validate_temporal(q: dict) -> List[str]:
    errors = []
    v = q["verification"]
    
    # Rule 1: Must be cross-camera
    if v["event_a"]["camera"] == v["event_b"]["camera"]:
        errors.append("Same camera (should be cross-camera)")
    
    # Rule 2: Gap must be reasonable
    gap = v["gap_sec"]
    if gap < 3.0:
        errors.append(f"Gap too small: {gap}s (min 3s)")
    if gap > 20.0:
        errors.append(f"Gap too large: {gap}s (max 20s)")
    
    # Rule 3: Event A must precede Event B
    if v["event_a"]["start_sec"] >= v["event_b"]["start_sec"]:
        errors.append("Event A does not precede Event B")
    
    return errors
```

**Spatial**:
```python
def validate_spatial(q: dict) -> List[str]:
    errors = []
    v = q["verification"]
    d = v["distance_meters"]
    prox = v["proximity"]
    
    # Rule 1: Distance must match proximity label
    if prox == "near" and d > 5.0:
        errors.append(f"Near but distance={d}m (should be ≤5m)")
    elif prox == "moderate" and not (5.0 < d <= 15.0):
        errors.append(f"Moderate but distance={d}m (should be 5-15m)")
    elif prox == "far" and d <= 15.0:
        errors.append(f"Far but distance={d}m (should be >15m)")
    
    return errors
```

### 8.2 Success Criteria (Section 13 of v6_todo.md)

**Quantitative**:
- ✅ 9 questions generated (3 per category)
- ✅ All questions pass validation (0 errors)
- ✅ Correct answer distribution (not all option 0)
- ✅ ≥2 cameras per question (cross-camera reasoning)
- ✅ Generation time <15 seconds per slot

**Qualitative**:
- ✅ Questions are non-trivial (can't answer from single camera)
- ✅ Distractors are plausible (no obviously wrong options)
- ✅ Language is natural (no template artifacts like "{{activity}}")
- ✅ Verification contains all debug info (cameras, timestamps, distances)

### 8.3 Output JSON Schema

```json
{
  "slot": "2018-03-11.11-25-00.school",
  "version": "v6",
  "annotation_source": "kitware",
  "entity_resolution_source": "mevid+heuristic",
  "generator": "v6_pipeline",
  "seed": 42,
  "difficulty": "easy",
  "cameras": ["G299", "G328", "G330", ...],
  "mevid_persons_in_slot": 41,
  "total_events": 147,
  "total_entities": 926,
  "cross_camera_clusters": 20,
  "total_questions": 9,
  "category_counts": {
    "temporal": 3,
    "spatial": 3,
    "perception": 3
  },
  "validation_issues": 0,
  "generation_time_sec": 7.9,
  "qa_pairs": [
    {
      "question_id": "v6_temporal_001",
      "category": "temporal",
      "difficulty": "easy",
      "question_template": "...",
      "options": [...],
      "correct_answer": 0,
      "distractor_reasoning": [...],
      "requires_cameras": ["G328", "G421"],
      "verification": {...}
    },
    ...
  ]
}
```

---

## Summary

The V6 pipeline is a **modular, memory-efficient, entity-centric** system for generating multi-camera QA pairs from surveillance video annotations. Key innovations:

1. **Streaming GEOM parser**: Regex-based line-by-line parsing avoids OOM on 50+ MB files
2. **Entity resolution**: Heuristic temporal handoff with anti-mega-cluster safeguards
3. **3D spatial reasoning**: KRTD camera calibration enables world-coordinate distance questions
4. **Category-specific deduplication**: Prevents asking same question twice within slot
5. **V7 full loading**: `sample_every=1` provides complete trajectories for better quality

**V7 enhancements** (next steps):
- Entity aliases (human-readable labels)
- Activity hierarchy (connected events)
- Debug markers (frame ranges, timestamps for manual verification)
- MEVID camera-level validation flags

**Performance**: 9 questions in 7.9 seconds, 0 validation errors, 100% pass rate

# V7 Pipeline: Frequently Asked Questions

> **Date**: 2026-02-12  
> **Purpose**: Clarify design decisions, limitations, and technical details

---

## Table of Contents

1. [GPS Data Usage](#1-gps-data-usage)
2. [Actor ID Persistence Problem](#2-actor-id-persistence-problem)
3. [Indoor Camera Handling](#3-indoor-camera-handling)
4. [Cross-Camera Entity Linking](#4-cross-camera-entity-linking)
5. [IoU (Intersection over Union)](#5-iou-intersection-over-union)
6. [KRTD Projection (Detailed)](#6-krtd-projection-detailed)
7. [Spatial Questions Across Cameras](#7-spatial-questions-across-cameras)
8. [Full GEOM Loading in V7](#8-full-geom-loading-in-v7)

---

## 1. GPS Data Usage

### Question: Does V7 use actor GPS data to create/validate questions?

**Short answer**: No, GPS data is **NOT** used in V6 or V7.

**Why not?**

GPS data in MEVA comes from **GPS-logged actors** (people wearing GPS trackers):
- 105 unique GPS logger IDs (G517-G625)
- Continuous tracks in GPX files: lat/lon/elevation every ~10 seconds
- Located in: `/nas/mars/dataset/MEVA/meva-data-repo/metadata/gps/`

**The fundamental problem**: There is **NO mapping** between:
- GPS logger IDs (G517-G625) ← Physical GPS devices
- MEVID person IDs (1-158) ← Re-identification annotations
- Kitware actor IDs (varies per camera) ← Activity annotations

**Example gap**:
```
GPS: G520 logged at (lat=38.9501, lon=-77.1037) on 2018-03-07 17:05:23
MEVID: Person 12 appears on cameras [G328, G421]
Kitware: Camera G328 has actor_id=143, Camera G421 has actor_id=122

Question: Is GPS logger G520 the same person as MEVID Person 12?
Answer: UNKNOWN - no mapping exists
```

**Potential future use**:
- If MEVID person → GPS logger mapping is inferred (via video analysis)
- GPS could validate spatial questions: "Are these two people within 5m?" → check GPS distance
- GPS could generate trajectory questions: "Was person A moving toward/away from person B?"

**Current V7 approach**: Use MEVID camera-level coverage only (see Section 4)

---

## 2. Actor ID Persistence Problem

### Question: Is actor_id stable within one camera? What about the `id1: 1 → id1: 1000` issue?

**Corrected understanding**: The documentation statement "actor_id is stable within one camera" is **MISLEADING**.

**The real behavior**:
- Kitware annotations are **activity-based**, not entity-tracking-based
- `id1` (actor_id) is assigned **per activity instance**, not per physical person
- If the same person performs two separate activities, they MAY get different `id1` values

**Example scenario**:
```yaml
# Camera G328, timestamp 100-120s
- act:
    id2: 101
    act2: {person_opens_facility_door: {}}
    actors:
      - id1: 1  # Person A

# Camera G328, timestamp 150-170s (same person, different activity)
- act:
    id2: 102
    act2: {person_enters_vehicle: {}}
    actors:
      - id1: 1000  # Same physical person, but different id1!
```

**Why does this happen?**
- Kitware annotators labeled activities independently
- No explicit within-camera person tracking requirement
- `id1` links to geom.yml (bounding boxes), which uses the same `id1` values
- If a person leaves FOV and returns, they might get a new `id1`

**Impact on V6/V7 pipeline**:
1. **Two Entity objects created**: `G328_actor_1` and `G328_actor_1000`
2. **These are NOT linked within-camera** (ResolvedGraph only links CROSS-camera)
3. **Questions about "Person A" may miss activities**: If we ask about actor 1's activities, we miss actor 1000's activities

**Is this a problem?**

**For V6/V7**: Partially mitigated
- Most questions are **cross-camera** (temporal questions require different cameras)
- Spatial questions project bbox to 3D → distance is still valid even if same person has multiple IDs
- Perception questions ask about specific activities → don't need full person history

**Still problematic for**:
- Within-camera temporal questions: "Did person A's first activity precede their second activity?" → CAN'T ANSWER if different IDs
- Trajectory analysis: "How did person A move over 5 minutes?" → Fragmented trajectory

**What would fix this?**

**Option 1: Within-camera entity resolution** (not implemented in V6/V7)
```python
# Link entities on SAME camera via:
# 1. Temporal proximity (gap < 5s between activities)
# 2. Spatial proximity (bbox IoU > 0.3 for overlapping time windows)
# 3. Appearance similarity (if using Re-ID features from bboxes)
```

**Option 2: Use geom.yml frame ranges**
- geom.yml has continuous bbox tracks (all frames where person visible)
- If `id1: 1` frames = [100, 500] and `id1: 1000` frames = [600, 900]
- Gap of 100 frames suggests different people (or same person who left/returned)
- Overlap of frames suggests annotation error → merge

**Current V7 approach**: Accept this limitation, document it clearly

---

## 3. Indoor Camera Handling

### Question: How does the pipeline handle indoor cameras?

**Indoor cameras in MEVA**:
- Defined in: `scripts/v6/utils/krtd.py` → `INDOOR_CAMERAS` set
- Examples: G639, G638, G506, G508 (mostly at hospital site)
- ~20% of cameras are indoor

**Key characteristics**:
- **No KRTD calibration models** (indoor = no outdoor ground plane reference)
- **No GPS coverage** (indoor locations)
- **Different FOV** (narrow hallways vs outdoor plazas)

**Pipeline handling**:

| Step | Indoor Camera Behavior |
|------|------------------------|
| Step 1: Parse annotations | ✅ Included (parse activities.yml normally) |
| Step 2: Build scene graph | ✅ Included (entities + bboxes from geom.yml) |
| Step 3: Entity resolution | ✅ Included (can be linked cross-camera via heuristic) |
| Step 4: Temporal questions | ✅ Included (temporal ordering doesn't need KRTD) |
| Step 5: Spatial questions | ❌ **EXCLUDED** (no KRTD → can't project to 3D) |
| Step 6: Perception questions | ✅ Included (visual perception doesn't need KRTD) |

**Code check**:
```python
# scripts/v6/generate_spatial.py, line 50
for cam_id in sg.cameras:
    if cam_id in INDOOR_CAMERAS:
        continue  # Skip spatial questions for indoor cameras
    model = load_camera_model(cam_id)
    if model is not None:
        camera_models[cam_id] = model
```

**Cross-camera linking**: Indoor ↔ Outdoor
- Temporal handoff works: Person exits indoor camera → enters outdoor camera within 10s
- Example: Person exits hospital G506 → appears on outdoor G421
- Creates cross-camera link for temporal questions

**Spatial questions workaround** (future):
- Could use **camera-center distances** (ENU positions from KRTD)
- "How far apart are camera G328 and G421?" → 15.3m
- Less precise than entity-level, but works for indoor cameras
- V6 uses this for camera-based fallback (see Section 7)

---

## 4. Cross-Camera Entity Linking

### Question: Can we link entities across cameras? How?

**Yes, via two methods**:

### Method 1: MEVID Ground Truth (Camera-Level Only)

**What MEVID provides**:
```python
# Example from mevid_data/mevid-v1-annotation-data/person-index-files/
Person 4: cameras ['G299', 'G328', 'G336', 'G419', 'G420', 'G421', 'G423']
Person 12: cameras ['G328', 'G421']
```

**What this means**:
- Person 4 appears on 7 cameras in this slot
- Ground truth: **some** entity on G299 is the same person as **some** entity on G328

**What MEVID does NOT provide**:
- Which specific `actor_id` on G299 corresponds to Person 4
- Which specific `actor_id` on G328 corresponds to Person 4

**V7 use case**: Camera-level validation
```python
# When generating temporal question:
# Is event_a.camera in same MEVID person's camera list as event_b.camera?
if 'G328' in mevid_persons[4] and 'G421' in mevid_persons[4]:
    question["verification"]["mevid_validated"] = True  # Plausible same person
```

**Why not entity-level MEVID?**
- Requires image matching: Extract crops from videos → match via Person Re-ID model
- Compute-intensive: 10 hours for 47 VSet7 slots
- See: `MEVID_IMPLEMENTATION_PLAN.md` for full technical approach

### Method 2: Heuristic Temporal Handoff

**Core idea**: If person exits camera A and enters camera B within 10 seconds, likely same person

**Algorithm**:
```python
def _resolve_heuristic(sg: SceneGraph):
    links = []
    
    # For each entity with EXIT activity
    for entity_a in sg.entities.values():
        if not has_exit_activity(entity_a):
            continue
        
        # Look for entities on OTHER cameras with ENTRY activity
        for entity_b in sg.entities.values():
            if entity_b.camera_id == entity_a.camera_id:
                continue  # Must be cross-camera
            
            if not has_entry_activity(entity_b):
                continue
            
            # Compute time gap
            gap = entity_b.first_sec - entity_a.last_sec
            
            if 0 < gap < MAX_HANDOFF_GAP:  # 10 seconds
                confidence = compute_confidence(gap, activities, entity_types)
                if confidence > 0.7:
                    links.append(CrossCameraLink(
                        entity_a=entity_a.entity_id,
                        entity_b=entity_b.entity_id,
                        link_type="temporal_handoff",
                        confidence=confidence,
                        time_gap_sec=gap
                    ))
    
    return links
```

**Confidence scoring**:
- **Time gap**: Smaller gap → higher confidence
  - 0-3s: very likely same person
  - 3-7s: likely same person
  - 7-10s: possible same person
- **Activity match**: exit → entry = +0.2 confidence
- **Entity type**: person → person required (person → vehicle = reject)

**Anti-mega-cluster safeguards** (learned from V6 debugging):
```python
MAX_HANDOFF_GAP = 10.0      # Don't link events >10s apart
MAX_LINKS_PER_ENTITY = 2    # Each entity links to at most 2 others
MIN_CONFIDENCE = 0.7        # Threshold for creating link
# Only link ACTIVE entities (those with annotated events, not background actors)
```

**Union-Find clustering**:
```python
# After creating links, merge into clusters
uf = UnionFind()
for link in links:
    uf.union(link.entity_a, link.entity_b)

clusters = uf.clusters()  # Connected components
# Example: {
#   "cluster_001": ["G328_actor_143", "G421_actor_122", "G299_actor_88"],
#   "cluster_002": ["G330_actor_55", "G336_actor_91"],
#   ...
# }
```

**Result**: 
- V6 on `2018-03-11.11-25-00.school`: 1330 links → 20 clusters
- Average cluster size: ~3 entities (person tracked across 3 cameras)

---

## 5. IoU (Intersection over Union)

### Question: How does IoU work in more depth?

**Purpose**: Measure overlap between two bounding boxes (0.0 = no overlap, 1.0 = perfect overlap)

**Mathematical definition**:
```
IoU = Area(Intersection) / Area(Union)
```

**Visual example**:
```
Bbox A: [100, 200, 150, 300]  →  50px wide × 100px tall = 5000px²
Bbox B: [120, 220, 170, 320]  →  50px wide × 100px tall = 5000px²

Intersection rectangle:
  x1 = max(100, 120) = 120
  y1 = max(200, 220) = 220
  x2 = min(150, 170) = 150
  y2 = min(300, 320) = 300
  → [120, 220, 150, 300] = 30px × 80px = 2400px²

Union = Area(A) + Area(B) - Area(Intersection)
      = 5000 + 5000 - 2400 = 7600px²

IoU = 2400 / 7600 = 0.316
```

**Code implementation** (`scripts/v6/utils/iou.py`):
```python
def compute_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox_a: [x1, y1, x2, y2]  (top-left and bottom-right corners)
        bbox_b: [x1, y1, x2, y2]
    
    Returns:
        IoU value [0.0, 1.0]
    """
    # 1. Find intersection rectangle
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    
    # 2. Check if boxes don't overlap
    if x2 <= x1 or y2 <= y1:
        return 0.0  # No overlap
    
    # 3. Compute areas
    inter = (x2 - x1) * (y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    
    # 4. Handle edge case
    if union <= 0:
        return 0.0
    
    # 5. Return ratio
    return inter / union
```

**Use cases in V6/V7**:

**1. Within-camera entity resolution** (not currently implemented, but proposed):
```python
# If two entities have overlapping time spans on same camera:
# Check if their bboxes overlap → likely same person with different actor_ids

entity_a_bbox = get_bbox_at_frame(geom_path, actor_id=1, frame=300)
entity_b_bbox = get_bbox_at_frame(geom_path, actor_id=1000, frame=300)
iou = compute_iou(entity_a_bbox, entity_b_bbox)

if iou > 0.5:
    # Likely same person with annotation inconsistency → merge entities
    union_entities(entity_a, entity_b)
```

**2. MEVID tracklet matching** (future work):
```python
# Match MEVID person crops to Kitware geom.yml bboxes
# Step 1: Extract crop at frame F from MEVID
# Step 2: Find Kitware bbox at same frame
# Step 3: Compute IoU to validate temporal alignment

mevid_crop_bbox = [120, 220, 170, 320]  # From MEVID tracklet
kitware_bbox = get_bbox_at_frame(geom_path, actor_id=143, frame=7100)
iou = compute_iou(mevid_crop_bbox, kitware_bbox)

if iou > 0.3:  # Threshold for spatial-temporal validation
    # Likely match → MEVID person_id maps to Kitware actor_id
```

**3. Activity overlap detection**:
```python
# Are two activities happening in same spatial region?
activity_a_bbox = entity_a.keyframe_bboxes[frame]
activity_b_bbox = entity_b.keyframe_bboxes[frame]
iou = compute_iou(activity_a_bbox, activity_b_bbox)

if iou > 0.1:
    # Activities spatially overlap → "interaction" or "co-location"
```

**IoU thresholds in practice**:
- **0.5+**: Strong overlap (likely same object tracked continuously)
- **0.3-0.5**: Moderate overlap (same region, could be same object or nearby objects)
- **0.1-0.3**: Weak overlap (nearby objects, edges touching)
- **<0.1**: Negligible overlap (effectively separate objects)

---

## 6. KRTD Projection (Detailed)

### Question: Can you explain KRTD projection in more depth?

**Goal**: Convert 2D image pixel coordinates (bbox) → 3D world coordinates (meters)

### 6.1 KRTD Parameters

**K**: Camera Intrinsic Matrix (3×3)
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```
- `fx, fy`: Focal length in pixels (horizontal and vertical)
- `cx, cy`: Principal point (image center) in pixels
- Example: fx=1800, fy=1800, cx=960, cy=540 (for 1920×1080 image)

**R**: Rotation Matrix (3×3)
```
R = camera_space → world_space rotation
```
- Converts vectors from camera coordinates to world coordinates
- Orthogonal matrix: R.T @ R = I
- Example: camera pointing northeast at 45° tilt

**T**: Translation Vector (3×1)
```
T = [tx, ty, tz]  # Camera position in world coordinates
```
- Camera center position in meters (ENU coordinates)
- Example: [123.4, 567.8, 3.2] = 123.4m East, 567.8m North, 3.2m Up

**D**: Distortion Coefficients (4×1 or 5×1)
```
D = [k1, k2, p1, p2, (k3)]
```
- `k1, k2, k3`: Radial distortion (barrel/pincushion)
- `p1, p2`: Tangential distortion
- Most outdoor cameras have moderate distortion (k1 ≈ -0.1 to -0.3)

### 6.2 Projection Pipeline (6 Steps)

**Input**: Bounding box `[x1, y1, x2, y2]` in image coordinates  
**Output**: 3D position `[East, North, Up]` in meters

**Step 1: Extract Foot Point**
```python
# Person's feet are at bottom-center of bbox
x_center = (bbox[0] + bbox[2]) / 2.0
y_bottom = bbox[3]  # Bottom edge = feet on ground

foot_point_image = [x_center, y_bottom]  # Image pixel coordinates
```

**Why feet?**
- Feet are reliably on ground plane (Z=0 in ENU)
- Bbox center is at person's waist (~1-1.5m above ground) → introduces Z error
- Head could be 1.7m above ground → even more error

**Step 2: Undistort (Remove Lens Distortion)**
```python
# OpenCV undistortPoints: image → normalized camera coordinates
pts = np.array([[x_center, y_bottom]], dtype=np.float32)
pts_undist = cv2.undistortPoints(pts, K, D, P=K)

x_undist = pts_undist[0, 0, 0]
y_undist = pts_undist[0, 0, 1]
```

**What undistortPoints does**:
1. Converts pixel → normalized image plane: `(x - cx) / fx`, `(y - cy) / fy`
2. Applies inverse distortion model: removes radial/tangential warping
3. Returns in pixel coordinates (if P=K) or normalized coordinates (if P=None)

**Step 3: Convert to Normalized Camera Coordinates**
```python
# From pixel coordinates → normalized [-1, 1] range
x_norm = (x_undist - K[0, 2]) / K[0, 0]  # (x - cx) / fx
y_norm = (y_undist - K[1, 2]) / K[1, 1]  # (y - cy) / fy

# Now we have a unit direction in camera space
ray_camera = np.array([x_norm, y_norm, 1.0])  # Z=1 convention
```

**Step 4: Rotate to World Frame**
```python
# Camera space → World space rotation
ray_world = R.T @ ray_camera  # R.T = inverse rotation (since R is orthogonal)

# ray_world is now a 3D direction vector in ENU coordinates
# Example: [0.707, 0.707, -0.1] means pointing NE at slight downward angle
```

**Step 5: Intersect with Ground Plane**
```python
# Ray equation: P(t) = camera_center + t * ray_world
# Ground plane: Z = 0 (everything on ground)
# Intersection: camera_center[2] + t * ray_world[2] = 0

camera_center = -R.T @ T  # Derive from R, T
t = -camera_center[2] / ray_world[2]

# t is the scale factor: how far along ray to reach ground
# If camera is 3m above ground and ray points down at 30°:
# t ≈ 3 / sin(30°) = 6m
```

**Step 6: Compute World Position**
```python
world_pos = camera_center + t * ray_world

# Result: [East, North, 0.0]
# Example: [131.2, 570.5, 0.0] = 131.2m East, 570.5m North, on ground
```

### 6.3 Error Sources

**1. Ground Plane Assumption**
- Assumes all people are on flat ground (Z=0)
- **Violation**: Hills, stairs, elevated platforms
- **Impact**: Position error proportional to Z deviation
- Example: Person on 1m platform → position error ~1-2m

**2. Indoor Cameras**
- No outdoor ground plane reference
- KRTD models may not exist or be unreliable
- **Solution**: Exclude indoor cameras from spatial questions

**3. Bbox Accuracy**
- geom.yml bboxes are manually adjusted or auto-tracked
- Error in bbox foot point → error in 3D position
- **Typical error**: ±10 pixels in 1920×1080 image
- **3D error**: ±0.5-1m at 30m distance

**4. Camera Calibration Drift**
- KRTD models calibrated once, cameras may shift over time (wind, vibration)
- **Impact**: Systematic position error (all entities shifted in same direction)
- **Mitigation**: Accept small errors (<2m) as inherent limitation

### 6.4 Validation Techniques

**1. Distance Sanity Checks**
```python
if distance > 500:
    # Likely projection error (cameras <200m apart in MEVA)
    discard_candidate()
```

**2. Cross-Camera Consistency**
```python
# If same person on two cameras, their 3D positions should be similar
pos_a = project_bbox(camera_a, bbox_a)
pos_b = project_bbox(camera_b, bbox_b)
distance = ||pos_a - pos_b||

if distance > 10:  # Same person shouldn't be 10m apart simultaneously
    # Annotation error or projection issue
```

**3. GPS Ground Truth** (future)
```python
# If GPS data available, compare projected positions to GPS
gps_pos = get_gps_position(actor_gps_id, timestamp)
projected_pos = project_bbox(camera, bbox)
error = ||gps_pos - projected_pos||

if error > 5:  # 5m threshold
    # Investigate: bbox error? KRTD error? GPS error?
```

---

## 7. Spatial Questions Across Cameras

### Question: Can spatial questions use multiple cameras?

**Yes, most spatial questions in V6/V7 are cross-camera.**

### 7.1 Cross-Camera Spatial Questions

**Typical question**:
```
"In the scene, are {activity_a} (camera {cam_a}) and {activity_b} (camera {cam_b}) 
close together or far apart?"
```

**Requirements**:
- Both cameras must have KRTD models
- Both entities must have bounding boxes
- Projection to 3D must succeed (feet on ground plane)

**Example**:
```python
# Entity A on camera G328
bbox_a = entity_a.keyframe_bboxes[7000]  # [100, 200, 150, 300]
model_a = load_camera_model("G328")
pos_a = model_a.bbox_foot_to_world(bbox_a)  # [123.4, 567.8, 0.0]

# Entity B on camera G421
bbox_b = entity_b.keyframe_bboxes[7100]  # [800, 400, 900, 600]
model_b = load_camera_model("G421")
pos_b = model_b.bbox_foot_to_world(bbox_b)  # [131.2, 570.5, 0.0]

# Compute distance
distance = ||pos_a - pos_b|| = √[(131.2-123.4)² + (570.5-567.8)²]
         = √[7.8² + 2.7²] = √[60.84 + 7.29] = √68.13 = 8.25m

# Classify
proximity = "moderate"  # 5-15m range
```

**V6 actual output**:
```json
{
  "question_id": "v6_spatial_002",
  "question_template": "In the scene, are person_opens_facility_door (camera G328) and person_talks_to_person (camera G421) close together or far apart?",
  "verification": {
    "entity_a": "G328_actor_143",
    "entity_b": "G421_actor_122",
    "camera_a": "G328",
    "camera_b": "G421",
    "distance_meters": 8.06,
    "proximity": "moderate",
    "world_pos_enu": {
      "entity_a": [123.4, 567.8, 0.0],
      "entity_b": [131.2, 570.5, 0.0]
    }
  }
}
```

### 7.2 Same-Camera Spatial Questions

**Also supported** (less common):
```
"How close are {activity_a} and {activity_b} in the scene visible on camera {cam}?"
```

**Example**:
```python
# Both entities on camera G330
entity_a = "G330_actor_88"   # Person opening door
entity_b = "G330_actor_91"   # Person carrying object

# Project both to 3D using SAME camera model
model = load_camera_model("G330")
pos_a = model.bbox_foot_to_world(bbox_a)  # [140.1, 572.3, 0.0]
pos_b = model.bbox_foot_to_world(bbox_b)  # [142.8, 573.1, 0.0]

distance = ||pos_a - pos_b|| = 2.87m
proximity = "near"
```

**V6 output**:
```json
{
  "question_id": "v6_spatial_001",
  "question_template": "How close are person_opens_facility_door and person_carries_heavy_object in the scene visible on camera G330?",
  "verification": {
    "camera_a": "G330",
    "camera_b": "G330",  # Same camera
    "distance_meters": 2.87,
    "proximity": "near"
  }
}
```

### 7.3 Camera-Center Fallback (Future Enhancement)

**For slots with NO entity projections** (all indoor, or KRTD issues):
```python
# Fallback: Use camera center positions instead of entity positions
camera_a_pos = model_a.camera_center  # [120.0, 565.0, 3.2]
camera_b_pos = model_b.camera_center  # [145.0, 580.0, 2.8]

distance = ||camera_a_pos - camera_b_pos|| = 26.3m

question = (
    f"Which cameras are closer together: {cam_a} and {cam_b}, "
    f"or {cam_c} and {cam_d}?"
)
```

**Not implemented in V6, but documented in V7 todo** (Section 10: Future Enhancements)

---

## 8. Full GEOM Loading in V7

### Question: Is there merit to loading the full geom file in V7?

**Yes, absolutely. V7 already implements this.**

### 8.1 V6 vs V7 Comparison

| Aspect | V6 | V7 |
|--------|----|----|
| **Sampling** | `sample_every=30` | `sample_every=1` |
| **Bboxes per actor** | ~300 (5-min clip) | ~9000 (5-min clip) |
| **Memory per camera** | ~240 KB | ~7 MB |
| **Total for 8 cameras** | ~2 MB | ~56 MB |
| **Trajectory completeness** | Sparse (1 per sec) | Complete (30 per sec) |
| **Spatial accuracy** | ±0.5s approximation | Exact frame |

### 8.2 Benefits of Full Loading

**1. Complete Trajectories**
```python
# V6: Only have bboxes at frames [0, 30, 60, 90, ...]
# Activity spans frames 45-75 → only 2 bboxes!

# V7: Have ALL bboxes [0, 1, 2, ..., 9000]
# Activity spans frames 45-75 → 30 bboxes (full trajectory)
```

**2. Exact Spatial Positioning**
```python
# V6: Need bbox at frame 7100
# Closest keyframe: 7110 (10 frames = 0.33s off)
# Position error: person walked ~1m in 0.33s → 1m spatial error

# V7: Have exact bbox at frame 7100
# No approximation error
```

**3. Temporal Precision**
```python
# Question: "When did person A start opening the door?"
# V6: Can only say "between frame 7080 and 7110" (30-frame = 1s window)
# V7: Can pinpoint exact frame 7093 (0.03s precision)
```

**4. Future Trajectory Questions**
```python
# "Was person A moving toward or away from person B?"
# V6: Only 2 positions → can't determine trajectory
# V7: 30 positions → compute velocity vector, direction of motion

# Example:
positions = [bbox_to_3d(bbox) for bbox in entity_a.keyframe_bboxes.values()]
velocities = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
avg_velocity = np.mean(velocities, axis=0)  # [vx, vy] in m/s

direction_to_b = pos_b - pos_a[0]  # Vector from A's start to B
dot_product = np.dot(avg_velocity, direction_to_b)

if dot_product > 0:
    answer = "Person A was moving toward person B"
else:
    answer = "Person A was moving away from person B"
```

**5. Activity Boundary Detection**
```python
# Detect exactly when person starts/stops activity
# V6: Activity endpoints quantized to 30-frame intervals
# V7: Exact frame where bbox motion changes
```

### 8.3 Memory Feasibility

**This machine**: 128 GB RAM available

**V7 memory usage**:
- Full GEOM load: 56 MB for 8 cameras
- Scene graph: ~20 MB
- Entity resolution: ~10 MB
- Question generation: ~5 MB
- **Total peak**: ~100 MB (0.08% of 128 GB)

**Even for 929-slot batch processing**:
- 929 slots × 100 MB = ~93 GB
- Still fits in 128 GB RAM with room for OS + other processes

**Conclusion**: Full loading is trivially feasible, benefits far outweigh costs.

### 8.4 V7 Implementation Status

✅ **Already implemented** in V7:
- [scripts/v7/build_scene_graph.py](scripts/v7/build_scene_graph.py#L147): `sample_every=1`
- [scripts/v7/utils/yaml_stream.py](scripts/v7/utils/yaml_stream.py#L63): Default `sample_every=1`
- [v7_todo.md Section 6](v7_todo.md): Full documentation of rationale

---

## Summary

| Question | Answer |
|----------|--------|
| **GPS data usage?** | No, no mapping to actor_ids |
| **Actor ID persistence?** | NOT guaranteed - can change within-camera (known limitation) |
| **Indoor cameras?** | Included in temporal/perception, excluded from spatial |
| **Cross-camera linking?** | Via MEVID (camera-level) + heuristic temporal handoff |
| **IoU?** | Intersection / Union ratio for bbox overlap (0-1) |
| **KRTD projection?** | 6-step pipeline: foot point → undistort → 3D ray → ground plane |
| **Spatial multi-camera?** | Yes, most spatial questions are cross-camera |
| **Full GEOM loading?** | Yes, implemented in V7, highly beneficial |

---

**Next Steps**: See v7_todo.md for V7 implementation plan (entity aliases, activity hierarchy, debug markers)

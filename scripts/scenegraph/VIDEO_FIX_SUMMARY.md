# Video Creation Fix Summary

## Problems

### Problem 1: No Objects Showing
The video creation script was not showing any objects because the scene graph JSON was missing critical fields needed for visualization.

### Problem 2: Bounding Boxes Appearing in Wrong Locations
Bounding boxes were appearing in far corners of images because object rotation was not being applied when creating 3D bounding boxes.

## Root Causes

### Issue 1: Missing Fields
The `SceneGraphNode` dataclass was missing three essential fields:
1. **`sample_token`** - Required to load camera images from nuScenes
2. **`visible_cameras`** - Required to filter which objects to draw on each camera view
3. **`rotation`** - Required to orient bounding boxes correctly

### Issue 2: Incorrect Coordinate Transformation
The video creation code had TWO problems with coordinate transformations:
1. **Missing rotation**: Not applying the object's rotation quaternion
2. **Incomplete transformation chain**: Going directly from global → camera, skipping the ego vehicle frame

The correct transformation chain is: **Global → Ego Vehicle → Camera**

## Fixes Applied

### 1. Updated `SceneGraphNode` (scenegraph.py)
Added missing fields to the dataclass:
```python
@dataclass
class SceneGraphNode:
    # ... existing fields ...
    sample_token: str  # Sample token for this frame
    visible_cameras: Optional[List[str]] = None  # Cameras where object is visible
```

### 2. Updated `build_frame_scene_graph` (scenegraph.py)
Now populates the new fields when creating nodes:
```python
node = SceneGraphNode(
    # ... existing fields ...
    sample_token=frame.sample_token,
    visible_cameras=obj.visible_cameras
)
```

### 3. Updated `export_to_json` (scenegraph.py)
- Added `scene_token` parameter
- Exports `sample_token` at frame level
- Exports `visible_cameras` for each object

### 4. Fixed Video Creation Coordinate Transforms (create_scenegraph_video.py)
- **Fixed coordinate transformation chain**: Now properly transforms Global → Ego → Camera
- **Added ego pose retrieval**: Gets ego vehicle pose for each frame
- **Applies object rotation**: Uses rotation quaternion to orient bounding boxes correctly
- **Added debug output**: Helps diagnose issues with object visibility and counts
- **Self-contained**: Uses object data directly from scene graph JSON (no redundant database queries)

**Transformation code**:
```python
# 1. Create corners in object's local frame
corners = create_box_corners(size)

# 2. Rotate to object's orientation
corners_rotated = rotation_quat.rotation_matrix @ corners

# 3. Translate to global position
corners_global = corners_rotated + position

# 4. Transform to ego vehicle frame
corners_ego = corners_global - ego_translation
corners_ego = ego_rotation.inverse.rotation_matrix @ corners_ego

# 5. Transform to camera frame
corners_cam = corners_ego - cam_translation
corners_cam = cam_rotation.inverse.rotation_matrix @ corners_cam

# 6. Project to 2D image
corners_2d = view_points(corners_cam, cam_intrinsic, normalize=True)
```

## Scene Graph JSON Format

After the fix, the exported JSON has this structure:

```json
{
  "scene_token": "scene_token_here",
  "frames": [
    {
      "frame_idx": 0,
      "sample_token": "sample_token_here",
      "timestamp": 1532402927647951,
      "objects": [
        {
          "object_id": "instance_token_123",
          "object_class": "vehicle.car",
          "position": [10.5, 5.2, 0.5],
          "size": [4.5, 1.8, 1.5],
          "velocity": [2.5, 0.3],
          "attributes": ["vehicle.moving"],
          "num_lidar_pts": 450,
          "visibility": 4,
          "sample_token": "sample_token_here",
          "visible_cameras": ["CAM_FRONT", "CAM_FRONT_LEFT"],
          "activity": null,
          "description": null,
          "caption": null
        }
      ],
      "relationships": [...]
    }
  ]
}
```

## How to Regenerate Scene Graph JSON

If you have an old scene graph JSON file, you need to regenerate it:

```bash
cd /path/to/scripts/scenegraph

# Run the scene graph builder
python scenegraph.py
```

Or use it programmatically:

```python
from nuscenes_dataloader import NuScenesLidarSegmentationLoader
from scenegraph import SceneGraphBuilder

# Initialize
loader = NuScenesLidarSegmentationLoader(
    dataroot='/nas/standard_datasets/nuscenes',
    version='v1.0-trainval'
)

builder = SceneGraphBuilder(loader)

# Build scene graph
scene_token = loader.get_scene_tokens()[0]
nodes, relationships = builder.build_scene_graphs(scene_token)

# Export with scene_token
builder.export_to_json(
    nodes, 
    relationships, 
    f'outputs/{scene_token}_scene_graph.json',
    scene_token=scene_token  # Important!
)
```

## Creating Videos

Once you have a properly formatted scene graph JSON:

```bash
python create_scenegraph_video.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --version v1.0-trainval \
    --input-json outputs/scene_token_scene_graph.json \
    --output-dir videos/ \
    --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
    --fps 2 \
    --show-bbox \
    --show-labels \
    --show-properties
```

## Debug Output

The video creation script now prints helpful debug information:
- Total frames in scene graph
- Frame structure (keys)
- Object structure (keys)
- Sample first object
- Number of objects per frame
- Number of visible objects per camera

This helps diagnose any remaining issues.

## Verification

To verify your scene graph JSON is correct:

```python
import json

with open('scene_graph.json', 'r') as f:
    data = json.load(f)

# Check structure
print(f"Scene token: {data.get('scene_token')}")
print(f"Number of frames: {len(data['frames'])}")

# Check first frame
frame = data['frames'][0]
print(f"Sample token: {frame.get('sample_token')}")
print(f"Number of objects: {len(frame['objects'])}")

# Check first object
if frame['objects']:
    obj = frame['objects'][0]
    print(f"Object class: {obj.get('object_class')}")
    print(f"Visible cameras: {obj.get('visible_cameras')}")
    print(f"Position: {obj.get('position')}")
    print(f"Size: {obj.get('size')}")
```

All these fields should be present and populated!


# nuScenes Lidar Segmentation Dataloader

This module provides a comprehensive dataloader for the nuScenes dataset to extract lidar segmentation maps and build scene graphs for VLM annotation.

## Features

- **LiDAR Segmentation**: Extract point clouds with instance-level segmentation labels
- **Object Properties**: Extract detailed object properties (position, size, velocity, attributes)
- **Scene Graphs**: Build structured scene graphs with spatial relationships
- **VLM Integration**: Prepare objects for Vision-Language Model annotation

## Installation

### Prerequisites

1. Install the nuScenes devkit:
```bash
pip install nuscenes-devkit
```

2. Download the nuScenes dataset from [nuScenes.org](https://www.nuscenes.org/)
   - For testing: `v1.0-mini` (3.4 GB)
   - For full dataset: `v1.0-trainval` (348 GB)

### Required Dependencies

The following packages are already included in your `pyproject.toml`:
- numpy
- opencv-python
- tqdm

Additional required:
- nuscenes-devkit

## Dataset Structure

The nuScenes dataset should be organized as follows:
```
/path/to/nuscenes/
├── maps/
├── samples/
│   ├── CAM_FRONT/
│   ├── CAM_BACK/
│   ├── LIDAR_TOP/
│   └── ...
├── sweeps/
├── v1.0-trainval/
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   └── ...
└── v1.0-mini/  (if using mini version)
```

## Usage

### 1. Basic LiDAR Segmentation Extraction

```python
from nuscenes_dataloader import NuScenesLidarSegmentationLoader

# Initialize dataloader
loader = NuScenesLidarSegmentationLoader(
    dataroot='/path/to/nuscenes',
    version='v1.0-mini',  # or 'v1.0-trainval'
    verbose=True
)

# Get all scenes
scene_tokens = loader.get_scene_tokens()

# Load a specific scene
scene_token = scene_tokens[0]
frames = loader.get_scene_frames(scene_token)

# Access frame data
for frame in frames:
    print(f"Frame: {frame.sample_token}")
    print(f"Points: {frame.lidar_points.shape}")
    print(f"Objects: {len(frame.objects)}")
    
    # Lidar points: (N, 5) - x, y, z, intensity, ring_index
    # Segmentation labels: (N,) - instance ID per point
```

### 2. Build Scene Graphs

```python
from scenegraph import SceneGraphBuilder
from nuscenes_dataloader import NuScenesLidarSegmentationLoader

# Initialize dataloader
loader = NuScenesLidarSegmentationLoader(
    dataroot='/path/to/nuscenes',
    version='v1.0-mini',
    verbose=True
)

# Initialize scene graph builder
builder = SceneGraphBuilder(
    dataloader=loader,
    extract_relationships=True,
    distance_threshold=10.0  # meters
)

# Build scene graphs for a scene
scene_token = loader.get_scene_tokens()[0]
all_nodes, all_relationships = builder.build_scene_graphs(scene_token)

# Export to JSON
builder.export_to_json(
    all_nodes,
    all_relationships,
    'outputs/scene_graph.json'
)
```

### 3. Prepare Objects for VLM Annotation

```python
# Get objects filtered by class and visibility
dynamic_classes = [
    'vehicle.car',
    'vehicle.truck',
    'vehicle.bus',
    'human.pedestrian',
    'vehicle.bicycle',
    'vehicle.motorcycle'
]

objects_for_vlm = builder.get_objects_for_vlm_annotation(
    all_nodes,
    filter_classes=dynamic_classes,
    min_visibility=2  # 0-4, where 4 is fully visible
)

# Save for VLM processing
import json
with open('outputs/vlm_objects.json', 'w') as f:
    json.dump(objects_for_vlm, f, indent=2)
```

### 4. Save Segmentation Data

```python
# Save individual frame data
for frame in frames:
    loader.save_frame(
        frame,
        output_dir='outputs/frames',
        save_points=True,      # Save point cloud (.npy)
        save_metadata=True     # Save metadata (.json)
    )
```

## Data Structures

### FrameData

Contains complete information for a single frame:

```python
@dataclass
class FrameData:
    scene_token: str                    # Scene identifier
    sample_token: str                   # Frame identifier
    timestamp: int                      # Unix timestamp
    lidar_points: np.ndarray           # (N, 5) array: x, y, z, intensity, ring
    segmentation_labels: np.ndarray    # (N,) instance labels
    objects: List[ObjectProperties]    # Detected objects
    ego_pose: Dict[str, Any]          # Vehicle pose
    camera_tokens: Dict[str, str]     # Camera sensor tokens
```

### ObjectProperties

Properties for each detected object:

```python
@dataclass
class ObjectProperties:
    token: str                          # Annotation token
    name: str                          # Object class (e.g., 'vehicle.car')
    position: Tuple[float, float, float]  # 3D position (x, y, z)
    size: Tuple[float, float, float]      # width, length, height
    rotation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    velocity: Optional[Tuple[float, float]]  # velocity in x, y
    num_lidar_pts: int                 # Number of lidar points
    visibility: int                    # Visibility level (0-4)
    attributes: List[str]              # Object attributes
    instance_token: str                # Unique instance identifier
```

### SceneGraphNode

Node in the scene graph representing an object:

```python
@dataclass
class SceneGraphNode:
    object_id: str           # Unique object instance token
    object_class: str        # Object category
    position: tuple          # 3D position
    size: tuple             # Object dimensions
    velocity: Optional[tuple]  # Object velocity
    attributes: List[str]    # Object attributes
    frame_idx: int          # Frame index
    timestamp: int          # Timestamp
    num_lidar_pts: int      # Number of lidar points
    visibility: int         # Visibility level
    
    # VLM-annotated properties (to be filled)
    activity: Optional[str] = None
    description: Optional[str] = None
    caption: Optional[str] = None
```

### SceneGraphRelationship

Spatial relationship between objects:

```python
@dataclass
class SceneGraphRelationship:
    source_id: str           # Source object instance token
    target_id: str          # Target object instance token
    relationship_type: str   # 'near', 'behind', 'in_front', 'left', 'right'
    distance: Optional[float]  # Distance in meters
    frame_idx: int          # Frame index
```

## Object Classes in nuScenes

The nuScenes dataset includes 23 object classes:

### Vehicles
- `vehicle.car`
- `vehicle.truck`
- `vehicle.bus`
- `vehicle.construction`
- `vehicle.emergency.ambulance`
- `vehicle.emergency.police`
- `vehicle.bicycle`
- `vehicle.motorcycle`
- `vehicle.trailer`

### Humans
- `human.pedestrian.adult`
- `human.pedestrian.child`
- `human.pedestrian.construction_worker`
- `human.pedestrian.police_officer`

### Movable Objects
- `movable_object.barrier`
- `movable_object.trafficcone`
- `movable_object.pushable_pullable`
- `movable_object.debris`

### Static Objects
- `static_object.bicycle_rack`

### Animals
- `animal`

## Spatial Relationships

The scene graph builder computes the following spatial relationships:

1. **Distance-based**: `near` (within distance threshold)
2. **Directional**:
   - `in_front` (angle: -45° to 45°)
   - `left` (angle: 45° to 135°)
   - `behind` (angle: 135° to -135°)
   - `right` (angle: -135° to -45°)

## Example Workflow

```python
import os
import json
from nuscenes_dataloader import NuScenesLidarSegmentationLoader
from scenegraph import SceneGraphBuilder

# 1. Initialize
loader = NuScenesLidarSegmentationLoader(
    dataroot='/data/nuscenes',
    version='v1.0-mini'
)

builder = SceneGraphBuilder(loader)

# 2. Process all scenes
for scene_token in loader.get_scene_tokens():
    # Build scene graphs
    nodes, rels = builder.build_scene_graphs(scene_token)
    
    # Export
    output_dir = f'outputs/{scene_token}'
    os.makedirs(output_dir, exist_ok=True)
    
    builder.export_to_json(
        nodes, rels,
        f'{output_dir}/scene_graph.json'
    )
    
    # Get objects for VLM
    vlm_objects = builder.get_objects_for_vlm_annotation(nodes)
    
    with open(f'{output_dir}/vlm_objects.json', 'w') as f:
        json.dump(vlm_objects, f, indent=2)
    
    # Save frame data
    frames = loader.get_scene_frames(scene_token)
    for frame in frames:
        loader.save_frame(frame, output_dir)
```

## Output Files

### Scene Graph JSON Format

```json
{
  "frames": [
    {
      "frame_idx": 0,
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
          "activity": null,
          "description": null
        }
      ],
      "relationships": [
        {
          "source_id": "instance_token_123",
          "target_id": "instance_token_456",
          "relationship_type": "near",
          "distance": 5.8,
          "frame_idx": 0
        }
      ]
    }
  ]
}
```

### VLM Objects JSON Format

```json
[
  {
    "object_id": "instance_token_123",
    "object_class": "vehicle.car",
    "frame_idx": 0,
    "timestamp": 1532402927647951,
    "position": [10.5, 5.2, 0.5],
    "size": [4.5, 1.8, 1.5],
    "velocity": [2.5, 0.3],
    "attributes": ["vehicle.moving"],
    "num_lidar_pts": 450,
    "activity": null,
    "description": null,
    "caption": null
  }
]
```

## Performance Tips

1. **Memory**: Process scenes one at a time for large datasets
2. **Speed**: Disable relationship extraction if not needed
3. **Storage**: Set `save_points=False` if only metadata is needed
4. **Filtering**: Use `filter_classes` and `min_visibility` to reduce object count

## Troubleshooting

### Import Error: nuscenes-devkit

```bash
pip install nuscenes-devkit
```

### Memory Issues

Process scenes sequentially and save to disk:

```python
for scene_token in loader.get_scene_tokens():
    frames = loader.get_scene_frames(scene_token)
    # Process and save
    for frame in frames:
        loader.save_frame(frame, output_dir)
    # Clear memory
    del frames
```

### Slow Loading

Use the mini version for testing:

```python
loader = NuScenesLidarSegmentationLoader(
    dataroot='/path/to/nuscenes',
    version='v1.0-mini'  # Only 10 scenes
)
```

## Next Steps: VLM Annotation

After extracting scene graphs, you can annotate objects with a Vision-Language Model:

1. Load `vlm_objects.json`
2. For each object, extract corresponding camera views
3. Generate prompts for VLM (e.g., "What activity is this car performing?")
4. Fill in `activity`, `description`, and `caption` fields
5. Save annotated results

See `waymo_caption.py` for an example of VLM-based captioning workflow.

## References

- [nuScenes Dataset](https://www.nuscenes.org/)
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
- [Dataset Paper](https://arxiv.org/abs/1903.11027)


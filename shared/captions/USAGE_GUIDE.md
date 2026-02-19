## Complete Usage Guide: VLM Annotation Pipeline

This guide walks you through the complete process of annotating nuScenes scene graphs with VLM.

## Table of Contents

1. [Setup](#setup)
2. [Quick Start](#quick-start)
3. [Single Scene Annotation](#single-scene-annotation)
4. [Batch Processing](#batch-processing)
5. [Visualization](#visualization)
6. [Advanced Usage](#advanced-usage)
7. [Output Format](#output-format)
8. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Prerequisites

```bash
# Install dependencies
pip install openai opencv-python numpy tqdm

# Verify nuScenes dataloader works
cd ..
python nuscenes_dataloader.py --help
```

### 2. Start vLLM Server

In a separate terminal:

```bash
# Example: Start LLaVA model on GPU 0
python -m vllm.entrypoints.openai.api_server \
    --model llava-hf/llava-1.5-7b-hf \
    --port 8001 \
    --gpu-memory-utilization 0.9
```

Or use existing vLLM scripts:
```bash
cd ../../vllm
bash vllm_serve0.sh  # or appropriate script
```

### 3. Test Setup

```bash
cd captions
python quick_test.py --dataroot /nas/standard_datasets/nuscenes
```

Expected output:
```
Quick Test: VLM Annotation Pipeline
============================================================
1. Initializing dataloader...
   ✓ Found 10 scenes
2. Loading first frame...
   ✓ Frame: abc123...
   ✓ Objects: 15
3. Objects with camera visibility:
   ✓ 12 objects visible in cameras
   1. car                   -> [CAM_FRONT, CAM_FRONT_LEFT]
   2. pedestrian           -> [CAM_FRONT]
   ...
5. Testing VLM connection...
   ✓ VLM server is running
   ✓ Available models: 1
     - llava-hf/llava-1.5-7b-hf

Ready for Annotation!
```

---

## Quick Start

### Minimal Example

```bash
# Annotate first scene, first 2 frames
python vlm_annotator.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --version v1.0-mini \
    --scene-idx 0 \
    --max-frames 2 \
    --output outputs/test
```

This will:
1. Load the first scene
2. Process 2 frames
3. Annotate visible objects with activity, description, caption
4. Save results to `outputs/test/`

---

## Single Scene Annotation

### Basic Usage

```bash
python vlm_annotator.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --version v1.0-trainval \
    --scene-idx 5 \
    --output outputs/scene_5
```

### With Options

```bash
python vlm_annotator.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --version v1.0-trainval \
    --vllm-api http://localhost:8001/v1 \
    --model llava-hf/llava-1.5-7b-hf \
    --scene-idx 5 \
    --max-frames 20 \
    --output outputs/scene_5_annotations
```

### Output Files

After running, you'll get:
- `{scene_token}_annotations.json` - Full annotations
- `{scene_token}_annotations_partial.json` - Intermediate saves (every 10 frames)

### Example Output

```
Annotating scene: scene-0005 (abc123def456...)
============================================================
Processing 20 frames...
Annotating frames: 100%|████| 20/20 [15:23<00:00, 46.2s/frame]

Annotation Statistics
============================================================
Scene: scene-0005
Frames processed: 20
Total annotations: 89
Avg annotations per frame: 4.4

Annotations by camera:
  CAM_FRONT                 :   52 (58.4%)
  CAM_FRONT_LEFT           :   22 (24.7%)
  CAM_FRONT_RIGHT          :   15 (16.9%)

Top 10 annotated object classes:
  vehicle.car                          :   34
  human.pedestrian.adult               :   18
  vehicle.truck                        :   12
  vehicle.bicycle                      :    8
  human.pedestrian.child               :    7
  vehicle.motorcycle                   :    5
  vehicle.bus                          :    3
  movable_object.trafficcone           :    2

Annotations saved to: outputs/scene_5/abc123def456_annotations.json
```

---

## Batch Processing

### Process Multiple Scenes

```bash
python batch_annotate.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --version v1.0-trainval \
    --max-scenes 10 \
    --max-frames 15 \
    --output outputs/batch_10scenes
```

### Process Specific Scenes

```bash
# Process scenes 0, 5, 10, 15
python batch_annotate.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --scene-indices 0 5 10 15 \
    --max-frames 20 \
    --output outputs/selected_scenes
```

### With Custom Camera Priority

```bash
python batch_annotate.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --max-scenes 5 \
    --cameras CAM_FRONT CAM_BACK CAM_FRONT_LEFT \
    --output outputs/custom_cameras
```

### Batch Output

```
Batch nuScenes Scene Annotation
============================================================
Dataset: /nas/standard_datasets/nuscenes
Version: v1.0-trainval
Output: outputs/batch_10scenes
VLM Server: http://localhost:8001/v1
Model: llava-hf/llava-1.5-7b-hf

Processing 10 scenes out of 850 total

============================================================
Scene 1/10: scene-0001
============================================================
...
✓ Scene completed in 920.3s
  Frames: 20
  Annotations: 87

============================================================
Scene 2/10: scene-0002
============================================================
...

Batch Processing Summary
============================================================
Total scenes processed: 10
  Successful: 10
  Failed: 0
Total frames: 200
Total annotations: 876
Total time: 154.2 minutes

Summary saved to: outputs/batch_10scenes/batch_summary.json
============================================================
```

---

## Visualization

### Visualize Single Frame

```bash
python visualize_annotations.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --annotations outputs/scene_5/abc123_annotations.json \
    --frame-idx 0
```

This will display an interactive window with annotations overlaid on camera images.

### Save Visualizations

```bash
python visualize_annotations.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --annotations outputs/scene_5/abc123_annotations.json \
    --frame-idx 0 \
    --output outputs/visualizations \
    --no-show
```

Saves images to `outputs/visualizations/`.

### Create Text Summary

```bash
python visualize_annotations.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --annotations outputs/scene_5/abc123_annotations.json \
    --summary
```

Creates `abc123_annotations_summary.md` with formatted annotations.

### Example Summary Output

```markdown
# Annotation Summary

Scene: scene-0005 (abc123def456)
Frames: 20
Total Annotations: 89

## Frame: frame_token_001

### 1. vehicle.car

- **Camera**: CAM_FRONT
- **Position**: (12.3, -2.1, 0.5) m
- **Speed**: 5.8 m/s
- **Activity**: driving straight in the middle lane
- **Description**: A silver sedan moving forward at moderate speed. The vehicle appears well-maintained and is positioned in the center of its lane. It's traveling in the same direction as the ego vehicle.
- **Caption**: silver car driving straight

### 2. human.pedestrian.adult

- **Camera**: CAM_FRONT
- **Position**: (8.5, 3.2, 0.0) m
- **Speed**: 1.2 m/s
- **Activity**: walking along the sidewalk
- **Description**: An adult pedestrian wearing dark clothing walking on the right sidewalk. The person appears to be walking casually and is not near the roadway.
- **Caption**: pedestrian walking sidewalk
```

---

## Advanced Usage

### Custom Annotation Function

```python
from vlm_annotator import VLMAnnotator
from nuscenes_dataloader import NuScenesLidarSegmentationLoader

# Initialize
loader = NuScenesLidarSegmentationLoader(
    dataroot='/nas/standard_datasets/nuscenes',
    version='v1.0-trainval'
)

annotator = VLMAnnotator(
    api_base='http://localhost:8001/v1',
    model='llava-hf/llava-1.5-7b-hf',
    dataloader=loader
)

# Get a frame
scene_token = loader.get_scene_tokens()[0]
frames = loader.get_scene_frames(scene_token)
frame = frames[0]

# Annotate specific objects
annotations = annotator.annotate_frame_objects(
    frame,
    camera_preference=['CAM_FRONT', 'CAM_FRONT_LEFT']
)

# Process annotations
for ann in annotations:
    print(f"{ann['name']}: {ann['activity']}")
```

### Filter and Process Annotations

```python
import json

# Load annotations
with open('outputs/scene_5/abc123_annotations.json', 'r') as f:
    data = json.load(f)

# Filter cars in front camera
front_cars = [
    ann for ann in data['annotations']
    if 'vehicle.car' in ann['name']
    and ann['annotation_camera'] == 'CAM_FRONT'
]

print(f"Found {len(front_cars)} cars in front camera")

# Group by activity
from collections import Counter
activities = Counter(ann['activity'] for ann in front_cars)
print("Activities:", activities.most_common(5))
```

### Merge with Scene Graph

```python
import json
from scenegraph import SceneGraphBuilder

# Load scene graph
builder = SceneGraphBuilder(loader)
nodes, relationships = builder.build_scene_graphs(scene_token)

# Load annotations
with open('annotations.json', 'r') as f:
    annotations = json.load(f)

# Create mapping
ann_map = {ann['instance_token']: ann for ann in annotations['annotations']}

# Update nodes
for frame_nodes in nodes:
    for node in frame_nodes:
        if node.object_id in ann_map:
            ann = ann_map[node.object_id]
            node.activity = ann['activity']
            node.description = ann['description']
            node.caption = ann['caption']

# Save updated scene graph
builder.export_to_json(nodes, relationships, 'scene_graph_annotated.json')
```

---

## Output Format

### Annotation JSON Structure

```json
{
  "scene_token": "abc123def456...",
  "scene_name": "scene-0005",
  "num_frames": 20,
  "num_annotations": 89,
  "frame_summaries": [
    {
      "frame_idx": 0,
      "sample_token": "frame_token_001",
      "timestamp": 1532402927647951,
      "num_objects": 15,
      "num_annotated": 8
    }
  ],
  "annotations": [
    {
      "token": "annotation_token_001",
      "instance_token": "instance_token_001",
      "name": "vehicle.car",
      "position": [12.3, -2.1, 0.5],
      "size": [4.5, 1.8, 1.5],
      "velocity": [5.8, 0.2],
      "num_lidar_pts": 450,
      "visibility": 4,
      "attributes": ["vehicle.moving"],
      "visible_cameras": ["CAM_FRONT", "CAM_FRONT_LEFT"],
      "annotation_camera": "CAM_FRONT",
      "activity": "driving straight in the middle lane",
      "description": "A silver sedan moving forward at moderate speed...",
      "caption": "silver car driving straight",
      "frame_token": "frame_token_001",
      "timestamp": 1532402927647951
    }
  ]
}
```

---

## Troubleshooting

### Issue: VLM Server Connection Failed

**Error:**
```
Error getting activity annotation: Connection refused
```

**Solution:**
```bash
# Check if server is running
curl http://localhost:8001/v1/models

# If not running, start it
python -m vllm.entrypoints.openai.api_server \
    --model llava-hf/llava-1.5-7b-hf \
    --port 8001
```

### Issue: Out of Memory

**Error:**
```
CUDA out of memory
```

**Solutions:**
1. Reduce batch size in vLLM server:
   ```bash
   --gpu-memory-utilization 0.7
   ```

2. Use smaller model:
   ```bash
   --model llava-hf/llava-1.5-7b-hf  # instead of 13b
   ```

3. Process fewer frames:
   ```bash
   --max-frames 5
   ```

### Issue: Slow Processing

**Problem:** Taking too long to annotate

**Solutions:**
1. Use GPU-accelerated vLLM (not CPU)
2. Reduce image resolution
3. Process fewer objects:
   ```python
   # Filter only important objects
   visible_objects = [
       obj for obj in frame.objects
       if obj.visible_cameras 
       and obj.visibility >= 3  # Only highly visible
       and 'vehicle' in obj.name  # Only vehicles
   ]
   ```

### Issue: No Objects Annotated

**Problem:** `num_annotated: 0` for all frames

**Possible Causes:**
1. Objects not visible in cameras
   ```bash
   # Check visibility
   python quick_test.py --dataroot /path/to/nuscenes
   ```

2. Camera images not found
   ```bash
   # Verify dataset structure
   ls /path/to/nuscenes/samples/CAM_FRONT/
   ```

### Issue: Invalid Annotations

**Problem:** VLM returns nonsensical responses

**Solutions:**
1. Check prompt format
2. Try different temperature:
   ```python
   temperature=0.1  # More deterministic
   ```

3. Use different model
4. Add more context in prompt

---

## Performance Benchmarks

### Single Scene (20 frames)

- **Time**: ~15-20 minutes
- **Annotations**: ~80-100 objects
- **GPU**: NVIDIA A100
- **Model**: LLaVA-1.5-7B

### Batch (10 scenes, 15 frames each)

- **Time**: ~2.5 hours
- **Annotations**: ~800 objects
- **Throughput**: ~5.3 annotations/minute

### Tips for Speed

1. **Use front cameras only**: 2x faster
2. **Sample frames**: Use `--max-frames 10` instead of all 40
3. **Filter objects**: Only annotate vehicles, ignore static objects
4. **Parallel processing**: Run multiple instances on different scenes

---

## Next Steps

1. **Integrate with Scene Graph**: Merge annotations into scene graph nodes
2. **Analysis**: Analyze activities, descriptions across dataset
3. **Visualization**: Create videos with annotations
4. **Fine-tuning**: Use annotations to fine-tune models

---

## Support

For issues:
1. Check this guide
2. Run `quick_test.py` for diagnostics
3. Check vLLM server logs
4. Verify dataset accessibility

Happy annotating! 🚗🤖


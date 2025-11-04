# VLM Annotator Improvements

## Summary of Changes

The VLM annotator has been significantly improved for better performance and efficiency:

### 1. **Best Camera Selection Based on Visibility** ✓
- Added `_select_best_camera()` method to intelligently select the best camera for each object
- Prioritizes cameras based on:
  1. User-specified camera preference list
  2. Object visibility in each camera
- Ensures each object is annotated using the camera with the best view

### 2. **Combined Activity + Description in One Prompt** ✓
- Merged `annotate_object_activity()` and `annotate_object_description()` into single `annotate_object_combined()` method
- **Benefits**:
  - Reduces API calls by 33% (2 calls → 1 call per object for activity + description)
  - Faster annotation time
  - Lower cost
  - More coherent responses (VLM sees context for both tasks together)
- Caption still generated separately as it's a different format

### 3. **Parallelized Requests for Maximum Efficiency** ✓
- Implemented `ThreadPoolExecutor` for parallel API calls
- **Key Features**:
  - Configurable `max_workers` parameter (default: 10 concurrent requests)
  - All objects in a frame are annotated in parallel
  - Camera images loaded once and reused for all objects in that camera
  - Progress tracking with `tqdm`
- **Performance Improvement**:
  - Sequential: ~3 API calls per object × N objects × latency
  - Parallel: ~2 API calls per object × latency (with 10 workers)
  - **Example**: 50 objects takes ~100 API calls, but with 10 workers running in parallel, this is ~10x faster!

## API Changes

### Updated Constructor

```python
VLMAnnotator(
    api_base: str,
    model: str,
    dataloader: NuScenesLidarSegmentationLoader,
    max_workers: int = 10  # NEW: Control parallelization
)
```

### New Method: `annotate_object_combined()`

```python
def annotate_object_combined(
    image: np.ndarray,
    obj: ObjectProperties,
    context_objects: List[ObjectProperties],
    camera: str
) -> Tuple[str, str]:
    """
    Generate both activity and description in one API call.
    
    Returns:
        Tuple of (activity, description)
    """
```

### New Method: `_select_best_camera()`

```python
def _select_best_camera(
    obj: ObjectProperties,
    frame: FrameData,
    camera_preference: List[str] = None
) -> Optional[str]:
    """
    Select the best camera to view an object based on visibility.
    """
```

### New Method: `_annotate_single_object()`

```python
def _annotate_single_object(
    obj: ObjectProperties,
    image: np.ndarray,
    camera: str,
    context_objects: List[ObjectProperties],
    frame: FrameData
) -> Optional[Dict[str, Any]]:
    """
    Annotate a single object (designed for parallel execution).
    """
```

### New Method: `cleanup()`

```python
def cleanup():
    """Clean up thread pool executor (call when done)."""
```

## Usage Examples

### Basic Usage (with defaults)

```bash
python vlm_annotator.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --version v1.0-trainval \
    --vllm-api http://localhost:8001/v1 \
    --model llava-hf/llava-1.5-7b-hf \
    --output outputs/annotations \
    --scene-idx 0
```

### Advanced Usage (custom parallelization and camera preference)

```bash
python vlm_annotator.py \
    --dataroot /nas/standard_datasets/nuscenes \
    --version v1.0-trainval \
    --vllm-api http://localhost:8001/v1 \
    --model llava-hf/llava-1.5-7b-hf \
    --output outputs/annotations \
    --scene-idx 0 \
    --max-workers 20 \
    --camera-preference CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
    --max-frames 10
```

### Programmatic Usage

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
    dataloader=loader,
    max_workers=15  # Customize parallelization
)

try:
    # Annotate scene with custom camera preference
    results = annotator.annotate_scene(
        scene_token='your_scene_token',
        output_dir='outputs/annotations',
        max_frames=None,  # Process all frames
        camera_preference=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    )
    
    print(f"Annotated {results['num_annotations']} objects across {results['num_frames']} frames")
    
finally:
    # Always cleanup
    annotator.cleanup()
```

## Performance Comparison

### Before (Sequential)
- 3 API calls per object (activity + description + caption)
- Sequential processing
- **Example**: 50 objects × 3 calls × 2s latency = **300 seconds**

### After (Parallelized with Combined Prompts)
- 2 API calls per object (combined activity+description + caption)
- Parallel processing with 10 workers
- **Example**: 50 objects × 2 calls × 2s latency / 10 workers = **20 seconds**
- **~15x faster!** 🚀

## Camera Selection Strategy

The best camera for each object is selected based on:

1. **Visibility**: Only cameras where the object is visible are considered
2. **Preference Order**: Cameras are selected in the order specified by `camera_preference`
3. **Default Order**: `['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']`

### Why This Matters

- Front cameras typically have better lighting and clearer views
- Reduces annotation errors from poor viewing angles
- More consistent annotations across objects

## Combined Prompt Format

The combined prompt asks for both activity and description in a structured format:

```
ACTIVITY: [brief activity description]
DESCRIPTION: [detailed description]
```

This ensures:
- Consistent parsing
- Better context for the VLM
- Coherent responses

## Thread Safety

- Uses `ThreadPoolExecutor` for thread-safe parallel execution
- Each thread gets its own HTTP connection to the vLLM server
- Progress tracking is thread-safe with `tqdm`
- All file I/O uses separate threads

## Best Practices

1. **Tune `max_workers`** based on your vLLM server capacity:
   - Default: 10 (good for most setups)
   - Higher (20-50): If you have a powerful vLLM server
   - Lower (5): If you have limited resources

2. **Camera Preference**: Order cameras by importance
   ```python
   camera_preference=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
   ```

3. **Always call `cleanup()`**: Use try/finally block to ensure thread pool shuts down properly

4. **Monitor vLLM server**: Watch server logs for any throttling or errors

## Backward Compatibility

The old methods are removed:
- ~~`annotate_object_activity()`~~ → Use `annotate_object_combined()`
- ~~`annotate_object_description()`~~ → Use `annotate_object_combined()`

The `annotate_frame_objects()` method signature remains the same, so existing code that calls this method will continue to work, just faster!

## Known Limitations

1. Caption is still separate (could be combined in future)
2. Requires proper visibility calculation in dataloader (already implemented)
3. Thread pool size limited by system resources

## Future Improvements

- [ ] Combine caption into the single prompt (3 calls → 1 call)
- [ ] Add retry logic for failed API calls
- [ ] Add caching for repeated annotations
- [ ] Support batch API calls if vLLM supports it
- [ ] Add progress estimation with ETA


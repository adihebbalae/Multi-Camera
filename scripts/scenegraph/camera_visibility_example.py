"""
Example: Get camera visibility for objects in nuScenes

This script demonstrates how to determine which cameras each object is visible from.
"""

from nuscenes_dataloader import NuScenesLidarSegmentationLoader
from nuscenes.utils.geometry_utils import BoxVisibility


def example_camera_visibility():
    """Example of getting camera visibility for objects."""
    
    # Initialize dataloader
    dataroot = '/nas/standard_datasets/nuscenes'
    version = 'v1.0-trainval'  # or 'v1.0-mini'
    
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=True
    )
    
    # Get first scene
    scene_tokens = loader.get_scene_tokens()
    scene_token = scene_tokens[0]
    
    print(f"\nProcessing scene: {scene_token}")
    
    # Get first frame
    sample_tokens = loader.get_samples_in_scene(scene_token)
    sample_token = sample_tokens[0]
    
    print(f"Sample token: {sample_token}\n")
    
    # Method 1: Get frame data (camera visibility is automatically included)
    print("=" * 60)
    print("Method 1: Using get_frame_data()")
    print("=" * 60)
    
    frame = loader.get_frame_data(sample_token)
    
    print(f"\nFound {len(frame.objects)} objects:")
    for idx, obj in enumerate(frame.objects, 1):
        print(f"\n{idx}. {obj.name}")
        print(f"   Token: {obj.token}")
        print(f"   Position: {obj.position}")
        print(f"   Visibility level: {obj.visibility}")
        print(f"   Visible in cameras: {obj.visible_cameras}")
        
        if obj.visible_cameras:
            print(f"   -> Visible in {len(obj.visible_cameras)} camera(s)")
        else:
            print(f"   -> Not visible in any camera")
    
    # Method 2: Get visibility for a specific object
    print("\n" + "=" * 60)
    print("Method 2: Get visibility for specific object")
    print("=" * 60)
    
    if frame.objects:
        obj = frame.objects[0]
        print(f"\nChecking object: {obj.name} (token: {obj.token})")
        
        # Get with different visibility thresholds
        for vis_level in [BoxVisibility.ANY, BoxVisibility.PARTIAL, BoxVisibility.MOST]:
            cameras = loader.get_object_camera_visibility(
                sample_token,
                obj.token,
                min_visibility=vis_level
            )
            print(f"  {vis_level.name}: {cameras}")
    
    # Method 3: Get visibility for all objects at once
    print("\n" + "=" * 60)
    print("Method 3: Get all objects visibility at once")
    print("=" * 60)
    
    visibility_map = loader.get_all_objects_camera_visibility(
        sample_token,
        min_visibility=BoxVisibility.ANY
    )
    
    print(f"\nVisibility map for {len(visibility_map)} objects:")
    for ann_token, cameras in list(visibility_map.items())[:5]:  # Show first 5
        print(f"  {ann_token[:8]}...: {cameras}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("Camera Visibility Statistics")
    print("=" * 60)
    
    camera_counts = {}
    for obj in frame.objects:
        if obj.visible_cameras:
            for camera in obj.visible_cameras:
                camera_counts[camera] = camera_counts.get(camera, 0) + 1
    
    print(f"\nObjects per camera:")
    for camera, count in sorted(camera_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {camera:20s}: {count:3d} objects")
    
    # Objects visible in multiple cameras
    multi_camera_objects = [obj for obj in frame.objects if obj.visible_cameras and len(obj.visible_cameras) > 1]
    print(f"\nObjects visible in multiple cameras: {len(multi_camera_objects)}")
    
    if multi_camera_objects:
        print("\nExamples:")
        for obj in multi_camera_objects[:3]:
            print(f"  - {obj.name}: {obj.visible_cameras}")
    
    # Objects not visible in any camera
    no_camera_objects = [obj for obj in frame.objects if not obj.visible_cameras]
    print(f"\nObjects NOT visible in any camera: {len(no_camera_objects)}")


def filter_by_camera_visibility():
    """Example: Filter objects by camera visibility."""
    
    dataroot = '/nas/standard_datasets/nuscenes'
    version = 'v1.0-trainval'
    
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=False
    )
    
    # Get a frame
    scene_token = loader.get_scene_tokens()[0]
    sample_token = loader.get_samples_in_scene(scene_token)[0]
    frame = loader.get_frame_data(sample_token)
    
    print("\n" + "=" * 60)
    print("Filtering Objects by Camera Visibility")
    print("=" * 60)
    
    # Filter 1: Objects visible in front cameras
    front_cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    front_visible = [
        obj for obj in frame.objects
        if obj.visible_cameras and any(cam in front_cameras for cam in obj.visible_cameras)
    ]
    print(f"\n1. Objects visible in front cameras: {len(front_visible)}")
    
    # Filter 2: Objects visible in CAM_FRONT only
    cam_front_only = [
        obj for obj in frame.objects
        if obj.visible_cameras and 'CAM_FRONT' in obj.visible_cameras
    ]
    print(f"2. Objects visible in CAM_FRONT: {len(cam_front_only)}")
    
    # Filter 3: Objects visible in at least 2 cameras
    multi_camera = [
        obj for obj in frame.objects
        if obj.visible_cameras and len(obj.visible_cameras) >= 2
    ]
    print(f"3. Objects visible in 2+ cameras: {len(multi_camera)}")
    
    # Filter 4: By object class and camera
    cars_in_front = [
        obj for obj in frame.objects
        if 'vehicle.car' in obj.name
        and obj.visible_cameras
        and 'CAM_FRONT' in obj.visible_cameras
    ]
    print(f"4. Cars visible in CAM_FRONT: {len(cars_in_front)}")
    
    # Show examples
    if cars_in_front:
        print("\n   Example car in CAM_FRONT:")
        car = cars_in_front[0]
        print(f"   - Name: {car.name}")
        print(f"   - Position: {car.position}")
        print(f"   - Visible in: {car.visible_cameras}")
        print(f"   - LiDAR points: {car.num_lidar_pts}")


if __name__ == "__main__":
    print("=" * 60)
    print("nuScenes Camera Visibility Examples")
    print("=" * 60)
    
    # Run examples
    example_camera_visibility()
    filter_by_camera_visibility()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


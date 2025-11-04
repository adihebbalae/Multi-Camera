"""
Example script to visualize objects on camera images.

Demonstrates the new visualize_objects_on_camera function.
"""

import argparse
from pathlib import Path

from nuscenes_dataloader import NuScenesLidarSegmentationLoader
from visualize import visualize_objects_on_camera


def main():
    parser = argparse.ArgumentParser(
        description='Visualize objects with categories on camera images'
    )
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                       help='Dataset version')
    parser.add_argument('--scene-idx', type=int, default=0,
                       help='Scene index to visualize')
    parser.add_argument('--frame-idx', type=int, default=0,
                       help='Frame index within the scene')
    parser.add_argument('--cameras', type=str, nargs='+',
                       default=['CAM_FRONT'],
                       help='Cameras to visualize (default: CAM_FRONT)')
    parser.add_argument('--output', type=str,
                       help='Output directory to save images (if not set, displays interactively)')
    parser.add_argument('--no-bbox', action='store_true',
                       help='Don\'t show bounding boxes')
    parser.add_argument('--no-labels', action='store_true',
                       help='Don\'t show object labels')
    parser.add_argument('--no-properties', action='store_true',
                       help='Don\'t show object properties (velocity, distance)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("nuScenes Camera Object Visualization")
    print("="*60)
    
    # Initialize dataloader
    print(f"\nLoading dataset from: {args.dataroot}")
    print(f"Version: {args.version}")
    
    loader = NuScenesLidarSegmentationLoader(
        dataroot=args.dataroot,
        version=args.version,
        verbose=False
    )
    
    # Get scene
    scene_tokens = loader.get_scene_tokens()
    if args.scene_idx >= len(scene_tokens):
        print(f"Error: Scene index {args.scene_idx} out of range (0-{len(scene_tokens)-1})")
        return
    
    scene_token = scene_tokens[args.scene_idx]
    scene = loader.nusc.get('scene', scene_token)
    print(f"\nScene {args.scene_idx}: {scene['name']} ({scene_token})")
    
    # Get frame
    sample_tokens = loader.get_samples_in_scene(scene_token)
    if args.frame_idx >= len(sample_tokens):
        print(f"Error: Frame index {args.frame_idx} out of range (0-{len(sample_tokens)-1})")
        return
    
    sample_token = sample_tokens[args.frame_idx]
    print(f"Frame {args.frame_idx}/{len(sample_tokens)}: {sample_token}")
    
    # Load frame data
    print("\nLoading frame data...")
    frame = loader.get_frame_data(sample_token)
    
    print(f"  Objects: {len(frame.objects)}")
    print(f"  Cameras: {list(frame.camera_tokens.keys())}")
    
    # Count objects visible in requested cameras
    visible_count = 0
    for obj in frame.objects:
        if obj.visible_cameras:
            for cam in args.cameras:
                if cam in obj.visible_cameras:
                    visible_count += 1
                    break
    
    print(f"  Objects visible in {args.cameras}: {visible_count}")
    
    # Visualize
    print(f"\nVisualizing cameras: {args.cameras}")
    if args.output:
        print(f"Saving to: {args.output}")
    else:
        print("Displaying interactively (press any key to close)")
    
    visualize_objects_on_camera(
        frame=frame,
        dataloader=loader,
        output_path=args.output,
        cameras=args.cameras,
        show_bbox=not args.no_bbox,
        show_labels=not args.no_labels,
        show_properties=not args.no_properties
    )
    
    print("\n✓ Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()


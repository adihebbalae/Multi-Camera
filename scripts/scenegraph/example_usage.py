"""
Example usage script for nuScenes Lidar Segmentation Dataloader

This script demonstrates how to:
1. Load the nuScenes dataset
2. Extract lidar segmentation maps
3. Build scene graphs
4. Prepare data for VLM annotation
"""

import os
import json
import argparse
from pathlib import Path

from nuscenes_dataloader import NuScenesLidarSegmentationLoader
from scenegraph import SceneGraphBuilder


def example_basic_loading(dataroot: str, version: str):
    """Example 1: Basic data loading and exploration."""
    print("\n" + "="*60)
    print("Example 1: Basic Loading and Exploration")
    print("="*60)
    
    # Initialize loader
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=True
    )
    
    # Get scene information
    scene_tokens = loader.get_scene_tokens()
    print(f"\nTotal scenes in dataset: {len(scene_tokens)}")
    
    # Get statistics for first scene
    if scene_tokens:
        scene_token = scene_tokens[0]
        print(f"\nAnalyzing scene: {scene_token}")
        
        # Get object statistics
        stats = loader.get_object_statistics(scene_token)
        print("\nObject class distribution:")
        for obj_class, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {obj_class:40s}: {count:3d}")
        
        # Load first frame
        sample_tokens = loader.get_samples_in_scene(scene_token)
        if sample_tokens:
            frame = loader.get_frame_data(sample_tokens[0])
            print(f"\nFirst frame details:")
            print(f"  Sample token: {frame.sample_token}")
            print(f"  Timestamp: {frame.timestamp}")
            print(f"  Total points: {frame.lidar_points.shape[0]}")
            print(f"  Segmented points: {(frame.segmentation_labels > 0).sum()}")
            print(f"  Number of objects: {len(frame.objects)}")
            
            print("\n  Objects in frame:")
            for obj in frame.objects[:5]:  # Show first 5
                print(f"    - {obj.name:30s}: {obj.num_lidar_pts:4d} pts, "
                      f"pos={obj.position}, vel={obj.velocity}")


def example_scene_graph_building(dataroot: str, version: str, output_dir: str):
    """Example 2: Build scene graphs with relationships."""
    print("\n" + "="*60)
    print("Example 2: Scene Graph Building")
    print("="*60)
    
    # Initialize loader
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=False
    )
    
    # Initialize scene graph builder
    builder = SceneGraphBuilder(
        dataloader=loader,
        extract_relationships=True,
        distance_threshold=10.0  # Consider objects within 10m as "near"
    )
    
    # Process first scene
    scene_tokens = loader.get_scene_tokens()
    if not scene_tokens:
        print("No scenes found!")
        return
    
    scene_token = scene_tokens[0]
    print(f"\nBuilding scene graphs for: {scene_token}")
    
    # Build scene graphs
    all_nodes, all_relationships = builder.build_scene_graphs(scene_token)
    
    # Print statistics
    total_objects = sum(len(nodes) for nodes in all_nodes)
    total_relationships = sum(len(rels) for rels in all_relationships)
    
    print(f"\nScene graph statistics:")
    print(f"  Frames: {len(all_nodes)}")
    print(f"  Total objects: {total_objects}")
    print(f"  Total relationships: {total_relationships}")
    print(f"  Avg objects per frame: {total_objects / len(all_nodes):.1f}")
    print(f"  Avg relationships per frame: {total_relationships / len(all_nodes):.1f}")
    
    # Show relationship types
    rel_types = {}
    for frame_rels in all_relationships:
        for rel in frame_rels:
            rel_types[rel.relationship_type] = rel_types.get(rel.relationship_type, 0) + 1
    
    print("\n  Relationship type distribution:")
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {rel_type:15s}: {count:4d}")
    
    # Export to JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{scene_token}_scene_graph.json')
    builder.export_to_json(all_nodes, all_relationships, output_path)
    print(f"\nScene graph saved to: {output_path}")
    
    return all_nodes, all_relationships


def example_vlm_preparation(dataroot: str, version: str, output_dir: str):
    """Example 3: Prepare objects for VLM annotation."""
    print("\n" + "="*60)
    print("Example 3: VLM Annotation Preparation")
    print("="*60)
    
    # Initialize loader and builder
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=False
    )
    
    builder = SceneGraphBuilder(
        dataloader=loader,
        extract_relationships=False  # Don't need relationships for VLM
    )
    
    # Build scene graphs
    scene_tokens = loader.get_scene_tokens()
    if not scene_tokens:
        print("No scenes found!")
        return
    
    scene_token = scene_tokens[0]
    print(f"\nPreparing objects from scene: {scene_token}")
    
    all_nodes, _ = builder.build_scene_graphs(scene_token)
    
    # Filter for dynamic objects that are interesting for VLM
    dynamic_classes = [
        'vehicle.car',
        'vehicle.truck',
        'vehicle.bus',
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'vehicle.bicycle',
        'vehicle.motorcycle',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer'
    ]
    
    # Get objects for VLM annotation
    vlm_objects = builder.get_objects_for_vlm_annotation(
        all_nodes,
        filter_classes=dynamic_classes,
        min_visibility=2  # Only reasonably visible objects
    )
    
    print(f"\nObjects prepared for VLM annotation: {len(vlm_objects)}")
    
    # Count by class
    class_counts = {}
    for obj in vlm_objects:
        class_counts[obj['object_class']] = class_counts.get(obj['object_class'], 0) + 1
    
    print("\n  Object class distribution:")
    for obj_class, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {obj_class:35s}: {count:3d}")
    
    # Save for VLM processing
    os.makedirs(output_dir, exist_ok=True)
    vlm_output_path = os.path.join(output_dir, f'{scene_token}_vlm_objects.json')
    with open(vlm_output_path, 'w') as f:
        json.dump(vlm_objects, f, indent=2)
    
    print(f"\nVLM objects saved to: {vlm_output_path}")
    
    # Show example object
    if vlm_objects:
        print("\n  Example object for VLM annotation:")
        example = vlm_objects[0]
        print(f"    Object ID: {example['object_id']}")
        print(f"    Class: {example['object_class']}")
        print(f"    Frame: {example['frame_idx']}")
        print(f"    Position: {example['position']}")
        print(f"    Velocity: {example['velocity']}")
        print(f"    Attributes: {example['attributes']}")
        print(f"    Fields for VLM to fill:")
        print(f"      - activity: {example['activity']}")
        print(f"      - description: {example['description']}")
        print(f"      - caption: {example['caption']}")


def example_save_frames(dataroot: str, version: str, output_dir: str, max_frames: int = 5):
    """Example 4: Save frame data to disk."""
    print("\n" + "="*60)
    print("Example 4: Save Frame Data")
    print("="*60)
    
    # Initialize loader
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=False
    )
    
    # Get first scene
    scene_tokens = loader.get_scene_tokens()
    if not scene_tokens:
        print("No scenes found!")
        return
    
    scene_token = scene_tokens[0]
    print(f"\nSaving frames from scene: {scene_token}")
    
    # Load frames
    frames = loader.get_scene_frames(scene_token)
    
    # Save first few frames
    frames_to_save = frames[:max_frames]
    
    frame_output_dir = os.path.join(output_dir, 'frames', scene_token)
    os.makedirs(frame_output_dir, exist_ok=True)
    
    print(f"\nSaving {len(frames_to_save)} frames to: {frame_output_dir}")
    
    for idx, frame in enumerate(frames_to_save):
        loader.save_frame(
            frame,
            output_dir=frame_output_dir,
            save_points=True,      # Save point clouds
            save_metadata=True     # Save metadata JSON
        )
        
        print(f"  Frame {idx+1}/{len(frames_to_save)}: {frame.sample_token}")
        print(f"    Points: {frame.lidar_points.shape[0]}")
        print(f"    Objects: {len(frame.objects)}")
    
    print(f"\nFiles saved:")
    print(f"  - *_points.npy: Point clouds (x, y, z, intensity, ring)")
    print(f"  - *_segmentation.npy: Instance segmentation labels")
    print(f"  - *_metadata.json: Frame metadata and object properties")


def main():
    parser = argparse.ArgumentParser(
        description='nuScenes Lidar Segmentation Dataloader Examples'
    )
    parser.add_argument(
        '--dataroot',
        type=str,
        required=True,
        help='Path to nuScenes dataset root directory'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
        help='Dataset version to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/examples',
        help='Output directory for results'
    )
    parser.add_argument(
        '--example',
        type=str,
        choices=['all', 'basic', 'scenegraph', 'vlm', 'save'],
        default='all',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    # Verify dataset exists
    if not os.path.exists(args.dataroot):
        print(f"Error: Dataset not found at {args.dataroot}")
        print("Please download nuScenes dataset from https://www.nuscenes.org/")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("nuScenes Lidar Segmentation Dataloader Examples")
    print("="*60)
    print(f"\nDataset: {args.dataroot}")
    print(f"Version: {args.version}")
    print(f"Output: {args.output}")
    
    # Run examples
    if args.example in ['all', 'basic']:
        example_basic_loading(args.dataroot, args.version)
    
    if args.example in ['all', 'scenegraph']:
        example_scene_graph_building(args.dataroot, args.version, args.output)
    
    if args.example in ['all', 'vlm']:
        example_vlm_preparation(args.dataroot, args.version, args.output)
    
    if args.example in ['all', 'save']:
        example_save_frames(args.dataroot, args.version, args.output, max_frames=3)
    
    print("\n" + "="*60)
    print("Examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()


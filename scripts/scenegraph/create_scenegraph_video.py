"""
Create videos from scene graph JSON with objects overlaid on camera images.

This script reads a scene graph JSON file and creates MP4 videos showing
all detected objects with their bounding boxes and labels on camera images.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MP4 video from scene graph JSON with objects overlaid on camera images."
    )
    parser.add_argument(
        "--dataroot", 
        required=True, 
        help="Path to nuScenes dataroot (e.g., /data/sets/nuscenes)"
    )
    parser.add_argument(
        "--version", 
        default="v1.0-trainval",
        help="nuScenes version (e.g., v1.0-trainval, v1.0-mini)"
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to scene graph JSON file"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for videos"
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["CAM_FRONT"],
        choices=[
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
        help="Cameras to create videos for"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Output video FPS (default 2)"
    )
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        default=True,
        help="Show 3D bounding boxes"
    )
    parser.add_argument(
        "--show-labels",
        action="store_true", 
        default=True,
        help="Show object labels"
    )
    parser.add_argument(
        "--show-properties",
        action="store_true",
        default=True,
        help="Show object properties (velocity, distance)"
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="Font scale for text"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Line thickness"
    )
    
    return parser.parse_args()


def load_scene_graph(input_json: str) -> Dict:
    """Load scene graph JSON file."""
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_object_color(obj_class: str) -> Tuple[int, int, int]:
    """Get BGR color for object class."""
    if 'vehicle' in obj_class:
        return (0, 0, 255)  # Red
    elif 'pedestrian' in obj_class or 'human' in obj_class:
        return (255, 0, 0)  # Blue
    elif 'bicycle' in obj_class or 'motorcycle' in obj_class:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 255, 0)  # Green


def draw_3d_box(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2
):
    """Draw 3D bounding box on image."""
    h, w = image.shape[:2]
    
    # Define box edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        pt1 = (int(corners_2d[0, edge[0]]), int(corners_2d[1, edge[0]]))
        pt2 = (int(corners_2d[0, edge[1]]), int(corners_2d[1, edge[1]]))
        
        # Only draw if points are within reasonable bounds
        if 0 <= pt1[0] < w*2 and 0 <= pt1[1] < h*2 and \
           0 <= pt2[0] < w*2 and 0 <= pt2[1] < h*2:
            cv2.line(image, pt1, pt2, color, thickness)


def draw_object_label(
    image: np.ndarray,
    corners_2d: np.ndarray,
    obj: Dict,
    color: Tuple[int, int, int],
    show_properties: bool,
    font_scale: float,
    thickness: int
):
    """Draw object label with properties."""
    h, w = image.shape[:2]
    
    # Get label position (center-bottom of box)
    center_2d = np.mean(corners_2d, axis=1)
    label_x = int(center_2d[0])
    label_y = int(np.max(corners_2d[1, :]))
    
    # Ensure label is within image bounds
    label_x = max(10, min(label_x, w - 10))
    label_y = max(30, min(label_y, h - 10))
    
    # Prepare label text
    class_name = obj['object_class'].split('.')[-1].replace('_', ' ')
    
    lines = [class_name]
    
    if show_properties:
        # Add velocity if available
        if obj.get('velocity') and obj['velocity'] != [0.0, 0.0]:
            speed = np.sqrt(obj['velocity'][0]**2 + obj['velocity'][1]**2)
            if speed > 0.5:
                lines.append(f"{speed:.1f} m/s")
        
        # Add distance
        if obj.get('position'):
            distance = np.sqrt(obj['position'][0]**2 + obj['position'][1]**2)
            lines.append(f"{distance:.1f}m")
    
    if not lines:
        return
    
    # Calculate text size
    max_width = 0
    total_height = 0
    line_heights = []
    
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        max_width = max(max_width, text_w)
        line_heights.append(text_h)
        total_height += text_h + 5
    
    # Draw semi-transparent background
    bg_x1 = max(0, label_x - 5)
    bg_y1 = max(0, label_y - total_height - 5)
    bg_x2 = min(w, label_x + max_width + 5)
    bg_y2 = min(h, label_y + 5)
    
    # Draw background
    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    
    # Draw border
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
    
    # Draw text lines
    y_offset = label_y - total_height
    for i, line in enumerate(lines):
        y_pos = y_offset + sum(line_heights[:i+1]) + (i * 5)
        cv2.putText(
            image, line, (label_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )


def draw_frame_info(
    image: np.ndarray,
    camera: str,
    frame_idx: int,
    total_frames: int,
    num_objects: int,
    font_scale: float,
    thickness: int
):
    """Draw frame information on image."""
    h, w = image.shape[:2]
    
    # Camera name at top
    cv2.putText(
        image, camera, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness
    )
    
    # Frame info at bottom
    info_text = f"Frame: {frame_idx+1}/{total_frames} | Objects: {num_objects}"
    cv2.putText(
        image, info_text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
    )


def create_video_for_camera(
    nusc: NuScenes,
    scene_graph: Dict,
    camera: str,
    output_path: str,
    args: argparse.Namespace
):
    """Create video for a specific camera."""
    print(f"\nCreating video for {camera}...")
    
    frames = scene_graph['frames']
    if not frames:
        print(f"  No frames found in scene graph")
        return
    
    print(f"  Total frames in scene graph: {len(frames)}")
    
    # Debug: Check first frame structure
    if frames:
        first_frame = frames[0]
        print(f"  First frame keys: {first_frame.keys()}")
        if 'objects' in first_frame and first_frame['objects']:
            print(f"  First object keys: {first_frame['objects'][0].keys()}")
            print(f"  Sample first object: {first_frame['objects'][0]}")
    
    # Initialize video writer
    writer = None
    
    try:
        for frame_idx, frame in enumerate(tqdm(frames, desc=f"  {camera}")):
            sample_token = frame['sample_token']
            
            try:
                sample = nusc.get('sample', sample_token)
            except:
                print(f"  Warning: Could not load sample {sample_token}")
                continue
            
            # Check if camera exists in this sample
            if camera not in sample['data']:
                continue
            
            camera_token = sample['data'][camera]
            cam_data = nusc.get('sample_data', camera_token)
            img_path = Path(args.dataroot) / cam_data['filename']
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  Warning: Could not load image {img_path}")
                continue
            
            # Initialize writer with first valid frame
            if writer is None:
                h, w = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, args.fps, (w, h))
            
            # Get ego pose for this frame (global → ego transformation)
            ego_pose_token = cam_data['ego_pose_token']
            ego_pose = nusc.get('ego_pose', ego_pose_token)
            ego_translation = np.array(ego_pose['translation'])
            ego_rotation = Quaternion(ego_pose['rotation'])
            
            # Get camera intrinsics and extrinsics (ego → camera transformation)
            cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            cam_translation = np.array(cs_record['translation'])
            cam_rotation = Quaternion(cs_record['rotation'])
            
            # Filter objects visible in this camera from scene graph
            visible_objects = []
            for obj in frame['objects']:
                if obj.get('visible_cameras') and camera in obj['visible_cameras']:
                    visible_objects.append(obj)
            
            # Debug output for first frame
            if frame_idx == 0:
                print(f"\n  Frame 0: Total objects = {len(frame['objects'])}, Visible in {camera} = {len(visible_objects)}")
            
            # Draw each object using data directly from scene graph
            for obj in visible_objects:
                try:
                    # Get position, size, and rotation from scene graph
                    position = np.array(obj['position'])  # (x, y, z) in global frame
                    size = np.array(obj['size'])  # (width, length, height)
                    rotation_quat = Quaternion(obj['rotation'])  # (w, x, y, z)
                    
                    # Create 8 corners of 3D bounding box
                    # Corners defined as: front-left, front-right, back-right, back-left (bottom), then top
                    w, l, h = size
                    
                    # Define corners in object's local frame (centered at origin)
                    corners = np.array([
                        [-w/2, -l/2, -h/2],  # 0: front-left-bottom
                        [ w/2, -l/2, -h/2],  # 1: front-right-bottom
                        [ w/2,  l/2, -h/2],  # 2: back-right-bottom
                        [-w/2,  l/2, -h/2],  # 3: back-left-bottom
                        [-w/2, -l/2,  h/2],  # 4: front-left-top
                        [ w/2, -l/2,  h/2],  # 5: front-right-top
                        [ w/2,  l/2,  h/2],  # 6: back-right-top
                        [-w/2,  l/2,  h/2],  # 7: back-left-top
                    ]).T  # Shape: (3, 8)
                    
                    # Rotate corners to object's orientation in global frame
                    corners_rotated = rotation_quat.rotation_matrix @ corners
                    
                    # Translate corners to object's global position
                    corners_global = corners_rotated + position.reshape(3, 1)
                    
                    # Transform from global to ego vehicle frame
                    corners_ego = corners_global - ego_translation.reshape(3, 1)
                    corners_ego = ego_rotation.inverse.rotation_matrix @ corners_ego
                    
                    # Transform from ego vehicle frame to camera frame
                    corners_cam = corners_ego - cam_translation.reshape(3, 1)
                    corners_cam = cam_rotation.inverse.rotation_matrix @ corners_cam
                    
                    # Check if box is in front of camera
                    if not np.any(corners_cam[2, :] > 0):
                        continue
                    
                    # Project 3D corners to 2D image plane
                    corners_2d = view_points(corners_cam, cam_intrinsic, normalize=True)[:2, :]
                    
                    # Get color for this object class
                    color = get_object_color(obj['object_class'])
                    
                    # Draw bounding box
                    if args.show_bbox:
                        draw_3d_box(image, corners_2d, color, args.thickness)
                    
                    # Draw label
                    if args.show_labels:
                        draw_object_label(
                            image, corners_2d, obj, color,
                            args.show_properties, args.font_scale, args.thickness
                        )
                    
                except Exception as e:
                    # Skip objects that can't be drawn
                    continue
            
            # Draw frame info
            draw_frame_info(
                image, camera, frame_idx, len(frames),
                len(visible_objects), args.font_scale, args.thickness
            )
            
            # Write frame
            writer.write(image)
    
    finally:
        if writer is not None:
            writer.release()
            print(f"  Saved: {output_path}")
        else:
            print(f"  No frames written for {camera}")


def main():
    args = parse_args()
    
    print("="*60)
    print("Scene Graph Video Creator")
    print("="*60)
    
    # Load nuScenes
    print(f"\nLoading nuScenes from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    # Load scene graph
    print(f"Loading scene graph from {args.input_json}...")
    scene_graph = load_scene_graph(args.input_json)
    
    print(f"Scene: {scene_graph.get('scene_name', 'unknown')}")
    print(f"Frames: {scene_graph.get('num_frames', 0)}")
    print(f"Annotations: {scene_graph.get('num_annotations', 0)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get scene name for output filename
    scene_name = scene_graph.get('scene_name', scene_graph.get('scene_token', 'scene'))
    
    # Create video for each camera
    for camera in args.cameras:
        output_path = output_dir / f"{scene_name}_{camera}.mp4"
        
        try:
            create_video_for_camera(
                nusc=nusc,
                scene_graph=scene_graph,
                camera=camera,
                output_path=str(output_path),
                args=args
            )
        except Exception as e:
            print(f"\nError creating video for {camera}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ Video creation complete!")
    print(f"Videos saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


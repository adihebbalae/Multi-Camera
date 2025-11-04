"""
Visualize VLM annotations on camera images.

Displays camera images with bounding boxes and annotations.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from nuscenes_dataloader import NuScenesLidarSegmentationLoader
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


class AnnotationVisualizer:
    """Visualize VLM annotations on images."""
    
    def __init__(self, dataloader: NuScenesLidarSegmentationLoader):
        """Initialize visualizer."""
        self.dataloader = dataloader
        self.nusc = dataloader.nusc
    
    def load_annotations(self, annotations_file: str) -> Dict[str, Any]:
        """Load annotations from JSON file."""
        with open(annotations_file, 'r') as f:
            return json.load(f)
    
    def visualize_frame(
        self,
        annotations: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize annotations for a frame.
        
        Args:
            annotations: List of annotated objects for this frame
            output_path: Path to save visualization
            show: Whether to display the image
        """
        if not annotations:
            print("No annotations to visualize")
            return
        
        # Group by camera
        by_camera = {}
        for ann in annotations:
            camera = ann['annotation_camera']
            if camera not in by_camera:
                by_camera[camera] = []
            by_camera[camera].append(ann)
        
        # Visualize each camera
        for camera, cam_annotations in by_camera.items():
            # Get frame token
            frame_token = cam_annotations[0]['frame_token']
            
            # Load camera image
            sample = self.nusc.get('sample', frame_token)
            camera_token = sample['data'][camera]
            cam_data = self.nusc.get('sample_data', camera_token)
            img_path = self.dataloader.dataroot / cam_data['filename']
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not load image: {img_path}")
                continue
            
            # Draw annotations
            for ann in cam_annotations:
                # Get simple class name
                class_name = ann['name'].split('.')[-1]
                
                # Get 2D bounding box (simplified - just put text)
                # For proper bounding box, would need to project 3D box to 2D
                
                # Position text in image
                # This is simplified - in practice you'd project the 3D position
                h, w = image.shape[:2]
                
                # Place text based on object index
                y_pos = 30 + (cam_annotations.index(ann) * 80)
                x_pos = 10
                
                # Create annotation text
                text_lines = [
                    f"{class_name}",
                    f"Activity: {ann['activity'][:50]}",
                    f"Caption: {ann['caption']}"
                ]
                
                # Draw semi-transparent background
                text_height = len(text_lines) * 25
                cv2.rectangle(image, (x_pos-5, y_pos-20), (w-10, y_pos+text_height), (0, 0, 0), -1)
                cv2.rectangle(image, (x_pos-5, y_pos-20), (w-10, y_pos+text_height), (255, 255, 255), 2)
                
                # Draw text
                for i, line in enumerate(text_lines):
                    y = y_pos + (i * 25)
                    if i == 0:  # Object name in larger font
                        cv2.putText(image, line, (x_pos, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(image, line, (x_pos, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add camera name
            cv2.putText(image, camera, (w - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Save or show
            if output_path:
                out_file = Path(output_path) / f"{frame_token}_{camera}.jpg"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_file), image)
                print(f"Saved: {out_file}")
            
            if show:
                cv2.imshow(f"Annotations - {camera}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    def create_annotation_summary(
        self,
        annotations_file: str,
        output_file: str
    ):
        """
        Create a summary document of annotations.
        
        Args:
            annotations_file: Path to annotations JSON
            output_file: Path to save summary
        """
        data = self.load_annotations(annotations_file)
        
        with open(output_file, 'w') as f:
            f.write(f"# Annotation Summary\n\n")
            f.write(f"Scene: {data['scene_name']} ({data['scene_token']})\n")
            f.write(f"Frames: {data['num_frames']}\n")
            f.write(f"Total Annotations: {data['num_annotations']}\n\n")
            
            # Group by frame
            frames = {}
            for ann in data['annotations']:
                frame_token = ann['frame_token']
                if frame_token not in frames:
                    frames[frame_token] = []
                frames[frame_token].append(ann)
            
            # Write annotations per frame
            for frame_token, frame_anns in frames.items():
                f.write(f"\n## Frame: {frame_token}\n\n")
                
                for i, ann in enumerate(frame_anns, 1):
                    f.write(f"### {i}. {ann['name']}\n\n")
                    f.write(f"- **Camera**: {ann['annotation_camera']}\n")
                    f.write(f"- **Position**: ({ann['position'][0]:.1f}, {ann['position'][1]:.1f}, {ann['position'][2]:.1f}) m\n")
                    
                    if ann.get('velocity'):
                        speed = np.sqrt(ann['velocity'][0]**2 + ann['velocity'][1]**2)
                        f.write(f"- **Speed**: {speed:.1f} m/s\n")
                    
                    f.write(f"- **Activity**: {ann['activity']}\n")
                    f.write(f"- **Description**: {ann['description']}\n")
                    f.write(f"- **Caption**: {ann['caption']}\n")
                    f.write(f"\n")
        
        print(f"Summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize VLM annotations')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--annotations', type=str, required=True, help='Path to annotations JSON file')
    parser.add_argument('--output', type=str, help='Output directory for visualizations')
    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--summary', action='store_true', help='Create text summary')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display images')
    
    args = parser.parse_args()
    
    # Initialize
    loader = NuScenesLidarSegmentationLoader(
        dataroot=args.dataroot,
        version=args.version,
        verbose=False
    )
    
    visualizer = AnnotationVisualizer(loader)
    
    # Load annotations
    data = visualizer.load_annotations(args.annotations)
    
    print(f"Loaded annotations for scene: {data['scene_name']}")
    print(f"Frames: {data['num_frames']}")
    print(f"Total annotations: {data['num_annotations']}")
    
    # Group by frame
    frames = {}
    for ann in data['annotations']:
        frame_token = ann['frame_token']
        if frame_token not in frames:
            frames[frame_token] = []
        frames[frame_token].append(ann)
    
    # Get frame to visualize
    frame_tokens = sorted(frames.keys())
    if args.frame_idx >= len(frame_tokens):
        print(f"Error: Frame index {args.frame_idx} out of range (0-{len(frame_tokens)-1})")
        return
    
    frame_token = frame_tokens[args.frame_idx]
    frame_annotations = frames[frame_token]
    
    print(f"\nVisualizing frame {args.frame_idx}: {frame_token}")
    print(f"Annotations: {len(frame_annotations)}")
    
    # Visualize
    visualizer.visualize_frame(
        annotations=frame_annotations,
        output_path=args.output,
        show=not args.no_show
    )
    
    # Create summary if requested
    if args.summary:
        summary_file = Path(args.annotations).parent / f"{Path(args.annotations).stem}_summary.md"
        visualizer.create_annotation_summary(args.annotations, str(summary_file))


if __name__ == "__main__":
    main()


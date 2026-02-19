"""
Improved VLM Annotator for nuScenes Scene Graph Objects (Instance-Based)

This module annotates object INSTANCES (not individual detections) using:
- Multiple frames per object instance (up to 8 best frames)
- Temporal context for better understanding of object behavior
- Bounding box visualization to help VLM focus on target object
- Parallelized processing for efficiency

Key improvement: Annotates each unique object instance once using multiple 
frames, rather than annotating the same object in every frame separately.
"""

import os
import sys
import json
import cv2
import base64
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from dataclasses import asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
# Add parent directory to path to import dataloader
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai not installed. Install with: pip install openai")

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from scenegraph.nuscenes_dataloader import NuScenesLidarSegmentationLoader, FrameData, ObjectProperties
from scenegraph.scenegraph import SceneGraphBuilder, SceneGraphNode


class InstanceAnnotator:
    """
    Annotates object instances using temporal context from multiple frames.
    """
    
    def __init__(
        self,
        api_base: str,
        model: str,
        dataloader: NuScenesLidarSegmentationLoader,
        max_workers: int = 10,
        max_frames_per_instance: int = 8
    ):
        """
        Initialize the instance-based VLM annotator.
        
        Args:
            api_base: Base URL for the vLLM server
            model: Model name
            dataloader: NuScenesLidarSegmentationLoader instance
            max_workers: Maximum number of parallel workers
            max_frames_per_instance: Maximum frames to use per object instance
        """
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model
        self.dataloader = dataloader
        self.nusc = dataloader.nusc
        self.max_workers = max_workers
        self.max_frames_per_instance = max_frames_per_instance
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """Encode a cv2 image to base64 string."""
        ret, buffer = cv2.imencode(".jpg", image)
        if not ret:
            raise ValueError("Could not encode image")
        return base64.b64encode(buffer).decode("utf-8")
    
    def _load_camera_image(self, camera_token: str) -> Optional[np.ndarray]:
        """Load camera image from token."""
        try:
            cam_data = self.nusc.get('sample_data', camera_token)
            img_path = self.dataloader.dataroot / cam_data['filename']
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                return None
            
            image = cv2.imread(str(img_path))
            return image
        except Exception as e:
            print(f"Error loading camera image: {e}")
            return None
    
    def _draw_bounding_box(
        self,
        image: np.ndarray,
        obj: ObjectProperties,
        sample_token: str,
        camera: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 3
    ) -> np.ndarray:
        """
        Draw bounding box on image for the target object.
        
        Args:
            image: Camera image
            obj: Object to highlight
            sample_token: Sample token
            camera: Camera name
            color: BGR color for box
            thickness: Line thickness
            
        Returns:
            Image with bounding box drawn
        """
        try:
            sample = self.nusc.get('sample', sample_token)
            camera_token = sample['data'][camera]
            cam_data = self.nusc.get('sample_data', camera_token)
            
            # Get ego pose
            ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
            ego_translation = np.array(ego_pose['translation'])
            ego_rotation = Quaternion(ego_pose['rotation'])
            
            # Get camera calibration
            cs_record = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            cam_translation = np.array(cs_record['translation'])
            cam_rotation = Quaternion(cs_record['rotation'])
            
            # Create 3D bounding box corners
            position = np.array(obj.position)
            size = np.array(obj.size)
            rotation_quat = Quaternion(obj.rotation)
            
            w, l, h = size
            corners = np.array([
                [-w/2, -l/2, -h/2], [ w/2, -l/2, -h/2],
                [ w/2,  l/2, -h/2], [-w/2,  l/2, -h/2],
                [-w/2, -l/2,  h/2], [ w/2, -l/2,  h/2],
                [ w/2,  l/2,  h/2], [-w/2,  l/2,  h/2],
            ]).T
            
            # Transform to camera frame
            corners_rotated = rotation_quat.rotation_matrix @ corners
            corners_global = corners_rotated + position.reshape(3, 1)
            corners_ego = corners_global - ego_translation.reshape(3, 1)
            corners_ego = ego_rotation.inverse.rotation_matrix @ corners_ego
            corners_cam = corners_ego - cam_translation.reshape(3, 1)
            corners_cam = cam_rotation.inverse.rotation_matrix @ corners_cam
            
            # Check if in front of camera
            if not np.any(corners_cam[2, :] > 0):
                return image
            
            # Project to 2D
            corners_2d = view_points(corners_cam, cam_intrinsic, normalize=True)[:2, :]
            
            # Draw box edges
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top
                [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
            ]
            
            img_copy = image.copy()
            h, w = img_copy.shape[:2]
            
            for edge in edges:
                pt1 = (int(corners_2d[0, edge[0]]), int(corners_2d[1, edge[0]]))
                pt2 = (int(corners_2d[0, edge[1]]), int(corners_2d[1, edge[1]]))
                
                if 0 <= pt1[0] < w*2 and 0 <= pt1[1] < h*2 and \
                   0 <= pt2[0] < w*2 and 0 <= pt2[1] < h*2:
                    cv2.line(img_copy, pt1, pt2, color, thickness)
            
            return img_copy
            
        except Exception as e:
            # Return original image if drawing fails
            print(f"Error drawing bounding box: {e}") 
            return image
    
    def collect_instance_data(
        self,
        scene_token: str
    ) -> Dict[str, List[Tuple[FrameData, ObjectProperties, str]]]:
        """
        Collect all frames for each object instance in the scene.
        
        Args:
            scene_token: Scene token to process
            
        Returns:
            Dictionary mapping instance_token → list of (frame, object, best_camera) tuples
        """
        print("Collecting instance data from frames...")
        
        # Load all frames
        frames = self.dataloader.get_scene_frames(scene_token)
        
        # Group by instance
        instance_data = defaultdict(list)
        
        for frame in tqdm(frames, desc="  Processing frames"):
            for obj in frame.objects:
                if not obj.visible_cameras:
                    continue
                
                # Select best camera for this object in this frame
                best_camera = obj.visible_cameras[0]  # Simple: take first visible camera
                
                instance_data[obj.instance_token].append((frame, obj, best_camera))
        
        return instance_data
    
    def select_best_frames(
        self,
        instance_frames: List[Tuple[FrameData, ObjectProperties, str]],
        max_frames: int
    ) -> List[Tuple[FrameData, ObjectProperties, str]]:
        """
        Select the best frames for an instance based on visibility and diversity.
        
        Args:
            instance_frames: All frames containing this instance
            max_frames: Maximum number of frames to select
            
        Returns:
            Selected frames (up to max_frames)
        """
        if len(instance_frames) <= max_frames:
            return instance_frames
        
        # Sort by visibility (higher is better)
        sorted_frames = sorted(
            instance_frames,
            key=lambda x: x[1].visibility,
            reverse=True
        )
        
        # Take top frames by visibility, but spread across time
        # Simple strategy: take evenly spaced frames from sorted list
        step = len(sorted_frames) / max_frames
        selected_indices = [int(i * step) for i in range(max_frames)]
        selected = [sorted_frames[i] for i in selected_indices]
        
        # Sort selected frames by timestamp for temporal coherence
        selected.sort(key=lambda x: x[0].timestamp)
        
        return selected
    
    def annotate_instance(
        self,
        instance_token: str,
        instance_frames: List[Tuple[FrameData, ObjectProperties, str]],
        object_class: str
    ) -> Optional[Dict[str, Any]]:
        """
        Annotate a single object instance using multiple frames.
        
        Args:
            instance_token: Unique instance identifier
            instance_frames: List of (frame, object, camera) tuples for this instance
            object_class: Object class name
            
        Returns:
            Annotation dictionary or None if failed
        """
        try:
            # Select best frames
            selected_frames = self.select_best_frames(
                instance_frames,
                self.max_frames_per_instance
            )
            
            if not selected_frames:
                return None
            
            # Prepare images with bounding boxes
            annotated_images = []
            frame_info = []
            
            for frame, obj, camera in selected_frames:
                # Load camera image
                camera_token = frame.camera_tokens.get(camera)
                if not camera_token:
                    continue
                
                image = self._load_camera_image(camera_token)
                if image is None:
                    continue
                
                # Draw bounding box on target object
                image_with_box = self._draw_bounding_box(
                    image, obj, frame.sample_token, camera,
                    color=(0, 255, 0), thickness=3
                )
                
                annotated_images.append(image_with_box)
                frame_info.append({
                    'camera': camera,
                    'timestamp': frame.timestamp,
                    'visibility': obj.visibility,
                    'velocity': obj.velocity
                })
            
            if not annotated_images:
                return None
            
            # Create multi-image prompt
            activity, description = self._annotate_with_multiframe(
                annotated_images,
                frame_info,
                object_class
            )
            

            return {
                'instance_token': instance_token,
                'object_class': object_class,
                'num_frames_used': len(annotated_images),
                'activity': activity,
                'description': description,
                'frames_info': frame_info
            }
            
        except Exception as e:
            print(f"Error annotating instance {instance_token}: {e}")
            return None
    
    def _annotate_with_multiframe(
        self,
        images: List[np.ndarray],
        frame_info: List[Dict],
        object_class: str
    ) -> Tuple[str, str]:
        """
        Annotate object using multiple frames.
        
        Args:
            images: List of images with bounding boxes
            frame_info: List of frame metadata
            object_class: Object class name
            
        Returns:
            Tuple of (activity, description)
        """
        # Encode all images
        encoded_images = [self._encode_image(img) for img in images]
        
        # Build prompt
        class_name = object_class.split('.')[-1].replace('_', ' ')
        
        # Create frame descriptions
        frame_descriptions = []
        for i, info in enumerate(frame_info, 1):
            vel_info = ""
            if info['velocity'] and info['velocity'] != (0.0, 0.0):
                speed = np.sqrt(info['velocity'][0]**2 + info['velocity'][1]**2)
                if speed > 0.5:
                    vel_info = f" (moving at {speed:.1f} m/s)"
            frame_descriptions.append(
                f"Frame {i} ({info['camera']}){vel_info}"
            )
        
        frames_text = "\n".join(frame_descriptions)
        
        prompt = f"""You are viewing {len(images)} frames showing the SAME {class_name} over time from an autonomous vehicle.
The {class_name} is highlighted with a GREEN bounding box in each frame.

{frames_text}

Please analyze this {class_name} across all frames and provide:

1. ACTIVITY: What is this {class_name} doing across these frames? (1-2 sentences)
   Consider its movement, behavior, and state changes over time.
   Examples: "driving forward and then turning left", "parked on the side of the road", "walking across the crosswalk"

2. DESCRIPTION: Detailed description of this {class_name} (2-3 sentences)
   Include:
   - Physical appearance (color, type, model if identifiable)
   - Consistent characteristics across frames
   - Notable features or state

Format your response as:
ACTIVITY: [activity description]
DESCRIPTION: [detailed description]"""

        # Build content with all images
        content = [{"type": "text", "text": prompt}]
        for encoded_img in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}
            })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.2,
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse response
            activity = "unknown"
            description = "No description available"
            
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('ACTIVITY:'):
                    activity = line.replace('ACTIVITY:', '').strip()
                elif line.startswith('DESCRIPTION:'):
                    description = line.replace('DESCRIPTION:', '').strip()
            
            return activity, description
            
        except Exception as e:
            print(f"Error calling VLM: {e}")
            return "unknown", "No description available"
    
    
    def annotate_scene(
        self,
        scene_token: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Annotate all object instances in a scene.
        
        Args:
            scene_token: Scene token to process
            output_dir: Directory to save annotations
            
        Returns:
            Dictionary containing all annotations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get scene info
        scene = self.nusc.get('scene', scene_token)
        scene_name = scene['name']
        
        print(f"\n{'='*60}")
        print(f"Annotating scene: {scene_name} ({scene_token})")
        print(f"{'='*60}")
        
        # Collect instance data
        instance_data = self.collect_instance_data(scene_token)
        
        print(f"\nFound {len(instance_data)} unique object instances")
        print(f"Using up to {self.max_frames_per_instance} frames per instance")
        print(f"Parallelizing with {self.max_workers} workers")
        
        # Get object classes
        instance_classes = {}
        for instance_token, frames in instance_data.items():
            # Use class from first frame
            instance_classes[instance_token] = frames[0][1].name
        
        # Submit annotation tasks
        print("\nSubmitting annotation tasks...")
        futures = []
        for instance_token, frames in instance_data.items():
            future = self.executor.submit(
                self.annotate_instance,
                instance_token,
                frames,
                instance_classes[instance_token]
            )
            futures.append((instance_token, future))
        
        # Collect results
        print("\nAnnotating instances...")
        annotations = []
        for instance_token, future in tqdm(futures, desc="  Progress"):
            result = future.result()
            if result is not None:
                annotations.append(result)
        
        # Save results
        results = {
            'scene_token': scene_token,
            'scene_name': scene_name,
            'num_instances': len(instance_data),
            'num_annotated': len(annotations),
            'max_frames_per_instance': self.max_frames_per_instance,
            'annotations': annotations
        }
        
        output_file = output_path / f'{scene_token}_instance_annotations.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnnotations saved to: {output_file}")
        
        # Print statistics
        self._print_statistics(results)
        
        return results
    
    def _print_statistics(self, results: Dict[str, Any]):
        """Print annotation statistics."""
        print(f"\n{'='*60}")
        print("Annotation Statistics")
        print(f"{'='*60}")
        print(f"Scene: {results['scene_name']}")
        print(f"Total instances: {results['num_instances']}")
        print(f"Successfully annotated: {results['num_annotated']}")
        print(f"Max frames per instance: {results['max_frames_per_instance']}")
        
        # Calculate average frames used
        if results['annotations']:
            avg_frames = np.mean([a['num_frames_used'] for a in results['annotations']])
            print(f"Average frames used per instance: {avg_frames:.1f}")
        
        # Object class statistics
        class_counts = defaultdict(int)
        for ann in results['annotations']:
            class_counts[ann['object_class']] += 1
        
        print(f"\nTop 10 annotated object classes:")
        for obj_class, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {obj_class:40s}: {count:4d}")
    
    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Annotate nuScenes object instances with VLM using temporal context'
    )
    parser.add_argument('--dataroot', type=str, required=True, help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--output', type=str, default='outputs/instance_annotations', help='Output directory')
    parser.add_argument('--vllm-api', type=str, default='http://localhost:8001/v1', help='vLLM server URL')
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3_5-14B', help='VLM model name')
    parser.add_argument('--num-scenes', type=int, default=250, help='Number of scenes to process')
    parser.add_argument('--max-workers', type=int, default=10, help='Max parallel workers')
    parser.add_argument('--max-frames', type=int, default=8, help='Max frames per instance')
    
    args = parser.parse_args()
    
    # Initialize dataloader
    print("Initializing nuScenes dataloader...")
    loader = NuScenesLidarSegmentationLoader(
        dataroot=args.dataroot,
        version=args.version,
        verbose=True
    )
    
    # Initialize instance annotator
    print(f"Initializing instance annotator...")
    annotator = InstanceAnnotator(
        api_base=args.vllm_api,
        model=args.model,
        dataloader=loader,
        max_workers=args.max_workers,
        max_frames_per_instance=args.max_frames
    )
    
    try:
        # Get scene
        for i in tqdm(range(args.num_scenes), desc="Processing scenes"):
            scene_tokens = loader.get_scene_tokens()
            print(f"Processing scene {i} of {args.num_scenes}")
            if i >= len(scene_tokens):
                print(f"Error: Scene index {i} out of range (0-{len(scene_tokens)-1})")
                return
            
            scene_token = scene_tokens[i]
            
            # Annotate scene
            results = annotator.annotate_scene(
                scene_token=scene_token,
                output_dir=args.output
            )
            
            print("\n✓ Annotation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        print("\nCleaning up...")
        annotator.cleanup()


if __name__ == "__main__":
    main()


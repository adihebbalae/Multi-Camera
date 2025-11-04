"""
VLM Annotator for nuScenes Scene Graph Objects

This module annotates objects with:
- Activity: What the object is doing
- Description: Detailed description of the object
- Caption: Short caption for the object

Uses camera frames to provide visual context to the VLM.
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

# Add parent directory to path to import dataloader
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai not installed. Install with: pip install openai")

from nuscenes_dataloader import NuScenesLidarSegmentationLoader, FrameData, ObjectProperties
from scenegraph import SceneGraphBuilder, SceneGraphNode


class VLMAnnotator:
    """
    Annotates scene graph objects using Vision-Language Models.
    """
    
    def __init__(
        self,
        api_base: str,
        model: str,
        dataloader: NuScenesLidarSegmentationLoader
    ):
        """
        Initialize the VLM annotator.
        
        Args:
            api_base: Base URL for the vLLM server (e.g., "http://localhost:8001/v1")
            model: Model name (e.g., "llava-hf/llava-1.5-7b-hf")
            dataloader: NuScenesLidarSegmentationLoader instance
        """
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model
        self.dataloader = dataloader
        self.nusc = dataloader.nusc
    
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
    
    def _create_object_list_text(self, objects: List[ObjectProperties], camera: str) -> str:
        """
        Create a text description of objects visible in the camera.
        
        Args:
            objects: List of objects visible in the camera
            camera: Camera name
            
        Returns:
            Formatted text listing the objects
        """
        if not objects:
            return "No objects detected in this view."
        
        text = f"Objects visible in {camera}:\n"
        for idx, obj in enumerate(objects, 1):
            # Extract simple class name
            class_name = obj.name.split('.')[-1].replace('_', ' ')
            
            # Add velocity info if available
            velocity_info = ""
            if obj.velocity and obj.velocity != (0.0, 0.0):
                speed = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
                if speed > 0.5:  # Moving
                    velocity_info = f" (moving at {speed:.1f} m/s)"
            
            # Add attributes
            attr_info = ""
            if obj.attributes:
                # Simplify attributes
                simple_attrs = [attr.split('.')[-1] for attr in obj.attributes]
                attr_info = f" - {', '.join(simple_attrs)}"
            
            text += f"{idx}. {class_name}{velocity_info}{attr_info}\n"
        
        return text
    
    def annotate_object_activity(
        self,
        image: np.ndarray,
        obj: ObjectProperties,
        context_objects: List[ObjectProperties],
        camera: str
    ) -> str:
        """
        Generate activity annotation for an object.
        
        Args:
            image: Camera image containing the object
            obj: Object to annotate
            context_objects: Other objects visible in the same camera
            camera: Camera name
            
        Returns:
            Activity description string
        """
        encoded_image = self._encode_image(image)
        
        # Get object class name
        class_name = obj.name.split('.')[-1].replace('_', ' ')
        
        # Create context
        context = self._create_object_list_text(context_objects, camera)
        
        # Construct prompt for activity
        prompt = f"""This is a view from {camera} of an autonomous vehicle.

{context}

Focus on the {class_name} at position {obj.position[0]:.1f}m forward, {obj.position[1]:.1f}m lateral.

What activity is this {class_name} performing? Provide a brief, specific description (1 sentence).
Examples: "driving straight", "turning left", "stopped at intersection", "walking across street", "parked"."""

        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=256,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting activity annotation: {e}")
            return "unknown"
    
    def annotate_object_description(
        self,
        image: np.ndarray,
        obj: ObjectProperties,
        camera: str
    ) -> str:
        """
        Generate detailed description for an object.
        
        Args:
            image: Camera image containing the object
            obj: Object to annotate
            camera: Camera name
            
        Returns:
            Description string
        """
        encoded_image = self._encode_image(image)
        
        class_name = obj.name.split('.')[-1].replace('_', ' ')
        
        # Construct prompt for description
        prompt = f"""This is a view from {camera} of an autonomous vehicle.

Focus on the {class_name} at position {obj.position[0]:.1f}m forward, {obj.position[1]:.1f}m lateral.

Describe this {class_name} in detail (2-3 sentences). Include:
- Appearance (color, type, model if identifiable)
- Position and orientation
- State or condition
"""

        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=512,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting description: {e}")
            return "No description available"
    
   
    def annotate_frame_objects(
        self,
        frame: FrameData,
        camera_preference: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Annotate all objects in a frame using camera images.
        
        Args:
            frame: FrameData object
            camera_preference: Ordered list of preferred cameras (e.g., ['CAM_FRONT', 'CAM_FRONT_LEFT'])
            
        Returns:
            List of annotated object dictionaries
        """
        if camera_preference is None:
            camera_preference = [
                'CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT'
            ]
        
        annotated_objects = []
        
        # Group objects by camera
        objects_by_camera = {}
        for obj in frame.objects:
            if not obj.visible_cameras:
                continue
            
            # Choose best camera for this object
            best_camera = None
            for cam in camera_preference:
                if cam in obj.visible_cameras:
                    best_camera = cam
                    break
            
            if best_camera:
                if best_camera not in objects_by_camera:
                    objects_by_camera[best_camera] = []
                objects_by_camera[best_camera].append(obj)
        
        # Process each camera
        for camera, camera_objects in objects_by_camera.items():
            # Load camera image
            camera_token = frame.camera_tokens.get(camera)
            if not camera_token:
                continue
            
            image = self._load_camera_image(camera_token)
            if image is None:
                continue
            
            # Annotate each object in this camera
            for obj in camera_objects:
                try:
                    # Get annotations
                    activity = self.annotate_object_activity(
                        image, obj, camera_objects, camera
                    )
                    
                    description = self.annotate_object_description(
                        image, obj, camera
                    )
                    
                    # Create annotated object dict
                    annotated_obj = {
                        'token': obj.token,
                        'instance_token': obj.instance_token,
                        'name': obj.name,
                        'position': obj.position,
                        'size': obj.size,
                        'velocity': obj.velocity,
                        'num_lidar_pts': obj.num_lidar_pts,
                        'visibility': obj.visibility,
                        'attributes': obj.attributes,
                        'visible_cameras': obj.visible_cameras,
                        'annotation_camera': camera,  # Camera used for annotation
                        'activity': activity,
                        'description': description,
                        'frame_token': frame.sample_token,
                        'timestamp': frame.timestamp
                    }
                    
                    annotated_objects.append(annotated_obj)
                    
                except Exception as e:
                    print(f"Error annotating object {obj.token}: {e}")
                    continue
        
        return annotated_objects
    
    def annotate_scene(
        self,
        scene_token: str,
        output_dir: str,
        max_frames: Optional[int] = None,
        camera_preference: List[str] = None
    ) -> Dict[str, Any]:
        """
        Annotate all objects in a scene.
        
        Args:
            scene_token: Scene token to process
            output_dir: Directory to save annotations
            max_frames: Maximum number of frames to process (None for all)
            camera_preference: Ordered list of preferred cameras
            
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
        
        # Load frames
        frames = self.dataloader.get_scene_frames(scene_token)
        
        if max_frames:
            frames = frames[:max_frames]
        
        print(f"Processing {len(frames)} frames...")
        
        # Annotate all frames
        all_annotations = []
        frame_summaries = []
        
        for frame_idx, frame in enumerate(tqdm(frames, desc="Annotating frames")):
            # Annotate objects in this frame
            annotated_objects = self.annotate_frame_objects(
                frame,
                camera_preference=camera_preference
            )
            
            all_annotations.extend(annotated_objects)
            
            # Frame summary
            frame_summary = {
                'frame_idx': frame_idx,
                'sample_token': frame.sample_token,
                'timestamp': frame.timestamp,
                'num_objects': len(frame.objects),
                'num_annotated': len(annotated_objects)
            }
            frame_summaries.append(frame_summary)
            
            # Save intermediate results every 10 frames
            if (frame_idx + 1) % 10 == 0:
                self._save_annotations(
                    scene_token,
                    scene_name,
                    all_annotations,
                    frame_summaries,
                    output_path,
                    partial=True
                )
        
        # Save final results
        results = self._save_annotations(
            scene_token,
            scene_name,
            all_annotations,
            frame_summaries,
            output_path,
            partial=False
        )
        
        # Print statistics
        self._print_statistics(results)
        
        return results
    
    def _save_annotations(
        self,
        scene_token: str,
        scene_name: str,
        annotations: List[Dict[str, Any]],
        frame_summaries: List[Dict[str, Any]],
        output_path: Path,
        partial: bool = False
    ) -> Dict[str, Any]:
        """Save annotations to disk."""
        results = {
            'scene_token': scene_token,
            'scene_name': scene_name,
            'num_frames': len(frame_summaries),
            'num_annotations': len(annotations),
            'frame_summaries': frame_summaries,
            'annotations': annotations
        }
        
        # Save main annotations file
        suffix = '_partial' if partial else ''
        annotations_file = output_path / f'{scene_token}_annotations{suffix}.json'
        
        with open(annotations_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not partial:
            print(f"\nAnnotations saved to: {annotations_file}")
        
        return results
    
    def _print_statistics(self, results: Dict[str, Any]):
        """Print annotation statistics."""
        print(f"\n{'='*60}")
        print("Annotation Statistics")
        print(f"{'='*60}")
        print(f"Scene: {results['scene_name']}")
        print(f"Frames processed: {results['num_frames']}")
        print(f"Total annotations: {results['num_annotations']}")
        print(f"Avg annotations per frame: {results['num_annotations'] / results['num_frames']:.1f}")
        
        # Camera usage statistics
        camera_counts = {}
        for ann in results['annotations']:
            cam = ann['annotation_camera']
            camera_counts[cam] = camera_counts.get(cam, 0) + 1
        
        print(f"\nAnnotations by camera:")
        for cam, count in sorted(camera_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cam:20s}: {count:4d} ({count/results['num_annotations']*100:.1f}%)")
        
        # Object class statistics
        class_counts = {}
        for ann in results['annotations']:
            obj_class = ann['name']
            class_counts[obj_class] = class_counts.get(obj_class, 0) + 1
        
        print(f"\nTop 10 annotated object classes:")
        for obj_class, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {obj_class:40s}: {count:4d}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotate nuScenes scene graph with VLM')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--output', type=str, default='outputs/annotations', help='Output directory')
    parser.add_argument('--vllm-api', type=str, default='http://localhost:8001/v1', help='vLLM server URL')
    parser.add_argument('--model', type=str, default='llava-hf/llava-1.5-7b-hf', help='VLM model name')
    parser.add_argument('--scene-idx', type=int, default=0, help='Scene index to process')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames per scene')
    
    args = parser.parse_args()
    
    # Initialize dataloader
    print("Initializing nuScenes dataloader...")
    loader = NuScenesLidarSegmentationLoader(
        dataroot=args.dataroot,
        version=args.version,
        verbose=True
    )
    
    # Initialize VLM annotator
    print("Initializing VLM annotator...")
    annotator = VLMAnnotator(
        api_base=args.vllm_api,
        model=args.model,
        dataloader=loader
    )
    
    # Get scene
    scene_tokens = loader.get_scene_tokens()
    if args.scene_idx >= len(scene_tokens):
        print(f"Error: Scene index {args.scene_idx} out of range (0-{len(scene_tokens)-1})")
        return
    
    scene_token = scene_tokens[args.scene_idx]
    
    # Annotate scene
    results = annotator.annotate_scene(
        scene_token=scene_token,
        output_dir=args.output,
        max_frames=args.max_frames,
        camera_preference=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    )
    
    print("\n✓ Annotation complete!")


if __name__ == "__main__":
    main()


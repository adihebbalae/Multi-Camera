"""
nuScenes Dataloader for LiDAR Segmentation Map Extraction

This module provides functionality to load nuScenes dataset and extract
lidar segmentation maps along with object properties per frame.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
except ImportError:
    print("Warning: nuscenes-devkit not installed. Install with: pip install nuscenes-devkit")


@dataclass
class ObjectProperties:
    """Properties of an object detected in a frame."""
    token: str
    name: str  # Object class name (e.g., 'car', 'pedestrian')
    position: Tuple[float, float, float]  # 3D position (x, y, z)
    size: Tuple[float, float, float]  # width, length, height
    rotation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    velocity: Optional[Tuple[float, float]]  # velocity in x, y
    num_lidar_pts: int  # Number of lidar points
    visibility: int  # Visibility level
    attributes: List[str]  # Object attributes
    instance_token: str  # Unique instance identifier
    visible_cameras: Optional[List[str]] = None  # List of camera names where object is visible


@dataclass
class FrameData:
    """Data for a single frame including lidar segmentation."""
    scene_token: str
    sample_token: str
    timestamp: int
    lidar_points: np.ndarray  # Shape: (N, 5) - x, y, z, intensity, ring_index
    segmentation_labels: np.ndarray  # Shape: (N,) - object instance per point
    objects: List[ObjectProperties]
    ego_pose: Dict[str, Any]  # Vehicle pose information
    camera_tokens: Dict[str, str]  # Camera sensor tokens for this frame


class NuScenesLidarSegmentationLoader:
    """
    Dataloader for nuScenes dataset to extract lidar segmentation maps.
    
    This loader processes the nuScenes dataset to extract:
    - LiDAR point clouds with segmentation labels
    - Object properties per frame
    - Scene graph relationships (optional)
    """
    
    def __init__(
        self,
        dataroot: str,
        version: str = 'v1.0-trainval',
        verbose: bool = True,
        load_annotations: bool = True
    ):
        """
        Initialize the nuScenes dataloader.
        
        Args:
            dataroot: Path to the nuScenes dataset root directory
            version: Dataset version (e.g., 'v1.0-trainval', 'v1.0-mini')
            verbose: Whether to print loading information
            load_annotations: Whether to load annotation data
        """
        self.dataroot = Path(dataroot)
        self.version = version
        self.verbose = verbose
        self.load_annotations = load_annotations
        
        if verbose:
            print(f"Initializing nuScenes dataloader from {dataroot}")
            print(f"Version: {version}")
        
        # Initialize nuScenes
        self.nusc = NuScenes(
            version=version,
            dataroot=str(dataroot),
            verbose=verbose
        )
        
        # Cache for frequently accessed data
        self._scene_cache = {}
        self._sample_cache = {}
        
    def get_scene_tokens(self) -> List[str]:
        """Get all scene tokens in the dataset."""
        return [scene['token'] for scene in self.nusc.scene]
    
    def get_samples_in_scene(self, scene_token: str) -> List[str]:
        """
        Get all sample tokens in a scene.
        
        Args:
            scene_token: Token of the scene
            
        Returns:
            List of sample tokens in chronological order
        """
        scene = self.nusc.get('scene', scene_token)
        sample_tokens = []
        
        # Start from first sample
        sample_token = scene['first_sample_token']
        
        while sample_token != '':
            sample_tokens.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']
            
        return sample_tokens
    
    def _get_lidar_segmentation(
        self,
        sample_token: str
    ) -> Tuple[np.ndarray, np.ndarray, List[ObjectProperties]]:
        """
        Extract lidar points with segmentation labels for a sample.
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            Tuple of (points, segmentation_labels, objects)
            - points: (N, 5) array of x, y, z, intensity, ring_index
            - segmentation_labels: (N,) array of instance indices
            - objects: List of ObjectProperties for detected objects
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Get lidar data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_filepath = self.dataroot / lidar_data['filename']
        
        # Load point cloud
        pc = LidarPointCloud.from_file(str(lidar_filepath))
        points = pc.points.T  # Shape: (N, 4) - x, y, z, intensity
        
        # Add ring index (if available, otherwise zeros)
        ring_index = np.zeros((points.shape[0], 1))
        points = np.hstack([points, ring_index])  # Shape: (N, 5)
        
        # Initialize segmentation labels (0 = background)
        segmentation_labels = np.zeros(points.shape[0], dtype=np.int32)
        
        # Get annotations
        objects = []
        if self.load_annotations:
            # Get all boxes for this sample
            _, boxes, _ = self.nusc.get_sample_data(lidar_token)
            
            # Get camera visibility for all objects once
            visibility_map = self.get_all_objects_camera_visibility(
                sample_token,
                min_visibility=BoxVisibility.ANY
            )
            
            for box_idx, box in enumerate(boxes, start=1):
                # Get object properties
                annotation_token = box.token
                annotation = self.nusc.get('sample_annotation', annotation_token)
                
                # Extract object information
                obj_props = ObjectProperties(
                    token=annotation_token,
                    name=annotation['category_name'],
                    position=tuple(annotation['translation']),
                    size=tuple(annotation['size']),
                    rotation=tuple(annotation['rotation']),
                    velocity=tuple(self.nusc.box_velocity(annotation_token)[:2]) 
                              if self.nusc.box_velocity(annotation_token) is not None else None,
                    num_lidar_pts=annotation['num_lidar_pts'],
                    visibility=int(annotation['visibility_token']),
                    attributes=[self.nusc.get('attribute', token)['name'] 
                               for token in annotation['attribute_tokens']],
                    instance_token=annotation['instance_token'],
                    visible_cameras=visibility_map.get(annotation_token, [])
                )
                objects.append(obj_props)
                
                # Segment points inside this bounding box
                # Transform points to box coordinate system
                points_in_box = self._points_in_box(points[:, :3], box)
                segmentation_labels[points_in_box] = box_idx
        
        return points, segmentation_labels, objects
    
    def _points_in_box(
        self,
        points: np.ndarray,
        box: Any
    ) -> np.ndarray:
        """
        Check which points are inside a bounding box.
        
        Args:
            points: (N, 3) array of point coordinates
            box: nuScenes Box object
            
        Returns:
            Boolean array of shape (N,) indicating points inside the box
        """
        # Transform points to box coordinate system
        points_centered = points - box.center
        
        # Rotate points to box frame
        rotation_matrix = box.rotation_matrix
        points_rotated = points_centered @ rotation_matrix.T
        
        # Check if points are within box dimensions
        half_size = np.array(box.wlh) / 2
        inside = (
            (np.abs(points_rotated[:, 0]) <= half_size[0]) &
            (np.abs(points_rotated[:, 1]) <= half_size[1]) &
            (np.abs(points_rotated[:, 2]) <= half_size[2])
        )
        
        return inside
    
    def get_object_camera_visibility(
        self,
        sample_token: str,
        annotation_token: str,
        min_visibility: BoxVisibility = BoxVisibility.ANY
    ) -> List[str]:
        """
        Determine which cameras an object is visible from.
        
        Args:
            sample_token: Token of the sample
            annotation_token: Token of the annotation
            min_visibility: Minimum visibility level (BoxVisibility.NONE, .ANY, .PARTIAL, .MOST, .ALL)
            
        Returns:
            List of camera names where the object is visible (e.g., ['CAM_FRONT', 'CAM_FRONT_LEFT'])
        """
        sample = self.nusc.get('sample', sample_token)
        annotation = self.nusc.get('sample_annotation', annotation_token)
        
        # Get the box in global coordinates
        box = self.nusc.get_box(annotation_token)
        
        visible_cameras = []
        
        # Check all camera channels
        camera_channels = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]
        
        for camera_channel in camera_channels:
            if camera_channel not in sample['data']:
                continue
            
            camera_token = sample['data'][camera_channel]
            
            # Check if box is visible in this camera
            try:
                _, boxes, camera_intrinsic = self.nusc.get_sample_data(
                    camera_token,
                    selected_anntokens=[annotation_token]
                )
                
                if len(boxes) > 0:
                    # Box is in the camera's field of view
                    # Now check if it meets the visibility requirement
                    visibility = self.nusc.get('sample_annotation', annotation_token)['visibility_token']
                    visibility_level = int(visibility)
                    
                    # BoxVisibility levels: NONE=0, ANY=1, PARTIAL=2, MOST=3, ALL=4
                    if visibility_level >= min_visibility.value:
                        visible_cameras.append(camera_channel)
            except:
                # Box not visible in this camera
                continue
        
        return visible_cameras
    
    def get_all_objects_camera_visibility(
        self,
        sample_token: str,
        min_visibility: BoxVisibility = BoxVisibility.ANY
    ) -> Dict[str, List[str]]:
        """
        Get camera visibility for all objects in a sample.
        
        Args:
            sample_token: Token of the sample
            min_visibility: Minimum visibility level
            
        Returns:
            Dictionary mapping annotation tokens to lists of visible camera names
        """
        sample = self.nusc.get('sample', sample_token)
        visibility_map = {}
        
        for ann_token in sample['anns']:
            visible_cameras = self.get_object_camera_visibility(
                sample_token,
                ann_token,
                min_visibility
            )
            visibility_map[ann_token] = visible_cameras
        
        return visibility_map
    
    def get_frame_data(self, sample_token: str) -> FrameData:
        """
        Get complete frame data including lidar segmentation.
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            FrameData object containing all frame information
        """
        sample = self.nusc.get('sample', sample_token)
        scene_token = sample['scene_token']
        
        # Get lidar segmentation
        points, seg_labels, objects = self._get_lidar_segmentation(sample_token)
        
        # Get ego pose
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose_token = lidar_data['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        
        # Get camera tokens
        camera_tokens = {
            sensor: sample['data'][sensor]
            for sensor in sample['data']
            if 'CAM' in sensor
        }
        
        frame_data = FrameData(
            scene_token=scene_token,
            sample_token=sample_token,
            timestamp=sample['timestamp'],
            lidar_points=points,
            segmentation_labels=seg_labels,
            objects=objects,
            ego_pose=ego_pose,
            camera_tokens=camera_tokens
        )
        
        return frame_data
    
    def get_scene_frames(self, scene_token: str) -> List[FrameData]:
        """
        Get all frames in a scene.
        
        Args:
            scene_token: Token of the scene
            
        Returns:
            List of FrameData objects for all samples in the scene
        """
        sample_tokens = self.get_samples_in_scene(scene_token)
        frames = []
        
        if self.verbose:
            print(f"Loading {len(sample_tokens)} frames from scene {scene_token}")
        
        for sample_token in sample_tokens:
            try:
                frame_data = self.get_frame_data(sample_token)
                frames.append(frame_data)
            except Exception as e:
                print(f"Error loading frame {sample_token}: {e}")
                continue
        
        return frames
    
    def export_frame_to_dict(self, frame: FrameData) -> Dict[str, Any]:
        """
        Export frame data to a dictionary format (for JSON serialization).
        
        Args:
            frame: FrameData object
            
        Returns:
            Dictionary containing frame data
        """
        return {
            'scene_token': frame.scene_token,
            'sample_token': frame.sample_token,
            'timestamp': frame.timestamp,
            'num_points': frame.lidar_points.shape[0],
            'objects': [
                {
                    'token': obj.token,
                    'name': obj.name,
                    'position': obj.position,
                    'size': obj.size,
                    'rotation': obj.rotation,
                    'velocity': obj.velocity,
                    'num_lidar_pts': obj.num_lidar_pts,
                    'visibility': obj.visibility,
                    'attributes': obj.attributes,
                    'instance_token': obj.instance_token,
                    'visible_cameras': obj.visible_cameras
                }
                for obj in frame.objects
            ],
            'ego_pose': frame.ego_pose,
            'camera_tokens': frame.camera_tokens
        }
    
    def save_frame(
        self,
        frame: FrameData,
        output_dir: str,
        save_points: bool = True,
        save_metadata: bool = True
    ):
        """
        Save frame data to disk.
        
        Args:
            frame: FrameData object to save
            output_dir: Directory to save data
            save_points: Whether to save point cloud and segmentation
            save_metadata: Whether to save metadata JSON
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sample_token = frame.sample_token
        
        # Save point cloud and segmentation
        if save_points:
            points_file = output_path / f"{sample_token}_points.npy"
            seg_file = output_path / f"{sample_token}_segmentation.npy"
            
            np.save(points_file, frame.lidar_points)
            np.save(seg_file, frame.segmentation_labels)
        
        # Save metadata
        if save_metadata:
            metadata_file = output_path / f"{sample_token}_metadata.json"
            metadata = self.export_frame_to_dict(frame)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def get_object_statistics(self, scene_token: Optional[str] = None) -> Dict[str, int]:
        """
        Get statistics of object classes in the dataset or a specific scene.
        
        Args:
            scene_token: Optional scene token. If None, computes for entire dataset
            
        Returns:
            Dictionary mapping object class names to counts
        """
        stats = {}
        
        if scene_token:
            sample_tokens = self.get_samples_in_scene(scene_token)
        else:
            sample_tokens = [sample['token'] for sample in self.nusc.sample]
        
        for sample_token in sample_tokens:
            sample = self.nusc.get('sample', sample_token)
            for ann_token in sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                category = ann['category_name']
                stats[category] = stats.get(category, 0) + 1
        
        return stats


def main():
    """Example usage of the NuScenesLidarSegmentationLoader."""
    # Configuration
    dataroot = '/path/to/nuscenes'  # Update this path
    version = 'v1.0-mini'  # Use 'v1.0-mini' for testing, 'v1.0-trainval' for full dataset
    output_dir = 'outputs/nuscenes_segmentation'
    
    # Initialize loader
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=True
    )
    
    # Get all scenes
    scene_tokens = loader.get_scene_tokens()
    print(f"\nFound {len(scene_tokens)} scenes")
    
    # Process first scene as example
    if scene_tokens:
        scene_token = scene_tokens[0]
        print(f"\nProcessing scene: {scene_token}")
        
        # Get object statistics for this scene
        stats = loader.get_object_statistics(scene_token)
        print("\nObject statistics:")
        for obj_class, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {obj_class}: {count}")
        
        # Load all frames in the scene
        frames = loader.get_scene_frames(scene_token)
        print(f"\nLoaded {len(frames)} frames")
        
        # Process and save each frame
        for idx, frame in enumerate(frames):
            print(f"\nFrame {idx + 1}/{len(frames)}:")
            print(f"  Timestamp: {frame.timestamp}")
            print(f"  Points: {frame.lidar_points.shape[0]}")
            print(f"  Objects: {len(frame.objects)}")
            
            # Print object details
            for obj in frame.objects:
                cameras_str = ', '.join(obj.visible_cameras) if obj.visible_cameras else 'None'
                print(f"    - {obj.name}: {obj.num_lidar_pts} points, visibility={obj.visibility}, cameras=[{cameras_str}]")
            
            # Save frame data
            loader.save_frame(
                frame,
                output_dir=output_dir,
                save_points=True,
                save_metadata=True
            )
        
        print(f"\nData saved to {output_dir}")


if __name__ == "__main__":
    main()


"""
Scene Graph Builder for nuScenes Dataset

This module obtains a list of objects per frame (optionally with their relationships)
and properties that we want to annotate for these objects with a VLM (activity, etc).
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from nuscenes_dataloader import (
    NuScenesLidarSegmentationLoader,
    FrameData,
    ObjectProperties
)


@dataclass
class SceneGraphNode:
    """A node in the scene graph representing an object."""
    annotation_token: str  # Annotation token
    object_id: str  # Unique object instance token
    object_class: str  # Object category name
    position: tuple  # 3D position
    size: tuple  # Object dimensions
    rotation: tuple  # Rotation quaternion (w, x, y, z)
    velocity: Optional[tuple]  # Object velocity
    attributes: List[str]  # Object attributes
    frame_idx: int  # Frame index
    timestamp: int  # Timestamp
    num_lidar_pts: int  # Number of lidar points
    visibility: int  # Visibility level
    sample_token: str  # Sample token for this frame
    visible_cameras: Optional[List[str]] = None  # Cameras where object is visible
    
    # VLM-annotated properties (to be filled later)
    activity: Optional[str] = None
    description: Optional[str] = None



@dataclass
class SceneGraphRelationship:
    """A relationship between two objects in the scene graph."""
    source_id: str  # Source object instance token
    target_id: str  # Target object instance token
    relationship_type: str  # Type of relationship (e.g., 'near', 'behind', 'interacting')
    distance: Optional[float] = None  # Distance between objects
    frame_idx: int = 0  # Frame index


class SceneGraphBuilder:
    """
    Builds scene graphs from nuScenes data for VLM annotation.
    
    This class processes lidar segmentation data and creates structured
    scene graphs with objects and their relationships per frame.
    """
    
    def __init__(
        self,
        dataloader: NuScenesLidarSegmentationLoader,
        extract_relationships: bool = True,
        distance_threshold: float = 10.0  # meters
    ):
        """
        Initialize the scene graph builder.
        
        Args:
            dataloader: NuScenesLidarSegmentationLoader instance
            extract_relationships: Whether to compute spatial relationships
            distance_threshold: Max distance to consider objects as "near"
        """
        self.dataloader = dataloader
        self.extract_relationships = extract_relationships
        self.distance_threshold = distance_threshold
        
    def build_frame_scene_graph(
        self,
        frame: FrameData,
        frame_idx: int
    ) -> tuple[List[SceneGraphNode], List[SceneGraphRelationship]]:
        """
        Build a scene graph for a single frame.
        
        Args:
            frame: FrameData object containing frame information
            frame_idx: Index of the frame in the sequence
            
        Returns:
            Tuple of (nodes, relationships) for the scene graph
        """
        nodes = []
        relationships = []
        
        # Create nodes for each object
        for obj in frame.objects:
            node = SceneGraphNode(
                annotation_token=obj.token,
                object_id=obj.instance_token,
                object_class=obj.name,
                position=obj.position,
                size=obj.size,
                rotation=obj.rotation,
                velocity=obj.velocity,
                attributes=obj.attributes,
                frame_idx=frame_idx,
                timestamp=frame.timestamp,
                num_lidar_pts=obj.num_lidar_pts,
                visibility=obj.visibility,
                sample_token=frame.sample_token,
                visible_cameras=obj.visible_cameras
            )
            nodes.append(node)
        
        # Extract relationships if enabled
        if self.extract_relationships and len(nodes) > 1:
            relationships = self._extract_spatial_relationships(nodes, frame_idx)
        
        return nodes, relationships
    
    def _extract_spatial_relationships(
        self,
        nodes: List[SceneGraphNode],
        frame_idx: int
    ) -> List[SceneGraphRelationship]:
        """
        Extract spatial relationships between objects.
        
        Args:
            nodes: List of scene graph nodes
            frame_idx: Frame index
            
        Returns:
            List of SceneGraphRelationship objects
        """
        relationships = []
        
        # Compute pairwise relationships
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                # Compute distance
                pos_a = np.array(node_a.position)
                pos_b = np.array(node_b.position)
                distance = np.linalg.norm(pos_a - pos_b)
                
                # Add "near" relationship if within threshold
                if distance < self.distance_threshold:
                    rel = SceneGraphRelationship(
                        source_id=node_a.object_id,
                        target_id=node_b.object_id,
                        relationship_type='near',
                        distance=float(distance),
                        frame_idx=frame_idx
                    )
                    relationships.append(rel)
                
                # Compute directional relationships (behind, in_front, left, right)
                rel_type = self._compute_directional_relationship(node_a, node_b)
                if rel_type:
                    rel = SceneGraphRelationship(
                        source_id=node_a.object_id,
                        target_id=node_b.object_id,
                        relationship_type=rel_type,
                        distance=float(distance),
                        frame_idx=frame_idx
                    )
                    relationships.append(rel)
        
        return relationships
    
    def _compute_directional_relationship(
        self,
        node_a: SceneGraphNode,
        node_b: SceneGraphNode
    ) -> Optional[str]:
        """
        Compute directional relationship (behind, in_front, left, right).
        
        Args:
            node_a: First node
            node_b: Second node
            
        Returns:
            Relationship type or None
        """
        pos_a = np.array(node_a.position[:2])  # x, y
        pos_b = np.array(node_b.position[:2])
        
        # Compute relative position
        rel_pos = pos_b - pos_a
        
        # Compute angle
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        angle_deg = np.degrees(angle)
        
        # Classify relationship based on angle
        if -45 <= angle_deg < 45:
            return 'in_front'
        elif 45 <= angle_deg < 135:
            return 'left'
        elif 135 <= angle_deg or angle_deg < -135:
            return 'behind'
        else:
            return 'right'
    
    def build_scene_graphs(
        self,
        scene_token: str
    ) -> tuple[List[List[SceneGraphNode]], List[List[SceneGraphRelationship]]]:
        """
        Build scene graphs for all frames in a scene.
        
        Args:
            scene_token: Token of the scene to process
            
        Returns:
            Tuple of (all_nodes, all_relationships) for the entire scene
        """
        frames = self.dataloader.get_scene_frames(scene_token)
        
        all_nodes = []
        all_relationships = []
        
        for frame_idx, frame in enumerate(frames):
            nodes, relationships = self.build_frame_scene_graph(frame, frame_idx)
            all_nodes.append(nodes)
            all_relationships.append(relationships)
        
        return all_nodes, all_relationships
    
    def export_to_json(
        self,
        nodes: List[List[SceneGraphNode]],
        relationships: List[List[SceneGraphRelationship]],
        output_path: str,
        scene_token: str = ''
    ):
        """
        Export scene graphs to JSON format.
        
        Args:
            nodes: List of node lists (per frame)
            relationships: List of relationship lists (per frame)
            output_path: Path to save JSON file
            scene_token: Scene token identifier
        """
        data = {
            'scene_token': scene_token,
            'frames': []
        }
        
        for frame_idx, (frame_nodes, frame_rels) in enumerate(zip(nodes, relationships)):
            frame_data = {
                'frame_idx': frame_idx,
                'sample_token': frame_nodes[0].sample_token if frame_nodes else '',
                'timestamp': frame_nodes[0].timestamp if frame_nodes else 0,
                'objects': [asdict(node) for node in frame_nodes],
                'relationships': [asdict(rel) for rel in frame_rels]
            }
            data['frames'].append(frame_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_objects_for_vlm_annotation(
        self,
        nodes: List[List[SceneGraphNode]],
        filter_classes: Optional[List[str]] = None,
        min_visibility: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Prepare object list for VLM annotation.
        
        Args:
            nodes: Scene graph nodes for all frames
            filter_classes: Optional list of object classes to include
            min_visibility: Minimum visibility level to include
            
        Returns:
            List of objects ready for VLM annotation with activity/description
        """
        objects_for_annotation = []
        
        for frame_idx, frame_nodes in enumerate(nodes):
            for node in frame_nodes:
                # Filter by class if specified
                if filter_classes and node.object_class not in filter_classes:
                    continue
                
                # Filter by visibility
                if node.visibility < min_visibility:
                    continue
                
                # Prepare object for annotation
                obj_data = {
                    'annotation_token': node.annotation_token,
                    'object_id': node.object_id,
                    'object_class': node.object_class,
                    'frame_idx': frame_idx,
                    'timestamp': node.timestamp,
                    'position': node.position,
                    'size': node.size,
                    'velocity': node.velocity,
                    'attributes': node.attributes,
                    'num_lidar_pts': node.num_lidar_pts,
                    # Fields to be filled by VLM
                    'activity': None,
                    'description': None,
                }
                objects_for_annotation.append(obj_data)
        
        return objects_for_annotation

def parse_args():
    parser = ArgumentParser(description='Build scene graphs for nuScenes dataset')
    parser.add_argument('--dataroot', type=str, required=False, help='Path to nuScenes dataset', default='/nas/standard_datasets/nuscenes')
    parser.add_argument('--version', type=str, required=False, help='Version of nuScenes dataset', default='v1.0-trainval')
    parser.add_argument('--output-dir', type=str, required=False, help='Path to output directory', default='outputs/scene_graphs')
    parser.add_argument('--num-scenes', type=int, required=False, help='Number of scenes to process', default=1)
    return parser.parse_args()

def main():
    """Example usage of the SceneGraphBuilder."""
    args = parse_args()
    # Configuration
    dataroot = args.dataroot
    version = args.version
    output_dir = args.output_dir
    
    # Initialize dataloader
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=True
    )
    
    # Initialize scene graph builder
    builder = SceneGraphBuilder(
        dataloader=loader,
        extract_relationships=True,
        distance_threshold=10.0
    )
    
    # Process first scene
    scene_tokens = loader.get_scene_tokens()
    for i in tqdm(range(args.num_scenes)):
        if scene_tokens:
            scene_token = scene_tokens[i]
            print(f"Building scene graphs for scene: {scene_token} {i+1}/{args.num_scenes}")
            
            # Build scene graphs
            all_nodes, all_relationships = builder.build_scene_graphs(scene_token)
            
            print(f"\nProcessed {len(all_nodes)} frames")
            
            # Print statistics
            total_objects = sum(len(nodes) for nodes in all_nodes)
            total_relationships = sum(len(rels) for rels in all_relationships)
            print(f"Total objects: {total_objects}")
            print(f"Total relationships: {total_relationships}")
            
            # Export to JSON
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, scene_token), exist_ok=True)
            output_path = os.path.join(output_dir, f'{scene_token}/scene_graph.json')
            builder.export_to_json(all_nodes, all_relationships, output_path, scene_token)
            print(f"\nScene graph saved to {output_path}")
            
            # Get objects for VLM annotation
            # Focus on dynamic objects like vehicles and pedestrians
            dynamic_classes = [
                'vehicle.car', 'vehicle.truck', 'vehicle.bus',
                'human.pedestrian', 'vehicle.bicycle', 'vehicle.motorcycle'
            ]
            
            objects_for_vlm = builder.get_objects_for_vlm_annotation(
                all_nodes,
                filter_classes=dynamic_classes,
                min_visibility=2
            )
            
            print(f"\nObjects ready for VLM annotation: {len(objects_for_vlm)}")
            
            # Save objects for VLM annotation
            vlm_output_path = os.path.join(output_dir, f'{scene_token}/vlm_objects.json')
            with open(vlm_output_path, 'w') as f:
                json.dump(objects_for_vlm, f, indent=2)
            
            print(f"VLM annotation list saved to {vlm_output_path}")


if __name__ == "__main__":
    main()
"""
Visualization utilities for nuScenes Lidar Segmentation

This module provides functions to visualize:
- LiDAR point clouds with segmentation
- Scene graphs with objects and relationships
- Object bounding boxes
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple
import json

try:
    import cv2
except ImportError:
    print("Warning: opencv-python not installed. Some visualization features may not work.")

from nuscenes_dataloader import FrameData, ObjectProperties
from scenegraph import SceneGraphNode, SceneGraphRelationship


def visualize_lidar_segmentation(
    frame: FrameData,
    output_path: Optional[str] = None,
    max_points: int = 10000,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize LiDAR point cloud with segmentation colors.
    
    Args:
        frame: FrameData object containing point cloud and segmentation
        output_path: Optional path to save the figure
        max_points: Maximum number of points to display (for performance)
        figsize: Figure size (width, height)
    """
    fig = plt.figure(figsize=figsize)
    
    # Sample points if too many
    points = frame.lidar_points
    seg_labels = frame.segmentation_labels
    
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[indices]
        seg_labels = seg_labels[indices]
    
    # Top-down view (Bird's Eye View)
    ax1 = fig.add_subplot(221)
    scatter1 = ax1.scatter(
        points[:, 0],  # x
        points[:, 1],  # y
        c=seg_labels,
        cmap='tab20',
        s=1,
        alpha=0.6
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top-Down View (Bird\'s Eye View)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Instance ID')
    
    # Front view
    ax2 = fig.add_subplot(222)
    scatter2 = ax2.scatter(
        points[:, 0],  # x
        points[:, 2],  # z
        c=seg_labels,
        cmap='tab20',
        s=1,
        alpha=0.6
    )
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Front View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Instance ID')
    
    # Side view
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(
        points[:, 1],  # y
        points[:, 2],  # z
        c=seg_labels,
        cmap='tab20',
        s=1,
        alpha=0.6
    )
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Instance ID')
    
    # 3D view
    ax4 = fig.add_subplot(224, projection='3d')
    scatter4 = ax4.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=seg_labels,
        cmap='tab20',
        s=1,
        alpha=0.4
    )
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D View')
    
    # Add frame info
    fig.suptitle(
        f'LiDAR Segmentation - Frame {frame.sample_token}\n'
        f'Points: {frame.lidar_points.shape[0]:,} | Objects: {len(frame.objects)}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_bev_with_boxes(
    frame: FrameData,
    output_path: Optional[str] = None,
    max_points: int = 20000,
    figsize: Tuple[int, int] = (12, 12),
    x_range: Tuple[float, float] = (-50, 50),
    y_range: Tuple[float, float] = (-50, 50)
):
    """
    Visualize Bird's Eye View with bounding boxes.
    
    Args:
        frame: FrameData object
        output_path: Optional path to save the figure
        max_points: Maximum number of points to display
        figsize: Figure size
        x_range: X-axis range (min, max) in meters
        y_range: Y-axis range (min, max) in meters
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sample and filter points
    points = frame.lidar_points
    seg_labels = frame.segmentation_labels
    
    # Filter by range
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    )
    points = points[mask]
    seg_labels = seg_labels[mask]
    
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[indices]
        seg_labels = seg_labels[indices]
    
    # Plot points
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=seg_labels,
        cmap='tab20',
        s=0.5,
        alpha=0.5
    )
    
    # Plot bounding boxes
    for obj in frame.objects:
        x, y, z = obj.position
        w, l, h = obj.size
        
        # Skip if outside range
        if not (x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]):
            continue
        
        # Draw bounding box (simplified as rectangle in BEV)
        rect = Rectangle(
            (x - w/2, y - l/2),
            w, l,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label
        label = obj.name.split('.')[-1]  # Get last part of class name
        ax.text(
            x, y + l/2 + 2,
            label,
            fontsize=8,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )
    
    # Plot ego vehicle
    ego_rect = Rectangle(
        (-2, -1), 4, 2,
        linewidth=2,
        edgecolor='blue',
        facecolor='lightblue',
        alpha=0.5
    )
    ax.add_patch(ego_rect)
    ax.text(0, 0, 'EGO', fontsize=10, ha='center', va='center', fontweight='bold')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f'Bird\'s Eye View - Frame {frame.sample_token}\n'
        f'Objects: {len(frame.objects)}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.colorbar(scatter, ax=ax, label='Instance ID')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"BEV visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_scene_graph(
    nodes: List[SceneGraphNode],
    relationships: List[SceneGraphRelationship],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Visualize scene graph with objects and relationships.
    
    Args:
        nodes: List of scene graph nodes
        relationships: List of relationships
        output_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Spatial layout
    node_positions = {}
    colors = []
    sizes = []
    
    # Define colors for different object classes
    class_colors = {
        'vehicle': 'red',
        'human': 'blue',
        'movable': 'green',
        'static': 'gray',
        'animal': 'orange'
    }
    
    for node in nodes:
        x, y, z = node.position
        node_positions[node.object_id] = (x, y)
        
        # Determine color by class
        color = 'purple'  # default
        for key, col in class_colors.items():
            if key in node.object_class.lower():
                color = col
                break
        colors.append(color)
        
        # Size based on actual object size
        w, l, h = node.size
        size = (w * l) * 100  # Scale for visibility
        sizes.append(size)
    
    # Plot nodes
    xs = [pos[0] for pos in node_positions.values()]
    ys = [pos[1] for pos in node_positions.values()]
    
    ax1.scatter(xs, ys, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Plot relationships
    for rel in relationships:
        if rel.source_id in node_positions and rel.target_id in node_positions:
            x1, y1 = node_positions[rel.source_id]
            x2, y2 = node_positions[rel.target_id]
            
            # Different line styles for different relationships
            if rel.relationship_type == 'near':
                style = '-'
                color = 'green'
                alpha = 0.3
            else:
                style = '--'
                color = 'blue'
                alpha = 0.2
            
            ax1.plot([x1, x2], [y1, y2], style, color=color, alpha=alpha, linewidth=1)
    
    # Add ego vehicle
    ax1.scatter([0], [0], marker='*', s=500, c='gold', edgecolors='black', linewidth=2, zorder=10)
    ax1.text(0, 0, 'EGO', fontsize=10, ha='center', va='center', fontweight='bold')
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Spatial Layout with Relationships', fontsize=12, fontweight='bold')
    
    # Right plot: Statistics
    ax2.axis('off')
    
    # Count statistics
    class_counts = {}
    for node in nodes:
        base_class = node.object_class.split('.')[0]
        class_counts[base_class] = class_counts.get(base_class, 0) + 1
    
    rel_type_counts = {}
    for rel in relationships:
        rel_type_counts[rel.relationship_type] = rel_type_counts.get(rel.relationship_type, 0) + 1
    
    # Display statistics
    stats_text = "Scene Graph Statistics\n" + "="*40 + "\n\n"
    stats_text += f"Total Objects: {len(nodes)}\n"
    stats_text += f"Total Relationships: {len(relationships)}\n\n"
    
    stats_text += "Object Classes:\n"
    for obj_class, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        stats_text += f"  {obj_class:20s}: {count:3d}\n"
    
    stats_text += "\nRelationship Types:\n"
    for rel_type, count in sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True):
        stats_text += f"  {rel_type:15s}: {count:3d}\n"
    
    # Legend for colors
    stats_text += "\nColor Legend:\n"
    for key, color in class_colors.items():
        stats_text += f"  {key.capitalize():15s}: {color}\n"
    
    ax2.text(
        0.1, 0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    frame_idx = nodes[0].frame_idx if nodes else 0
    fig.suptitle(
        f'Scene Graph Visualization - Frame {frame_idx}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Scene graph visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_objects_on_camera(
    frame: FrameData,
    dataloader,
    output_path: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    show_bbox: bool = True,
    show_labels: bool = True,
    show_properties: bool = True
):
    """
    Visualize objects with their categories on camera images.
    
    Args:
        frame: FrameData object containing objects and camera info
        dataloader: NuScenesLidarSegmentationLoader instance
        output_path: Optional directory to save visualizations
        cameras: List of cameras to visualize (None for all)
        show_bbox: Whether to show bounding boxes
        show_labels: Whether to show object labels
        show_properties: Whether to show object properties (velocity, etc.)
    """
    from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
    from pyquaternion import Quaternion
    import os
    from pathlib import Path
    
    nusc = dataloader.nusc
    
    # Default to all cameras if not specified
    if cameras is None:
        cameras = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
    
    # Filter cameras that exist in this frame
    available_cameras = [cam for cam in cameras if cam in frame.camera_tokens]
    
    if not available_cameras:
        print("No cameras available in this frame")
        return
    
    # Create output directory if saving
    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each camera
    for camera in available_cameras:
        camera_token = frame.camera_tokens[camera]
        
        # Load camera image
        cam_data = nusc.get('sample_data', camera_token)
        img_path = dataloader.dataroot / cam_data['filename']
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Failed to load image: {img_path}")
            continue
        
        # Get camera intrinsics and extrinsics
        cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        
        # Get objects visible in this camera
        visible_objects = [obj for obj in frame.objects if obj.visible_cameras and camera in obj.visible_cameras]
        
        # Draw each object
        for obj in visible_objects:
            try:
                # Get the 3D box
                annotation = nusc.get('sample_annotation', obj.token)
                box = nusc.get_box(obj.token)
                
                # Move box to sensor coordinate system
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)
                
                # Check if box is in front of camera
                if not np.any(box.corners()[2, :] > 0):
                    continue
                
                # Project 3D box to 2D
                corners_3d = box.corners()
                corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
                
                # Get image dimensions
                h, w = image.shape[:2]
                
                # Check if box is visible in image
                if not np.all(corners_2d[0, :] >= 0) or not np.all(corners_2d[0, :] < w) or \
                   not np.all(corners_2d[1, :] >= 0) or not np.all(corners_2d[1, :] < h):
                    # Box partially or fully outside image, but we can still try to draw visible parts
                    pass
                
                # Draw 3D bounding box
                if show_bbox:
                    # Define box edges (which corners to connect)
                    edges = [
                        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                    ]
                    
                    # Choose color based on object class
                    if 'vehicle' in obj.name:
                        color = (0, 0, 255)  # Red
                    elif 'pedestrian' in obj.name or 'human' in obj.name:
                        color = (255, 0, 0)  # Blue
                    elif 'bicycle' in obj.name or 'motorcycle' in obj.name:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 255, 0)  # Green
                    
                    # Draw edges
                    for edge in edges:
                        pt1 = (int(corners_2d[0, edge[0]]), int(corners_2d[1, edge[0]]))
                        pt2 = (int(corners_2d[0, edge[1]]), int(corners_2d[1, edge[1]]))
                        
                        # Only draw if points are within reasonable bounds
                        if 0 <= pt1[0] < w*2 and 0 <= pt1[1] < h*2 and \
                           0 <= pt2[0] < w*2 and 0 <= pt2[1] < h*2:
                            cv2.line(image, pt1, pt2, color, 2)
                
                # Draw label and properties
                if show_labels or show_properties:
                    # Get position for label (center-bottom of box in 2D)
                    center_2d = np.mean(corners_2d, axis=1)
                    label_x = int(center_2d[0])
                    label_y = int(np.max(corners_2d[1, :]))  # Bottom of box
                    
                    # Ensure label is within image bounds
                    label_x = max(10, min(label_x, w - 10))
                    label_y = max(30, min(label_y, h - 10))
                    
                    # Prepare label text
                    class_name = obj.name.split('.')[-1].replace('_', ' ')
                    
                    lines = []
                    if show_labels:
                        lines.append(class_name)
                    
                    if show_properties:
                        # Add velocity if available
                        if obj.velocity and obj.velocity != (0.0, 0.0):
                            speed = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
                            if speed > 0.5:
                                lines.append(f"{speed:.1f} m/s")
                        
                        # Add distance
                        distance = np.sqrt(obj.position[0]**2 + obj.position[1]**2)
                        lines.append(f"{distance:.1f}m")
                    
                    # Draw background box for text
                    if lines:
                        # Calculate text size
                        max_width = 0
                        total_height = 0
                        line_heights = []
                        
                        for line in lines:
                            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            max_width = max(max_width, text_w)
                            line_heights.append(text_h)
                            total_height += text_h + 5
                        
                        # Draw semi-transparent background
                        bg_x1 = label_x - 5
                        bg_y1 = label_y - total_height - 5
                        bg_x2 = label_x + max_width + 5
                        bg_y2 = label_y + 5
                        
                        # Ensure background is within image
                        bg_x1 = max(0, bg_x1)
                        bg_y1 = max(0, bg_y1)
                        bg_x2 = min(w, bg_x2)
                        bg_y2 = min(h, bg_y2)
                        
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
                            cv2.putText(image, line, (label_x, y_pos),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Error drawing object {obj.token} in {camera}: {e}")
                continue
        
        # Add camera name and frame info
        cv2.putText(image, camera, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        info_text = f"Frame: {frame.sample_token[:8]}... | Objects: {len(visible_objects)}/{len(frame.objects)}"
        cv2.putText(image, info_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save or show image
        if output_path:
            output_file = output_dir / f"{frame.sample_token}_{camera}.jpg"
            cv2.imwrite(str(output_file), image)
            print(f"Saved: {output_file}")
        else:
            # Display image
            cv2.imshow(f"{camera} - {frame.sample_token[:8]}", image)
    
    # Wait for key press if displaying
    if not output_path:
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_object_statistics(
    scene_graph_json_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize statistics from a scene graph JSON file.
    
    Args:
        scene_graph_json_path: Path to scene graph JSON file
        output_path: Optional path to save the figure
        figsize: Figure size
    """
    # Load scene graph
    with open(scene_graph_json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    
    # Collect statistics
    objects_per_frame = []
    relationships_per_frame = []
    class_counts = {}
    velocity_data = []
    
    for frame in frames:
        objects_per_frame.append(len(frame['objects']))
        relationships_per_frame.append(len(frame['relationships']))
        
        for obj in frame['objects']:
            class_name = obj['object_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            if obj['velocity'] is not None:
                velocity = np.linalg.norm(obj['velocity'])
                velocity_data.append((class_name, velocity))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Objects per frame
    ax1 = axes[0, 0]
    ax1.plot(objects_per_frame, linewidth=2, color='blue')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Number of Objects')
    ax1.set_title('Objects per Frame')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relationships per frame
    ax2 = axes[0, 1]
    ax2.plot(relationships_per_frame, linewidth=2, color='green')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Number of Relationships')
    ax2.set_title('Relationships per Frame')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Object class distribution
    ax3 = axes[1, 0]
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1][:10]  # Top 10
    top_classes = [classes[i].split('.')[-1] for i in sorted_indices]
    top_counts = [counts[i] for i in sorted_indices]
    
    ax3.barh(top_classes, top_counts, color='coral')
    ax3.set_xlabel('Count')
    ax3.set_title('Top 10 Object Classes')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Velocity distribution
    ax4 = axes[1, 1]
    if velocity_data:
        velocities = [v for _, v in velocity_data]
        ax4.hist(velocities, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Velocity (m/s)')
        ax4.set_ylabel('Count')
        ax4.set_title('Object Velocity Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=ax4.transAxes)
    
    fig.suptitle('Scene Graph Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Statistics visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Example usage of visualization functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize nuScenes data')
    parser.add_argument('--mode', choices=['lidar', 'bev', 'graph', 'stats', 'camera'], required=True,
                       help='Visualization mode')
    parser.add_argument('--dataroot', type=str, 
                       help='Path to nuScenes dataset (required for lidar, bev, camera modes)')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       help='Dataset version (default: v1.0-trainval)')
    parser.add_argument('--scene-idx', type=int, default=0,
                       help='Scene index (default: 0)')
    parser.add_argument('--frame-idx', type=int, default=0,
                       help='Frame index within scene (default: 0)')
    parser.add_argument('--input', type=str,
                       help='Input file (scene graph JSON for stats/graph modes)')
    parser.add_argument('--output', type=str, help='Output path for saving visualization')
    parser.add_argument('--cameras', type=str, nargs='+', default=['CAM_FRONT'],
                       help='Cameras to visualize (for camera mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'stats':
        if not args.input:
            print("Error: --input required for stats mode (path to scene graph JSON)")
            return
        visualize_object_statistics(args.input, args.output)
        
    elif args.mode == 'graph':
        if not args.input:
            print("Error: --input required for graph mode (path to scene graph JSON)")
            return
        # Load scene graph JSON
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        if not data['frames']:
            print("Error: No frames in scene graph")
            return
        
        # Visualize first frame
        frame_data = data['frames'][0]
        
        # Convert to SceneGraphNode and SceneGraphRelationship objects
        from scenegraph import SceneGraphNode, SceneGraphRelationship
        
        nodes = []
        for obj in frame_data['objects']:
            # Try different field names for object ID
            object_id = obj.get('instance_token') or obj.get('token') or obj.get('object_id', 'unknown')
            
            node = SceneGraphNode(
                object_id=object_id,
                object_class=obj.get('object_class', 'unknown'),
                position=tuple(obj.get('position', (0, 0, 0))),
                size=tuple(obj.get('size', (1, 1, 1))),
                velocity=tuple(obj['velocity']) if obj.get('velocity') else None,
                attributes=obj.get('attributes', []),
                frame_idx=frame_data.get('frame_idx', 0),
                timestamp=frame_data.get('timestamp', 0),
                num_lidar_pts=obj.get('num_lidar_pts', 0),
                visibility=obj.get('visibility', 0)
            )
            nodes.append(node)
        
        relationships = []
        for rel in frame_data.get('relationships', []):
            relationship = SceneGraphRelationship(
                source_id=rel['source_id'],
                target_id=rel['target_id'],
                relationship_type=rel['relationship_type'],
                distance=rel.get('distance'),
                frame_idx=rel.get('frame_idx', 0)
            )
            relationships.append(relationship)
        
        visualize_scene_graph(nodes, relationships, args.output)
        
    elif args.mode in ['lidar', 'bev', 'camera']:
        if not args.dataroot:
            print(f"Error: --dataroot required for {args.mode} mode")
            return
        
        # Load nuScenes data
        from nuscenes_dataloader import NuScenesLidarSegmentationLoader
        
        print(f"Loading nuScenes dataset from {args.dataroot}...")
        loader = NuScenesLidarSegmentationLoader(
            dataroot=args.dataroot,
            version=args.version,
            verbose=False
        )
        
        # Get scene and frame
        scene_tokens = loader.get_scene_tokens()
        if args.scene_idx >= len(scene_tokens):
            print(f"Error: Scene index {args.scene_idx} out of range (0-{len(scene_tokens)-1})")
            return
        
        scene_token = scene_tokens[args.scene_idx]
        sample_tokens = loader.get_samples_in_scene(scene_token)
        
        if args.frame_idx >= len(sample_tokens):
            print(f"Error: Frame index {args.frame_idx} out of range (0-{len(sample_tokens)-1})")
            return
        
        sample_token = sample_tokens[args.frame_idx]
        
        print(f"Loading frame {args.frame_idx} from scene {args.scene_idx}...")
        frame = loader.get_frame_data(sample_token)
        
        print(f"Frame: {frame.sample_token}")
        print(f"Objects: {len(frame.objects)}")
        
        # Perform visualization
        if args.mode == 'lidar':
            print("Visualizing LiDAR segmentation...")
            visualize_lidar_segmentation(frame, args.output)
        elif args.mode == 'bev':
            print("Visualizing Bird's Eye View...")
            visualize_bev_with_boxes(frame, args.output)
        elif args.mode == 'camera':
            print(f"Visualizing objects on cameras: {args.cameras}...")
            visualize_objects_on_camera(
                frame=frame,
                dataloader=loader,
                output_path=args.output,
                cameras=args.cameras
            )
    else:
        print(f"Unknown mode: {args.mode}")
        print("Available modes: lidar, bev, camera, graph, stats")


if __name__ == "__main__":
    main()


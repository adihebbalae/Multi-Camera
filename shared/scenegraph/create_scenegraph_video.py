"""
Create videos from scene graph JSON with objects overlaid on camera images.

This script reads a scene graph JSON file and creates MP4 videos showing
all detected objects with their bounding boxes and labels on camera images.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
import traceback

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
        "--scene-token",
        required=True,
        help="Scene token"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing scene_graphs/, captions/, instance_annotations/. If not set, uses --output-dir."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output videos will be written (videos/ subdir will be created here)"
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
    parser.add_argument(
        "--panoramic",
        action="store_true",
        help="Export a single video with all camera views stitched in panoramic order (front-left → front → front-right → back-right → back → back-left)"
    )
    
    return parser.parse_args()


# Camera order for panoramic stitch (left-to-right = clockwise around vehicle)
PANORAMA_CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


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
    corners_2d: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
    thickness: int = 2
):
    """Draw 3D bounding box on image."""
    h, w = image.shape[:2]
    
    # Define box edges
    # edges = [
    #     [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
    #     [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
    #     [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    #]
    
    # Draw edges
    # # for edge in edges:
    lines = [
       [[0, 1], [2,1]],
       [[0, 1], [0,3]],
       [[2, 3], [2,1]],
       [[2, 3], [0,3]]# Bottom face
    ]
    for line in lines:
        pt1 = (int(corners_2d[line[0][0]]), int(corners_2d[line[0][1]]))
        pt2 = (int(corners_2d[line[1][0]]), int(corners_2d[line[1][1]]))
        
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
    thickness: int,
    annotations: Dict = None,
):
    """Draw object label with properties."""
    h, w = image.shape[:2]
    
    # Get label position (center-bottom of box)
    center_2d = (corners_2d[0] + corners_2d[2]) / 2,(corners_2d[1] + corners_2d[3]) / 2
    label_x = int(center_2d[0])
    label_y = int(corners_2d[1])
    
    # Ensure label is within image bounds
    label_x = max(10, min(label_x, w - 10))
    label_y = max(30, min(label_y, h - 10))
    
    # Prepare label text
    class_name = obj['object_class'].split('.')[-1].replace('_', ' ')
    
    lines = [class_name]
    
    if show_properties:
        # Add velocity if available
        # if obj.get('velocity') and obj['velocity'] != [0.0, 0.0]:
        #     speed = np.sqrt(obj['velocity'][0]**2 + obj['velocity'][1]**2)
        #     if speed > 0.5:
        #         lines.append(f"{speed:.1f} m/s")
        
        # # Add distance
        # if obj.get('position'):
        #     distance = np.sqrt(obj['position'][0]**2 + obj['position'][1]**2)
        #     lines.append(f"{distance:.1f}m")
        if annotations is not None:
            if obj.get('activity') or annotations.get('activity'):
                lines.append(f"activity : {annotations['activity']}")
            if obj.get('description') or annotations.get('description'):
                lines.append(f"description : {annotations['description']}")

    
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

def post_process_coords(corner_coords: List,
                        imsize: Tuple[int, int] = (1600, 900)) -> Union[Tuple[float, float, float, float], None]:
    """
    Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no
    intersection.
    :param corner_coords: Corner coordinates of reprojected bounding box.
    :param imsize: Size of the image canvas.
    :return: Intersection of the convex hull of the 2D box corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
    
def draw_frame_info(
    image: np.ndarray,
    camera: str,
    frame_idx: int,
    total_frames: int,
    num_objects: int,
    font_scale: float,
    thickness: int,
    captions: Dict = None
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
    if captions is not None:
        info_text += f" | Caption: {captions}"
    info_text = info_text.replace('\n', ' ')
    info_texts = info_text.split('.')
    info_texts = info_texts[::-1]
    for i, info_text in enumerate(info_texts):
        cv2.putText(
            image, info_text, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
        )
        h -= 15


def render_frame_for_camera(
    nusc: NuScenes,
    scene_graph: Dict,
    frame: Dict,
    frame_idx: int,
    total_frames: int,
    camera: str,
    args: argparse.Namespace,
    annotations: Dict = None,
    captions: Dict = None,
) -> Optional[np.ndarray]:
    """
    Render a single frame for one camera: load image, draw boxes/labels, return BGR image.
    Returns None if the sample/camera/image cannot be loaded.
    """
    sample_token = frame["sample_token"]
    try:
        sample = nusc.get("sample", sample_token)
    except Exception:
        return None
    if camera not in sample["data"]:
        return None
    camera_token = sample["data"][camera]
    cam_data = nusc.get("sample_data", camera_token)
    img_path = Path(args.dataroot) / cam_data["filename"]
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    image = image.copy()

    ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
    cs_record = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    cam_intrinsic = np.array(cs_record["camera_intrinsic"])

    visible_objects = [
        obj
        for obj in frame["objects"]
        if obj.get("visible_cameras") and camera in obj["visible_cameras"]
    ]

    for obj in visible_objects:
        try:
            box = nusc.get_box(obj["annotation_token"])
            box.translate(-np.array(ego_pose["translation"]))
            box.rotate(Quaternion(ego_pose["rotation"]).inverse)
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)
            corners_3d = box.corners()
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            corners = corners_3d[:, in_front]
            corner_coords = view_points(corners, cam_intrinsic, normalize=True).T[:, :2]
            corners_2d = post_process_coords(corner_coords)
            if corners_2d is None:
                continue
            color = get_object_color(obj["object_class"])
            if args.show_bbox:
                draw_3d_box(image, corners_2d, color, args.thickness)
            if args.show_labels:
                obj_annotations = annotations.get(obj["object_id"]) if annotations else None
                draw_object_label(
                    image,
                    np.array(corners_2d),
                    obj,
                    color,
                    args.show_properties,
                    args.font_scale,
                    args.thickness,
                    obj_annotations,
                )
        except Exception:
            continue

    caption = captions.get(frame["sample_token"]) if captions else None
    draw_frame_info(
        image,
        camera,
        frame_idx,
        total_frames,
        len(visible_objects),
        args.font_scale,
        args.thickness,
        caption,
    )
    return image


def create_video_for_camera(
    nusc: NuScenes,
    scene_graph: Dict,
    camera: str,
    output_path: str,
    args: argparse.Namespace,
    annotations: Dict = None,
    captions: Dict = None
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
    
    writer = None
    total_frames = len(frames)
    
    try:
        for frame_idx, frame in enumerate(tqdm(frames, desc=f"  {camera}")):
            image = render_frame_for_camera(
                nusc, scene_graph, frame, frame_idx, total_frames,
                camera, args, annotations, captions
            )
            if image is None:
                continue
            if writer is None:
                h, w = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, args.fps, (w, h))
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
            print(f"  Saved: {output_path}")
        else:
            print(f"  No frames written for {camera}")


def _panorama_camera_list(requested: List[str]) -> List[str]:
    """Return requested cameras in panoramic order (front-left → … → back-left)."""
    ordered = [c for c in PANORAMA_CAMERA_ORDER if c in requested]
    # Add any requested cameras not in the default order at the end
    for c in requested:
        if c not in ordered:
            ordered.append(c)
    return ordered


def create_panoramic_video(
    nusc: NuScenes,
    scene_graph: Dict,
    output_path: str,
    args: argparse.Namespace,
    annotations: Dict = None,
    captions: Dict = None,
):
    """Create a single video with all camera views stitched in panoramic order."""
    frames = scene_graph["frames"]
    if not frames:
        print("  No frames found in scene graph")
        return

    cameras = _panorama_camera_list(args.cameras)
    print(f"\nCreating panoramic video with cameras (left→right): {cameras}")

    writer = None
    total_frames = len(frames)
    target_height = 400  # uniform height for panorama strip

    try:
        for frame_idx, frame in enumerate(tqdm(frames, desc="  Panorama")):
            tiles = []
            for camera in cameras:
                img = render_frame_for_camera(
                    nusc,
                    scene_graph,
                    frame,
                    frame_idx,
                    total_frames,
                    camera,
                    args,
                    annotations,
                    captions,
                )
                if img is None:
                    # Placeholder so layout stays fixed
                    img = np.zeros((target_height, 320, 3), dtype=np.uint8)
                    img[:] = (40, 40, 40)
                    cv2.putText(
                        img,
                        camera,
                        (20, target_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 180, 180),
                        2,
                    )
                else:
                    h, w = img.shape[:2]
                    scale = target_height / h
                    new_w = int(round(w * scale))
                    img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
                tiles.append(img)
            stitched = np.concatenate(tiles, axis=1)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    args.fps,
                    (stitched.shape[1], stitched.shape[0]),
                )
            writer.write(stitched)
    finally:
        if writer is not None:
            writer.release()
            print(f"  Saved: {output_path}")
        else:
            print("  No frames written for panoramic video")


def load_annotations(annotations_json: str) -> Dict:
    """Load annotations from JSON file."""
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    instance_annotations = data['annotations']
    annotations = {}
    for annotation in instance_annotations:
        annotations[annotation['instance_token']] = {'activity': annotation['activity'].split('.')[0], 'description': annotation['description'].split('.')[0]}

    return annotations

def load_captions(captions_json: str) -> Dict:
    """Load captions from JSON file."""
    with open(captions_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions_data = data['captions']
    captions = {}
    for caption in captions_data:
        captions[caption['sample_token']] = caption['caption']

    return captions

def main():
    args = parse_args()
    
    print("="*60)
    print("Scene Graph Video Creator")
    print("="*60)
    
    # Load nuScenes
    print(f"\nLoading nuScenes from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    # Where to read scene graph, annotations, captions from
    data_dir = args.data_dir if args.data_dir is not None else args.output_dir
    args.input_json = os.path.join(data_dir, f"scene_graphs/{args.scene_token}/scene_graph.json")
    args.annotations_json = os.path.join(data_dir, f"instance_annotations/{args.scene_token}_instance_annotations.json")
    args.captions_json = os.path.join(data_dir, f"captions/{args.scene_token}_captions.json")
    print(f"Loading scene graph from {args.input_json}...")
    scene_graph = load_scene_graph(args.input_json)
    annotations = load_annotations(args.annotations_json)
    captions = load_captions(args.captions_json)
    
    print(f"Scene: {scene_graph.get('scene_name', 'unknown')}")
    print(f"Frames: {scene_graph.get('num_frames', 0)}")
    print(f"Annotations: {scene_graph.get('num_annotations', 0)}")
    
    # Where to write videos (may be different from data_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    
    # Get scene name for output filename
    scene_name = scene_graph.get('scene_name', scene_graph.get('scene_token', 'scene'))
    
    # When panoramic, default to all 6 cameras in spatial order
    if args.panoramic and args.cameras == ["CAM_FRONT"]:
        args.cameras = list(PANORAMA_CAMERA_ORDER)
    
    if args.panoramic:
        # Single stitched panoramic video (all requested cameras in spatial order)
        output_path = output_dir / "videos" / f"{scene_name}_panoramic.mp4"
        try:
            create_panoramic_video(
                nusc=nusc,
                scene_graph=scene_graph,
                output_path=str(output_path),
                args=args,
                annotations=annotations,
                captions=captions,
            )
        except Exception as e:
            print(f"\nError creating panoramic video: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Create video for each camera
        for camera in args.cameras:
            output_path = output_dir / "videos" / f"{scene_name}_{camera}.mp4"
            try:
                create_video_for_camera(
                    nusc=nusc,
                    scene_graph=scene_graph,
                    camera=camera,
                    output_path=str(output_path),
                    args=args,
                    annotations=annotations,
                    captions=captions
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


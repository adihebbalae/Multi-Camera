#!/usr/bin/env python3
"""
Render question validation videos: multi-camera grid with geom overlays.

Creates a video showing all cameras in a question side-by-side/grid, with:
- Bounding boxes from geom.yml overlayed
- Time span limited to the answer frame range
- Camera labels and timestamps

Usage:
    python3 render_question_validation.py --qa-file qa.json --question-id 0 --output validated.mp4
    python3 render_question_validation.py --slot 2018-03-07.17-05.school --question-id 5 -v
"""

import argparse
import json
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import subprocess
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Tuple

# Paths
MEVA_MP4_ROOT = Path("/nas/mars/dataset/MEVA/mp4s")
KITWARE_BASE = Path("/nas/mars/dataset/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware")
KITWARE_TRAINING_BASE = Path("/nas/mars/dataset/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware-meva-training")
QA_OUTPUT_DIR = Path("/home/ah66742/data/qa_pairs")
VIDEO_OUTPUT_DIR = Path("/home/ah66742/output/validation_videos")

VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color constants
COLOR_BOX = (0, 255, 0)     # Green
COLOR_TEXT = (255, 255, 255)  # White
COLOR_BGND = (50, 50, 50)   # Dark gray background

# Geom parsing (copied from extract_entity_descriptions.py)
_RE_ID1 = re.compile(r"['\"]?id1['\"]?\s*:\s*['\"]?(\d+)")
_RE_TS0 = re.compile(r"['\"]?ts0['\"]?\s*:\s*['\"]?(\d+)")
_RE_G0  = re.compile(r"['\"]?g0['\"]?\s*:\s*['\"]?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")


def parse_geom_boxes(geom_file: Path) -> Dict[int, list]:
    """Extract bounding boxes from geom.yml by frame number.
    
    Returns {frame_number: [(x1, y1, x2, y2, actor_id), ...]}
    """
    boxes_by_frame = defaultdict(list)
    if not geom_file.exists():
        return boxes_by_frame
    
    with open(geom_file) as f:
        content = f.read()
    
    # Parse lines (each line is a geom entry)
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Extract matches from this line
        match_id1 = _RE_ID1.search(line)
        match_ts0 = _RE_TS0.search(line)
        match_g0 = _RE_G0.search(line)
        
        if match_id1 and match_ts0 and match_g0:
            actor_id = int(match_id1.group(1))
            frame = int(match_ts0.group(1))
            x1, y1, x2, y2 = map(int, match_g0.groups())
            boxes_by_frame[frame].append((x1, y1, x2, y2, actor_id))
    
    return dict(boxes_by_frame)


def find_geom_file(date: str, hour: str, start_time: str, site: str, camera: str) -> Path:
    """Find geom.yml file for a camera."""
    prefix = f"{date}.{start_time}"
    for kitware_dir in [KITWARE_BASE, KITWARE_TRAINING_BASE]:
        ann_dir = kitware_dir / date / hour
        if not ann_dir.exists():
            continue
        matches = list(ann_dir.glob(f"{prefix}*.{site}.{camera}.geom.yml"))
        if matches:
            return matches[0]
    return None


def find_mp4(date: str, hour: str, start_time: str, site: str, end_time: str, camera: str) -> Path:
    """Find MP4 file for a camera."""
    slot_dir = MEVA_MP4_ROOT / date / hour / f"{date}.{start_time}.{site}"
    if not slot_dir.exists():
        return None
    
    # Try to find exact match or fuzzy match
    pattern = f"{date}.{start_time}*.{end_time}*.{site}.{camera}*.r13.mp4"
    matches = list(slot_dir.glob(pattern))
    if matches:
        return matches[0]
    return None


def extract_clip_timing(clip_file: str) -> tuple:
    """Extract date, hour, start_time, end_time, site from clip filename."""
    # Format: 2018-03-07.17-05-00.17-10-00.school.G330.r13.mp4
    parts = clip_file.replace(".r13.mp4", "").split(".")
    if len(parts) >= 5:
        date = parts[0]
        start_time = parts[1]
        end_time = parts[2]
        site = parts[3]
        hour = start_time.split("-")[0]
        return date, hour, start_time, end_time, site
    return None


def load_video_frames(mp4_path: Path, frame_start: int, frame_end: int, target_w: int = 640, target_h: int = 360):
    """Load video frames as numpy arrays, resized to target dimensions."""
    if not mp4_path.exists():
        return None, None
    
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    for i in range(frame_end - frame_start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to target
        frame = cv2.resize(frame, (target_w, target_h))
        frames.append(frame)
    
    cap.release()
    return frames, fps


def overlay_boxes(frame: np.ndarray, boxes: list, actor_ids: list = None) -> np.ndarray:
    """Draw bounding boxes on frame."""
    output = frame.copy()
    for box in boxes:
        x1, y1, x2, y2, actor_id = box
        # Scale boxes to match resized frame (this is approximate — better to pass scale factor)
        label = f"A{actor_id}"
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_BOX, 2)
        cv2.putText(output, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    return output


def compose_grid(frames_dict: Dict[str, list], cameras: list, fps: float = 30) -> list:
    """Compose multiple camera views into a grid layout.
    
    Returns list of composed frames (all cameras aligned by frame index).
    """
    # Grid layout: 2x2 for 4 cameras, 1x2 for 2, etc.
    n_cams = len(cameras)
    if n_cams <= 2:
        grid_h, grid_w = 1, n_cams
    elif n_cams <= 4:
        grid_h, grid_w = 2, 2
    elif n_cams <= 6:
        grid_h, grid_w = 2, 3
    else:
        grid_h, grid_w = 3, 3
    
    # Frame dimensions
    frame_h, frame_w = 360, 640
    comp_h = grid_h * frame_h + 30 * (grid_h + 1)  # 30px padding
    comp_w = grid_w * frame_w + 30 * (grid_w + 1)
    
    # Get the number of frames (min across all cameras)
    n_frames = min(len(v) for v in frames_dict.values() if v) if frames_dict else 0
    
    composed_frames = []
    for frame_idx in range(n_frames):
        # Create blank canvas
        canvas = np.ones((comp_h, comp_w, 3), dtype=np.uint8) * 50
        
        for cam_idx, camera in enumerate(cameras):
            if camera not in frames_dict or not frames_dict[camera]:
                continue
            
            frame = frames_dict[camera][frame_idx]
            grid_row = cam_idx // grid_w
            grid_col = cam_idx % grid_w
            
            x = 30 + grid_col * (frame_w + 30)
            y = 30 + grid_row * (frame_h + 30)
            
            # Place frame
            canvas[y:y + frame_h, x:x + frame_w] = frame
            
            # Add camera label
            label = f"{camera} [{frame_idx+1}/{n_frames}]"
            cv2.putText(canvas, label, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        
        composed_frames.append(canvas)
    
    return composed_frames


def write_video(frames: list, output_path: Path, fps: float = 30) -> bool:
    """Write frames to MP4 video file."""
    if not frames:
        return False
    
    h, w = frames[0].shape[:2]
    
    # Use ffmpeg for better codec support
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-y',
        str(output_path)
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    
    return output_path.exists()


def render_question(qa_data: dict, question_idx: int, output_path: Path = None):
    """Render a single question into a validation video.
    
    Collects all events from debug_info and renders them in a single timeline grid.
    """
    if question_idx >= len(qa_data["qa_pairs"]):
        print(f"ERROR: Question index {question_idx} out of range (max {len(qa_data['qa_pairs']) - 1})")
        return None
    
    q = qa_data["qa_pairs"][question_idx]
    slot = qa_data["slot"]
    cameras = q.get("requires_cameras", [])
    
    print(f"\n{'='*60}")
    print(f"Rendering Q{question_idx}: {q['category'].upper()}")
    print(f"Question: {q['question_template'][:80]}...")
    print(f"Cameras: {cameras}")
    print(f"{'='*60}")
    
    # Parse debug_info to extract events
    debug = q.get("debug_info", {})
    if not debug:
        print("ERROR: No debug_info in question")
        return None
    
    # Collect events and their timing
    events = {}
    frame_ranges_by_camera = defaultdict(list)
    timings_by_camera = {}
    
    for key, value in debug.items():
        if key in ["gap_sec", "connection_type", "connection_strength", "connection_score", 
                   "relationship", "cluster_id", "mevid_validated", "mevid_person_id"]:
            continue  # Skip metadata keys
        
        if isinstance(value, dict) and "camera" in value:
            camera = value["camera"]
            frame_range = value.get("frame_range")
            clip_file = value.get("clip_file")
            fps = value.get("fps", 30.0)
            
            if frame_range and clip_file:
                events[key] = value
                frame_ranges_by_camera[camera].append((frame_range[0], frame_range[1], value.get("activity", key)))
                timings_by_camera[camera] = (clip_file, fps)
    
    if not events:
        print("ERROR: No events in debug_info")
        return None
    
    # Compute overall frame range for each camera (covers all events for that camera)
    frame_bounds = {}
    for camera, ranges in frame_ranges_by_camera.items():
        if ranges:
            min_frame = min(r[0] for r in ranges)
            max_frame = max(r[1] for r in ranges)
            frame_bounds[camera] = (min_frame, max_frame)
    
    print(f"Events: {len(events)} ({', '.join(events.keys())})")
    for camera, frame_range in frame_bounds.items():
        frame_start, frame_end = frame_range
        clip_file, fps = timings_by_camera[camera]
        duration_sec = (frame_end - frame_start) / fps
        print(f"  {camera}: frames {frame_start}-{frame_end} ({duration_sec:.1f}s) from {Path(clip_file).name}")
    
    # Load frames for each camera
    frames_dict = {}
    geom_boxes_dict = {}
    frame_offsets = {}  # Map camera -> (date, hour, start_time, end_time, site, frame_offset)
    
    for camera in frame_bounds:
        print(f"  Loading {camera}...", end="", flush=True)
        
        clip_file, fps = timings_by_camera[camera]
        timing = extract_clip_timing(clip_file)
        if not timing:
            print(f" [ERROR: Could not parse clip file: {clip_file}]")
            continue
        
        date, hour, start_time, end_time, site = timing
        frame_start, frame_end = frame_bounds[camera]
        
        # Find MP4
        mp4_path = find_mp4(date, hour, start_time, site, end_time, camera)
        if not mp4_path:
            print(f" [NOT FOUND]")
            continue
        
        # Load frames
        frames, vid_fps = load_video_frames(mp4_path, frame_start, frame_end)
        if not frames:
            print(" [FAILED TO LOAD]")
            continue
        
        frames_dict[camera] = frames
        frame_offsets[camera] = (date, hour, start_time, end_time, site, frame_start)
        
        # Load geom boxes
        geom_file = find_geom_file(date, hour, start_time, site, camera)
        if geom_file:
            boxes_by_frame = parse_geom_boxes(geom_file)
            geom_boxes_dict[camera] = boxes_by_frame
            print(f" [OK: {len(frames)} frames, {len(boxes_by_frame)} geom frames]")
        else:
            print(f" [OK: {len(frames)} frames, no geom]")
    
    if not frames_dict:
        print("ERROR: No frames loaded")
        return None
    
    # Overlay geom boxes on frames
    print(f"  Overlaying geom boxes...")
    for camera in frames_dict:
        boxes_by_frame = geom_boxes_dict.get(camera, {})
        frame_offset = frame_offsets[camera][5]  # frame_start offset
        for frame_idx, frame in enumerate(frames_dict[camera]):
            actual_frame_num = frame_offset + frame_idx
            if actual_frame_num in boxes_by_frame:
                frames_dict[camera][frame_idx] = overlay_boxes(frame, boxes_by_frame[actual_frame_num])
    
    # Compose grid
    print(f"  Composing grid for {len(cameras)} cameras...")
    composed = compose_grid(frames_dict, cameras, 30.0)
    
    # Write video
    if not output_path:
        output_path = VIDEO_OUTPUT_DIR / f"{slot}_q{question_idx}_{q['category']}.mp4"
    
    print(f"  Writing to {output_path}...", end="", flush=True)
    if write_video(composed, output_path, 30.0):
        print(" [OK]")
        return output_path
    else:
        print(" [FAILED]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Render question validation videos")
    parser.add_argument("--slot", help="Slot name (e.g., 2018-03-07.17-05.school)")
    parser.add_argument("--qa-file", help="QA JSON file path")
    parser.add_argument("--question-id", type=int, help="Question index to render")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    
    # Load QA data
    if args.qa_file:
        qa_path = Path(args.qa_file)
    elif args.slot:
        qa_path = QA_OUTPUT_DIR / f"{args.slot}.final.raw.json"
    else:
        parser.error("Must provide --slot or --qa-file")
    
    if not qa_path.exists():
        print(f"ERROR: {qa_path} not found")
        return
    
    with open(qa_path) as f:
        qa_data = json.load(f)
    
    print(f"Loaded: {qa_path}")
    print(f"Slot: {qa_data['slot']}")
    print(f"Total questions: {qa_data.get('total_questions', len(qa_data['qa_pairs']))}")
    
    q_idx = args.question_id if args.question_id is not None else 0
    
    output = render_question(qa_data, q_idx, Path(args.output) if args.output else None)
    if output:
        print(f"\nSaved: {output}")
        if args.verbose:
            print(f"Play with: ffplay {output}")


if __name__ == "__main__":
    main()

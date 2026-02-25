#!/usr/bin/env python3
"""
V8 Entity Description Extractor — Extract visual descriptions from raw video + geom.yml.

For EVERY annotated actor in a slot, this script:
  1. Parses geom.yml → per-actor bounding boxes per frame
  2. Extracts 5 representative crops from the MP4 video
  3. Runs YOLO color analysis on each crop (upper/lower body colors, carried objects)
  4. Aggregates via majority vote across crops
  5. Generates template description: "a person in blue top and black pants carrying a backpack"

This gives EVERY entity a visual description (not just the 10% with MEVID matches),
solving the temporal disambiguation problem where "a person enters scene" is ambiguous
when there are 100+ such events.

Cost: $0 (all local, no API calls)
Time: ~3-4 min per slot (mostly video decode + YOLO inference)

Usage:
    python3 scripts/v8/extract_entity_descriptions.py --slot 2018-03-11.11-25-00.school -v
    python3 scripts/v8/extract_entity_descriptions.py --slot 2018-03-11.11-25-00.school --dry-run
"""

import argparse
import json
import re
import sys
import time
import glob
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np

# ============================================================================
# Paths
# ============================================================================

KITWARE_BASE = Path("/nas/mars/dataset/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware")
KITWARE_TRAINING_BASE = Path("/nas/mars/dataset/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware-meva-training")
AVI_BASE = Path("/nas/mars/dataset/MEVA/avis")    # Raw AVIs — lossless, better color
MP4_BASE = Path("/nas/mars/dataset/MEVA/mp4s")     # Fallback (CRF 32 re-encode)
OUTPUT_DIR = Path("/home/ah66742/data/entity_descriptions")

# ============================================================================
# Constants
# ============================================================================

CROPS_PER_ACTOR = 5           # Crops to extract per actor track
MIN_BBOX_HEIGHT = 144         # Min bbox height in pixels for usable crop (~consistent with 2% area filter)
MIN_BBOX_WIDTH = 144          # Min bbox width (~consistent with 2% area filter)
YOLO_CONF = 0.25              # YOLO detection confidence threshold
YOLO_MODEL = "yolov8n.pt"    # Nano model (fast, sufficient for crops)

# COCO carried-object classes
CARRIED_OBJECTS = {
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
    28: "suitcase", 39: "bottle", 63: "laptop", 67: "cell phone",
    73: "book", 74: "clock",
}

# ============================================================================
# Geom Parsing (regex-based, memory efficient)
# ============================================================================

# Order-independent field extractors: handle both kitware and kitware-training formats
# kitware:  - { geom: {id1: 5193, id0: 1, ts0: 51, g0: 881 438 947 603, keyframe: true } }
# training: - {'geom': {'g0': '282 499 432 764', 'id0': 115, 'id1': 3, 'ts0': 1019}}
_RE_ID1 = re.compile(r"['\"]?id1['\"]?\s*:\s*['\"]?(\d+)")
_RE_TS0 = re.compile(r"['\"]?ts0['\"]?\s*:\s*['\"]?(\d+)")
_RE_G0  = re.compile(r"['\"]?g0['\"]?\s*:\s*['\"]?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")


def parse_geom(geom_path: Path) -> Dict[int, Dict[int, Tuple[int, int, int, int]]]:
    """
    Parse geom.yml → Dict[actor_id → Dict[frame_num → (x1, y1, x2, y2)]].
    Uses regex line-by-line (no YAML load, handles 100K+ line files).
    Handles both field orderings (kitware vs kitware-training).
    """
    actors = defaultdict(dict)
    with open(geom_path) as f:
        for line in f:
            id1_m = _RE_ID1.search(line)
            ts0_m = _RE_TS0.search(line)
            g0_m  = _RE_G0.search(line)
            if not (id1_m and ts0_m and g0_m):
                continue
            actor_id = int(id1_m.group(1))
            frame = int(ts0_m.group(1))
            bbox = (int(g0_m.group(1)), int(g0_m.group(2)),
                    int(g0_m.group(3)), int(g0_m.group(4)))
            actors[actor_id][frame] = bbox
    return dict(actors)


# ============================================================================
# Video Crop Extraction
# ============================================================================

def extract_crops(video_path: Path,
                  actors: Dict[int, Dict[int, Tuple[int, int, int, int]]],
                  max_crops: int = CROPS_PER_ACTOR,
                  min_h: int = MIN_BBOX_HEIGHT,
                  min_w: int = MIN_BBOX_WIDTH,
                  ) -> Dict[int, List[np.ndarray]]:
    """
    Extract bbox crops for all actors from a single video.

    Strategy selection:
      - Few target frames (< 200) spread across the video → random seek
      - Many target frames or dense clustering → sequential read (skip non-target)

    Sequential read is ~10-50x faster than random seek on H.264 MP4s because
    seeks must decode from the nearest keyframe, while sequential just grabs
    the next already-decoded frame.

    For each actor, samples `max_crops` frames evenly across their track,
    filtering out tiny bboxes. Returns Dict[actor_id → [crop_bgr, ...]].
    """
    if not actors:
        return {}

    # Build frame → [(actor_id, bbox)] mapping, sampling per actor
    frame_to_actors: Dict[int, List[Tuple[int, Tuple]]] = defaultdict(list)

    for actor_id, keyframes in actors.items():
        # Filter to usable bboxes
        usable = {f: bb for f, bb in keyframes.items()
                  if (bb[2] - bb[0]) >= min_w and (bb[3] - bb[1]) >= min_h}
        if not usable:
            continue

        frames = sorted(usable.keys())
        if len(frames) > max_crops:
            indices = np.linspace(0, len(frames) - 1, max_crops, dtype=int)
            frames = [frames[i] for i in indices]

        for fn in frames:
            frame_to_actors[fn].append((actor_id, usable[fn]))

    if not frame_to_actors:
        return {}

    target_frames = sorted(frame_to_actors.keys())
    target_set = set(target_frames)
    results: Dict[int, List[np.ndarray]] = defaultdict(list)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    WARNING: Cannot open {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_target = target_frames[-1]

    # Decide strategy: sequential if many frames or if target span covers >30%
    # of the video (seeking back-and-forth is slower than just reading through)
    span = last_target - target_frames[0] + 1
    use_sequential = (len(target_frames) > 150 or
                      span > 0 and len(target_frames) / span > 0.02)

    def _crop_frame(frame_bgr, frame_idx):
        """Extract all actor crops from a decoded frame."""
        h, w = frame_bgr.shape[:2]
        for actor_id, bbox in frame_to_actors[frame_idx]:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                results[actor_id].append(crop)

    if use_sequential:
        # Sequential read: read every frame from first target to last target,
        # only decode+crop on target frames. cap.grab() is fast (no decode),
        # cap.retrieve() decodes only when needed.
        start_frame = target_frames[0]
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current = start_frame
        collected = 0
        while current <= last_target and current < total_frames:
            if current in target_set:
                ret, frame = cap.read()
                if ret:
                    _crop_frame(frame, current)
                    collected += 1
            else:
                cap.grab()  # Advance without decoding — very fast
            current += 1
    else:
        # Random seek: fewer frames, worth the per-seek cost
        for target_frame in target_frames:
            if target_frame >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                continue
            _crop_frame(frame, target_frame)

    cap.release()
    return dict(results)


# ============================================================================
# Color Analysis (HSV-based, same as extract_attributes_yolo.py)
# ============================================================================

def _hsv_to_color(h: float, s: float, v: float) -> str:
    """Convert OpenCV HSV (H:0-180, S:0-255, V:0-255) to color name."""
    if s < 40:
        if v < 60:
            return "black"
        elif v < 150:
            return "gray"
        else:
            return "white"
    if v < 40:
        return "black"
    if h < 10 or h > 170:
        return "red"
    elif h < 22:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 78:
        return "green"
    elif h < 131:
        return "blue"
    elif h < 155:
        return "purple"
    elif h <= 170:
        return "pink"
    return "unknown"


def _extract_region_color(crop_bgr: np.ndarray) -> str:
    """Extract dominant color from a BGR crop using center-weighted HSV mean."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return "unknown"

    # Inner 80% to avoid background bleed
    my, mx = max(1, h // 10), max(1, w // 10)
    inner = crop_bgr[my:h - my, mx:w - mx]

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    return _hsv_to_color(h_mean, s_mean, v_mean)


def analyze_crops_color_only(crops: List[np.ndarray]) -> Dict:
    """
    Analyze a list of person-crop BGR images using HSV color analysis only.
    No YOLO needed — faster, simpler, works on any size crop.

    Returns dict with upper_color, lower_color.
    """
    upper_colors = []
    lower_colors = []

    for crop in crops:
        h, w = crop.shape[:2]
        if h < 10:
            continue

        # Upper body: 10-45% of height (skip head)
        u_y1 = int(h * 0.10)
        u_y2 = int(h * 0.45)
        # Lower body: 55-90% (skip feet)
        l_y1 = int(h * 0.55)
        l_y2 = int(h * 0.90)

        upper_colors.append(_extract_region_color(crop[u_y1:u_y2, :]))
        lower_colors.append(_extract_region_color(crop[l_y1:l_y2, :]))

    upper = _majority_vote(upper_colors)
    lower = _majority_vote(lower_colors)

    return {"upper_color": upper, "lower_color": lower}


def _majority_vote(colors: List[str]) -> str:
    """Majority vote ignoring 'unknown'."""
    filtered = [c for c in colors if c != "unknown"]
    if not filtered:
        return "unknown"
    return Counter(filtered).most_common(1)[0][0]


# ============================================================================
# YOLO Analysis (optional, richer — detects carried objects)
# ============================================================================

_yolo_model = None


def _get_yolo():
    """Lazy-load YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def analyze_crops_yolo(crops: List[np.ndarray]) -> Dict:
    """
    Analyze crops with YOLO for person detection, colors, and carried objects.
    More expensive than color-only but detects backpacks, bottles, phones, etc.
    """
    model = _get_yolo()

    upper_colors = []
    lower_colors = []
    all_objects = []

    for crop in crops:
        h, w = crop.shape[:2]
        if h < 15 or w < 8:
            continue

        # Run YOLO
        results = model(crop, conf=YOLO_CONF, verbose=False)

        # Find person bbox and carried objects
        person_box = None
        person_conf = 0.0

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                if cls_id == 0 and conf > person_conf:
                    coords = box.xyxy[0].cpu().numpy()
                    person_box = (int(coords[0]), int(coords[1]),
                                  int(coords[2]), int(coords[3]))
                    person_conf = conf
                if cls_id in CARRIED_OBJECTS and conf > 0.3:
                    all_objects.append(CARRIED_OBJECTS[cls_id])

        # Use person bbox if found, else full crop
        if person_box:
            px1, py1, px2, py2 = person_box
        else:
            px1, py1, px2, py2 = 0, 0, w, h

        ph = py2 - py1
        if ph < 10:
            continue

        # Upper/lower body color
        u_y1 = py1 + int(ph * 0.10)
        u_y2 = py1 + int(ph * 0.45)
        l_y1 = py1 + int(ph * 0.55)
        l_y2 = py1 + int(ph * 0.90)

        upper_colors.append(_extract_region_color(crop[u_y1:u_y2, px1:px2]))
        lower_colors.append(_extract_region_color(crop[l_y1:l_y2, px1:px2]))

    upper = _majority_vote(upper_colors)
    lower = _majority_vote(lower_colors)

    # Objects seen in >= 2 crops (or any if only 1 crop)
    obj_counts = Counter(all_objects)
    objects = sorted(set(
        obj for obj, cnt in obj_counts.items()
        if cnt >= 2 or len(crops) <= 2
    ))

    return {"upper_color": upper, "lower_color": lower, "carried_objects": objects}


# ============================================================================
# Description Generation (template-based, free)
# ============================================================================

def build_description(attrs: Dict) -> str:
    """
    Build a natural description from structured attributes.

    Examples:
      {"upper_color": "blue", "lower_color": "black", "carried_objects": ["backpack"]}
      → "a person in a blue top and black pants carrying a backpack"

      {"upper_color": "gray", "lower_color": "green", "carried_objects": []}
      → "a person in a gray top and green pants"
    """
    parts = []

    upper = attrs.get("upper_color", "unknown")
    lower = attrs.get("lower_color", "unknown")

    if upper != "unknown" and lower != "unknown":
        parts.append(f"a person in a {upper} top and {lower} pants")
    elif upper != "unknown":
        parts.append(f"a person in a {upper} top")
    elif lower != "unknown":
        parts.append(f"a person in {lower} pants")
    else:
        parts.append("a person")

    objects = attrs.get("carried_objects", [])
    if objects:
        obj_str = " and ".join(objects[:2])  # Max 2 objects
        parts[0] += f" carrying a {obj_str}"

    return parts[0]


# ============================================================================
# Slot Processing Pipeline
# ============================================================================

def find_slot_files(slot: str) -> List[Dict]:
    """
    Find geom.yml + video pairs for a slot.
    Searches BOTH kitware/ and kitware-meva-training/ directories.
    Prefers raw AVI (lossless) over MP4 (CRF 32 re-encode) for better color.
    Returns list of {camera, geom_path, video_path, act_path, video_format, source}.
    """
    # Parse slot: "2018-03-11.11-25.school"
    parts = slot.split(".")
    if len(parts) < 3:
        raise ValueError(f"Invalid slot format: {slot} (expected date.time.site)")

    date = parts[0]
    time_part = parts[1]  # HH-MM (no seconds)
    site = parts[2]

    hour = time_part.split("-")[0]  # e.g. "11" from "11-25"

    # Check both annotation sources
    search_dirs = [
        (KITWARE_BASE / date / hour, "kitware"),
        (KITWARE_TRAINING_BASE / date / hour, "kitware-training"),
    ]
    
    avi_dir = AVI_BASE / date / hour
    mp4_dir = MP4_BASE / date / hour

    # Find geom files matching slot pattern
    prefix = f"{date}.{time_part}"
    results = []

    for kitware_dir, source in search_dirs:
        if not kitware_dir.is_dir():
            continue
        
        for gf in sorted(kitware_dir.glob(f"{prefix}*.{site}.*.geom.yml")):
            name = gf.name
            # Extract camera: ...school.G328.geom.yml
            cam_match = re.search(rf'\.{site}\.(G\d+)\.geom\.yml$', name)
            if not cam_match:
                continue
            cam = cam_match.group(1)

            # Geom basename: 2018-03-11.11-25-00.11-30-00.school.G328
            base_name = name.replace(".geom.yml", "")

            # Prefer raw AVI over MP4
            # AVI naming: {base_name}.r13.avi (exact match or fuzzy on end-time)
            video_path = None
            video_fmt = None

            if avi_dir.is_dir():
                avi_candidates = sorted(avi_dir.glob(f"{base_name}.r13.avi")) + \
                                 sorted(avi_dir.glob(f"{prefix}*.{site}.{cam}.r13.avi"))
                if avi_candidates:
                    video_path = avi_candidates[0]
                    video_fmt = "avi"

            # Fallback to MP4
            if video_path is None and mp4_dir.is_dir():
                mp4_candidates = sorted(mp4_dir.glob(f"{base_name}*.mp4")) + \
                                 sorted(mp4_dir.glob(f"{prefix}*.{site}.{cam}*.mp4"))
                if mp4_candidates:
                    video_path = mp4_candidates[0]
                    video_fmt = "mp4"

            # Activity file
            act_path = gf.parent / name.replace(".geom.yml", ".activities.yml")
            if not act_path.exists():
                act_path = None

            results.append({
                "camera": cam,
                "geom_path": gf,
                "video_path": video_path,
                "video_format": video_fmt,
                "act_path": act_path,
                "source": source,
            })

    return results


def process_slot(slot: str, use_yolo: bool = True,
                 verbose: bool = False) -> Dict:
    """
    Full pipeline: extract descriptions for all actors in a slot.

    Returns dict ready for JSON output:
    {
      "slot": "...",
      "cameras": {...},
      "actors": {actor_id_str: {camera, upper_color, lower_color, objects, description}},
      "stats": {...}
    }
    """
    t0 = time.time()
    files = find_slot_files(slot)

    if verbose:
        print(f"\n  Slot: {slot}")
        print(f"  Found {len(files)} cameras with geom annotations")

    if use_yolo:
        if verbose:
            print(f"  Loading YOLO model...", end="", flush=True)
        _get_yolo()
        if verbose:
            print(" done.")

    all_actors = {}
    cam_stats = {}

    for cf in files:
        cam = cf["camera"]
        geom_path = cf["geom_path"]
        video_path = cf["video_path"]
        video_fmt = cf.get("video_format", "unknown")

        if verbose:
            print(f"\n  Camera {cam}:")
            print(f"    Geom: {geom_path.name}")

        if video_path is None or not video_path.exists():
            if verbose:
                print(f"    SKIP: No video found (checked AVI + MP4)")
            cam_stats[cam] = {"actors": 0, "usable": 0, "skipped": "no_video"}
            continue

        if verbose:
            print(f"    Video: {video_path.name} ({video_fmt})")

        # Parse geom
        actors = parse_geom(geom_path)
        if verbose:
            print(f"    Actors: {len(actors)} total")

        if not actors:
            cam_stats[cam] = {"actors": 0, "usable": 0, "skipped": "no_actors"}
            continue

        # Extract crops
        t1 = time.time()
        crops_by_actor = extract_crops(
            video_path, actors,
            max_crops=CROPS_PER_ACTOR,
            min_h=MIN_BBOX_HEIGHT, min_w=MIN_BBOX_WIDTH,
        )
        decode_time = time.time() - t1

        usable = len(crops_by_actor)
        total_crops = sum(len(c) for c in crops_by_actor.values())
        if verbose:
            print(f"    Usable actors: {usable}/{len(actors)} ({total_crops} crops, {decode_time:.1f}s decode)")

        # Analyze each actor
        t2 = time.time()
        for actor_id, crops in crops_by_actor.items():
            if not crops:
                continue

            if use_yolo:
                attrs = analyze_crops_yolo(crops)
            else:
                attrs = analyze_crops_color_only(crops)

            desc = build_description(attrs)

            # Store by camera_actorID (matching V8 entity ID format)
            entity_key = f"{cam}_actor_{actor_id}"
            all_actors[entity_key] = {
                "actor_id": actor_id,
                "camera": cam,
                "upper_color": attrs.get("upper_color", "unknown"),
                "lower_color": attrs.get("lower_color", "unknown"),
                "carried_objects": attrs.get("carried_objects", []),
                "description": desc,
                "num_crops": len(crops),
                "avg_crop_height": int(np.mean([c.shape[0] for c in crops])),
            }

        analyze_time = time.time() - t2
        if verbose:
            print(f"    Analysis: {analyze_time:.1f}s ({'YOLO' if use_yolo else 'color-only'})")

        cam_stats[cam] = {
            "actors": len(actors),
            "usable": usable,
            "total_crops": total_crops,
            "decode_sec": round(decode_time, 1),
            "analyze_sec": round(analyze_time, 1),
        }

    total_time = time.time() - t0

    # Summary stats
    described = sum(1 for a in all_actors.values() if a["description"] != "a person")
    color_dist = Counter(a["upper_color"] for a in all_actors.values())

    result = {
        "slot": slot,
        "method": "yolo" if use_yolo else "color_only",
        "total_actors": len(all_actors),
        "actors_with_colors": described,
        "actors_without_colors": len(all_actors) - described,
        "upper_color_distribution": dict(color_dist.most_common()),
        "cameras": cam_stats,
        "processing_time_sec": round(total_time, 1),
        "actors": all_actors,
    }

    if verbose:
        print(f"\n  === Summary ===")
        print(f"  Total actors: {len(all_actors)}")
        print(f"  With color descriptions: {described}")
        print(f"  Without: {len(all_actors) - described}")
        print(f"  Color distribution: {dict(color_dist.most_common(5))}")
        print(f"  Total time: {total_time:.1f}s")

    return result


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V8 Entity Description Extractor — Geom + Video → YOLO descriptions",
    )
    parser.add_argument("--slot", "-s", required=True,
        help="Slot to process (e.g., 2018-03-11.11-25.school)")
    parser.add_argument("--no-yolo", action="store_true",
        help="Color-only analysis (no YOLO, faster but no carried objects)")
    parser.add_argument("--dry-run", action="store_true",
        help="Show what would be processed without extracting")
    parser.add_argument("--output", "-o",
        help="Output path (default: data/entity_descriptions/{slot}.json)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.dry_run:
        files = find_slot_files(args.slot)
        print(f"\n  Slot: {args.slot}")
        print(f"  Cameras with geom: {len(files)}")
        for cf in files:
            cam = cf["camera"]
            actors = parse_geom(cf["geom_path"])
            usable = sum(1 for aid in actors
                        for frames in [actors[aid].values()]
                        if any((bb[2]-bb[0]) >= MIN_BBOX_WIDTH and
                               (bb[3]-bb[1]) >= MIN_BBOX_HEIGHT
                               for bb in frames))
            vp = cf["video_path"]
            has_video = vp and vp.exists()
            vfmt = cf.get("video_format", "none")
            print(f"    {cam}: {len(actors)} actors, {usable} usable, video={'YES' if has_video else 'NO'} ({vfmt})")
        return

    result = process_slot(args.slot, use_yolo=not args.no_yolo, verbose=args.verbose)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.slot}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Output: {out_path}")

    # Show sample descriptions
    print(f"\n  === Sample Descriptions ===")
    seen = set()
    for eid, info in sorted(result["actors"].items()):
        desc = info["description"]
        if desc in seen or desc == "a person":
            continue
        seen.add(desc)
        print(f"    {info['camera']} actor ...{str(info['actor_id'])[-6:]}: {desc}")
        if len(seen) >= 10:
            break


if __name__ == "__main__":
    main()

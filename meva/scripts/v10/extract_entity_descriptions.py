#!/usr/bin/env python3
"""
V10 Entity Description Extractor — Extract visual descriptions from raw video + geom.yml.

For EVERY annotated actor in a slot, this script:
  1. Parses geom.yml → per-actor bounding boxes per frame
  2. Extracts 5 representative crops from the video
  3. Runs SegFormer human parsing (default) or YOLO+color analysis on each crop
  4. Aggregates via majority vote across crops
  5. Generates rich description: "a person with dark hair, wearing a blue top and black pants"

Methods:
  segformer  — SegFormer human parsing (18 body-part classes, best quality) [default]
  yolo       — YOLO person detection + fixed-split colors + carried objects
  color-only — Fixed vertical splits (fastest, no model needed)

Cost: $0 (all local, no API calls)
Time: ~2-3 min per slot (segformer on GPU), ~3-4 min (YOLO)

Usage:
    python3 scripts/v10/extract_entity_descriptions.py --slot 2018-03-11.11-25.school -v
    python3 scripts/v10/extract_entity_descriptions.py --slot 2018-03-11.11-25.school --method yolo
    python3 scripts/v10/extract_entity_descriptions.py --slot 2018-03-11.11-25.school --dry-run
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
# Entity descriptions output — matches reader default in person_descriptions.py
# Override with MEVA_ENTITY_DESC_DIR env var
OUTPUT_DIR = Path(os.environ.get("MEVA_ENTITY_DESC_DIR") or "/nas/mars/dataset/MEVA/entity_descriptions")

# ============================================================================
# Constants
# ============================================================================

CROPS_PER_ACTOR = 5           # Crops to extract per actor track
MIN_BBOX_HEIGHT = 144         # Min bbox height for SegFormer (needs detail for segmentation)
MIN_BBOX_HEIGHT_COLOR = 40    # Min bbox height for HSV color-only fallback (just needs colors)
MIN_BBOX_WIDTH = 48           # Min bbox width for SegFormer crops
MIN_BBOX_WIDTH_COLOR = 16     # Min bbox width for HSV color-only fallback
YOLO_CONF = 0.25              # YOLO detection confidence threshold
YOLO_MODEL = "yolov8n.pt"    # Nano model (fast, sufficient for crops)

# SegFormer human parsing model
SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
MIN_REGION_PIXELS = 50        # Min pixels for a body region to count

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
            # Prefer middle-of-track frames (#5): more stable pose/lighting,
            # less likely to be entering/exiting frame. Sample from inner 80%
            # of the track, with endpoints only if needed.
            n = len(frames)
            inner_start = max(0, int(n * 0.1))
            inner_end = min(n - 1, int(n * 0.9))
            inner_frames = frames[inner_start:inner_end + 1]
            if len(inner_frames) >= max_crops:
                indices = np.linspace(0, len(inner_frames) - 1, max_crops, dtype=int)
                frames = [inner_frames[i] for i in indices]
            else:
                # Track too short — use uniform across full range
                indices = np.linspace(0, n - 1, max_crops, dtype=int)
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
    """Convert OpenCV HSV (H:0-180, S:0-255, V:0-255) to ~25 CSS-friendly color names.

    Expanded vocabulary for better entity disambiguation.  VLMs trained on web
    data recognise names like "navy", "olive", "khaki" better than raw HSV.

    Color map (OpenCV hue 0-180):
      achromatic (S<40): black / charcoal / dark gray / gray / silver / ivory / white
      chromatic by hue band:
        0-10,170-180 red   → maroon / crimson / red
        10-22        orange → rust / orange
        22-35        yellow → khaki / gold / yellow
        35-55        green  → olive / green
        55-78        teal   → teal
        78-95        blue   → teal-blue (low H), navy (low V)
        95-115       blue   → blue
        115-131      indigo → indigo
        131-155      purple → plum / purple
        155-170      pink   → mauve / pink
    """
    # --- Achromatic: low saturation ---
    if s < 40:
        if v < 30:
            return "black"
        elif v < 60:
            return "charcoal"
        elif v < 100:
            return "dark gray"
        elif v < 130:
            return "gray"
        elif v < 150:
            return "silver"
        else:
            # Slight warm tint → ivory
            if 15 <= h <= 35 and s >= 15:
                return "ivory"
            return "white"
    # Very dark with some saturation
    if v < 40:
        return "black"

    # --- Chromatic: hue-based ---
    # Red (H wraps: 0-10 and 170-180)
    if h < 10 or h > 170:
        if v < 100:
            return "maroon"
        elif s > 150:
            return "crimson"
        return "red"

    # Orange (10-22)
    elif h < 22:
        if v < 120:
            return "rust"
        return "orange"

    # Yellow (22-35)
    elif h < 35:
        if s < 80:
            return "khaki"
        elif s < 150:
            return "gold"
        return "yellow"

    # Green (35-55)
    elif h < 55:
        if v < 120:
            return "olive"
        return "green"

    # Teal-green (55-78)
    elif h < 78:
        return "teal"

    # Blue range (78-131)
    elif h < 95:
        if s > 100 and v > 100:
            return "teal"
        if v < 100:
            return "navy"
        return "blue"
    elif h < 115:
        if v < 100:
            return "navy"
        return "blue"
    elif h < 131:
        return "indigo"

    # Purple (131-155)
    elif h < 155:
        if v < 100:
            return "plum"
        return "purple"

    # Pink (155-170)
    elif h <= 170:
        if s < 100:
            return "mauve"
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


def _majority_vote_with_confidence(colors: List[str]) -> Tuple[str, float]:
    """Majority vote returning (winner, confidence 0-1) ignoring 'unknown'."""
    filtered = [c for c in colors if c != "unknown"]
    if not filtered:
        return "unknown", 0.0
    counter = Counter(filtered)
    winner, count = counter.most_common(1)[0]
    confidence = count / len(filtered)
    return winner, confidence


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
# SegFormer Human Parsing (semantic body-part segmentation)
# ============================================================================

_segformer_processor = None
_segformer_model = None


def _get_segformer():
    """Lazy-load SegFormer human parsing model (GPU if available)."""
    global _segformer_processor, _segformer_model
    if _segformer_model is None:
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        _segformer_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
        if torch.cuda.is_available():
            _segformer_model = _segformer_model.to("cuda")
        _segformer_model.eval()
    return _segformer_processor, _segformer_model


def _get_segmentation_map(crop_bgr: np.ndarray, processor, model) -> np.ndarray:
    """Run SegFormer inference on a BGR crop → per-pixel class ID map (H, W)."""
    import torch
    from PIL import Image
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, 18, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits, size=crop_bgr.shape[:2], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1).squeeze().cpu().numpy()


def _extract_mask_color(crop_bgr: np.ndarray, mask: np.ndarray) -> str:
    """Extract dominant color from BGR pixels where mask is True."""
    if mask.sum() < MIN_REGION_PIXELS:
        return "unknown"
    # Gather masked pixels as (N, 3)
    pixels = crop_bgr[mask]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, 0, 0]))
    s_mean = float(np.mean(hsv[:, 0, 1]))
    v_mean = float(np.mean(hsv[:, 0, 2]))
    return _hsv_to_color(h_mean, s_mean, v_mean)


def _detect_texture(crop_bgr: np.ndarray, mask: np.ndarray) -> Dict:
    """
    Detect clothing texture/pattern within a segmentation mask.

    Analyzes:
      - Solid vs patterned: low intra-mask color variance = solid
      - Striped: high directional gradient variance (horizontal or vertical bands)
      - Light/dark qualifier: based on mean V channel value

    Returns dict: {"texture": "solid"|"patterned"|"striped", "brightness": "light"|"dark"|""}
    """
    result = {"texture": "", "brightness": ""}
    if mask.sum() < MIN_REGION_PIXELS * 2:  # need enough pixels for texture
        return result

    pixels = crop_bgr[mask]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)

    # --- Brightness qualifier ---
    v_mean = float(np.mean(hsv[:, 0, 2]))
    s_mean = float(np.mean(hsv[:, 0, 1]))
    # Only add qualifier for chromatic colors (skip achromatic = low saturation)
    # Also skip very dark regions (V < 60) — brightness is noise at that level
    if s_mean >= 40 and v_mean >= 60:
        if v_mean < 90:
            result["brightness"] = "dark"
        elif v_mean > 190:
            result["brightness"] = "light"

    # --- Texture detection via color variance within mask ---
    # Use HSV Hue + Saturation channels for variance (ignore brightness variations
    # from shading which don't indicate pattern)
    h_std = float(np.std(hsv[:, 0, 0]))
    s_std = float(np.std(hsv[:, 0, 1]))

    # Skip texture detection on very dark regions (V_mean < 70) — dark clothing
    # creates compression noise that falsely triggers pattern detection
    if v_mean < 70:
        result["texture"] = "solid"  # assume solid for very dark clothing
        return result

    # High hue variance = multi-color pattern (raised thresholds to reduce FP)
    if h_std > 45 or s_std > 60:
        # Check for stripes: look for strong directional gradients
        # Get the bounding box of the mask region for structured analysis
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        region = crop_bgr[y_min:y_max+1, x_min:x_max+1]
        region_mask = mask[y_min:y_max+1, x_min:x_max+1]
        rh, rw = region.shape[:2]

        if rh > 10 and rw > 10:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Apply mask: set non-mask pixels to mean to avoid edge artifacts
            mean_val = float(gray[region_mask].mean()) if region_mask.any() else 128
            gray[~region_mask] = mean_val

            # Horizontal gradient (detects vertical stripes)
            grad_h = np.abs(np.diff(gray, axis=1))
            # Vertical gradient (detects horizontal stripes)
            grad_v = np.abs(np.diff(gray, axis=0))

            # Stripe detection: one direction should have much stronger gradients
            h_energy = float(grad_h.mean())
            v_energy = float(grad_v.mean())

            if max(h_energy, v_energy) > 15:  # significant edge energy
                ratio = max(h_energy, v_energy) / (min(h_energy, v_energy) + 1e-6)
                if ratio > 1.5:
                    result["texture"] = "striped"
                else:
                    result["texture"] = "patterned"
            else:
                result["texture"] = "patterned"
        else:
            result["texture"] = "patterned"
    else:
        result["texture"] = "solid"

    return result


# SegFormer class IDs → semantic groups
_SEG_HAIR = 2
_SEG_UPPER = 4
_SEG_SKIRT = 5
_SEG_PANTS = 6
_SEG_DRESS = 7
_SEG_LSHOE = 9
_SEG_RSHOE = 10
_SEG_HAT = 1
_SEG_SUNGLASSES = 3
_SEG_BAG = 16
_SEG_SCARF = 17


def analyze_crops_segformer(crops: List[np.ndarray]) -> Dict:
    """
    Analyze crops with SegFormer human parsing (18 body-part classes).

    Segments each crop into semantic regions (hair, upper-clothes, pants/skirt/dress,
    shoes, etc.), extracts HSV color per region, detects accessories and texture.
    Majority-votes across crops with confidence tracking for robust results.

    Returns dict with:
      hair_color, upper_color, lower_color, lower_type,
      shoe_color, accessories, carried_objects,
      upper_texture, lower_texture, upper_brightness, lower_brightness,
      confidence (per-attribute confidence scores)
    """
    processor, model = _get_segformer()

    hair_colors = []
    upper_colors = []
    lower_colors = []
    shoe_colors = []
    lower_types = []
    accessories_per_crop = []
    # Texture/brightness per crop (#2)
    upper_textures = []
    lower_textures = []
    upper_brightness_list = []
    lower_brightness_list = []

    for crop in crops:
        h, w = crop.shape[:2]
        if h < 15 or w < 8:
            continue

        seg_map = _get_segmentation_map(crop, processor, model)

        # Hair (class 2)
        hair_colors.append(_extract_mask_color(crop, seg_map == _SEG_HAIR))

        # Upper-clothes (class 4)
        upper_mask = seg_map == _SEG_UPPER
        upper_colors.append(_extract_mask_color(crop, upper_mask))
        # Texture analysis for upper clothing (#2)
        upper_tex = _detect_texture(crop, upper_mask)
        upper_textures.append(upper_tex["texture"])
        upper_brightness_list.append(upper_tex["brightness"])

        # Lower body: Pants(6), Skirt(5), Dress(7) — pick dominant
        pants_px = (seg_map == _SEG_PANTS).sum()
        skirt_px = (seg_map == _SEG_SKIRT).sum()
        dress_px = (seg_map == _SEG_DRESS).sum()

        lower_mask = None
        if dress_px > max(pants_px, skirt_px) and dress_px >= MIN_REGION_PIXELS:
            lower_mask = seg_map == _SEG_DRESS
            lower_colors.append(_extract_mask_color(crop, lower_mask))
            lower_types.append("dress")
        elif skirt_px > pants_px and skirt_px >= MIN_REGION_PIXELS:
            lower_mask = seg_map == _SEG_SKIRT
            lower_colors.append(_extract_mask_color(crop, lower_mask))
            lower_types.append("skirt")
        elif pants_px >= MIN_REGION_PIXELS:
            lower_mask = seg_map == _SEG_PANTS
            lower_colors.append(_extract_mask_color(crop, lower_mask))
            lower_types.append("pants")
        else:
            lower_colors.append("unknown")
            lower_types.append("unknown")

        # Texture for lower clothing (#2)
        if lower_mask is not None:
            lower_tex = _detect_texture(crop, lower_mask)
            lower_textures.append(lower_tex["texture"])
            lower_brightness_list.append(lower_tex["brightness"])
        else:
            lower_textures.append("")
            lower_brightness_list.append("")

        # Shoes (left 9 + right 10)
        shoe_mask = (seg_map == _SEG_LSHOE) | (seg_map == _SEG_RSHOE)
        shoe_colors.append(_extract_mask_color(crop, shoe_mask))

        # Accessories
        crop_acc = []
        if (seg_map == _SEG_HAT).sum() >= MIN_REGION_PIXELS:
            crop_acc.append("hat")
        if (seg_map == _SEG_SUNGLASSES).sum() >= MIN_REGION_PIXELS:
            crop_acc.append("sunglasses")
        if (seg_map == _SEG_BAG).sum() >= MIN_REGION_PIXELS:
            crop_acc.append("bag")
        if (seg_map == _SEG_SCARF).sum() >= MIN_REGION_PIXELS:
            crop_acc.append("scarf")
        accessories_per_crop.append(crop_acc)

    # Majority votes with confidence (#5)
    hair, hair_conf = _majority_vote_with_confidence(hair_colors)
    upper, upper_conf = _majority_vote_with_confidence(upper_colors)
    lower, lower_conf = _majority_vote_with_confidence(lower_colors)
    shoes, shoes_conf = _majority_vote_with_confidence(shoe_colors)
    lower_type = _majority_vote(lower_types)

    # Texture/brightness votes (#2)
    upper_texture = _majority_vote([t for t in upper_textures if t])
    lower_texture = _majority_vote([t for t in lower_textures if t])
    upper_brightness = _majority_vote([b for b in upper_brightness_list if b])
    lower_brightness = _majority_vote([b for b in lower_brightness_list if b])

    # Accessories: keep if seen in ≥2 crops (or any if ≤2 total)
    acc_counter = Counter(a for crop_acc in accessories_per_crop for a in crop_acc)
    threshold = 2 if len(crops) > 2 else 1
    accessories = sorted(a for a, cnt in acc_counter.items() if cnt >= threshold)

    # Confidence dict (#5): per-attribute agreement score
    confidence = {
        "hair": round(hair_conf, 2),
        "upper": round(upper_conf, 2),
        "lower": round(lower_conf, 2),
        "shoes": round(shoes_conf, 2),
    }

    return {
        "hair_color": hair,
        "upper_color": upper,
        "lower_color": lower,
        "lower_type": lower_type if lower_type != "unknown" else "pants",
        "shoe_color": shoes,
        "accessories": accessories,
        "carried_objects": [],  # SegFormer detects bags; other objects need YOLO
        "upper_texture": upper_texture if upper_texture != "unknown" else "",
        "lower_texture": lower_texture if lower_texture != "unknown" else "",
        "upper_brightness": upper_brightness if upper_brightness != "unknown" else "",
        "lower_brightness": lower_brightness if lower_brightness != "unknown" else "",
        "confidence": confidence,
    }


# ============================================================================
# Description Generation (template-based, free)
# ============================================================================

def _article(word: str) -> str:
    """Return 'an' if word starts with a vowel sound, else 'a'."""
    return "an" if word and word[0].lower() in "aeiou" else "a"


def build_description(attrs: Dict, include_position: bool = False) -> str:
    """
    Build a natural description from structured attributes.

    Handles both old-style (upper_color/lower_color only) and new segformer-style
    (hair_color, lower_type, shoe_color, accessories) attributes.

    Optional positional/height hints for disambiguation.

    Examples (segformer):
      → "a person with dark hair, wearing a navy top and khaki pants, silver shoes"
      → "a person wearing a crimson top and gray skirt, carrying a bag"

    Examples (with position):
      → "a tall person with dark hair, wearing a navy top and khaki pants, on the left side"
    """
    hair = attrs.get("hair_color")
    upper = attrs.get("upper_color", "unknown")
    lower = attrs.get("lower_color", "unknown")
    lower_type = attrs.get("lower_type", "pants")
    shoes = attrs.get("shoe_color")
    accessories = attrs.get("accessories", [])
    carried = attrs.get("carried_objects", [])
    confidence = attrs.get("confidence", {})

    # Texture/brightness qualifiers (#2)
    upper_texture = attrs.get("upper_texture", "")
    lower_texture = attrs.get("lower_texture", "")
    upper_brightness = attrs.get("upper_brightness", "")
    lower_brightness = attrs.get("lower_brightness", "")

    # Drop low-confidence attributes (#5): if agreement < 40%, omit to avoid
    # wrong colors. Better to say nothing than to say the wrong color.
    min_conf = 0.4
    if confidence.get("hair", 1.0) < min_conf:
        hair = "unknown"
    if confidence.get("upper", 1.0) < min_conf:
        upper = "unknown"
    if confidence.get("lower", 1.0) < min_conf:
        lower = "unknown"
    if confidence.get("shoes", 1.0) < min_conf:
        shoes = "unknown"

    # Relative height from bbox (tall/medium/short)
    height_hint = attrs.get("height_category")  # set by enrich step if available

    desc = _article(height_hint if height_hint and height_hint != "medium" else "person")
    if height_hint and height_hint != "medium":
        desc += f" {height_hint}"
    desc += " person"

    # Hair color
    if hair and hair != "unknown":
        desc += f" with {hair} hair"

    # Clothing — include brightness + texture qualifiers (#2)
    clothing_parts = []
    # Colors that are inherently dark/light — don't add redundant qualifiers
    _DARK_COLORS = {"black", "charcoal", "navy", "maroon", "dark gray"}
    _LIGHT_COLORS = {"white", "ivory", "silver"}

    if upper != "unknown":
        upper_desc = upper
        # Add brightness: "light blue", "dark green" — skip if color is already dark/light
        if upper_brightness == "dark" and upper not in _DARK_COLORS:
            upper_desc = f"dark {upper}"
        elif upper_brightness == "light" and upper not in _LIGHT_COLORS:
            upper_desc = f"light {upper}"
        # Add texture: "striped blue top", "patterned red top"
        if upper_texture and upper_texture not in ("solid", ""):
            clothing_parts.append(f"{_article(upper_texture)} {upper_texture} {upper_desc} top")
        else:
            clothing_parts.append(f"{_article(upper_desc)} {upper_desc} top")
    if lower != "unknown":
        lower_desc = lower
        if lower_brightness == "dark" and lower not in _DARK_COLORS:
            lower_desc = f"dark {lower}"
        elif lower_brightness == "light" and lower not in _LIGHT_COLORS:
            lower_desc = f"light {lower}"
        if lower_texture and lower_texture not in ("solid", ""):
            clothing_parts.append(f"{lower_texture} {lower_desc} {lower_type}")
        else:
            clothing_parts.append(f"{lower_desc} {lower_type}")

    if clothing_parts:
        desc += ", wearing " + " and ".join(clothing_parts)

    # Shoes
    if shoes and shoes != "unknown":
        desc += f", {shoes} shoes"

    # Accessories (worn items: hat, sunglasses, scarf)
    worn = [a for a in accessories if a in ("hat", "sunglasses", "scarf")]
    if worn:
        desc += f", with {_article(worn[0])} {' and '.join(worn)}"

    # Carried items (bag from segformer + YOLO objects)
    carried_items = (["bag"] if "bag" in accessories else []) + list(carried[:2])
    if carried_items:
        desc += f", carrying {_article(carried_items[0])} {' and '.join(carried_items[:2])}"

    # Spatial position hint (for disambiguation)
    if include_position:
        position = attrs.get("frame_position")  # "left", "center", "right"
        if position and position != "center":
            desc += f", on the {position} side"

    return desc


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


def process_slot(slot: str, method: str = "segformer",
                 verbose: bool = False) -> Dict:
    """
    Full pipeline: extract descriptions for all actors in a slot.

    Args:
        method: "segformer" (default, richest), "yolo", or "color-only"

    Returns dict ready for JSON output:
    {
      "slot": "...",
      "cameras": {...},
      "actors": {actor_id_str: {camera, upper_color, lower_color, ..., description}},
      "stats": {...}
    }
    """
    t0 = time.time()
    files = find_slot_files(slot)

    if verbose:
        print(f"\n  Slot: {slot}")
        print(f"  Found {len(files)} cameras with geom annotations")
        print(f"  Method: {method}")

    if method == "yolo":
        if verbose:
            print(f"  Loading YOLO model...", end="", flush=True)
        _get_yolo()
        if verbose:
            print(" done.")
    elif method == "segformer":
        if verbose:
            print(f"  Loading SegFormer model...", end="", flush=True)
        _get_segformer()
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

        # Extract crops — use lower threshold to capture distant actors too
        t1 = time.time()
        # Two-tier: extract at lower threshold, then decide analysis method per actor
        extraction_min_h = MIN_BBOX_HEIGHT_COLOR if method == "segformer" else MIN_BBOX_HEIGHT
        extraction_min_w = MIN_BBOX_WIDTH_COLOR if method == "segformer" else MIN_BBOX_WIDTH
        crops_by_actor = extract_crops(
            video_path, actors,
            max_crops=CROPS_PER_ACTOR,
            min_h=extraction_min_h, min_w=extraction_min_w,
        )
        decode_time = time.time() - t1

        usable = len(crops_by_actor)
        total_crops = sum(len(c) for c in crops_by_actor.values())
        segformer_count = 0
        color_fallback_count = 0
        if verbose:
            print(f"    Usable actors: {usable}/{len(actors)} ({total_crops} crops, {decode_time:.1f}s decode)")

        # Analyze each actor
        t2 = time.time()
        for actor_id, crops in crops_by_actor.items():
            if not crops:
                continue

            # Run SegFormer on ALL actors regardless of size.
            # Safety layers handle small crops: internal skip <15x8px,
            # MIN_REGION_PIXELS=50 for color extraction, confidence filtering.
            avg_h = float(np.mean([c.shape[0] for c in crops]))
            if method == "segformer":
                attrs = analyze_crops_segformer(crops)
                segformer_count += 1
            elif method == "yolo":
                attrs = analyze_crops_yolo(crops)
            else:
                attrs = analyze_crops_color_only(crops)
                if method == "segformer":
                    color_fallback_count += 1

            desc = build_description(attrs)

            # Store by camera_actorID (matching entity ID format)
            entity_key = f"{cam}_actor_{actor_id}"
            all_actors[entity_key] = {
                "actor_id": actor_id,
                "camera": cam,
                "hair_color": attrs.get("hair_color", "unknown"),
                "upper_color": attrs.get("upper_color", "unknown"),
                "lower_color": attrs.get("lower_color", "unknown"),
                "lower_type": attrs.get("lower_type", "pants"),
                "shoe_color": attrs.get("shoe_color", "unknown"),
                "accessories": attrs.get("accessories", []),
                "carried_objects": attrs.get("carried_objects", []),
                "upper_texture": attrs.get("upper_texture", ""),
                "lower_texture": attrs.get("lower_texture", ""),
                "upper_brightness": attrs.get("upper_brightness", ""),
                "lower_brightness": attrs.get("lower_brightness", ""),
                "confidence": attrs.get("confidence", {}),
                "description": desc,
                "num_crops": len(crops),
                "avg_crop_height": int(np.mean([c.shape[0] for c in crops])),
            }

        analyze_time = time.time() - t2
        if verbose:
            tier_info = ""
            if method == "segformer" and color_fallback_count > 0:
                tier_info = f" ({segformer_count} segformer, {color_fallback_count} color-fallback)"
            print(f"    Analysis: {analyze_time:.1f}s ({method}){tier_info}")

        cam_stats[cam] = {
            "actors": len(actors),
            "usable": usable,
            "total_crops": total_crops,
            "decode_sec": round(decode_time, 1),
            "analyze_sec": round(analyze_time, 1),
            "segformer_count": segformer_count,
            "color_fallback_count": color_fallback_count,
        }

    total_time = time.time() - t0

    # Summary stats
    described = sum(1 for a in all_actors.values() if a["description"] != "a person")
    color_dist = Counter(a["upper_color"] for a in all_actors.values())

    result = {
        "slot": slot,
        "method": method,
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
        description="V10 Entity Description Extractor — Geom + Video → rich descriptions",
    )
    parser.add_argument("--slot", "-s", required=True,
        help="Slot to process (e.g., 2018-03-11.11-25.school)")
    parser.add_argument("--method", "-m", default="segformer",
        choices=["segformer", "yolo", "color-only"],
        help="Analysis method: segformer (best, default), yolo, color-only")
    parser.add_argument("--no-yolo", action="store_true",
        help="DEPRECATED: use --method color-only instead")
    parser.add_argument("--dry-run", action="store_true",
        help="Show what would be processed without extracting")
    parser.add_argument("--output", "-o",
        help=f"Output path (default: {OUTPUT_DIR}/{{slot}}.json)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Handle deprecated --no-yolo flag
    method = args.method
    if args.no_yolo and method == "segformer":
        method = "color-only"

    args = parser.parse_args()

    if args.dry_run:
        files = find_slot_files(args.slot)
        print(f"\n  Slot: {args.slot}")
        print(f"  Cameras with geom: {len(files)}")
        for cf in files:
            cam = cf["camera"]
            actors = parse_geom(cf["geom_path"])
            usable_segformer = sum(1 for aid in actors
                        for frames in [actors[aid].values()]
                        if any((bb[2]-bb[0]) >= MIN_BBOX_WIDTH and
                               (bb[3]-bb[1]) >= MIN_BBOX_HEIGHT
                               for bb in frames))
            usable_color = sum(1 for aid in actors
                        for frames in [actors[aid].values()]
                        if any((bb[2]-bb[0]) >= MIN_BBOX_WIDTH_COLOR and
                               (bb[3]-bb[1]) >= MIN_BBOX_HEIGHT_COLOR
                               for bb in frames))
            vp = cf["video_path"]
            has_video = vp and vp.exists()
            vfmt = cf.get("video_format", "none")
            print(f"    {cam}: {len(actors)} actors, {usable_segformer} segformer-ready, {usable_color} color-ready, video={'YES' if has_video else 'NO'} ({vfmt})")
        return

    result = process_slot(args.slot, method=method, verbose=args.verbose)

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
        print(f"    {info['camera']} actor {info['actor_id']:>6}: {desc}")
        if len(seen) >= 15:
            break


if __name__ == "__main__":
    main()

"""
generate_spatial.py — Closest-approach spatial questions.

Asks: "How close do two people come to each other in this scene?"

Instead of a single snapshot distance, computes the MINIMUM distance
(closest approach) over the overlapping time window when both entities
are visible on the same camera. This handles moving actors correctly.

Answer options: close together / moderate distance / far apart / cross paths
Distance buckets: <=5m = near, 5-15m = moderate, >15m = far
"Cross paths" = entities whose minimum distance is <=2m (they physically
pass through the same location at overlapping times).
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .parse_annotations import Event, find_clips_for_slot, DEFAULT_FRAMERATE
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .activity_hierarchy import humanize_activity, humanize_activity_gerund
from .utils.krtd import (
    load_camera_model, CameraModel,
    classify_proximity, INDOOR_CAMERAS,
)
from .utils.yaml_stream import stream_geom_records

DEFAULT_FPS = 30.0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_EDGE_MARGIN = 10  # pixels

# Sampling: sample every N frames for distance computation
SAMPLE_EVERY = 30  # ~1 second at 30fps

# Cross-paths threshold: if minimum distance <= this, entities "cross paths"
CROSS_PATHS_THRESHOLD_M = 2.0

# Same-person dedup thresholds
SAME_PERSON_MIN_DIST_M = 0.5     # closest approach < this = likely same person
SAME_PERSON_BBOX_IOU = 0.3       # bbox IoU above this = same person
SAME_PERSON_TEMPORAL_OVERLAP = 0.5  # frame overlap ratio threshold

# Maximum distance to consider (skip absurd projections)
MAX_REASONABLE_DISTANCE_M = 500.0


def _compute_bbox_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Compute Intersection over Union of two bounding boxes [x1, y1, x2, y2]."""
    if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_bbox_clipping_frame(bbox: List[int], margin: int = FRAME_EDGE_MARGIN) -> bool:
    """Check if a bounding box is clipping the frame edge."""
    if not bbox or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[:4]
    return (x1 <= margin or y1 <= margin or
            x2 >= FRAME_WIDTH - margin or y2 >= FRAME_HEIGHT - margin)


def _disambiguate_description(desc: str, entity: Entity, sg: SceneGraph,
                               other_desc: str) -> str:
    """Add disambiguating context when two entities share the same description."""
    primary_activity = None
    for evt in sg.events:
        if evt.camera_id == entity.camera_id:
            for actor in evt.actors:
                if actor["actor_id"] == entity.actor_id:
                    primary_activity = evt.activity
                    break
            if primary_activity:
                break
    if primary_activity:
        act_gerund = humanize_activity_gerund(primary_activity)
        return f"{desc} ({act_gerund.lower()})"
    return desc


# ============================================================================
# Geom-based closest approach computation
# ============================================================================

def _load_geom_bboxes_for_actors(geom_path: Path,
                                  actor_ids: Set[int],
                                  sample_every: int = SAMPLE_EVERY
                                  ) -> Dict[int, Dict[int, List[int]]]:
    """Load sampled bounding boxes for specific actors from geom.yml.
    
    Returns {actor_id: {frame: [x1,y1,x2,y2], ...}, ...}
    Samples every `sample_every` frames for efficiency.
    """
    result: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
    for rec in stream_geom_records(geom_path):
        aid = rec['id1']
        if aid not in actor_ids:
            continue
        frame = rec['ts0']
        if frame % sample_every != 0:
            continue
        result[aid][frame] = rec['g0']
    return dict(result)


def _compute_closest_approach(
    bboxes_a: Dict[int, List[int]],
    bboxes_b: Dict[int, List[int]],
    camera_model: CameraModel,
) -> Tuple[Optional[float], Optional[int], Optional[List[int]], Optional[List[int]]]:
    """Compute closest approach between two entities on the same camera.
    
    Finds the frame where the 3D distance between the two entities is minimized
    across their overlapping time window.
    
    Returns: (min_distance_m, closest_frame, bbox_a_at_frame, bbox_b_at_frame)
             or (None, None, None, None) if no valid overlap.
    """
    # Find overlapping frames (both actors have bboxes)
    frames_a = set(bboxes_a.keys())
    frames_b = set(bboxes_b.keys())
    common_frames = sorted(frames_a & frames_b)
    
    if not common_frames:
        # No exact frame matches - try nearest-neighbor matching
        # within a tolerance of 2*SAMPLE_EVERY frames
        tolerance = 2 * SAMPLE_EVERY
        paired_frames = []
        sorted_b = sorted(frames_b)
        for fa in sorted(frames_a):
            best_fb = None
            best_dist = tolerance + 1
            for fb in sorted_b:
                d = abs(fa - fb)
                if d < best_dist:
                    best_dist = d
                    best_fb = fb
                if fb > fa + tolerance:
                    break
            if best_fb is not None and best_dist <= tolerance:
                paired_frames.append((fa, best_fb))
        
        if not paired_frames:
            return None, None, None, None
        
        min_dist = float('inf')
        closest_frame = None
        closest_bbox_a = None
        closest_bbox_b = None
        
        for fa, fb in paired_frames:
            ba = bboxes_a[fa]
            bb = bboxes_b[fb]
            if _is_bbox_clipping_frame(ba) or _is_bbox_clipping_frame(bb):
                continue
            pos_a = camera_model.bbox_foot_to_world(ba)
            pos_b = camera_model.bbox_foot_to_world(bb)
            if pos_a is None or pos_b is None:
                continue
            dist = float(np.linalg.norm(pos_a - pos_b))
            if dist > MAX_REASONABLE_DISTANCE_M:
                continue
            if dist < min_dist:
                min_dist = dist
                closest_frame = fa
                closest_bbox_a = ba
                closest_bbox_b = bb
        
        if closest_frame is None:
            return None, None, None, None
        return min_dist, closest_frame, closest_bbox_a, closest_bbox_b
    
    # Common frames exist - compute distance at each
    min_dist = float('inf')
    closest_frame = None
    closest_bbox_a = None
    closest_bbox_b = None
    
    for frame in common_frames:
        ba = bboxes_a[frame]
        bb = bboxes_b[frame]
        if _is_bbox_clipping_frame(ba) or _is_bbox_clipping_frame(bb):
            continue
        pos_a = camera_model.bbox_foot_to_world(ba)
        pos_b = camera_model.bbox_foot_to_world(bb)
        if pos_a is None or pos_b is None:
            continue
        dist = float(np.linalg.norm(pos_a - pos_b))
        if dist > MAX_REASONABLE_DISTANCE_M:
            continue
        if dist < min_dist:
            min_dist = dist
            closest_frame = frame
            closest_bbox_a = ba
            closest_bbox_b = bb
    
    if closest_frame is None:
        return None, None, None, None
    return min_dist, closest_frame, closest_bbox_a, closest_bbox_b


# ============================================================================
# Candidate Finding (closest-approach version)
# ============================================================================

def _find_spatial_candidates(sg: SceneGraph, verbose: bool = False) -> List[Dict]:
    """Find entity pairs with valid closest-approach distances.
    
    For each pair of person entities on the same camera:
    1. Load sampled bboxes from geom.yml
    2. Compute closest approach distance over co-visible frames
    3. Record minimum distance, proximity, and reference frame
    """
    # Load camera models (skip indoor)
    camera_models: Dict[str, CameraModel] = {}
    for cam_id in sg.cameras:
        if cam_id in INDOOR_CAMERAS:
            continue
        model = load_camera_model(cam_id)
        if model is not None:
            camera_models[cam_id] = model
    
    if verbose:
        print(f"  Spatial: {len(camera_models)} cameras with KRTD models")
    
    if not camera_models:
        return []
    
    # Find geom.yml paths per camera
    clips = find_clips_for_slot(sg.slot)
    geom_paths: Dict[str, Path] = {}
    for clip in clips:
        cam = clip["camera_id"]
        if cam not in camera_models:
            continue
        act_path = Path(clip["activities_file"])
        geom_path = act_path.with_name(
            act_path.name.replace(".activities.yml", ".geom.yml")
        )
        if geom_path.exists() and geom_path.stat().st_size > 100:
            geom_paths[cam] = geom_path
    
    if verbose:
        print(f"  Spatial: {len(geom_paths)} cameras with geom.yml files")
    
    # Group person entities by camera
    cam_entities: Dict[str, List[Entity]] = defaultdict(list)
    for eid, ent in sg.entities.items():
        if ent.entity_type != "person":
            continue
        if ent.camera_id in camera_models and ent.camera_id in geom_paths:
            cam_entities[ent.camera_id].append(ent)
    
    candidates = []
    
    for cam_id, entities in cam_entities.items():
        if len(entities) < 2:
            continue
        
        model = camera_models[cam_id]
        geom_path = geom_paths[cam_id]
        
        # Collect actor IDs we need
        actor_ids = {ent.actor_id for ent in entities}
        
        # Load sampled bboxes for all actors on this camera
        all_bboxes = _load_geom_bboxes_for_actors(geom_path, actor_ids)
        
        if verbose:
            print(f"    Camera {cam_id}: {len(entities)} entities, "
                  f"{len(all_bboxes)} with geom bboxes")
        
        # Compare all pairs
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ent_a = entities[i]
                ent_b = entities[j]
                
                bboxes_a = all_bboxes.get(ent_a.actor_id, {})
                bboxes_b = all_bboxes.get(ent_b.actor_id, {})
                
                if not bboxes_a or not bboxes_b:
                    continue
                
                # Check temporal overlap
                overlap_start = max(ent_a.first_frame, ent_b.first_frame)
                overlap_end = min(ent_a.last_frame, ent_b.last_frame)
                if overlap_end <= overlap_start:
                    continue  # no temporal overlap
                
                # Compute closest approach
                min_dist, closest_frame, bbox_a, bbox_b = _compute_closest_approach(
                    bboxes_a, bboxes_b, model
                )
                
                if min_dist is None:
                    continue
                
                # Classify proximity
                proximity = classify_proximity(min_dist)
                
                # Check if entities cross paths (very close approach)
                crosses_paths = min_dist <= CROSS_PATHS_THRESHOLD_M
                
                candidates.append({
                    "entity_a": ent_a.entity_id,
                    "entity_b": ent_b.entity_id,
                    "camera": cam_id,
                    "min_distance_m": round(min_dist, 2),
                    "closest_frame": closest_frame,
                    "bbox_a": bbox_a,
                    "bbox_b": bbox_b,
                    "proximity": proximity,
                    "crosses_paths": crosses_paths,
                    "overlap_frames": (overlap_start, overlap_end),
                    "entity_a_obj": ent_a,
                    "entity_b_obj": ent_b,
                })
    
    return candidates


# ============================================================================
# Same-person filtering
# ============================================================================

def _is_likely_same_person(cand: Dict, resolved: ResolvedGraph) -> bool:
    """Check if two entities are likely the same person."""
    eid_a = cand["entity_a"]
    eid_b = cand["entity_b"]
    
    # Check 1: Same entity cluster
    for cluster in resolved.entity_clusters:
        if eid_a in cluster.entities and eid_b in cluster.entities:
            return True
    
    # Check 2: Bbox IoU at closest frame
    bbox_iou = _compute_bbox_iou(cand["bbox_a"], cand["bbox_b"])
    if bbox_iou > SAME_PERSON_BBOX_IOU:
        return True
    
    # Check 3: Very close proximity with high temporal overlap
    if cand["min_distance_m"] < SAME_PERSON_MIN_DIST_M:
        ent_a = cand["entity_a_obj"]
        ent_b = cand["entity_b_obj"]
        start_a, end_a = ent_a.first_frame, ent_a.last_frame
        start_b, end_b = ent_b.first_frame, ent_b.last_frame
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)
        if overlap_end > overlap_start:
            duration_a = max(end_a - start_a, 1)
            duration_b = max(end_b - start_b, 1)
            overlap_ratio = (overlap_end - overlap_start) / min(duration_a, duration_b)
            if overlap_ratio > SAME_PERSON_TEMPORAL_OVERLAP:
                return True
    
    return False


# ============================================================================
# Question Generation
# ============================================================================

def generate_spatial_qa(sg: SceneGraph, resolved: ResolvedGraph,
                        entity_descs: Dict[str, str],
                        rng: random.Random, count: int = 3,
                        verbose: bool = False,
                        fallback_eids: Optional[Set[str]] = None) -> List[Dict]:
    """
    Generate spatial closest-approach questions.
    
    Asks how close two people come to each other in the scene.
    Uses minimum distance over co-visible time window (not a single snapshot).
    """
    candidates = _find_spatial_candidates(sg, verbose)
    
    if verbose:
        print(f"  Spatial: {len(candidates)} candidate pairs")
    
    if not candidates:
        return []
    
    # Filter same-person pairs
    before = len(candidates)
    candidates = [c for c in candidates if not _is_likely_same_person(c, resolved)]
    if verbose and before != len(candidates):
        print(f"    Filtered {before - len(candidates)} same-person pairs "
              f"-> {len(candidates)} remaining")
    
    if not candidates:
        return []
    
    # Filter out pairs with fallback (non-visual) descriptions
    if fallback_eids:
        before = len(candidates)
        candidates = [
            c for c in candidates
            if c["entity_a"] not in fallback_eids and c["entity_b"] not in fallback_eids
        ]
        if verbose and before != len(candidates):
            print(f"    Filtered {before - len(candidates)} fallback-description pairs")
    
    # Try to disambiguate identical descriptions
    filtered = []
    for c in candidates:
        desc_a = entity_descs.get(c["entity_a"], "")
        desc_b = entity_descs.get(c["entity_b"], "")
        if desc_a and desc_b and desc_a == desc_b:
            ent_a = c["entity_a_obj"]
            ent_b = c["entity_b_obj"]
            new_a = _disambiguate_description(desc_a, ent_a, sg, desc_b)
            new_b = _disambiguate_description(desc_b, ent_b, sg, desc_a)
            if new_a != new_b:
                c["disambiguated_a"] = new_a
                c["disambiguated_b"] = new_b
                filtered.append(c)
            elif verbose:
                print(f"    Skipping: cannot disambiguate '{desc_a}'")
            continue
        filtered.append(c)
    candidates = filtered
    
    if not candidates:
        return []
    
    # Sort by distance diversity - pick from each proximity bucket
    near = [c for c in candidates if c["proximity"] == "near"]
    moderate = [c for c in candidates if c["proximity"] == "moderate"]
    far = [c for c in candidates if c["proximity"] == "far"]
    cross = [c for c in candidates if c["crosses_paths"]]
    
    rng.shuffle(near)
    rng.shuffle(moderate)
    rng.shuffle(far)
    rng.shuffle(cross)
    
    # Dedup: track (desc_a, desc_b, camera) tuples
    def _dedup_key(c):
        da = c.get("disambiguated_a") or entity_descs.get(c["entity_a"], "")
        db = c.get("disambiguated_b") or entity_descs.get(c["entity_b"], "")
        cam = c["camera"]
        pair = tuple(sorted([da, db]))
        return (pair, cam)
    
    seen_keys = set()
    def _try_add(c, selected_list):
        key = _dedup_key(c)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        selected_list.append(c)
        return True
    
    selected = []
    
    # Priority: cross-paths first (most interesting), then one from each bucket
    for c in cross:
        if len(selected) >= count:
            break
        _try_add(c, selected)
    
    for bucket in [near, moderate, far]:
        if len(selected) >= count:
            break
        for b in bucket:
            if _try_add(b, selected):
                break
    
    # Fill remaining
    remaining = near + moderate + far
    rng.shuffle(remaining)
    for c in remaining:
        if len(selected) >= count:
            break
        _try_add(c, selected)
    
    # Generate questions
    qa_pairs = []
    
    for idx, cand in enumerate(selected[:count]):
        ent_a = cand["entity_a_obj"]
        ent_b = cand["entity_b_obj"]
        proximity = cand["proximity"]
        min_dist = cand["min_distance_m"]
        crosses = cand["crosses_paths"]
        
        desc_a = cand.get("disambiguated_a") or entity_descs.get(cand["entity_a"], "a person")
        desc_b = cand.get("disambiguated_b") or entity_descs.get(cand["entity_b"], "a person")
        
        # Skip if descriptions are still identical
        if desc_a == desc_b:
            if verbose:
                print(f"    Skipping: identical descriptions '{desc_a}'")
            continue
        
        question = (
            f"How close do {desc_a} and {desc_b} come to each other in the scene?"
        )
        
        # Build options - crosses_paths is the "special" answer
        options = [
            "They come close together (within a few meters)",
            "They stay at a moderate distance (5-15 meters apart)",
            "They remain far apart (more than 15 meters)",
            "They cross paths (pass very close to each other)",
        ]
        
        if crosses:
            correct_idx = 3  # cross paths
        elif proximity == "near":
            correct_idx = 0
        elif proximity == "moderate":
            correct_idx = 1
        else:
            correct_idx = 2
        
        # Build reasoning
        if crosses:
            reasoning = (
                f"{desc_a.capitalize()} and {desc_b} come within "
                f"{min_dist:.1f} meters of each other, crossing paths "
                f"during the scene."
            )
        else:
            proximity_text = {
                "near": "close together",
                "moderate": "at a moderate distance",
                "far": "far apart",
            }
            reasoning = (
                f"The closest {desc_a} and {desc_b} come to each other is "
                f"approximately {min_dist:.1f} meters, placing them "
                f"{proximity_text.get(proximity, proximity)}."
            )
        
        # Find clip_file for the camera
        def _entity_clip_file(entity_id):
            ent = sg.entities.get(entity_id)
            if not ent:
                return ""
            event_map = {e.event_id: e for e in sg.events}
            for evid in ent.events:
                evt = event_map.get(evid)
                if evt and evt.video_file:
                    return evt.video_file.replace(".avi", ".mp4")
            return ""
        
        clip_file = _entity_clip_file(cand["entity_a"])
        if not clip_file:
            clip_file = _entity_clip_file(cand["entity_b"])
        
        overlap_start, overlap_end = cand["overlap_frames"]
        
        debug_info = {
            "entity_a": {
                "entity_id": cand["entity_a"],
                "camera": cand["camera"],
                "description": desc_a,
                "bbox_at_closest": cand["bbox_a"],
                "timestamp": f"{ent_a.first_sec:.2f}-{ent_a.last_sec:.2f}s",
            },
            "entity_b": {
                "entity_id": cand["entity_b"],
                "camera": cand["camera"],
                "description": desc_b,
                "bbox_at_closest": cand["bbox_b"],
                "timestamp": f"{ent_b.first_sec:.2f}-{ent_b.last_sec:.2f}s",
            },
            "min_distance_meters": min_dist,
            "closest_frame": cand["closest_frame"],
            "proximity": proximity,
            "crosses_paths": crosses,
            "overlap_frames": f"{overlap_start}-{overlap_end}",
            "projection_method": "krtd_bbox_foot_closest_approach",
            "clip_file": clip_file,
        }
        if clip_file:
            debug_info["clip_files"] = [clip_file]
        
        qa = {
            "question_id": f"v10_spatial_{idx+1:03d}",
            "category": "spatial",
            "difficulty": "easy" if crosses or proximity == "near" else "medium",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "reasoning": reasoning,
            "requires_cameras": [cand["camera"]],
            "requires_multi_camera": False,
            "verification": {
                "entity_a": cand["entity_a"],
                "entity_b": cand["entity_b"],
                "entity_a_desc": desc_a,
                "entity_b_desc": desc_b,
                "min_distance_meters": min_dist,
                "closest_frame": cand["closest_frame"],
                "proximity": proximity,
                "crosses_paths": crosses,
                "projection_method": "krtd_bbox_foot_closest_approach",
            },
            "debug_info": debug_info,
        }
        qa_pairs.append(qa)
    
    if verbose:
        print(f"  Spatial: {len(qa_pairs)} questions generated")
    
    return qa_pairs

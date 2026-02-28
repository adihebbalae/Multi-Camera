"""
V8 generate_spatial.py — Spatial entity distance questions with MEVID descriptions.

V8 CHANGES from V7:
- Entity descriptions from MEVID (GPT/YOLO) instead of actor ID aliases
- Questions use natural person descriptions for spatial references
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .parse_annotations import Event, find_clips_for_slot, DEFAULT_FRAMERATE
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .person_descriptions import enrich_entities
from .activity_hierarchy import humanize_activity, humanize_activity_gerund
from .utils.krtd import (
    load_camera_model, CameraModel, compute_entity_distance,
    classify_proximity, INDOOR_CAMERAS,
)
from .utils.yaml_stream import get_bbox_at_frame

# Scene context (optional — graceful degradation)
try:
    from .scene_context import get_scene_context, enrich_description_with_location
    _HAS_SCENE_CONTEXT = True
except ImportError:
    _HAS_SCENE_CONTEXT = False

DEFAULT_FPS = 30.0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_EDGE_MARGIN = 10  # pixels

# Issue 8: Same-person dedup thresholds
SAME_PERSON_PROXIMITY_M = 0.5    # entities closer than this are likely same person
SAME_PERSON_BBOX_IOU = 0.3       # bbox IoU above this = same person
SAME_PERSON_TEMPORAL_OVERLAP = 0.5  # frame overlap ratio threshold


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


def _is_likely_same_person(cand: Dict, resolved: ResolvedGraph) -> bool:
    """
    Issue 8: Check if two entities in a spatial candidate are likely the same person.

    Uses three independent checks (any one triggering = skip):
    1. Entity cluster membership (same resolved identity)
    2. Bounding box IoU > 0.3
    3. Distance < 0.5m with overlapping frame ranges
    """
    ent_a = cand["entity_a_obj"]
    ent_b = cand["entity_b_obj"]
    eid_a = cand["entity_a"]
    eid_b = cand["entity_b"]

    # Check 1: Same entity cluster
    for cluster in resolved.entity_clusters:
        if eid_a in cluster.entities and eid_b in cluster.entities:
            return True

    # Check 2: Bbox IoU
    bbox_iou = _compute_bbox_iou(cand["bbox_a"], cand["bbox_b"])
    if bbox_iou > SAME_PERSON_BBOX_IOU:
        return True

    # Check 3: Proximity + temporal overlap (same camera only)
    if cand["camera_a"] == cand["camera_b"] and cand["distance_m"] < SAME_PERSON_PROXIMITY_M:
        # Check frame overlap
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


def _is_bbox_clipping_frame(bbox: List[int], margin: int = FRAME_EDGE_MARGIN) -> bool:
    """Check if a bounding box is clipping the frame edge."""
    if not bbox or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[:4]
    return (x1 <= margin or y1 <= margin or
            x2 >= FRAME_WIDTH - margin or y2 >= FRAME_HEIGHT - margin)


def _disambiguate_description(desc: str, entity: Entity, sg: SceneGraph,
                               other_desc: str) -> str:
    """Add disambiguating context when two entities share the same description.
    
    Appends activity context to make descriptions unique.
    Camera IDs and raw timestamps are forbidden (Issue 12).
    """
    # Find primary activity for this entity
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
    # Cannot disambiguate without activity — return original
    return desc


# ============================================================================
# Spatial Candidate Finding (from V7, unchanged logic)
# ============================================================================

def _find_spatial_candidates(sg: SceneGraph, verbose: bool = False) -> List[Dict]:
    """Find entity pairs with valid KRTD projections for spatial questions."""
    candidates = []
    
    camera_models: Dict[str, CameraModel] = {}
    for cam_id in sg.cameras:
        if cam_id in INDOOR_CAMERAS:
            continue
        model = load_camera_model(cam_id)
        if model is not None:
            camera_models[cam_id] = model
    
    if verbose:
        print(f"  Spatial: {len(camera_models)} cameras with KRTD models")
    
    clips = find_clips_for_slot(sg.slot)
    clip_by_camera = {c["camera_id"]: c for c in clips}
    
    entity_positions: Dict[str, Dict] = {}
    
    for eid, entity in sg.entities.items():
        if entity.camera_id not in camera_models:
            continue
        if entity.entity_type != "person":
            continue
        
        model = camera_models[entity.camera_id]
        mid_frame = (entity.first_frame + entity.last_frame) // 2
        
        bbox = None
        if entity.keyframe_bboxes:
            closest_frame = min(entity.keyframe_bboxes.keys(),
                              key=lambda f: abs(int(f) - mid_frame))
            bbox = entity.keyframe_bboxes[closest_frame]
        
        if bbox is None and entity.camera_id in clip_by_camera:
            clip = clip_by_camera[entity.camera_id]
            geom_path = Path(clip["activities_file"]).with_name(
                Path(clip["activities_file"]).name.replace(".activities.yml", ".geom.yml")
            )
            if geom_path.exists():
                bbox = get_bbox_at_frame(geom_path, entity.actor_id, mid_frame, tolerance=15)
        
        if bbox is None:
            continue
        
        pos = model.bbox_foot_to_world(bbox)
        if pos is None:
            continue
        
        # V10: Skip entities whose bbox clips the frame edge
        if _is_bbox_clipping_frame(bbox):
            if verbose:
                print(f"    Skipping {eid}: bbox clips frame edge {bbox}")
            continue
        
        entity_positions[eid] = {
            "position": pos,
            "frame": mid_frame,
            "bbox": bbox,
            "camera_id": entity.camera_id,
            "entity": entity,
        }
    
    if verbose:
        print(f"  Spatial: {len(entity_positions)} entities with 3D positions")
    
    entity_ids = sorted(entity_positions.keys())
    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            eid_a = entity_ids[i]
            eid_b = entity_ids[j]
            
            pos_a = entity_positions[eid_a]
            pos_b = entity_positions[eid_b]
            
            # Same-camera only: both entities must be on the same camera
            # so the spatial relationship is visually verifiable in one frame
            if pos_a["camera_id"] != pos_b["camera_id"]:
                continue
            
            distance = float(np.linalg.norm(pos_a["position"] - pos_b["position"]))
            
            if distance > 500:
                continue
            
            proximity = classify_proximity(distance)
            
            candidates.append({
                "entity_a": eid_a,
                "entity_b": eid_b,
                "camera_a": pos_a["camera_id"],
                "camera_b": pos_b["camera_id"],
                "position_a": pos_a["position"].tolist(),
                "position_b": pos_b["position"].tolist(),
                "bbox_a": pos_a["bbox"],
                "bbox_b": pos_b["bbox"],
                "frame_a": pos_a["frame"],
                "frame_b": pos_b["frame"],
                "distance_m": round(distance, 2),
                "proximity": proximity,
                "entity_a_obj": pos_a["entity"],
                "entity_b_obj": pos_b["entity"],
            })
    
    return candidates


# ============================================================================
# Question Generation
# ============================================================================

def generate_spatial_qa(sg: SceneGraph, resolved: ResolvedGraph,
                        entity_descs: Dict[str, str],
                        rng: random.Random, count: int = 2,
                        verbose: bool = False,
                        fallback_eids: Optional[Set[str]] = None) -> List[Dict]:
    """
    Generate spatial entity distance questions with MEVID descriptions.
    
    V8: Uses entity_descs for person descriptions instead of actor ID aliases.
    """
    candidates = _find_spatial_candidates(sg, verbose)
    
    if verbose:
        print(f"  Spatial: {len(candidates)} candidate pairs")
    
    if not candidates:
        return []
    
    # Issue 8: Filter out pairs that are likely the same person
    before_same_person = len(candidates)
    candidates = [c for c in candidates if not _is_likely_same_person(c, resolved)]
    if verbose and before_same_person != len(candidates):
        print(f"    Filtered {before_same_person - len(candidates)} same-person pairs "
              f"→ {len(candidates)} remaining")
    
    if not candidates:
        return []

    # Filter out pairs with identical descriptions (indistinguishable entities)
    # V10: Instead of just filtering, try to disambiguate with activity/time context
    filtered = []
    for c in candidates:
        desc_a = entity_descs.get(c["entity_a"], "")
        desc_b = entity_descs.get(c["entity_b"], "")
        if desc_a and desc_b and desc_a == desc_b:
            # Try to disambiguate
            ent_a = c["entity_a_obj"]
            ent_b = c["entity_b_obj"]
            new_a = _disambiguate_description(desc_a, ent_a, sg, desc_b)
            new_b = _disambiguate_description(desc_b, ent_b, sg, desc_a)
            if new_a != new_b:
                # Store disambiguated descriptions for use in question text
                c["disambiguated_a"] = new_a
                c["disambiguated_b"] = new_b
                filtered.append(c)
            elif verbose:
                print(f"    Filtering spatial pair: cannot disambiguate '{desc_a}'")
            continue
        filtered.append(c)
    candidates = filtered

    # Filter out pairs where either entity has a fallback (non-visual) description
    if fallback_eids:
        before = len(candidates)
        candidates = [
            c for c in candidates
            if c["entity_a"] not in fallback_eids and c["entity_b"] not in fallback_eids
        ]
        if verbose and before != len(candidates):
            print(f"    Filtered {before - len(candidates)} spatial pairs (fallback descriptions)")

    # Sort by distance diversity
    near = [c for c in candidates if c["proximity"] == "near"]
    moderate = [c for c in candidates if c["proximity"] == "moderate"]
    far = [c for c in candidates if c["proximity"] == "far"]
    
    rng.shuffle(near)
    rng.shuffle(moderate)
    rng.shuffle(far)
    
    # Dedup: track (desc_a, desc_b, camera) tuples to avoid identical-looking questions
    def _dedup_key(c):
        da = c.get("disambiguated_a") or entity_descs.get(c["entity_a"], "")
        db = c.get("disambiguated_b") or entity_descs.get(c["entity_b"], "")
        cam = c["camera_a"]
        # Normalize order so (A,B) == (B,A)
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
    for bucket in [near, moderate, far]:
        if bucket and len(selected) < count:
            for b in bucket:
                if _try_add(b, selected):
                    break
    
    remaining = near[1:] + moderate[1:] + far[1:]
    rng.shuffle(remaining)
    for c in remaining:
        if len(selected) >= count:
            break
        _try_add(c, selected)
    
    qa_pairs = []
    
    for idx, cand in enumerate(selected[:count]):
        ent_a = cand["entity_a_obj"]
        ent_b = cand["entity_b_obj"]
        proximity = cand["proximity"]
        distance = cand["distance_m"]
        
        # All spatial questions are same-camera (filtered in _find_spatial_candidates)
        
        # Issue 12: Use descriptions WITHOUT camera IDs or raw timestamps
        desc_a = cand.get("disambiguated_a") or entity_descs.get(cand["entity_a"], "a person")
        desc_b = cand.get("disambiguated_b") or entity_descs.get(cand["entity_b"], "a person")
        
        # V10: Enrich with spatial location context if available
        if _HAS_SCENE_CONTEXT:
            parts = sg.slot.split(".")
            if len(parts) >= 3:
                site = parts[2]
                cam_model = load_camera_model(cand["camera_a"])
                if cam_model is not None:
                    # Get 3D point for entity A
                    if ent_a.keyframe_bboxes:
                        mid_a = min(ent_a.keyframe_bboxes.keys(),
                                    key=lambda f: abs(f - (ent_a.first_frame + ent_a.last_frame)//2))
                        pt_a = cam_model.bbox_foot_to_world(ent_a.keyframe_bboxes[mid_a])
                        if pt_a is not None:
                            desc_a = enrich_description_with_location(desc_a, pt_a, site)
                    if ent_b.keyframe_bboxes:
                        mid_b = min(ent_b.keyframe_bboxes.keys(),
                                    key=lambda f: abs(f - (ent_b.first_frame + ent_b.last_frame)//2))
                        pt_b = cam_model.bbox_foot_to_world(ent_b.keyframe_bboxes[mid_b])
                        if pt_b is not None:
                            desc_b = enrich_description_with_location(desc_b, pt_b, site)
        
        # Issue 12: No camera IDs or raw timestamps in question text
        # Skip pairs with identical descriptions (can't distinguish without camera IDs)
        if desc_a == desc_b:
            if verbose:
                print(f"    Skipping spatial pair: identical descriptions '{desc_a}'")
            continue
        
        question = (
            f"How close are {desc_a} and {desc_b} in the scene?"
        )
        
        options = [
            "They are near each other (within a few meters)",
            "They are at a moderate distance (5-15 meters)",
            "They are far apart (more than 15 meters)",
            "They are at the same location",
        ]
        
        if proximity == "near":
            correct_idx = 0
        elif proximity == "moderate":
            correct_idx = 1
        else:
            correct_idx = 2
        
        # Find clip_files for each entity from their events
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

        clip_a = _entity_clip_file(cand["entity_a"])
        clip_b = _entity_clip_file(cand["entity_b"])

        debug_info = {
            "entity_a": {
                "entity_id": cand["entity_a"],
                "camera": cand["camera_a"],
                "description": desc_a,
                "bbox": cand["bbox_a"],
                "frame": cand["frame_a"],
                "timestamp": f"{ent_a.first_sec:.2f}-{ent_a.last_sec:.2f}s",
                "world_pos_enu": cand["position_a"],
                "clip_file": clip_a,
            },
            "entity_b": {
                "entity_id": cand["entity_b"],
                "camera": cand["camera_b"],
                "description": desc_b,
                "bbox": cand["bbox_b"],
                "frame": cand["frame_b"],
                "timestamp": f"{ent_b.first_sec:.2f}-{ent_b.last_sec:.2f}s",
                "world_pos_enu": cand["position_b"],
                "clip_file": clip_b,
            },
            "distance_meters": distance,
            "proximity": proximity,
            "projection_method": "krtd_bbox_foot",
        }
        
        qa = {
            "question_id": f"v8_spatial_{idx+1:03d}",
            "category": "spatial",
            "difficulty": "easy",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "requires_cameras": [cand["camera_a"]],
            "requires_multi_camera": False,
            "verification": {
                "entity_a": cand["entity_a"],
                "entity_b": cand["entity_b"],
                "entity_a_desc": desc_a,
                "entity_b_desc": desc_b,
                "world_pos_a_enu": cand["position_a"],
                "world_pos_b_enu": cand["position_b"],
                "distance_meters": distance,
                "proximity": proximity,
                "projection_method": "krtd_bbox_foot",
            },
            "debug_info": debug_info,
        }
        qa_pairs.append(qa)
    
    if verbose:
        print(f"  Spatial: {len(qa_pairs)} questions generated")
    
    return qa_pairs

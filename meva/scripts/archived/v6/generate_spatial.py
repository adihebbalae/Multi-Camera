"""
V6 generate_spatial.py — Step 5: Spatial entity distance questions.

CONSTRAINTS:
- Ask about entity distances (not camera distances)
- Use bbox→3D projection via KRTD
- Ideally 2+ cameras, but 1 camera OK
- Mostly simple questions + 1 complex per slot for eval
- All easy difficulty by default
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .parse_annotations import Event, find_clips_for_slot
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .utils.krtd import (
    load_camera_model, CameraModel, compute_entity_distance,
    classify_proximity, INDOOR_CAMERAS,
)
from .utils.yaml_stream import get_bbox_at_frame


# ============================================================================
# Spatial Candidate Finding
# ============================================================================

def _find_spatial_candidates(sg: SceneGraph, verbose: bool = False) -> List[Dict]:
    """
    Find entity pairs with valid KRTD projections for spatial questions.
    
    For each pair of entities (same or different cameras), both with KRTD coverage
    and bounding boxes, project their foot points to 3D and compute distance.
    """
    candidates = []
    
    # Load camera models
    camera_models: Dict[str, CameraModel] = {}
    for cam_id in sg.cameras:
        if cam_id in INDOOR_CAMERAS:
            continue
        model = load_camera_model(cam_id)
        if model is not None:
            camera_models[cam_id] = model
    
    if verbose:
        print(f"  Spatial: {len(camera_models)} cameras with KRTD models")
    
    # Get clips for geom.yml access
    clips = find_clips_for_slot(sg.slot)
    clip_by_camera = {c["camera_id"]: c for c in clips}
    
    # For each entity with KRTD coverage and bboxes, compute world position
    entity_positions: Dict[str, Dict] = {}  # entity_id -> {position, frame, bbox, camera}
    
    for eid, entity in sg.entities.items():
        if entity.camera_id not in camera_models:
            continue
        if entity.entity_type != "person":
            continue
        
        model = camera_models[entity.camera_id]
        
        # Get a representative bbox (midpoint of entity's time span)
        mid_frame = (entity.first_frame + entity.last_frame) // 2
        
        # Try keyframe bboxes first
        bbox = None
        if entity.keyframe_bboxes:
            # Pick bbox closest to mid_frame
            closest_frame = min(entity.keyframe_bboxes.keys(),
                              key=lambda f: abs(int(f) - mid_frame))
            bbox = entity.keyframe_bboxes[closest_frame]
        
        # If no keyframe bbox, try streaming from geom.yml
        if bbox is None and entity.camera_id in clip_by_camera:
            clip = clip_by_camera[entity.camera_id]
            geom_path = Path(clip["activities_file"]).with_name(
                Path(clip["activities_file"]).name.replace(".activities.yml", ".geom.yml")
            )
            if geom_path.exists():
                bbox = get_bbox_at_frame(geom_path, entity.actor_id, mid_frame, tolerance=15)
        
        if bbox is None:
            continue
        
        # Project to 3D
        pos = model.bbox_foot_to_world(bbox)
        if pos is None:
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
    
    # Find pairs with valid distances
    entity_ids = sorted(entity_positions.keys())
    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            eid_a = entity_ids[i]
            eid_b = entity_ids[j]
            
            pos_a = entity_positions[eid_a]
            pos_b = entity_positions[eid_b]
            
            distance = float(np.linalg.norm(pos_a["position"] - pos_b["position"]))
            
            # Filter out unreasonable distances (> 500m probably projection error)
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
                        rng: random.Random, count: int = 3,
                        verbose: bool = False) -> List[Dict]:
    """
    Generate spatial entity distance questions.
    
    Types:
    - Simple (2 per slot): "Are entity A and entity B close or far apart?"
    - Complex (1 per slot): "From the perspective of entity A, where is entity B?"
    
    Args:
        sg: Scene graph
        resolved: Entity resolution results
        rng: Random number generator
        count: Target number of questions
        verbose: Print progress
    
    Returns:
        List of QA pair dicts
    """
    candidates = _find_spatial_candidates(sg, verbose)
    
    if verbose:
        print(f"  Spatial: {len(candidates)} candidate pairs")
    
    if not candidates:
        return []
    
    # Sort by distance diversity: pick one near, one far, one moderate
    near = [c for c in candidates if c["proximity"] == "near"]
    moderate = [c for c in candidates if c["proximity"] == "moderate"]
    far = [c for c in candidates if c["proximity"] == "far"]
    
    rng.shuffle(near)
    rng.shuffle(moderate)
    rng.shuffle(far)
    
    # Select diverse set
    selected = []
    for bucket in [near, moderate, far]:
        if bucket and len(selected) < count:
            selected.append(bucket[0])
    
    # Fill remaining from any bucket
    remaining = near[1:] + moderate[1:] + far[1:]
    rng.shuffle(remaining)
    for c in remaining:
        if len(selected) >= count:
            break
        selected.append(c)
    
    # Get activity names for entities
    def get_entity_activity(entity) -> str:
        """Get the most distinctive activity for an entity."""
        for evt in sg.events:
            if evt.camera_id == entity.camera_id:
                for actor in evt.actors:
                    if actor["actor_id"] == entity.actor_id:
                        return evt.activity
        return "person"
    
    qa_pairs = []
    
    for idx, cand in enumerate(selected[:count]):
        ent_a = cand["entity_a_obj"]
        ent_b = cand["entity_b_obj"]
        act_a = get_entity_activity(ent_a)
        act_b = get_entity_activity(ent_b)
        proximity = cand["proximity"]
        distance = cand["distance_m"]
        
        is_cross_camera = cand["camera_a"] != cand["camera_b"]
        
        # Simple proximity question
        if is_cross_camera:
            question = (
                f"In the scene, are {act_a} (camera {cand['camera_a']}) and "
                f"{act_b} (camera {cand['camera_b']}) close together or far apart?"
            )
        else:
            question = (
                f"How close are {act_a} and {act_b} "
                f"in the scene visible on camera {cand['camera_a']}?"
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
        
        qa = {
            "question_id": f"v6_spatial_{idx+1:03d}",
            "category": "spatial",
            "difficulty": "easy",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "requires_cameras": sorted(set([cand["camera_a"], cand["camera_b"]])),
            "requires_multi_camera": is_cross_camera,
            "verification": {
                "entity_a": cand["entity_a"],
                "entity_b": cand["entity_b"],
                "world_pos_a_enu": cand["position_a"],
                "world_pos_b_enu": cand["position_b"],
                "distance_meters": distance,
                "proximity": proximity,
                "projection_method": "krtd_bbox_foot",
            },
        }
        qa_pairs.append(qa)
    
    return qa_pairs

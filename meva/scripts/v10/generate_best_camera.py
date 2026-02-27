"""
FINAL generate_best_camera.py — Camera Transition Logic questions.

From the paper: "Identifying which specific camera in the synchronized network
first captures the entrance or appearance of an entity."

Example question: "Which camera first captures the entrance of a person
wearing a gray top into the scene?"

Uses `person_enters_scene_through_structure` activity annotations to
determine which camera first detects a specific entity entering, then
asks the user to identify that camera from a set of options.

Two question sub-types:
  1. first_entrance — "Which camera first captures the entrance of {entity}?"
  2. last_entrance — "Which camera last captures the entrance of {entity}?"
     (inverse — checks if model can distinguish temporal extremes)
"""

import random
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

from .parse_annotations import Event
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .person_descriptions import enrich_entities, get_mevid_persons_with_cameras
from .distractor_bank import get_camera_distractors
from .activity_hierarchy import humanize_activity, humanize_activity_gerund


# ============================================================================
# Constants
# ============================================================================

# Activities that indicate "entering the scene"
ENTRANCE_ACTIVITIES = {
    "person_enters_scene_through_structure",
    "person_exits_vehicle",       # effectively enters the scene on foot
}

# Minimum time gap between first and second camera entrance (seconds)
MIN_SEPARATION_SEC = 2.0

# Global pool of MEVA camera IDs (used as distractor padding when slot has few cameras)
_MEVA_CAMERA_POOL = [
    "G299", "G300", "G328", "G330", "G336", "G339", "G341", "G419", "G420",
    "G421", "G423", "G424", "G436", "G638", "G639", "G503", "G504", "G505",
    "G506", "G507", "G508", "G509",
]


# ============================================================================
# Entrance Event Collection
# ============================================================================

def _collect_entrance_events(sg: SceneGraph) -> List[Event]:
    """Collect all entrance-type events from the scene graph."""
    return [e for e in sg.events if e.activity in ENTRANCE_ACTIVITIES]


def _group_entrances_by_entity(entrance_events: List[Event],
                                sg: SceneGraph,
                                resolved: ResolvedGraph
                                ) -> Dict[str, List[Tuple[Event, str]]]:
    """
    Group entrance events by resolved entity cluster.
    
    Returns:
        { cluster_id: [(event, camera_id), ...] }  sorted by start_sec
    """
    # Map (camera, actor_id) -> entity_id -> cluster_id
    entity_to_cluster = {}
    for cluster in resolved.entity_clusters:
        for eid in cluster.entities:
            entity_to_cluster[eid] = cluster.cluster_id

    # Map entity_id -> (camera, actor_id)
    eid_to_key = {}
    for eid, entity in sg.entities.items():
        eid_to_key[eid] = (entity.camera_id, entity.actor_id)

    # Reverse: (camera, actor_id) -> cluster_id
    key_to_cluster = {}
    for eid, key in eid_to_key.items():
        if eid in entity_to_cluster:
            key_to_cluster[key] = entity_to_cluster[eid]

    # Group events by cluster
    cluster_events: Dict[str, List[Tuple[Event, str]]] = defaultdict(list)
    for event in entrance_events:
        for actor in event.actors:
            key = (event.camera_id, actor["actor_id"])
            cid = key_to_cluster.get(key)
            if cid:
                cluster_events[cid].append((event, event.camera_id))
            else:
                # No cluster — use camera+actor as standalone key
                standalone = f"standalone_{event.camera_id}_{actor['actor_id']}"
                cluster_events[standalone].append((event, event.camera_id))

    # Sort each cluster's events by time
    for cid in cluster_events:
        cluster_events[cid].sort(key=lambda x: x[0].start_sec)

    return dict(cluster_events)


def _get_entity_description(cluster_id: str, sg: SceneGraph,
                             entity_descs: Dict[str, str],
                             resolved: ResolvedGraph) -> str:
    """Get the best available VISUAL description for an entity cluster.
    
    Only returns clothing/appearance descriptions (from MEVID or geom-color).
    Filters out activity-verb fallbacks like 'a person puts down object'.
    """
    # Find entity IDs in this cluster
    for cluster in resolved.entity_clusters:
        if cluster.cluster_id == cluster_id:
            for eid in cluster.entities:
                desc = entity_descs.get(eid)
                if not desc or desc == "a person" or "someone" in desc.lower():
                    continue
                # Filter out activity-verb fallbacks — they contain verbs
                # Visual descriptions contain words like "top", "pants", "wearing"
                desc_lower = desc.lower()
                is_visual = any(kw in desc_lower for kw in 
                    ["top", "pants", "wearing", "shirt", "jacket", "color",
                     "blue", "red", "black", "white", "green", "gray", "yellow",
                     "brown", "orange", "purple", "backpack", "bag"])
                if is_visual:
                    return desc
            break
    return "a person"


# ============================================================================
# Question Generation
# ============================================================================

def generate_best_camera_qa(sg: SceneGraph, resolved: ResolvedGraph,
                             entity_descs: Dict[str, str],
                             rng: random.Random, count: int = 2,
                             verbose: bool = False) -> List[Dict]:
    """
    Generate Camera Transition Logic questions.
    
    "Which camera first captures the entrance of {entity} into the scene?"
    """
    entrance_events = _collect_entrance_events(sg)
    
    if len(entrance_events) < 2:
        if verbose:
            print("  Best Camera: Too few entrance events")
        return []
    
    # Group by entity cluster
    cluster_entrances = _group_entrances_by_entity(entrance_events, sg, resolved)
    
    # Filter to entities that appear on 2+ cameras (multi-camera entrance)
    multi_cam_clusters = {}
    for cid, events_cams in cluster_entrances.items():
        cameras_seen = set(cam for _, cam in events_cams)
        if len(cameras_seen) >= 2:
            multi_cam_clusters[cid] = events_cams
    
    # If no multi-camera entities, fall back to any entity with entrance events
    # and use all cameras in slot as options
    use_multi_cam = len(multi_cam_clusters) >= 1
    candidates = multi_cam_clusters if use_multi_cam else cluster_entrances
    
    if not candidates:
        if verbose:
            print("  Best Camera: No entrance events to use")
        return []
    
    all_cameras = sorted(sg.cameras.keys())
    qa_pairs = []
    used_clusters = set()
    
    # Attempt to generate `count` questions
    candidate_list = list(candidates.items())
    
    # V10: Sort candidates so entities with visual descriptions come first
    # This avoids generating generic "a person" questions when better options exist
    def _has_visual_desc(item):
        cid, _ = item
        desc = _get_entity_description(cid, sg, entity_descs, resolved)
        return 0 if desc != "a person" else 1
    candidate_list.sort(key=_has_visual_desc)
    
    # Shuffle within each priority group (visual first, then generic)
    visual_cands = [c for c in candidate_list if _has_visual_desc(c) == 0]
    generic_cands = [c for c in candidate_list if _has_visual_desc(c) == 1]
    rng.shuffle(visual_cands)
    rng.shuffle(generic_cands)
    candidate_list = visual_cands + generic_cands
    
    for cluster_id, events_cams in candidate_list:
        if len(qa_pairs) >= count:
            break
        if cluster_id in used_clusters:
            continue
        
        # Get entity description
        desc = _get_entity_description(cluster_id, sg, entity_descs, resolved)
        
        # Get cameras sorted by first entrance time
        cam_times: Dict[str, float] = {}
        cam_events: Dict[str, Event] = {}
        for event, cam in events_cams:
            if cam not in cam_times or event.start_sec < cam_times[cam]:
                cam_times[cam] = event.start_sec
                cam_events[cam] = event
        
        sorted_cams = sorted(cam_times.keys(), key=lambda c: cam_times[c])
        
        if len(sorted_cams) < 1:
            continue
        
        # ----- Sub-type 1: first_entrance -----
        if len(qa_pairs) < count:
            first_cam = sorted_cams[0]
            first_time = cam_times[first_cam]
            first_event = cam_events[first_cam]
            
            # Build options: correct camera + distractors
            other_cams = [c for c in all_cameras if c != first_cam]
            if len(other_cams) < 3:
                # Not enough cameras for 4 options
                distractor_cams = other_cams
            else:
                # Prefer cameras that DO have entrance events (harder distractors)
                entrance_cams = [c for c in sorted_cams[1:] if c != first_cam]
                non_entrance_cams = [c for c in other_cams if c not in entrance_cams]
                
                # Mix: 1-2 entrance cams + 1-2 non-entrance cams
                rng.shuffle(entrance_cams)
                rng.shuffle(non_entrance_cams)
                distractor_cams = (entrance_cams[:2] + non_entrance_cams[:2])[:3]
            
            options = [f"Camera {first_cam}"] + [f"Camera {c}" for c in distractor_cams]
            while len(options) < 4:
                # Pad with remaining slot cameras first, then global pool
                remaining = [c for c in all_cameras if f"Camera {c}" not in options]
                if not remaining:
                    remaining = [c for c in _MEVA_CAMERA_POOL if f"Camera {c}" not in options]
                if remaining:
                    options.append(f"Camera {rng.choice(remaining)}")
                else:
                    break
            
            rng.shuffle(options)
            correct_idx = options.index(f"Camera {first_cam}")
            
            # Build clip_file from the event's video_file
            clip_file = (first_event.video_file or '').replace('.avi', '.mp4')
            
            question_template = (
                f"Which camera first captures the entrance of {desc} into the scene?"
            )
            
            qa = {
                "question_id": "",  # will be renumbered
                "category": "best_camera",
                "subcategory": "first_entrance",
                "question_template": question_template,
                "options": options,
                "correct_answer": options[correct_idx],
                "correct_answer_index": correct_idx,
                "requires_cameras": sorted_cams + [c for c in all_cameras if c not in sorted_cams],
                "difficulty": "medium" if len(sorted_cams) >= 3 else "easy",
                "verification": {
                    "question_type": "first_entrance",
                    "correct_camera": first_cam,
                    "entrance_time_sec": round(first_time, 2),
                    "entity_description": desc,
                    "cluster_id": cluster_id,
                    "all_entrance_cameras": {
                        cam: round(cam_times[cam], 2) for cam in sorted_cams
                    },
                    "activity": first_event.activity,
                },
                "debug_info": {
                    "representative_event": {
                        "clip_file": clip_file,
                        "camera": first_cam,
                        "activity": first_event.activity,
                        "start_sec": round(first_time, 2),
                    },
                },
            }
            qa_pairs.append(qa)
            used_clusters.add(cluster_id)
        
        # ----- Sub-type 2: last_entrance (if enough cameras) -----
        if len(qa_pairs) < count and len(sorted_cams) >= 2:
            last_cam = sorted_cams[-1]
            last_time = cam_times[last_cam]
            last_event = cam_events[last_cam]
            
            # Ensure meaningful separation from first
            if last_time - cam_times[sorted_cams[0]] < MIN_SEPARATION_SEC:
                continue
            
            other_cams = [c for c in all_cameras if c != last_cam]
            rng.shuffle(other_cams)
            distractor_cams = other_cams[:3]
            
            options = [f"Camera {last_cam}"] + [f"Camera {c}" for c in distractor_cams]
            while len(options) < 4:
                remaining = [c for c in all_cameras if f"Camera {c}" not in options]
                if not remaining:
                    remaining = [c for c in _MEVA_CAMERA_POOL if f"Camera {c}" not in options]
                if remaining:
                    options.append(f"Camera {rng.choice(remaining)}")
                else:
                    break
            
            rng.shuffle(options)
            correct_idx = options.index(f"Camera {last_cam}")
            
            clip_file = (last_event.video_file or '').replace('.avi', '.mp4')
            
            question_template = (
                f"Of all cameras capturing {desc} entering the scene, "
                f"which camera captures this last?"
            )
            
            qa = {
                "question_id": "",
                "category": "best_camera",
                "subcategory": "last_entrance",
                "question_template": question_template,
                "options": options,
                "correct_answer": options[correct_idx],
                "correct_answer_index": correct_idx,
                "requires_cameras": sorted_cams + [c for c in all_cameras if c not in sorted_cams],
                "difficulty": "hard",
                "verification": {
                    "question_type": "last_entrance",
                    "correct_camera": last_cam,
                    "entrance_time_sec": round(last_time, 2),
                    "entity_description": desc,
                    "cluster_id": cluster_id,
                    "all_entrance_cameras": {
                        cam: round(cam_times[cam], 2) for cam in sorted_cams
                    },
                    "activity": last_event.activity,
                },
                "debug_info": {
                    "representative_event": {
                        "clip_file": clip_file,
                        "camera": last_cam,
                        "activity": last_event.activity,
                        "start_sec": round(last_time, 2),
                    },
                },
            }
            qa_pairs.append(qa)
    
    if verbose:
        print(f"  Best Camera: {len(qa_pairs)} questions generated "
              f"(from {len(entrance_events)} entrance events, "
              f"{len(multi_cam_clusters)} multi-cam entities)")
    
    return qa_pairs[:count]

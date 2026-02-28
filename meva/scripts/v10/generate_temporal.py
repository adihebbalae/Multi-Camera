"""
V10 generate_temporal.py — Multi-camera temporal cross-camera questions.

V10 CHANGES from V8:
- REMOVED entity-cluster linkage as scoring factor —
  deliberately pairs unrelated events across different cameras so VLMs
  can't "cheat" by inferring causal/narrative answers.
- Scoring now driven by CAMERA PROXIMITY: adjacent cameras at the same
  site get highest priority (events *require* multi-camera reasoning).
- Added "What happened before/after X?" question format (alongside
  "which occurred first?") to match ego-exo4d/agibot breadth.
- Uses new camera_proximity utility for indoor-aware spatial reasoning,
  including cameras without KRTD (admin, school hallways, bus indoor).
- Connection type metadata preserved for debug but not used for selection.
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .parse_annotations import Event, find_clips_for_slot
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .person_descriptions import enrich_entities, get_mevid_persons_with_cameras
from .distractor_bank import get_distractors
from .activity_hierarchy import (
    are_related, get_relationship, get_relationship_strength, humanize_activity,
    get_activity_entity_type,
)
from .utils.mevid import find_mevid_persons_for_slot
from .utils.krtd import load_camera_model, CameraModel, INDOOR_CAMERAS
from .utils.yaml_stream import get_bbox_at_frame

# Camera proximity (new V10) — graceful degradation
try:
    from .utils.camera_proximity import (
        score_camera_pair_for_temporal, get_proximity_tier,
    )
    _HAS_PROXIMITY = True
except ImportError:
    _HAS_PROXIMITY = False

# Scene context (optional — graceful degradation)
try:
    from .scene_context import get_scene_context, enrich_description_with_location
    _HAS_SCENE_CONTEXT = True
except ImportError:
    _HAS_SCENE_CONTEXT = False

# ============================================================================
# Constants
# ============================================================================

MIN_GAP = 1.0
MAX_GAP = 10.0
FALLBACK_MAX_GAP = 15.0
DEFAULT_FPS = 30.0
# Cross-camera duplicate detection: if two events have same activity,
# 3D positions within this distance AND time within this window,
# they are likely the same real-world event seen by different cameras.
CROSS_CAM_DEDUP_DISTANCE_M = 5.0    # meters
CROSS_CAM_DEDUP_TIME_SEC = 8.0      # seconds
# Frame boundary margin for bbox-in-frame validation
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_EDGE_MARGIN = 10  # pixels from edge to consider "clipping"


# ============================================================================
# Cross-Camera Event Deduplication
# ============================================================================

def _is_bbox_clipping_frame(bbox: List[int], margin: int = FRAME_EDGE_MARGIN) -> bool:
    """Check if a bounding box is clipping the frame edge (entity likely out of view)."""
    if not bbox or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = bbox[:4]
    return (x1 <= margin or y1 <= margin or 
            x2 >= FRAME_WIDTH - margin or y2 >= FRAME_HEIGHT - margin)


def _get_event_3d_position(event: Event, sg: SceneGraph) -> Optional[np.ndarray]:
    """Get approximate 3D world position for an event using KRTD projection.
    
    Uses the first person actor's bbox at the event's mid-frame.
    Returns ENU coordinates or None if projection fails.
    """
    if event.camera_id in INDOOR_CAMERAS:
        return None
    model = load_camera_model(event.camera_id)
    if model is None:
        return None
    
    # Find bbox for first person actor
    mid_frame = (event.start_frame + event.end_frame) // 2
    for eid, entity in sg.entities.items():
        if entity.camera_id != event.camera_id:
            continue
        for actor in event.actors:
            if actor["actor_id"] == entity.actor_id and entity.entity_type == "person":
                # Look for bbox near mid_frame
                if entity.keyframe_bboxes:
                    closest = min(entity.keyframe_bboxes.keys(),
                                 key=lambda f: abs(int(f) - mid_frame))
                    if abs(int(closest) - mid_frame) <= 30:  # within 1 second
                        bbox = entity.keyframe_bboxes[closest]
                        pos = model.bbox_foot_to_world(bbox)
                        return pos
    return None


# Camera overlap detection for aggressive dedup on overlapping FOVs
try:
    from .utils.camera_overlap import cameras_overlap as _cameras_overlap
    _HAS_OVERLAP = True
except ImportError:
    _HAS_OVERLAP = False


def _is_likely_duplicate_event(ea: Event, eb: Event, sg: SceneGraph) -> bool:
    """Check if two cross-camera events are likely the same real-world event.
    
    Two events are duplicates if they:
    1. Have the same activity type
    2. Are temporally close (within CROSS_CAM_DEDUP_TIME_SEC)
    3. Have similar 3D positions (within CROSS_CAM_DEDUP_DISTANCE_M)
    
    For cameras with known FOV overlap, the time window is applied more
    aggressively (any temporal proximity = likely duplicate).
    """
    # Must be same activity for duplicate detection
    if ea.activity != eb.activity:
        return False
    
    # Must be on different cameras (same-camera = not duplicates)
    if ea.camera_id == eb.camera_id:
        return False
    
    # Check if cameras have known overlap
    overlap = _HAS_OVERLAP and _cameras_overlap(ea.camera_id, eb.camera_id)
    
    # Check temporal proximity
    time_gap = abs(ea.start_sec - eb.start_sec)
    if overlap:
        # Overlapping cameras: wider time window, skip 3D check
        if time_gap <= CROSS_CAM_DEDUP_TIME_SEC:
            return True
        return False
    
    if time_gap > CROSS_CAM_DEDUP_TIME_SEC:
        return False
    
    # If we can get 3D positions, check spatial proximity
    pos_a = _get_event_3d_position(ea, sg)
    pos_b = _get_event_3d_position(eb, sg)
    
    if pos_a is not None and pos_b is not None:
        dist = float(np.linalg.norm(pos_a - pos_b))
        if dist < CROSS_CAM_DEDUP_DISTANCE_M:
            return True  # Same place, same time, same activity = duplicate
        # If they're far apart, they're distinct events even with same activity
        return False
    
    # Without 3D positions, use heuristic: same activity + close time = likely dup
    # Be conservative — only flag if very close temporally
    if time_gap <= 3.0:
        return True
    
    return False


def _event_has_visible_bbox(event: Event, sg: SceneGraph) -> bool:
    """Check if the event's primary actor has a bbox that is fully within the frame.
    
    Rejects actors whose bounding boxes clip the frame edge, since they
    may not be visually identifiable from the camera angle.
    """
    mid_frame = (event.start_frame + event.end_frame) // 2
    for eid, entity in sg.entities.items():
        if entity.camera_id != event.camera_id:
            continue
        for actor in event.actors:
            if actor["actor_id"] == entity.actor_id:
                if entity.keyframe_bboxes:
                    closest = min(entity.keyframe_bboxes.keys(),
                                 key=lambda f: abs(int(f) - mid_frame))
                    if abs(int(closest) - mid_frame) <= 30:
                        bbox = entity.keyframe_bboxes[closest]
                        if not _is_bbox_clipping_frame(bbox):
                            return True
                        return False  # bbox clips edge
    return True  # No bbox data — assume visible (don't block)


# ============================================================================
# Connection Scoring (V10: camera-proximity-driven, entity-cluster removed)
# ============================================================================

def _score_connection(event_a: Event, event_b: Event,
                      sg: SceneGraph, resolved: ResolvedGraph,
                      mevid_person_cameras: Dict[int, Set[str]]) -> Dict:
    """Score connection strength between two cross-camera events.

    V10 philosophy: Score is driven by CAMERA PROXIMITY, not entity linkage.
    Adjacent cameras at the same site produce the best multi-camera temporal
    questions. Entity-cluster and activity-relationship info is recorded in
    metadata for debugging but does NOT influence the score — we deliberately
    want unrelated event pairs so VLMs can't shortcut the answer via causal
    reasoning.
    """
    score = 0.0
    connection_type = "unrelated"
    connection_strength = "weak"
    mevid_validated = False
    mevid_person_id = None
    relationship = None
    cluster_id = None
    proximity_tier = None

    # Primary scoring signal: camera spatial proximity
    if _HAS_PROXIMITY:
        proximity_score = score_camera_pair_for_temporal(
            event_a.camera_id, event_b.camera_id
        )
        proximity_tier = get_proximity_tier(event_a.camera_id, event_b.camera_id)
        score += proximity_score
    else:
        # Fallback: any cross-camera pair on same site gets base score
        score += 1.0

    # Bonus: different activities = more interesting question (harder to guess)
    if event_a.activity != event_b.activity:
        score += 1.0

    # Record (but do NOT score) entity cluster linkage — metadata only
    for cluster_obj in resolved.entity_clusters:
        a_entities = set()
        b_entities = set()
        for eid in cluster_obj.entities:
            entity = sg.entities.get(eid)
            if not entity:
                continue
            if entity.camera_id == event_a.camera_id:
                for actor in event_a.actors:
                    if actor["actor_id"] == entity.actor_id:
                        a_entities.add(eid)
            if entity.camera_id == event_b.camera_id:
                for actor in event_b.actors:
                    if actor["actor_id"] == entity.actor_id:
                        b_entities.add(eid)
        if a_entities and b_entities:
            connection_type = "same_entity_cluster"
            cluster_id = cluster_obj.cluster_id
            break

    # Record (but do NOT score) activity relationships — metadata only
    rel = get_relationship(event_a.activity, event_b.activity)
    if rel:
        relationship = rel
        if connection_type == "unrelated":
            connection_type = f"related_activities_{rel}"

    # Record MEVID validation — metadata only (no score bonus)
    for pid, cameras in mevid_person_cameras.items():
        if event_a.camera_id in cameras and event_b.camera_id in cameras:
            mevid_validated = True
            mevid_person_id = pid
            break

    # Derive connection_strength from proximity tier (for debug display)
    if proximity_tier == "adjacent":
        connection_strength = "strong"
    elif proximity_tier == "same_site":
        connection_strength = "medium"
    else:
        connection_strength = "weak"

    return {
        "connection_type": connection_type,
        "connection_strength": connection_strength,
        "score": score,
        "mevid_validated": mevid_validated,
        "mevid_person_id": mevid_person_id,
        "relationship": relationship,
        "cluster_id": cluster_id,
        "proximity_tier": proximity_tier,
    }


def _build_debug_info(event: Event, sg: SceneGraph,
                      entity_descs: Dict[str, str]) -> Dict:
    """Build debug info for one event, using MEVID descriptions."""
    clip_file = event.video_file
    if clip_file.endswith(".avi"):
        clip_file = clip_file.replace(".avi", ".mp4")
    
    desc = None
    for eid, entity in sg.entities.items():
        if entity.camera_id == event.camera_id:
            for actor in event.actors:
                if actor["actor_id"] == entity.actor_id:
                    desc = entity_descs.get(eid, entity.alias)
                    break
            if desc:
                break
    
    return {
        "camera": event.camera_id,
        "activity": event.activity,
        "actor_ids": [a["actor_id"] for a in event.actors],
        "frame_range": [event.start_frame, event.end_frame],
        "timestamp": f"{event.start_sec:.2f}-{event.end_sec:.2f}s",
        "fps": DEFAULT_FPS,
        "clip_file": clip_file,
        "entity_description": desc,
    }


def _get_event_description(event: Event, sg: SceneGraph,
                           entity_descs: Dict[str, str],
                           fallback_eids: Optional[Set[str]] = None) -> str:
    """Get a human-readable description for an event, using MEVID if available.
    
    Prefers visual (non-fallback) descriptions. Falls back to 'someone {activity}'
    if only fallback descriptions are available.
    """
    from .person_descriptions import is_visual_description
    short_act = humanize_activity(event.activity)
    
    # Try to get entity description — prefer visual descriptions
    best_desc = None
    for eid, entity in sg.entities.items():
        if entity.camera_id == event.camera_id:
            for actor in event.actors:
                if actor["actor_id"] == entity.actor_id:
                    desc = entity_descs.get(eid)
                    if desc:
                        if fallback_eids and eid in fallback_eids:
                            # Record fallback but keep searching for visual
                            if best_desc is None:
                                best_desc = desc
                            continue
                        # Found a visual description — use it
                        if short_act in desc:
                            return desc
                        return f"{desc} {short_act}"
    
    # If only fallback was found, use it (better than "someone")
    if best_desc:
        if short_act in best_desc:
            return best_desc
        return f"{best_desc} {short_act}"
    
    # Last resort: use any available description from entity_descs
    for eid, entity in sg.entities.items():
        if entity.camera_id == event.camera_id:
            for actor in event.actors:
                if actor["actor_id"] == entity.actor_id:
                    desc = entity_descs.get(eid)
                    if desc and desc not in ("a person", "a vehicle", "someone"):
                        if short_act in desc:
                            return desc
                        return f"{desc} {short_act}"
    
    # Absolute fallback: use correct entity type (person vs vehicle)
    entity_type = get_activity_entity_type(event.activity)
    return f"a {entity_type} {short_act}"


def _enrich_with_location(desc: str, event: Event, sg: SceneGraph) -> str:
    """Optionally append location context to an event description using 3D scene models.

    e.g. "a person in navy top opens vehicle door" →
         "a person in navy top opens vehicle door near the school"
    """
    if not _HAS_SCENE_CONTEXT:
        return desc
    # Extract site from slot name
    parts = sg.slot.split(".")
    if len(parts) < 3:
        return desc
    site = parts[2]
    # Get entity bbox for 3D projection
    cam_model = load_camera_model(event.camera_id)
    if cam_model is None:
        return desc
    # Use midpoint frame bbox
    mid_frame = int((event.start_frame + event.end_frame) / 2)
    bbox = None
    for actor in event.actors:
        eid = f"{event.camera_id}_actor_{actor['actor_id']}"
        entity = sg.entities.get(eid)
        if entity and entity.keyframe_bboxes:
            # Find nearest keyframe
            nearest = min(entity.keyframe_bboxes.keys(),
                          key=lambda f: abs(f - mid_frame))
            bbox = entity.keyframe_bboxes[nearest]
            break
    if bbox is None:
        return desc
    point_3d = cam_model.bbox_foot_to_world(bbox)
    if point_3d is None:
        return desc
    return enrich_description_with_location(desc, point_3d, site)


def _short_option_label(desc: str, activity: str) -> str:
    """Build a short label for an option from the event description.
    
    E.g. 'a person in gray top entering through a structure' -> 
         'The person in gray top entering through a structure'
    """
    label = desc.strip()
    # Capitalize first letter for option text
    if label.startswith("a "):
        label = "The " + label[2:]
    elif label.startswith("someone "):
        label = "Someone " + label[8:]
    else:
        label = label[0].upper() + label[1:]
    return label


# ============================================================================
# Uniqueness Gate
# ============================================================================

DESC_UNIQUENESS_LONG_EVENT_SEC = 30.0  # align with event_ordering first-instance threshold


def _build_uniqueness_index(
    events: List[Event], sg: SceneGraph, entity_descs: Dict[str, str],
    fallback_eids: Optional[Set[str]] = None,
) -> Dict[Tuple[str, str], int]:
    """Build index of how many visually-indistinguishable entities share the
    same (camera, activity).  Returns {(camera_id, activity): count_of_entities}
    where count > 1 means ambiguity.

    Two entities on the same camera doing the same activity are ambiguous if
    their descriptions are identical (or both missing).
    """
    # (camera, activity) -> set of descriptions seen
    cam_act_descs: Dict[Tuple[str, str], Dict[str, int]] = {}

    for evt in events:
        key = (evt.camera_id, evt.activity)
        if key not in cam_act_descs:
            cam_act_descs[key] = {}
        for actor in evt.actors:
            eid = f"{evt.camera_id}_actor_{actor['actor_id']}"
            desc = entity_descs.get(eid, "")
            if not desc or desc in ("a person", "a vehicle", "someone"):
                desc = "__generic__"
            cam_act_descs[key][desc] = cam_act_descs[key].get(desc, 0) + 1

    # For each (cam, activity), record the max entity count sharing one description
    ambiguity: Dict[Tuple[str, str], int] = {}
    for key, desc_counts in cam_act_descs.items():
        ambiguity[key] = max(desc_counts.values()) if desc_counts else 0
    return ambiguity


def _event_is_unique(
    event: Event, sg: SceneGraph, entity_descs: Dict[str, str],
    uniqueness_index: Dict[Tuple[str, str], int],
) -> bool:
    """Return True if this event's (camera, activity, description) combination
    is unambiguous — no other entity on the same camera does the same activity
    with an indistinguishable description."""
    key = (event.camera_id, event.activity)
    return uniqueness_index.get(key, 0) <= 1


# ============================================================================
# Candidate Selection
# ============================================================================

def _find_temporal_candidates(events: List[Event], sg: SceneGraph,
                              resolved: ResolvedGraph,
                              mevid_person_cameras: Dict[int, Set[str]],
                              max_gap: float = MAX_GAP) -> List[Dict]:
    """Find cross-camera event pairs within temporal gap constraints.
    
    V10 additions:
    - Rejects pairs that are likely the same real-world event (cross-camera dedup)
    - Rejects events whose actors have bbox clipping the frame edge
    - First-instance filter only for long events (>30s), aligning with event_ordering
    """
    # First-instance filter: only for long-duration events (>30s).
    # Short discrete actions (<30s) may be genuinely different occurrences
    # by different people, so they're preserved — the uniqueness gate
    # handles ambiguity at selection time.
    first_instance: Dict[Tuple[str, str], Event] = {}
    short_events: List[Event] = []
    for evt in events:
        if evt.duration_sec > DESC_UNIQUENESS_LONG_EVENT_SEC:
            key = (evt.activity, evt.camera_id)
            if key not in first_instance or evt.start_sec < first_instance[key].start_sec:
                first_instance[key] = evt
        else:
            short_events.append(evt)
    # Merge: first instances of long events + all short events
    merged = list(first_instance.values()) + short_events
    # Skip events in the first 5 seconds (camera stabilization period)
    events = [e for e in merged if e.start_sec >= 5.0]
    events.sort(key=lambda e: e.start_sec)

    candidates = []
    seen = set()
    
    for i, ea in enumerate(events):
        for j in range(i + 1, len(events)):
            eb = events[j]
            if ea.camera_id == eb.camera_id:
                continue
            
            gap = eb.start_sec - ea.end_sec
            gap_rev = ea.start_sec - eb.end_sec
            
            first, second, actual_gap = None, None, None
            
            if gap >= MIN_GAP and gap <= max_gap:
                first, second, actual_gap = ea, eb, gap
            elif gap_rev >= MIN_GAP and gap_rev <= max_gap:
                first, second, actual_gap = eb, ea, gap_rev
            else:
                continue
            
            key = (first.activity, first.camera_id, second.activity, second.camera_id)
            if key in seen:
                continue
            seen.add(key)
            
            # V10: Cross-camera event dedup — reject duplicate events
            if _is_likely_duplicate_event(first, second, sg):
                continue
            
            # V10: Bbox-in-frame validation — reject actors clipping frame edge
            if not _event_has_visible_bbox(first, sg) or not _event_has_visible_bbox(second, sg):
                continue
            
            conn = _score_connection(first, second, sg, resolved, mevid_person_cameras)
            candidates.append({
                "event_a": first,
                "event_b": second,
                "gap_sec": round(actual_gap, 2),
                **conn,
            })
    
    candidates.sort(key=lambda c: (-c["score"], c["gap_sec"]))
    return candidates


# ============================================================================
# Question Generation
# ============================================================================

def _event_has_visual_desc(event: Event, sg: SceneGraph,
                           fallback_eids: Set[str]) -> bool:
    """Check if at least one actor in this event has a visual description."""
    for eid, entity in sg.entities.items():
        if entity.camera_id == event.camera_id:
            for actor in event.actors:
                if actor["actor_id"] == entity.actor_id:
                    if eid not in fallback_eids:
                        return True
    return False


# ============================================================================
# Question Format Builders (V10)
# ============================================================================

def _build_which_first_question(
    desc_a: str, desc_b: str, short_a: str, short_b: str,
    rng: random.Random,
) -> Tuple[str, List[str], int]:
    """Build a 'which occurred first?' question (original V8 format).

    Returns (question_text, options_list, correct_answer_index).
    event_a is always the one that occurred first.
    """
    question = f"{desc_a} and {desc_b} -- which occurred first?"
    options = [
        f"{short_a} occurred first",
        f"{short_b} occurred first",
        "They occurred simultaneously",
        "Cannot be determined",
    ]
    correct_idx = 0

    if rng.random() < 0.5:
        question = f"{desc_b} and {desc_a} -- which occurred first?"
        options = [
            f"{short_b} occurred first",
            f"{short_a} occurred first",
            "They occurred simultaneously",
            "Cannot be determined",
        ]
        correct_idx = 1

    return question, options, correct_idx


def generate_temporal_qa(sg: SceneGraph, resolved: ResolvedGraph,
                         entity_descs: Dict[str, str],
                         rng: random.Random, count: int = 2,
                         verbose: bool = False,
                         fallback_eids: Optional[Set[str]] = None) -> List[Dict]:
    """
    Generate temporal cross-camera questions.

    V10 changes:
      - Entity-cluster linkage removed from scoring — camera proximity drives
        pair selection instead (adjacent cams at same site = best).
      - Deliberately pairs unrelated events across cameras to prevent VLMs
        from shortcutting via causal/narrative reasoning.
      - All questions use "which occurred first?" format.
      - Uniqueness gate: rejects pairs where the activity+description is
        ambiguous on that camera (prevents indistinguishable-entity confusion).
      - Same_area relaxation: 2-camera sites (admin) allowed since sparse.
      - Max gap 10s (FALLBACK_MAX_GAP 15s).
    """
    slot_cameras = list(sg.cameras.keys())
    mevid_person_cameras = find_mevid_persons_for_slot(sg.slot, slot_cameras)

    # Build uniqueness index for the gate
    uniqueness_index = _build_uniqueness_index(
        sg.events, sg, entity_descs, fallback_eids
    )

    candidates = _find_temporal_candidates(
        sg.events, sg, resolved, mevid_person_cameras, MAX_GAP
    )

    if len(candidates) < count:
        candidates = _find_temporal_candidates(
            sg.events, sg, resolved, mevid_person_cameras, FALLBACK_MAX_GAP
        )

    if verbose:
        print(f"  Temporal: {len(candidates)} candidate pairs")

    if not candidates:
        return []

    # Filter out candidates where EITHER event uses fallback (non-visual) descriptions
    if fallback_eids:
        visual_candidates = [
            c for c in candidates
            if _event_has_visual_desc(c["event_a"], sg, fallback_eids)
            and _event_has_visual_desc(c["event_b"], sg, fallback_eids)
        ]
        if verbose and len(visual_candidates) < len(candidates):
            print(f"    Filtered {len(candidates) - len(visual_candidates)} "
                  f"fallback-only pairs → {len(visual_candidates)} remaining")
        candidates = visual_candidates

    # Uniqueness gate: reject pairs where either event's (cam, activity)
    # has multiple entities with indistinguishable descriptions
    pre_uniq = len(candidates)
    candidates = [
        c for c in candidates
        if _event_is_unique(c["event_a"], sg, entity_descs, uniqueness_index)
        and _event_is_unique(c["event_b"], sg, entity_descs, uniqueness_index)
    ]
    if verbose and len(candidates) < pre_uniq:
        print(f"    Uniqueness gate: {pre_uniq - len(candidates)} ambiguous "
              f"pairs rejected → {len(candidates)} remaining")

    if not candidates:
        return []

    # Determine if this site has only 2 cameras (sparse site like admin)
    # If so, allow same_area pairs as fallback since there's no alternative
    n_cameras = len(slot_cameras)
    allow_same_area = (n_cameras <= 2)

    # ----------------------------------------------------------------
    # Selection: camera-proximity-based, single pass (no tiered passes)
    # Prefer: adjacent cameras > same-site > same-area, diverse activities
    # ----------------------------------------------------------------
    used_event_ids: Set[str] = set()
    used_activity_names: Set[str] = set()
    selected = []

    for c in candidates:
        if len(selected) >= count:
            break
        ea_id = c["event_a"].event_id
        eb_id = c["event_b"].event_id
        if ea_id in used_event_ids or eb_id in used_event_ids:
            continue
        if (c["event_a"].activity in used_activity_names
                or c["event_b"].activity in used_activity_names):
            continue
        # Skip same_area pairs unless sparse site (≤2 cameras)
        if _HAS_PROXIMITY and not allow_same_area:
            tier = get_proximity_tier(c["event_a"].camera_id, c["event_b"].camera_id)
            if tier == "same_area":
                continue
        selected.append(c)
        used_event_ids.add(ea_id)
        used_event_ids.add(eb_id)
        used_activity_names.add(c["event_a"].activity)
        used_activity_names.add(c["event_b"].activity)

    # If still short, relax the activity-uniqueness constraint
    if len(selected) < count:
        for c in candidates:
            if len(selected) >= count:
                break
            if c in selected:
                continue
            ea_id = c["event_a"].event_id
            eb_id = c["event_b"].event_id
            if ea_id in used_event_ids or eb_id in used_event_ids:
                continue
            if _HAS_PROXIMITY and not allow_same_area:
                tier = get_proximity_tier(c["event_a"].camera_id, c["event_b"].camera_id)
                if tier == "same_area":
                    continue
            selected.append(c)
            used_event_ids.add(ea_id)
            used_event_ids.add(eb_id)

    # ----------------------------------------------------------------
    # Generate QA pairs — always "which occurred first?" format
    # ----------------------------------------------------------------
    qa_pairs = []

    for idx, cand in enumerate(selected[:count]):
        ea = cand["event_a"]
        eb = cand["event_b"]
        gap = cand["gap_sec"]

        desc_a = _get_event_description(ea, sg, entity_descs, fallback_eids)
        desc_b = _get_event_description(eb, sg, entity_descs, fallback_eids)

        # Enrich with spatial location context (e.g. "near the school")
        desc_a = _enrich_with_location(desc_a, ea, sg)
        desc_b = _enrich_with_location(desc_b, eb, sg)

        # If descriptions identical after enrichment, skip (can't distinguish)
        if desc_a == desc_b:
            if verbose:
                print(f"    Skipping temporal pair: identical descriptions '{desc_a}'")
            continue

        short_a = _short_option_label(desc_a, ea.activity)
        short_b = _short_option_label(desc_b, eb.activity)

        q_format = "which_first"
        question, options, correct_idx = _build_which_first_question(
            desc_a, desc_b, short_a, short_b, rng
        )

        debug_info = {
            "event_a": _build_debug_info(ea, sg, entity_descs),
            "event_b": _build_debug_info(eb, sg, entity_descs),
            "gap_sec": gap,
            "connection_type": cand["connection_type"],
            "connection_strength": cand["connection_strength"],
            "connection_score": cand["score"],
            "relationship": cand["relationship"],
            "cluster_id": cand.get("cluster_id"),
            "proximity_tier": cand.get("proximity_tier"),
            "mevid_validated": cand["mevid_validated"],
            "mevid_person_id": cand["mevid_person_id"],
            "question_format": "which_first",
        }

        qa = {
            "question_id": f"v10_temporal_{idx+1:03d}",
            "category": "temporal",
            "difficulty": "easy",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "requires_cameras": sorted(set([ea.camera_id, eb.camera_id])),
            "requires_multi_camera": True,
            "verification": {
                "event_a": {
                    "activity": ea.activity,
                    "camera": ea.camera_id,
                    "start_sec": ea.start_sec,
                    "end_sec": ea.end_sec,
                    "actor_ids": [a["actor_id"] for a in ea.actors],
                    "description": desc_a,
                },
                "event_b": {
                    "activity": eb.activity,
                    "camera": eb.camera_id,
                    "start_sec": eb.start_sec,
                    "end_sec": eb.end_sec,
                    "actor_ids": [a["actor_id"] for a in eb.actors],
                    "description": desc_b,
                },
                "gap_sec": gap,
                "entity_link": "mevid_validated" if cand["mevid_validated"] else "heuristic",
                "same_person": cand["connection_type"] == "same_entity_cluster",
            },
            "debug_info": debug_info,
        }
        qa_pairs.append(qa)

    if verbose:
        prox_counts = {}
        for q in qa_pairs:
            tier = q["debug_info"].get("proximity_tier", "unknown")
            prox_counts[tier] = prox_counts.get(tier, 0) + 1
        print(f"  Temporal: {len(qa_pairs)} questions "
              f"(proximity: {prox_counts})")

    return qa_pairs

"""
V8 generate_temporal.py — Temporal cross-camera questions with MEVID person descriptions.

V8 CHANGES from V7:
- Entity aliases replaced with MEVID person descriptions when available
- Questions prioritize events involving described persons
- Description-enriched question text for better naturalization
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .parse_annotations import Event
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .person_descriptions import enrich_entities, get_mevid_persons_with_cameras
from .distractor_bank import get_distractors
from .activity_hierarchy import (
    are_related, get_relationship, get_relationship_strength, humanize_activity,
)
from .utils.mevid import find_mevid_persons_for_slot

# ============================================================================
# Constants
# ============================================================================

MIN_GAP = 1.0
MAX_GAP = 15.0
FALLBACK_MAX_GAP = 20.0
DEFAULT_FPS = 30.0


# ============================================================================
# Connection Scoring (from V7, unchanged)
# ============================================================================

def _score_connection(event_a: Event, event_b: Event,
                      sg: SceneGraph, resolved: ResolvedGraph,
                      mevid_person_cameras: Dict[int, Set[str]]) -> Dict:
    """Score connection strength between two events."""
    score = 0.0
    connection_type = "unrelated"
    connection_strength = "weak"
    mevid_validated = False
    mevid_person_id = None
    relationship = None
    cluster_id = None
    
    for cluster in resolved.entity_clusters:
        a_entities = set()
        b_entities = set()
        for eid in cluster.entities:
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
            connection_strength = "strong"
            score += 3.0
            cluster_id = cluster.cluster_id
            break
    
    rel = get_relationship(event_a.activity, event_b.activity)
    if rel:
        relationship = rel
        rel_score = get_relationship_strength(event_a.activity, event_b.activity)
        score += rel_score * 2.0
        if connection_type == "unrelated":
            connection_type = f"related_activities_{rel}"
            connection_strength = "medium" if rel_score >= 0.7 else "weak"
    
    for pid, cameras in mevid_person_cameras.items():
        if event_a.camera_id in cameras and event_b.camera_id in cameras:
            mevid_validated = True
            mevid_person_id = pid
            score += 1.5
            break
    
    if event_a.activity != event_b.activity:
        score += 0.5
    
    return {
        "connection_type": connection_type,
        "connection_strength": connection_strength,
        "score": score,
        "mevid_validated": mevid_validated,
        "mevid_person_id": mevid_person_id,
        "relationship": relationship,
        "cluster_id": cluster_id,
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
    
    # If only fallback was found, use generic form
    if best_desc and (not fallback_eids):
        if short_act in best_desc:
            return best_desc
        return f"{best_desc} {short_act}"
    
    # Fallback: V7-style
    return f"someone {short_act}"


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
# Candidate Selection
# ============================================================================

def _find_temporal_candidates(events: List[Event], sg: SceneGraph,
                              resolved: ResolvedGraph,
                              mevid_person_cameras: Dict[int, Set[str]],
                              max_gap: float = MAX_GAP) -> List[Dict]:
    """Find cross-camera event pairs within temporal gap constraints."""
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


def generate_temporal_qa(sg: SceneGraph, resolved: ResolvedGraph,
                         entity_descs: Dict[str, str],
                         rng: random.Random, count: int = 2,
                         verbose: bool = False,
                         fallback_eids: Optional[Set[str]] = None) -> List[Dict]:
    """
    Generate temporal cross-camera questions with MEVID person descriptions.
    
    V8 changes:
      - entity_descs parameter provides MEVID descriptions
      - Question text uses natural person descriptions instead of actor IDs
      - Prioritizes events involving described persons
    """
    slot_cameras = list(sg.cameras.keys())
    mevid_person_cameras = find_mevid_persons_for_slot(sg.slot, slot_cameras)
    
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
    # Both events appear in the question text, so both must have visual descriptions
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
    
    if not candidates:
        return []
    
    # Diversify selection: strong > medium > weak, MEVID-validated preferred
    used_pairs = set()
    used_activities = set()
    used_event_ids: Set[str] = set()        # No event reuse across questions
    used_activity_names: Set[str] = set()   # No activity string reuse across questions
    selected = []
    
    # Pass 1: strong connection + MEVID-validated (best quality)
    for c in candidates:
        if len(selected) >= count:
            break
        if c["connection_strength"] == "strong" and c["mevid_validated"]:
            ea_id = c["event_a"].event_id
            eb_id = c["event_b"].event_id
            if ea_id in used_event_ids or eb_id in used_event_ids:
                continue
            if c["event_a"].activity in used_activity_names or c["event_b"].activity in used_activity_names:
                continue
            cam_pair = (c["event_a"].camera_id, c["event_b"].camera_id)
            act_pair = (c["event_a"].activity, c["event_b"].activity)
            if cam_pair not in used_pairs or act_pair not in used_activities:
                used_pairs.add(cam_pair)
                used_activities.add(act_pair)
                selected.append(c)
                used_event_ids.add(ea_id)
                used_event_ids.add(eb_id)
                used_activity_names.add(c["event_a"].activity)
                used_activity_names.add(c["event_b"].activity)
    
    # Pass 2: strong connection (entity cluster linked)
    for c in candidates:
        if len(selected) >= count:
            break
        if c in selected:
            continue
        if c["connection_strength"] == "strong":
            ea_id = c["event_a"].event_id
            eb_id = c["event_b"].event_id
            if ea_id in used_event_ids or eb_id in used_event_ids:
                continue
            if c["event_a"].activity in used_activity_names or c["event_b"].activity in used_activity_names:
                continue
            cam_pair = (c["event_a"].camera_id, c["event_b"].camera_id)
            act_pair = (c["event_a"].activity, c["event_b"].activity)
            if cam_pair not in used_pairs or act_pair not in used_activities:
                used_pairs.add(cam_pair)
                used_activities.add(act_pair)
                selected.append(c)
                used_event_ids.add(ea_id)
                used_event_ids.add(eb_id)
                used_activity_names.add(c["event_a"].activity)
                used_activity_names.add(c["event_b"].activity)
    
    # Pass 3: medium connection (related activities)
    for c in candidates:
        if len(selected) >= count:
            break
        if c in selected:
            continue
        ea_id = c["event_a"].event_id
        eb_id = c["event_b"].event_id
        if ea_id in used_event_ids or eb_id in used_event_ids:
            continue
        if c["event_a"].activity in used_activity_names or c["event_b"].activity in used_activity_names:
            continue
        if c["connection_strength"] == "medium":
            selected.append(c)
            used_event_ids.add(ea_id)
            used_event_ids.add(eb_id)
            used_activity_names.add(c["event_a"].activity)
            used_activity_names.add(c["event_b"].activity)
    
    # Pass 4: fill remaining from any candidates (score-sorted order)
    for c in candidates:
        if len(selected) >= count:
            break
        if c not in selected:
            ea_id = c["event_a"].event_id
            eb_id = c["event_b"].event_id
            if ea_id in used_event_ids or eb_id in used_event_ids:
                continue
            if c["event_a"].activity in used_activity_names or c["event_b"].activity in used_activity_names:
                continue
            selected.append(c)
            used_event_ids.add(ea_id)
            used_event_ids.add(eb_id)
            used_activity_names.add(c["event_a"].activity)
            used_activity_names.add(c["event_b"].activity)
    
    # Generate QA pairs
    qa_pairs = []
    
    for idx, cand in enumerate(selected[:count]):
        ea = cand["event_a"]
        eb = cand["event_b"]
        gap = cand["gap_sec"]
        
        desc_a = _get_event_description(ea, sg, entity_descs, fallback_eids)
        desc_b = _get_event_description(eb, sg, entity_descs, fallback_eids)
        
        # Build short option labels from descriptions (no camera IDs)
        short_a = _short_option_label(desc_a, ea.activity)
        short_b = _short_option_label(desc_b, eb.activity)
        
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
        
        debug_info = {
            "event_a": _build_debug_info(ea, sg, entity_descs),
            "event_b": _build_debug_info(eb, sg, entity_descs),
            "gap_sec": gap,
            "connection_type": cand["connection_type"],
            "connection_strength": cand["connection_strength"],
            "connection_score": cand["score"],
            "relationship": cand["relationship"],
            "cluster_id": cand.get("cluster_id"),
            "mevid_validated": cand["mevid_validated"],
            "mevid_person_id": cand["mevid_person_id"],
        }
        
        qa = {
            "question_id": f"v8_temporal_{idx+1:03d}",
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
        mevid_count = sum(1 for q in qa_pairs 
                          if q["debug_info"]["mevid_validated"])
        print(f"  Temporal: {len(qa_pairs)} questions ({mevid_count} MEVID-validated)")
    
    return qa_pairs

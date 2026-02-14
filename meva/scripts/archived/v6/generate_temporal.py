"""
V6 generate_temporal.py — Step 4: Temporal cross-camera questions.

CONSTRAINTS:
- 2+ cameras REQUIRED (strict enforcement)
- Temporal gap: minimum 3 seconds, maximum 15 seconds (fallback: 20s if <3 candidates)
- Uses raw annotation activity text
- Uses linked entities from entity resolution when available
"""

import random
from typing import Any, Dict, List, Optional, Set

from .parse_annotations import Event
from .build_scene_graph import SceneGraph
from .entity_resolution import ResolvedGraph
from .distractor_bank import get_distractors

# ============================================================================
# Constants
# ============================================================================

MIN_GAP = 3.0       # minimum seconds between events
MAX_GAP = 15.0      # maximum seconds between events  
FALLBACK_MAX_GAP = 20.0  # used only if <3 candidates at MAX_GAP


# ============================================================================
# Candidate Selection
# ============================================================================

def _find_temporal_candidates(events: List[Event], 
                              max_gap: float = MAX_GAP) -> List[Dict]:
    """
    Find cross-camera event pairs within temporal gap constraints.
    
    Returns list of {event_a, event_b, gap_sec} dicts.
    """
    candidates = []
    seen = set()  # avoid duplicate activity pairs
    
    for i, ea in enumerate(events):
        for j in range(i + 1, len(events)):
            eb = events[j]
            
            # Must be cross-camera
            if ea.camera_id == eb.camera_id:
                continue
            
            # Gap: B starts after A ends
            gap = eb.start_sec - ea.end_sec
            
            # Also check reverse
            gap_rev = ea.start_sec - eb.end_sec
            
            # Use the positive gap direction
            if gap >= MIN_GAP and gap <= max_gap:
                key = (ea.activity, ea.camera_id, eb.activity, eb.camera_id)
                if key not in seen:
                    seen.add(key)
                    candidates.append({
                        "event_a": ea,
                        "event_b": eb,
                        "gap_sec": round(gap, 2),
                        "a_first": True,
                    })
            elif gap_rev >= MIN_GAP and gap_rev <= max_gap:
                key = (eb.activity, eb.camera_id, ea.activity, ea.camera_id)
                if key not in seen:
                    seen.add(key)
                    candidates.append({
                        "event_a": eb,
                        "event_b": ea,
                        "gap_sec": round(gap_rev, 2),
                        "a_first": True,
                    })
    
    return candidates


# ============================================================================
# Question Generation
# ============================================================================

def generate_temporal_qa(sg: SceneGraph, resolved: ResolvedGraph,
                         rng: random.Random, count: int = 3,
                         verbose: bool = False) -> List[Dict]:
    """
    Generate temporal cross-camera questions.
    
    Template: "A {activity_a} event on camera {cam_a} and a {activity_b} event
              on camera {cam_b} — which occurred first?"
    Options: [A first, B first, simultaneously, cannot determine]
    
    Args:
        sg: Scene graph
        resolved: Entity resolution results 
        rng: Random number generator
        count: Target number of questions
        verbose: Print progress
    
    Returns:
        List of QA pair dicts
    """
    # Find candidates with standard gap
    candidates = _find_temporal_candidates(sg.events, MAX_GAP)
    
    # Fallback: if not enough, ease to 20s
    if len(candidates) < count:
        candidates = _find_temporal_candidates(sg.events, FALLBACK_MAX_GAP)
        if verbose and len(candidates) >= count:
            print(f"  Temporal: used fallback max_gap={FALLBACK_MAX_GAP}s "
                  f"({len(candidates)} candidates)")
    
    if verbose:
        print(f"  Temporal: {len(candidates)} candidate pairs")
    
    if not candidates:
        return []
    
    # Sort by gap (prefer tighter gaps)
    candidates.sort(key=lambda c: c["gap_sec"])
    
    # Diversify: prefer different camera pairs and activity types
    # Group by (cam_a, cam_b) and pick at most 1 per pair
    used_pairs = set()
    used_activities = set()
    selected = []
    
    for c in candidates:
        cam_pair = (c["event_a"].camera_id, c["event_b"].camera_id)
        act_pair = (c["event_a"].activity, c["event_b"].activity)
        
        if cam_pair in used_pairs and len(selected) < count:
            continue  # skip same camera pair unless we need more
        if act_pair in used_activities and len(selected) < count:
            continue
        
        used_pairs.add(cam_pair)
        used_activities.add(act_pair)
        selected.append(c)
        
        if len(selected) >= count:
            break
    
    # If not enough after diversification, fill from remaining
    if len(selected) < count:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= count:
                break
    
    # Generate QA pairs
    qa_pairs = []
    slot_activities = set(e.activity for e in sg.events)
    
    for idx, cand in enumerate(selected[:count]):
        ea = cand["event_a"]
        eb = cand["event_b"]
        gap = cand["gap_sec"]
        
        # Check if entities are linked (same person across cameras)
        same_person = False
        mevid_pid = None
        for cluster in resolved.entity_clusters:
            a_entities = [eid for eid in cluster.entities if ea.camera_id in eid]
            b_entities = [eid for eid in cluster.entities if eb.camera_id in eid]
            if a_entities and b_entities:
                same_person = True
                mevid_pid = cluster.mevid_person_id
                break
        
        # Build question text using raw annotation names
        question = (
            f"A {ea.activity} event on camera {ea.camera_id} and a "
            f"{eb.activity} event on camera {eb.camera_id} — which occurred first?"
        )
        
        options = [
            f"{ea.activity} on {ea.camera_id} occurred first",
            f"{eb.activity} on {eb.camera_id} occurred first",
            "They occurred simultaneously",
            "Cannot be determined",
        ]
        correct_idx = 0  # A is always first (by construction)
        
        # Randomly swap A and B presentation order (but keep correct answer accurate)
        if rng.random() < 0.5:
            # Swap: present B first in question
            question = (
                f"A {eb.activity} event on camera {eb.camera_id} and a "
                f"{ea.activity} event on camera {ea.camera_id} — which occurred first?"
            )
            options = [
                f"{eb.activity} on {eb.camera_id} occurred first",
                f"{ea.activity} on {ea.camera_id} occurred first",
                "They occurred simultaneously",
                "Cannot be determined",
            ]
            correct_idx = 1  # A (now second option) is first
        
        qa = {
            "question_id": f"v6_temporal_{idx+1:03d}",
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
                },
                "event_b": {
                    "activity": eb.activity,
                    "camera": eb.camera_id,
                    "start_sec": eb.start_sec,
                    "end_sec": eb.end_sec,
                    "actor_ids": [a["actor_id"] for a in eb.actors],
                },
                "gap_sec": gap,
                "entity_link": "mevid_ground_truth" if mevid_pid else "heuristic",
                "same_person": same_person,
            },
        }
        qa_pairs.append(qa)
    
    return qa_pairs

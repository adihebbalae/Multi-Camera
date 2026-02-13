"""
V6 generate_perception.py — Step 6: Perception (activity presence) questions.

CONSTRAINTS:
- Affirming questions ONLY (prevents false negatives)
- Ideally 2+ cameras, but 1 camera OK
- Uses raw annotation activity text
- All easy difficulty
"""

import random
from typing import Any, Dict, List, Set
from collections import defaultdict

from .parse_annotations import Event
from .build_scene_graph import SceneGraph
from .entity_resolution import ResolvedGraph
from .distractor_bank import get_distractors, get_camera_distractors


# ============================================================================
# Question Generation
# ============================================================================

def generate_perception_qa(sg: SceneGraph, resolved: ResolvedGraph,
                           rng: random.Random, count: int = 3,
                           verbose: bool = False) -> List[Dict]:
    """
    Generate perception questions about activity presence.
    
    Types:
    1. "Which camera captures a {activity} event?" (camera selection)
    2. "{activity} is occurring. Which cameras capture this activity?" (multi-camera)
    3. "What activity is occurring on camera {cam}?" (activity identification)
    
    Args:
        sg: Scene graph
        resolved: Entity resolution results
        rng: Random number generator
        count: Target number of questions
        verbose: Print progress
    
    Returns:
        List of QA pair dicts
    """
    all_cameras = sorted(sg.cameras.keys())
    slot_activities = set(e.activity for e in sg.events)
    
    # Build activity → cameras mapping
    activity_cameras: Dict[str, Set[str]] = defaultdict(set)
    for evt in sg.events:
        activity_cameras[evt.activity].add(evt.camera_id)
    
    # Build camera → activities mapping
    camera_activities: Dict[str, Set[str]] = defaultdict(set)
    for evt in sg.events:
        camera_activities[evt.camera_id].add(evt.activity)
    
    qa_pairs = []
    used_activities = set()
    used_cameras = set()
    
    # Type 1: "Which camera captures X?" (1 question, prefer activity on 1-2 cameras)
    type1_pool = [
        (act, cams) for act, cams in activity_cameras.items()
        if 1 <= len(cams) <= 3 and act not in used_activities
    ]
    rng.shuffle(type1_pool)
    
    for act, correct_cams in type1_pool:
        if len(qa_pairs) >= 1:  # only 1 type-1 question
            break
        if act in used_activities:
            continue
        
        correct_cam = sorted(correct_cams)[0]
        distractors = get_camera_distractors([correct_cam], all_cameras, rng, n=3)
        
        if len(distractors) < 2:
            continue
        
        options = [correct_cam] + distractors[:3]
        rng.shuffle(options)
        correct_idx = options.index(correct_cam)
        
        question = f"Which camera captures a {act} event?"
        
        qa = {
            "question_id": f"v6_perception_{len(qa_pairs)+1:03d}",
            "category": "perception",
            "difficulty": "easy",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "requires_cameras": sorted(correct_cams),
            "requires_multi_camera": len(correct_cams) > 1,
            "verification": {
                "question_type": "which_camera",
                "activity": act,
                "cameras_with_activity": sorted(correct_cams),
                "total_cameras_in_slot": len(all_cameras),
            },
        }
        qa_pairs.append(qa)
        used_activities.add(act)
    
    # Type 2: "What activity is occurring on camera X?" (1 question)
    type2_pool = [
        (cam, acts) for cam, acts in camera_activities.items()
        if len(acts) >= 2 and cam not in used_cameras
    ]
    rng.shuffle(type2_pool)
    
    for cam, correct_acts in type2_pool:
        if len(qa_pairs) >= 2:  # 1 type-1 + 1 type-2
            break
        if cam in used_cameras:
            continue
        
        correct_act = rng.choice(sorted(correct_acts))
        distractors = get_distractors(correct_act, slot_activities, rng, n=3)
        
        if len(distractors) < 2:
            continue
        
        options = [correct_act] + distractors[:3]
        rng.shuffle(options)
        correct_idx = options.index(correct_act)
        
        question = f"What activity is occurring on camera {cam}?"
        
        qa = {
            "question_id": f"v6_perception_{len(qa_pairs)+1:03d}",
            "category": "perception",
            "difficulty": "easy",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "requires_cameras": [cam],
            "requires_multi_camera": False,
            "verification": {
                "question_type": "activity_identification",
                "camera": cam,
                "correct_activity": correct_act,
                "all_activities_on_camera": sorted(correct_acts),
            },
        }
        qa_pairs.append(qa)
        used_cameras.add(cam)
    
    # Type 3: Multi-camera confirmation (if we still need more)
    if len(qa_pairs) < count:
        multi_cam_acts = [
            (act, cams) for act, cams in activity_cameras.items()
            if len(cams) >= 2 and act not in used_activities
        ]
        rng.shuffle(multi_cam_acts)
        
        for act, correct_cams in multi_cam_acts:
            if len(qa_pairs) >= count:
                break
            
            sorted_cams = sorted(correct_cams)
            other_cams = [c for c in all_cameras if c not in correct_cams]
            
            if len(sorted_cams) >= 2:
                cam_str = f"{sorted_cams[0]} and {sorted_cams[1]}"
                option_both = f"Both {sorted_cams[0]} and {sorted_cams[1]}"
                option_a_only = f"{sorted_cams[0]} only"
                option_b_only = f"{sorted_cams[1]} only"
                option_neither = "Neither"
                
                options = [option_both, option_a_only, option_b_only, option_neither]
                correct_idx = 0
                
                question = (
                    f"{act} is occurring. Which cameras capture this activity?"
                )
                
                qa = {
                    "question_id": f"v6_perception_{len(qa_pairs)+1:03d}",
                    "category": "perception",
                    "difficulty": "easy",
                    "question_template": question,
                    "options": options,
                    "correct_answer_index": correct_idx,
                    "correct_answer": options[correct_idx],
                    "requires_cameras": sorted_cams[:2],
                    "requires_multi_camera": True,
                    "verification": {
                        "question_type": "multi_camera_confirmation",
                        "activity": act,
                        "cameras_with_activity": sorted_cams,
                    },
                }
                qa_pairs.append(qa)
                used_activities.add(act)
    
    if verbose:
        print(f"  Perception: {len(qa_pairs)} questions generated")
    
    return qa_pairs[:count]

"""
V8 generate_perception.py — Perception questions with MEVID attribute verification.

V8 CHANGES from V7:
- New subtype: attribute_verification ("What is the person wearing?")
- Entity descriptions from MEVID (GPT/YOLO) instead of actor ID aliases
- Attribute verification requires MEVID color data (not generic fallback)
- 4 question types total (V7 had 3)

Types:
1. "Which camera captures a {activity} event?" (which_camera) — V7 unchanged
2. "What activity is occurring on camera {cam}?" (activity_identification) — V7 unchanged
3. "{activity} is occurring. Which cameras capture this?" (multi_camera_confirmation) — V7 unchanged
4. NEW: "What is the person on camera {cam} wearing?" (attribute_verification) — V8
"""

import random
from typing import Any, Dict, List, Set
from collections import defaultdict

from .parse_annotations import Event
from .build_scene_graph import SceneGraph
from .entity_resolution import ResolvedGraph
from .person_descriptions import (
    load_person_database, get_person_description, get_person_short_label,
    get_mevid_persons_with_cameras, enrich_entities,
)
from .distractor_bank import get_distractors, get_camera_distractors
from .activity_hierarchy import humanize_activity, humanize_activity_gerund


# ============================================================================
# V8 NEW: Attribute Distractor Colors
# ============================================================================

UPPER_COLORS = ["black", "white", "blue", "red", "green", "gray", "yellow",
                "brown", "orange", "purple", "navy", "beige", "khaki"]
LOWER_COLORS = ["black", "blue", "dark", "gray", "brown", "khaki", "white",
                "green", "red", "beige", "navy"]
CARRIED_OBJECTS = ["backpack", "bag", "purse", "briefcase", "water bottle",
                   "umbrella", "phone", "laptop bag"]


def _build_appearance_options(person_data: Dict, rng: random.Random) -> Dict:
    """
    Build MCQ options for attribute verification.
    
    Returns dict with:
      - question_text: what we ask about
      - options: list of 4 strings
      - correct_answer_index: int
      - attribute_type: "upper_color" | "lower_color" | "carried_object"
    """
    upper = person_data.get("primary_upper_color", "unknown")
    lower = person_data.get("primary_lower_color", "unknown")
    objects = person_data.get("all_carried_objects", [])
    
    # Pick the best attribute to ask about (prefer colors with actual data)
    candidates = []
    
    if upper != "unknown":
        candidates.append(("upper_color", upper,
                           "upper body clothing",
                           UPPER_COLORS))
    if lower != "unknown":
        candidates.append(("lower_color", lower,
                           "lower body clothing",
                           LOWER_COLORS))
    # Objects are rarer — only use if we have colors too
    if objects and len(candidates) >= 1:
        candidates.append(("carried_object", objects[0],
                           "carried item",
                           CARRIED_OBJECTS))
    
    if not candidates:
        return None
    
    attr_type, correct_val, label, distractor_pool = rng.choice(candidates)
    
    # Build distractors
    dist_pool = [c for c in distractor_pool if c.lower() != correct_val.lower()]
    rng.shuffle(dist_pool)
    distractors = dist_pool[:3]
    
    if len(distractors) < 3:
        return None  # Not enough distractors
    
    options = [correct_val.capitalize()] + [d.capitalize() for d in distractors]
    rng.shuffle(options)
    correct_idx = next(i for i, o in enumerate(options)
                       if o.lower() == correct_val.lower())
    
    return {
        "attribute_type": attr_type,
        "correct_value": correct_val,
        "label": label,
        "options": options,
        "correct_answer_index": correct_idx,
    }


# ============================================================================
# Question Generation
# ============================================================================

def generate_perception_qa(sg: SceneGraph, resolved: ResolvedGraph,
                           entity_descs: Dict[str, str],
                           rng: random.Random, count: int = 3,
                           verbose: bool = False) -> List[Dict]:
    """
    Generate perception questions with MEVID attribute verification.
    
    V8: Takes entity_descs parameter. Adds attribute_verification subtype.
    Target: 1 of each type if possible, up to `count` total.
    """
    all_cameras = sorted(sg.cameras.keys())
    slot_activities = set(e.activity for e in sg.events)
    
    # Build activity/camera mappings (same as V7)
    activity_cameras: Dict[str, Set[str]] = defaultdict(set)
    activity_events: Dict[str, List[Event]] = defaultdict(list)
    for evt in sg.events:
        activity_cameras[evt.activity].add(evt.camera_id)
        activity_events[evt.activity].append(evt)
    
    camera_activities: Dict[str, Set[str]] = defaultdict(set)
    camera_events: Dict[str, List[Event]] = defaultdict(list)
    for evt in sg.events:
        camera_activities[evt.camera_id].add(evt.activity)
        camera_events[evt.camera_id].append(evt)
    
    qa_pairs = []
    used_activities = set()
    used_cameras = set()
    
    # ------------------------------------------------------------------
    # Type 1: "Which camera captures X?" (1 question, V7 logic)
    # ------------------------------------------------------------------
    type1_pool = [
        (act, cams) for act, cams in activity_cameras.items()
        if 1 <= len(cams) <= 3 and act not in used_activities
    ]
    rng.shuffle(type1_pool)
    
    for act, correct_cams in type1_pool:
        if len(qa_pairs) >= 1:
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
        
        human_act = humanize_activity(act)
        question = f"Which camera captures a {human_act} event?"
        
        rep_event = activity_events[act][0] if activity_events[act] else None
        
        debug_info = {
            "question_type": "which_camera",
            "activity": act,
            "activity_alias": human_act,
            "correct_camera": correct_cam,
            "cameras_with_activity": sorted(correct_cams),
        }
        if rep_event:
            debug_info["representative_event"] = {
                "camera": rep_event.camera_id,
                "frame_range": [rep_event.start_frame, rep_event.end_frame],
                "timestamp": f"{rep_event.start_sec:.2f}-{rep_event.end_sec:.2f}s",
                "clip_file": rep_event.video_file.replace(".avi", ".mp4"),
            }
        
        qa = {
            "question_id": f"v8_perception_{len(qa_pairs)+1:03d}",
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
                "activity_alias": human_act,
                "cameras_with_activity": sorted(correct_cams),
                "total_cameras_in_slot": len(all_cameras),
            },
            "debug_info": debug_info,
        }
        qa_pairs.append(qa)
        used_activities.add(act)
    
    # ------------------------------------------------------------------
    # Type 4 (V8 NEW): Attribute Verification — "What is the person wearing?"
    # ------------------------------------------------------------------
    person_cameras = get_mevid_persons_with_cameras(sg.slot)
    
    if person_cameras:
        # Pick a person with color data
        db = load_person_database()
        persons = db.get("persons", {})
        
        attr_candidates = []
        for pid, cameras in person_cameras.items():
            if pid not in persons:
                continue
            pdata = persons[pid]
            upper = pdata.get("primary_upper_color", "unknown")
            lower = pdata.get("primary_lower_color", "unknown")
            if upper == "unknown" and lower == "unknown":
                continue
            
            # Only use cameras that are in our slot
            slot_cams = [c for c in cameras if c in sg.cameras]
            if slot_cams:
                attr_candidates.append((pid, pdata, slot_cams))
        
        rng.shuffle(attr_candidates)
        
        for pid, pdata, slot_cams in attr_candidates:
            if len(qa_pairs) >= count:
                break
            
            cam = rng.choice(slot_cams)
            attr_opts = _build_appearance_options(pdata, rng)
            
            if attr_opts is None:
                continue
            
            desc = get_person_description(pid)
            gpt_desc = pdata.get("gpt_description", "")
            
            # Frame the question: show the person on a camera, ask about attribute
            if attr_opts["attribute_type"] == "upper_color":
                question = (
                    f"A person is visible on camera {cam}. "
                    f"What color is their upper body clothing?"
                )
            elif attr_opts["attribute_type"] == "lower_color":
                question = (
                    f"A person is visible on camera {cam}. "
                    f"What color are they wearing on their lower body?"
                )
            else:  # carried_object
                question = (
                    f"A person is visible on camera {cam}. "
                    f"What object are they carrying?"
                )
            
            debug_info = {
                "question_type": "attribute_verification",
                "mevid_person_id": pid,
                "camera": cam,
                "person_description": desc,
                "gpt_description": gpt_desc,
                "attribute_type": attr_opts["attribute_type"],
                "correct_value": attr_opts["correct_value"],
                "all_cameras_for_person": sorted(slot_cams),
                "source": "MEVID YOLO+GPT person database",
            }
            
            qa = {
                "question_id": f"v8_perception_{len(qa_pairs)+1:03d}",
                "category": "perception",
                "subcategory": "attribute_verification",
                "difficulty": "medium",
                "question_template": question,
                "options": attr_opts["options"],
                "correct_answer_index": attr_opts["correct_answer_index"],
                "correct_answer": attr_opts["options"][attr_opts["correct_answer_index"]],
                "requires_cameras": [cam],
                "requires_multi_camera": False,
                "verification": {
                    "question_type": "attribute_verification",
                    "mevid_person_id": pid,
                    "attribute_type": attr_opts["attribute_type"],
                    "correct_value": attr_opts["correct_value"],
                    "camera": cam,
                    "person_description": desc,
                },
                "debug_info": debug_info,
            }
            qa_pairs.append(qa)
            used_cameras.add(cam)
    
    # ------------------------------------------------------------------
    # Type 2: "What activity is occurring on camera X?" (V7 logic)
    # ------------------------------------------------------------------
    type2_pool = [
        (cam, acts) for cam, acts in camera_activities.items()
        if len(acts) >= 2 and cam not in used_cameras
    ]
    rng.shuffle(type2_pool)
    
    for cam, correct_acts in type2_pool:
        if len(qa_pairs) >= count:
            break
        if cam in used_cameras:
            continue
        
        correct_act = rng.choice(sorted(correct_acts))
        distractors = get_distractors(correct_act, slot_activities, rng, n=3)
        
        if len(distractors) < 2:
            continue
        
        human_correct = humanize_activity(correct_act)
        human_distractors = [humanize_activity(d) for d in distractors[:3]]
        
        options = [human_correct] + human_distractors
        rng.shuffle(options)
        correct_idx = options.index(human_correct)
        
        question = f"What activity is occurring on camera {cam}?"
        
        cam_evts = [e for e in camera_events[cam] if e.activity == correct_act]
        rep_event = cam_evts[0] if cam_evts else None
        
        debug_info = {
            "question_type": "activity_identification",
            "camera": cam,
            "correct_activity": correct_act,
            "correct_activity_alias": human_correct,
            "all_activities_on_camera": sorted(correct_acts),
        }
        if rep_event:
            debug_info["representative_event"] = {
                "camera": rep_event.camera_id,
                "frame_range": [rep_event.start_frame, rep_event.end_frame],
                "timestamp": f"{rep_event.start_sec:.2f}-{rep_event.end_sec:.2f}s",
                "clip_file": rep_event.video_file.replace(".avi", ".mp4"),
            }
        
        qa = {
            "question_id": f"v8_perception_{len(qa_pairs)+1:03d}",
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
                "correct_activity_alias": human_correct,
                "all_activities_on_camera": sorted(correct_acts),
            },
            "debug_info": debug_info,
        }
        qa_pairs.append(qa)
        used_cameras.add(cam)
    
    # ------------------------------------------------------------------
    # Type 3: Multi-camera confirmation (V7 logic)
    # ------------------------------------------------------------------
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
            human_act = humanize_activity(act)
            gerund_act = humanize_activity_gerund(act)
            
            if len(sorted_cams) >= 2:
                option_both = f"Both {sorted_cams[0]} and {sorted_cams[1]}"
                option_a_only = f"{sorted_cams[0]} only"
                option_b_only = f"{sorted_cams[1]} only"
                option_neither = "Neither"
                
                options = [option_both, option_a_only, option_b_only, option_neither]
                correct_idx = 0
                
                question = f"{gerund_act} is occurring. Which cameras capture this activity?"
                
                debug_info = {
                    "question_type": "multi_camera_confirmation",
                    "activity": act,
                    "activity_alias": human_act,
                    "cameras_with_activity": sorted_cams,
                }
                for ci, cam_id in enumerate(sorted_cams[:2]):
                    cam_evts = [e for e in activity_events[act] if e.camera_id == cam_id]
                    if cam_evts:
                        evt = cam_evts[0]
                        debug_info[f"camera_{ci+1}_event"] = {
                            "camera": evt.camera_id,
                            "frame_range": [evt.start_frame, evt.end_frame],
                            "timestamp": f"{evt.start_sec:.2f}-{evt.end_sec:.2f}s",
                            "clip_file": evt.video_file.replace(".avi", ".mp4"),
                        }
                
                qa = {
                    "question_id": f"v8_perception_{len(qa_pairs)+1:03d}",
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
                        "activity_alias": human_act,
                        "cameras_with_activity": sorted_cams,
                    },
                    "debug_info": debug_info,
                }
                qa_pairs.append(qa)
                used_activities.add(act)
    
    if verbose:
        print(f"  Perception: {len(qa_pairs)} questions generated "
              f"(incl. {sum(1 for q in qa_pairs if q.get('subcategory') == 'attribute_verification')} attribute verification)")
    
    return qa_pairs[:count]

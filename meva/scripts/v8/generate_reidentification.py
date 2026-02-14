"""
V8 generate_reidentification.py — Cross-camera person re-identification questions.

NEW in V8: Questions that test whether a model can identify the same person
appearing on different cameras within the same slot.

Uses MEVID ground-truth person IDs to know which persons appear on which cameras,
combined with YOLO+GPT descriptions to make questions answerable from visual features.

Question types:
  1. "Does the person in [description] on camera A also appear on camera B?"
  2. "Which camera also shows [description person]?"
  3. "A person is seen on camera A wearing [X]. On camera B, which person matches?"
"""

import random
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from .build_scene_graph import SceneGraph
from .entity_resolution import ResolvedGraph
from .person_descriptions import (
    get_person_description, get_person_short_label,
    get_mevid_persons_with_cameras, load_person_database,
)
from .activity_hierarchy import humanize_activity
from .distractor_bank import get_camera_distractors


# ============================================================================
# Question Generation
# ============================================================================

def generate_reidentification_qa(sg: SceneGraph, resolved: ResolvedGraph,
                                  entity_descs: Dict[str, str],
                                  rng: random.Random, count: int = 2,
                                  verbose: bool = False) -> List[Dict]:
    """
    Generate cross-camera re-identification questions.
    
    These require recognizing that the same person appears on multiple cameras
    based on their visual appearance (clothing, carried objects, etc.).
    
    Selection:
      - Only uses MEVID persons (known cross-camera ground truth)
      - Prefers persons with distinctive descriptions
      - Requires person to appear on 2+ cameras in the slot
    """
    slot = sg.slot
    all_cameras = sorted(sg.cameras.keys())
    
    # Get MEVID persons for this slot with their cameras
    person_cameras = get_mevid_persons_with_cameras(slot)
    
    if not person_cameras:
        if verbose:
            print("  Re-ID: No MEVID persons in slot — skipping")
        return []
    
    # Filter to persons on 2+ cameras AND with good descriptions
    db = load_person_database()
    persons_data = db.get("persons", {})
    
    candidates = []
    for pid, cameras in person_cameras.items():
        if len(cameras) < 2:
            continue
        pdata = persons_data.get(pid, {})
        has_gpt = bool(pdata.get("gpt_description"))
        has_colors = (pdata.get("primary_upper_color", "unknown") != "unknown")
        
        if has_gpt or has_colors:
            candidates.append({
                "person_id": pid,
                "cameras": sorted(cameras),
                "has_gpt": has_gpt,
                "description": get_person_description(pid),
                "short_label": get_person_short_label(pid),
                "upper_color": pdata.get("primary_upper_color", "unknown"),
                "lower_color": pdata.get("primary_lower_color", "unknown"),
                "objects": pdata.get("all_carried_objects", []),
            })
    
    # Sort: prefer GPT descriptions, then by number of cameras (more = better)
    candidates.sort(key=lambda c: (-int(c["has_gpt"]), -len(c["cameras"])))
    
    if verbose:
        print(f"  Re-ID: {len(candidates)} MEVID persons with 2+ cameras and descriptions")
    
    if not candidates:
        return []
    
    qa_pairs = []
    used_persons = set()
    
    # Type 1: "Does person X on camera A also appear on camera B?" (Yes/No with camera options)
    for cand in candidates:
        if len(qa_pairs) >= count:
            break
        if cand["person_id"] in used_persons:
            continue
        
        pid = cand["person_id"]
        cams = cand["cameras"]
        desc = cand["description"]
        
        # Pick two cameras this person appears on
        cam_pair = rng.sample(cams, min(2, len(cams)))
        cam_a, cam_b = cam_pair[0], cam_pair[1] if len(cam_pair) > 1 else cam_pair[0]
        
        if cam_a == cam_b:
            continue
        
        # Find cameras where this person does NOT appear (for distractors)
        other_cams = [c for c in all_cameras if c not in cams]
        
        if len(qa_pairs) % 2 == 0:
            # Type 1: "Which camera also shows this person?"
            distractors = rng.sample(other_cams, min(2, len(other_cams))) if other_cams else []
            if not distractors:
                # Use all cameras as options if needed
                distractors = [c for c in all_cameras if c != cam_a and c != cam_b][:2]
            
            options = [cam_b] + distractors + ["None of these cameras"]
            rng.shuffle(options[:3])  # Shuffle camera options but keep "None" at end
            correct_idx = options.index(cam_b)
            
            question = (
                f"On camera {cam_a}, {desc} is visible. "
                f"Which other camera also shows this same person?"
            )
            
            debug_info = {
                "question_type": "which_camera_reid",
                "mevid_person_id": pid,
                "source_camera": cam_a,
                "target_camera": cam_b,
                "all_person_cameras": cams,
                "person_description": desc,
            }
            
            qa = {
                "question_id": f"v8_reid_{len(qa_pairs)+1:03d}",
                "category": "re_identification",
                "difficulty": "medium",
                "question_template": question,
                "options": options,
                "correct_answer_index": correct_idx,
                "correct_answer": options[correct_idx],
                "requires_cameras": sorted([cam_a, cam_b]),
                "requires_multi_camera": True,
                "verification": {
                    "question_type": "which_camera_reid",
                    "mevid_person_id": pid,
                    "person_description": desc,
                    "source_camera": cam_a,
                    "correct_target_camera": cam_b,
                    "all_person_cameras": cams,
                },
                "debug_info": debug_info,
            }
        else:
            # Type 2: "Is the person in [desc] on camera A the same as person on camera B?"
            # Correct answer: Yes (same MEVID person)
            question = (
                f"{desc.capitalize()} is observed on camera {cam_a}. "
                f"Is this the same person visible on camera {cam_b}?"
            )
            
            options = [
                "Yes, it is the same person",
                "No, they are different people",
                "Cannot be determined from the footage",
                "The person is not visible on the second camera",
            ]
            correct_idx = 0
            
            debug_info = {
                "question_type": "same_person_confirmation",
                "mevid_person_id": pid,
                "camera_a": cam_a,
                "camera_b": cam_b,
                "all_person_cameras": cams,
                "person_description": desc,
            }
            
            qa = {
                "question_id": f"v8_reid_{len(qa_pairs)+1:03d}",
                "category": "re_identification",
                "difficulty": "medium",
                "question_template": question,
                "options": options,
                "correct_answer_index": correct_idx,
                "correct_answer": options[correct_idx],
                "requires_cameras": sorted([cam_a, cam_b]),
                "requires_multi_camera": True,
                "verification": {
                    "question_type": "same_person_confirmation",
                    "mevid_person_id": pid,
                    "person_description": desc,
                    "camera_a": cam_a,
                    "camera_b": cam_b,
                    "all_person_cameras": cams,
                    "ground_truth": "same_person",
                },
                "debug_info": debug_info,
            }
        
        qa_pairs.append(qa)
        used_persons.add(pid)
    
    if verbose:
        print(f"  Re-ID: {len(qa_pairs)} questions generated")
    
    return qa_pairs[:count]

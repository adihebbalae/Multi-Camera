"""
V6 generate_counting.py — Step 7: Numerical/Counting questions.

CONSTRAINTS:
- Count activity occurrences in the scene
- Prioritize activities that occur more than once
- Fall back to activities that occur once if needed
- Uses raw annotation activity text
- All easy difficulty
"""

import random
from typing import Any, Dict, List, Set
from collections import defaultdict

from .parse_annotations import Event
from .build_scene_graph import SceneGraph
from .entity_resolution import ResolvedGraph


# ============================================================================
# Question Generation
# ============================================================================

def generate_counting_qa(sg: SceneGraph, resolved: ResolvedGraph,
                         rng: random.Random, count: int = 3,
                         verbose: bool = False) -> List[Dict]:
    """
    Generate counting questions about activity occurrences.
    
    Questions ask: "How many times does {activity} occur in this scene?"
    
    Strategy:
    1. Count occurrences of each activity type across all events
    2. Prioritize activities with count > 1 (more interesting)
    3. Fall back to activities with count = 1 if needed
    4. Generate numerical distractors (wrong counts)
    
    Args:
        sg: Scene graph
        resolved: Entity resolution results (not used but kept for consistency)
        rng: Random number generator
        count: Target number of questions
        verbose: Print progress
    
    Returns:
        List of QA pair dicts
    """
    # Count activity occurrences
    activity_counts: Dict[str, int] = defaultdict(int)
    activity_events: Dict[str, List[Event]] = defaultdict(list)
    
    for evt in sg.events:
        activity_counts[evt.activity] += 1
        activity_events[evt.activity].append(evt)
    
    if verbose:
        print(f"  Counting: Found {len(activity_counts)} unique activities")
        top_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for act, cnt in top_activities:
            print(f"    {act}: {cnt} occurrence(s)")
    
    # Separate activities by count
    multiple_occurrences = [
        (act, cnt) for act, cnt in activity_counts.items() if cnt > 1
    ]
    single_occurrence = [
        (act, cnt) for act, cnt in activity_counts.items() if cnt == 1
    ]
    
    # Sort by count (descending) for prioritization
    multiple_occurrences.sort(key=lambda x: x[1], reverse=True)
    single_occurrence.sort(key=lambda x: x[0])  # Alphabetical for consistency
    
    # Prioritize multiple occurrences, fall back to single
    candidate_activities = multiple_occurrences + single_occurrence
    
    if not candidate_activities:
        if verbose:
            print("  Counting: No activities found")
        return []
    
    qa_pairs = []
    used_activities = set()
    
    for act, correct_count in candidate_activities:
        if len(qa_pairs) >= count:
            break
        
        if act in used_activities:
            continue
        
        # Generate numerical distractors
        # Strategy: Generate counts that are plausible but wrong
        # Options: correct_count, correct_count±1, correct_count±2, 0
        distractors = []
        
        # Add counts near the correct answer
        for offset in [1, 2, -1, -2]:
            candidate = correct_count + offset
            if candidate >= 0 and candidate != correct_count:
                distractors.append(candidate)
        
        # Always include 0 as a distractor (common wrong answer)
        if 0 not in distractors:
            distractors.append(0)
        
        # Remove duplicates and sort
        distractors = sorted(set(distractors))
        
        # Ensure we have at least 3 distractors (for 4 total options)
        while len(distractors) < 3:
            # Add more distant counts
            max_count = max(activity_counts.values())
            for candidate in [max_count + 1, max_count + 2, correct_count + 3]:
                if candidate not in distractors and candidate != correct_count:
                    distractors.append(candidate)
                    break
        
        # Select 3 distractors (to have 4 total options)
        rng.shuffle(distractors)
        selected_distractors = distractors[:3]
        
        # Build options: correct answer + distractors
        options = [str(correct_count)] + [str(d) for d in selected_distractors]
        rng.shuffle(options)
        correct_idx = options.index(str(correct_count))
        
        # Format activity name for question (replace underscores with spaces)
        activity_display = act.replace("_", " ")
        
        question = f"How many times does {activity_display} occur in this scene?"
        
        # Get cameras where this activity occurs
        cameras_with_activity = sorted(set(
            evt.camera_id for evt in activity_events[act]
        ))
        
        qa = {
            "question_id": f"v6_counting_{len(qa_pairs)+1:03d}",
            "category": "counting",
            "difficulty": "easy",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "requires_cameras": cameras_with_activity,
            "requires_multi_camera": len(cameras_with_activity) > 1,
            "verification": {
                "activity": act,
                "activity_display": activity_display,
                "correct_count": correct_count,
                "total_occurrences": correct_count,
                "cameras_with_activity": cameras_with_activity,
                "event_ids": [evt.event_id for evt in activity_events[act]],
                "event_count_per_camera": {
                    cam: sum(1 for evt in activity_events[act] if evt.camera_id == cam)
                    for cam in cameras_with_activity
                },
            },
        }
        qa_pairs.append(qa)
        used_activities.add(act)
    
    if verbose:
        print(f"  Counting: {len(qa_pairs)} questions generated")
    
    return qa_pairs[:count]

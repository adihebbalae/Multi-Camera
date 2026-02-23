"""
V8 generate_scene_summary.py — Scene-level summary questions.

NEW in V8: Questions that test holistic understanding of the entire
multi-camera scene. Requires aggregating information across all cameras
and understanding the overall activity pattern.

Generates summaries from annotation density: activity types, counts,
temporal flow, and cross-camera patterns.
"""

import random
from typing import Any, Dict, List, Set
from collections import defaultdict, Counter

from .parse_annotations import Event
from .build_scene_graph import SceneGraph
from .entity_resolution import ResolvedGraph
from .activity_hierarchy import humanize_activity, humanize_activity_gerund


# ============================================================================
# Scene Analysis
# ============================================================================

def _analyze_scene(sg: SceneGraph) -> Dict:
    """
    Compute scene-level statistics for summary question generation.
    """
    events = sg.events
    if not events:
        return {}
    
    # Activity frequency
    activity_counts = Counter(e.activity for e in events)
    
    # Camera activity density
    camera_event_counts = Counter(e.camera_id for e in events)
    
    # Temporal span
    min_sec = min(e.start_sec for e in events)
    max_sec = max(e.end_sec for e in events)
    duration = max_sec - min_sec
    
    # Activity categories
    person_activities = [e for e in events if e.activity.startswith("person_")]
    vehicle_activities = [e for e in events if e.activity.startswith("vehicle_")]
    
    # Cross-camera activities (same activity on 2+ cameras)
    activity_cameras: Dict[str, Set[str]] = defaultdict(set)
    for e in events:
        activity_cameras[e.activity].add(e.camera_id)
    cross_camera_acts = {act: cams for act, cams in activity_cameras.items() 
                         if len(cams) >= 2}
    
    # Most active camera
    busiest_camera = camera_event_counts.most_common(1)[0] if camera_event_counts else (None, 0)
    
    # Unique entity count (person entities across all cameras)
    unique_entities = len(sg.entities)
    person_entities = sum(1 for e in sg.entities.values() if e.entity_type == "person")
    
    # Dominant activity pattern
    top_3_activities = activity_counts.most_common(3)
    
    return {
        "total_events": len(events),
        "unique_activities": len(activity_counts),
        "activity_counts": dict(activity_counts),
        "camera_event_counts": dict(camera_event_counts),
        "num_cameras": len(sg.cameras),
        "duration_sec": round(duration, 1),
        "person_event_count": len(person_activities),
        "vehicle_event_count": len(vehicle_activities),
        "cross_camera_activities": {a: sorted(c) for a, c in cross_camera_acts.items()},
        "busiest_camera": busiest_camera[0],
        "busiest_camera_count": busiest_camera[1],
        "unique_entities": unique_entities,
        "person_entities": person_entities,
        "top_3_activities": top_3_activities,
    }


def _build_scene_description(analysis: Dict) -> str:
    """Build a natural-language scene description from analysis."""
    parts = []
    
    n_cams = analysis.get("num_cameras", 0)
    n_events = analysis.get("total_events", 0)
    n_acts = analysis.get("unique_activities", 0)
    
    parts.append(f"The scene spans {n_cams} cameras with {n_events} total activity events")
    
    top_3 = analysis.get("top_3_activities", [])
    if top_3:
        top_descs = []
        for act, count in top_3:
            human = humanize_activity(act)
            top_descs.append(f"{human} ({count} occurrences)")
        parts.append(f"The most frequent activities are: {', '.join(top_descs)}")
    
    cross_cam = analysis.get("cross_camera_activities", {})
    if cross_cam:
        n_cross = len(cross_cam)
        parts.append(f"{n_cross} activities occur on multiple cameras")
    
    busiest = analysis.get("busiest_camera")
    if busiest:
        parts.append(f"Camera {busiest} is the most active with {analysis['busiest_camera_count']} events")
    
    return ". ".join(parts) + "."


# ============================================================================
# Question Generation
# ============================================================================

def generate_scene_summary_qa(sg: SceneGraph, resolved: ResolvedGraph,
                               entity_descs: Dict[str, str],
                               rng: random.Random, count: int = 1,
                               verbose: bool = False) -> List[Dict]:
    """
    Generate scene-level summary questions.
    
    Types:
      1. "Which best describes the overall scene?" (scene characterization)
      2. "Which camera is most active?" (activity density)
      3. "What is the dominant activity type?" (activity distribution)
    """
    analysis = _analyze_scene(sg)
    
    if not analysis or analysis.get("total_events", 0) < 5:
        if verbose:
            print("  Scene Summary: Too few events for summary questions")
        return []
    
    qa_pairs = []
    
    # Type 1: Scene characterization
    # "Which description best matches the overall scene?"
    description = _build_scene_description(analysis)
    
    top_3 = analysis.get("top_3_activities", [])
    n_events = analysis.get("total_events", 0)
    n_cams = analysis.get("num_cameras", 0)
    n_person = analysis.get("person_event_count", 0)
    n_vehicle = analysis.get("vehicle_event_count", 0)
    
    # Build correct answer based on dominant activity pattern
    if n_person > n_vehicle * 2:
        scene_type = "pedestrian-dominant"
    elif n_vehicle > n_person * 2:
        scene_type = "vehicle-dominant"
    else:
        scene_type = "mixed activity"
    
    _cam_word = 'camera' if n_cams == 1 else 'cameras'
    
    if top_3:
        top_act = humanize_activity_gerund(top_3[0][0]).lower()
        correct_desc = (
            f"A {scene_type} scene across {n_cams} {_cam_word}, "
            f"primarily featuring {top_act}"
        )
    else:
        correct_desc = f"A {scene_type} scene across {n_cams} {_cam_word}"
    
    # Generate plausible but wrong descriptions
    wrong_descs = []
    if scene_type == "pedestrian-dominant":
        wrong_descs.append(
            f"A vehicle-focused scene with mostly parking and driving activity"
        )
    else:
        wrong_descs.append(
            f"A scene dominated by people entering and exiting buildings"
        )
    
    _empty_cam_count = max(1, n_cams - 3)
    wrong_descs.append(
        f"An empty scene with minimal activity, captured on {_empty_cam_count} {'camera' if _empty_cam_count == 1 else 'cameras'}"
    )
    wrong_descs.append(
        f"A single-camera scene showing only indoor activities"
    )
    
    options = [correct_desc] + wrong_descs[:3]
    rng.shuffle(options)
    correct_idx = options.index(correct_desc)
    
    question = (
        f"Considering all {n_cams} {_cam_word} in this slot, "
        f"which description best characterizes the overall scene?"
    )
    
    # Collect all clip_files across all cameras
    all_clip_files = sorted(set(
        e.video_file.replace(".avi", ".mp4")
        for e in sg.events if e.video_file
    ))

    debug_info = {
        "question_type": "scene_characterization",
        "scene_analysis": analysis,
        "scene_description": description,
        "scene_type": scene_type,
        "clip_files": all_clip_files,
    }
    
    qa = {
        "question_id": f"v8_summary_{len(qa_pairs)+1:03d}",
        "category": "scene_summary",
        "difficulty": "hard",
        "question_template": question,
        "options": options,
        "correct_answer_index": correct_idx,
        "correct_answer": options[correct_idx],
        "requires_cameras": sorted(sg.cameras.keys()),
        "requires_multi_camera": True,
        "verification": {
            "question_type": "scene_characterization",
            "total_events": n_events,
            "num_cameras": n_cams,
            "person_events": n_person,
            "vehicle_events": n_vehicle,
            "scene_type": scene_type,
            "top_activity": top_3[0][0] if top_3 else None,
            "top_activity_count": top_3[0][1] if top_3 else 0,
        },
        "debug_info": debug_info,
    }
    qa_pairs.append(qa)
    
    # Type 2: Busiest camera (if count > 1)
    if count > 1 and analysis.get("busiest_camera"):
        busiest = analysis["busiest_camera"]
        busiest_count = analysis["busiest_camera_count"]
        
        other_cameras = [c for c in sg.cameras.keys() if c != busiest]
        distractors = rng.sample(other_cameras, min(3, len(other_cameras)))
        
        options = [busiest] + distractors
        rng.shuffle(options)
        correct_idx = options.index(busiest)
        
        question = (
            f"Across all cameras in this scene, which camera captures "
            f"the most activity events?"
        )
        
        qa2 = {
            "question_id": f"v8_summary_{len(qa_pairs)+1:03d}",
            "category": "scene_summary",
            "difficulty": "hard",
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "requires_cameras": sorted(sg.cameras.keys()),
            "requires_multi_camera": True,
            "verification": {
                "question_type": "busiest_camera",
                "correct_camera": busiest,
                "event_count": busiest_count,
                "all_camera_counts": analysis["camera_event_counts"],
            },
            "debug_info": {
                "question_type": "busiest_camera",
                "camera_event_counts": analysis["camera_event_counts"],
                "clip_files": all_clip_files,
            },
        }
        qa_pairs.append(qa2)
    
    if verbose:
        print(f"  Scene Summary: {len(qa_pairs)} questions generated "
              f"(scene_type={analysis.get('scene_type', scene_type)}, "
              f"{n_events} events, {n_cams} cameras)")
    
    return qa_pairs[:count]

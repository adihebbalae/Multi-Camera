"""
FINAL generate_numerical.py — Numerical/counting questions across cameras.

Tests a model's ability to count activities from a multi-camera scene.

**activity_counting**: "How many times does [activity] occur across all cameras?"
   → Count event instances of that activity type from sg.events, with
     cross-camera temporal deduplication (events of the same activity on
     different cameras within ±2 seconds are counted as one instance).

Distractors are generated arithmetically (±1, ±2, ×2) so that wrong answers
are plausible.  All options are stringified integers > 0, sorted numerically.

Guard rails: skip any candidate whose correct count is < 2 (trivial) or > 20
(unreasonable for video QA).  Preferred difficulty sweet-spot is [3, 10].
"""

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple

from .parse_annotations import Event
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .activity_hierarchy import humanize_activity, humanize_activity_gerund

# Issue 6: Import 3D position matching for improved cross-camera dedup
try:
    from .generate_temporal import _get_event_3d_position, CROSS_CAM_DEDUP_DISTANCE_M
    _HAS_3D_DEDUP = True
except ImportError:
    _HAS_3D_DEDUP = False


# ============================================================================
# Constants
# ============================================================================

MIN_COUNT = 2    # minimum correct count to consider
MAX_COUNT = 20   # maximum correct count to consider
SWEET_LOW = 3    # preferred range lower bound (for scoring)
SWEET_HIGH = 10  # preferred range upper bound


# ============================================================================
# Distractor Generation
# ============================================================================

def _make_distractors(correct: int, rng: random.Random) -> List[int]:
    """
    Build 3 distinct distractor values for a counting question.

    Candidate pool: correct ± 1, correct ± 2, correct × 2.
    All values must be > 0 and != correct.  If pool is too small we
    widen with correct + 3, correct + 4, etc.
    """
    pool = set()
    for delta in (-2, -1, 1, 2):
        v = correct + delta
        if v > 0:
            pool.add(v)
    doubled = correct * 2
    if doubled > 0 and doubled != correct:
        pool.add(doubled)

    pool.discard(correct)

    # Widen if we still don't have 3
    extend = 3
    while len(pool) < 3:
        extend += 1
        v = correct + extend
        if v > 0 and v != correct:
            pool.add(v)
        v = correct - extend
        if v > 0 and v != correct:
            pool.add(v)

    pool_list = sorted(pool)
    rng.shuffle(pool_list)
    return pool_list[:3]


def _build_options(correct: int, rng: random.Random) -> Tuple[List[str], int]:
    """
    Return (options, correct_answer_index) with 4 string-ified integers
    sorted in ascending numerical order.
    """
    distractors = _make_distractors(correct, rng)
    all_vals = sorted(set([correct] + distractors))
    options = [str(v) for v in all_vals]
    correct_idx = all_vals.index(correct)
    return options, correct_idx


# ============================================================================
# Candidate Builders
# ============================================================================

def _dedup_activity_count(events_for_activity: list,
                          sg: SceneGraph = None) -> Tuple[int, List[str], List[Dict]]:
    """Count distinct instances of an activity with cross-camera temporal dedup.

    Events on DIFFERENT cameras whose start_sec is within ±2 seconds are
    merged into a single cluster (counted as one occurrence).  Events on
    the SAME camera are always counted separately.

    Issue 6: Also uses 3D position matching when available for more accurate dedup.
    Issue 10: Returns cluster details for key_frames in QA output.

    Returns (deduped_count, list_of_event_ids, cluster_details).
    cluster_details: List[Dict] with one entry per cluster, each containing
    {camera, start_sec, end_sec, start_frame, event_id, clip_file}.
    """
    sorted_evts = sorted(events_for_activity, key=lambda e: e.start_sec)
    clusters: List[list] = []
    for evt in sorted_evts:
        merged = False
        for cluster in clusters:
            for c_evt in cluster:
                if evt.camera_id == c_evt.camera_id:
                    continue  # Same camera = always distinct

                # Time-based check
                if abs(evt.start_sec - c_evt.start_sec) > 2.0:
                    continue

                # Issue 6: 3D position check when available
                if _HAS_3D_DEDUP and sg is not None:
                    import numpy as np
                    pos_a = _get_event_3d_position(evt, sg)
                    pos_b = _get_event_3d_position(c_evt, sg)
                    if pos_a is not None and pos_b is not None:
                        dist = float(np.linalg.norm(pos_a - pos_b))
                        if dist > CROSS_CAM_DEDUP_DISTANCE_M:
                            continue  # Far apart, not duplicates despite time proximity

                cluster.append(evt)
                merged = True
                break
            if merged:
                break
        if not merged:
            clusters.append([evt])

    all_ids = [e.event_id for e in sorted_evts]

    # Issue 10: Build cluster details (key_frames) — one representative per cluster
    cluster_details = []
    for cluster in clusters:
        # Use earliest event as representative
        rep = min(cluster, key=lambda e: e.start_sec)
        clip_file = rep.video_file
        if clip_file and clip_file.endswith(".avi"):
            clip_file = clip_file.replace(".avi", ".mp4")
        cluster_details.append({
            "camera": rep.camera_id,
            "start_frame": rep.start_frame,
            "start_sec": round(rep.start_sec, 2),
            "end_sec": round(rep.end_sec, 2),
            "event_id": rep.event_id,
            "clip_file": clip_file or "",
            "cluster_size": len(cluster),
        })
    # Sort chronologically
    cluster_details.sort(key=lambda d: d["start_sec"])

    return len(clusters), all_ids, cluster_details


def _activity_counting_candidates(sg: SceneGraph) -> List[Dict]:
    """
    For each activity type, count event instances across all cameras
    with cross-camera temporal deduplication (±2 s).
    """
    # Group events by activity
    activity_groups: Dict[str, list] = defaultdict(list)
    activity_cameras: Dict[str, Set[str]] = defaultdict(set)

    for e in sg.events:
        activity_groups[e.activity].append(e)
        activity_cameras[e.activity].add(e.camera_id)

    candidates = []
    for act, evts in activity_groups.items():
        cnt, event_ids, cluster_details = _dedup_activity_count(evts, sg)
        if cnt < MIN_COUNT or cnt > MAX_COUNT:
            continue
        candidates.append({
            "subtype": "activity_counting",
            "activity": act,
            "correct_count": cnt,
            "cameras_involved": sorted(activity_cameras[act]),
            "event_ids": event_ids,
            "cross_camera": len(activity_cameras[act]) >= 2,
            "key_frames": cluster_details,  # Issue 10: per-instance details
        })
    return candidates



# ============================================================================
# Candidate Scoring
# ============================================================================

def _score_candidate(cand: Dict) -> float:
    """
    Score a candidate – higher is better.

    Prefers:
    - Counts in the [3, 10] sweet-spot
    - Cross-camera occurrence
    - entity_counting slightly preferred (scene-level understanding)
    """
    score = 0.0
    cnt = cand["correct_count"]

    # Sweet-spot bonus
    if SWEET_LOW <= cnt <= SWEET_HIGH:
        score += 3.0
    elif MIN_COUNT <= cnt < SWEET_LOW:
        score += 1.5
    else:
        score += 0.5

    # Cross-camera bonus
    if cand["cross_camera"]:
        score += 2.0

    # Subtype: activity_counting is the only subtype
    score += 0.5

    return score


# ============================================================================
# Question Text Templates
# ============================================================================

def _make_question_text(cand: Dict) -> str:
    """Return the natural-language question string for a candidate."""
    subtype = cand["subtype"]

    if subtype == "activity_counting":
        act_gerund = humanize_activity_gerund(cand["activity"])
        act_lower = act_gerund[0].lower() + act_gerund[1:]  # lowercase first letter
        return (
            f"How many times does someone perform the action of "
            f"{act_lower} across all cameras in this slot?"
        )

    return "How many?"


# ============================================================================
# Difficulty Classification
# ============================================================================

def _classify_difficulty(cand: Dict) -> str:
    """
    easy   : count ≤ 5 and ≤ 3 cameras
    medium : otherwise
    """
    cnt = cand["correct_count"]
    n_cams = len(cand["cameras_involved"])
    if cnt <= 5 and n_cams <= 3:
        return "easy"
    return "medium"


# ============================================================================
# Issue 9: Deterministic Reasoning Generation
# ============================================================================

def _build_reasoning(cand: Dict) -> str:
    """
    Build deterministic reasoning text for a counting question.

    Uses the deduped cluster details to produce structured reasoning with
    per-camera breakdown and the correct count. This reasoning is passed
    to naturalize.py so GPT only polishes grammar, not numbers.

    Example: "The action of opening a vehicle door was observed 8 times
    across cameras G299 and G330: G299 contributed 5 occurrences and G330
    contributed 3 occurrences (0 cross-camera duplicates were removed)."
    """
    activity = humanize_activity(cand.get("activity", "unknown"))
    correct = cand["correct_count"]
    cameras = cand["cameras_involved"]
    key_frames = cand.get("key_frames", [])

    # Per-camera breakdown from key_frames
    cam_counts: Dict[str, int] = defaultdict(int)
    for kf in key_frames:
        cam_counts[kf["camera"]] += 1

    # Count cross-camera dedup removals
    total_raw_events = len(cand.get("event_ids", []))
    dedup_removed = total_raw_events - correct

    if len(cameras) == 1:
        breakdown = f"all on camera {cameras[0]}"
    else:
        cam_parts = [f"{cam}: {cam_counts.get(cam, 0)}" for cam in sorted(cam_counts.keys())]
        breakdown = ", ".join(cam_parts)

    reasoning = (
        f"The action of {activity} was observed {correct} time{'s' if correct != 1 else ''} "
        f"across {len(cameras)} camera{'s' if len(cameras) != 1 else ''} "
        f"({breakdown})"
    )
    if dedup_removed > 0:
        reasoning += f". {dedup_removed} cross-camera duplicate{'s' if dedup_removed != 1 else ''} removed"
    reasoning += "."

    return reasoning


# ============================================================================
# Public API
# ============================================================================

def generate_numerical_qa(
    sg: SceneGraph,
    resolved: ResolvedGraph,
    entity_descs: Dict[str, str],
    rng: random.Random,
    count: int = 1,
    verbose: bool = False,
) -> List[Dict]:
    """
    Generate numerical/counting questions for a multi-camera slot.

    Args:
        sg:           Scene graph (events, entities, cameras, slot).
        resolved:     Resolved cross-camera entity graph.
        entity_descs: entity_id → human-readable description string.
        rng:          Seeded RNG for reproducibility.
        count:        Target number of questions (default 1).
        verbose:      Print debug info.

    Returns:
        List of QA dicts in the standard V8 format.
    """
    if not sg.events:
        if verbose:
            print("  Numerical: no events – skipping")
        return []

    # ------------------------------------------------------------------
    # 1. Collect all candidates from three subtypes
    # ------------------------------------------------------------------
    all_candidates: List[Dict] = []
    all_candidates.extend(_activity_counting_candidates(sg))

    if not all_candidates:
        if verbose:
            print("  Numerical: no valid candidates (counts out of range)")
        return []

    # ------------------------------------------------------------------
    # 2. Score and sort
    # ------------------------------------------------------------------
    for c in all_candidates:
        c["_score"] = _score_candidate(c)
    all_candidates.sort(key=lambda c: c["_score"], reverse=True)

    if verbose:
        print(f"  Numerical: {len(all_candidates)} activity_counting candidates")

    # ------------------------------------------------------------------
    # 3. Diversified selection: no two Qs with same subtype or same activity
    # ------------------------------------------------------------------
    used_subtypes: Set[str] = set()
    used_activities: Set[str] = set()
    selected: List[Dict] = []

    for cand in all_candidates:
        if len(selected) >= count:
            break

        sub = cand["subtype"]
        act = cand.get("activity")

        # Diversity: skip if we already used this subtype
        if sub in used_subtypes:
            continue
        # Diversity: skip if we already asked about this activity
        if act and act in used_activities:
            continue

        used_subtypes.add(sub)
        if act:
            used_activities.add(act)
        selected.append(cand)

    # If we still need more, relax the subtype constraint (keep activity unique)
    if len(selected) < count:
        for cand in all_candidates:
            if len(selected) >= count:
                break
            if cand in selected:
                continue
            act = cand.get("activity")
            if act and act in used_activities:
                continue
            if act:
                used_activities.add(act)
            selected.append(cand)

    # ------------------------------------------------------------------
    # 4. Build QA dicts
    # ------------------------------------------------------------------
    qa_pairs: List[Dict] = []

    for idx, cand in enumerate(selected[:count]):
        correct = cand["correct_count"]
        options, correct_idx = _build_options(correct, rng)
        question = _make_question_text(cand)
        difficulty = _classify_difficulty(cand)

        all_cameras = sorted(sg.cameras.keys())
        requires_cams = cand["cameras_involved"] if cand["cameras_involved"] else all_cameras

        verification: Dict[str, Any] = {
            "question_type": cand["subtype"],
            "correct_count": correct,
            "cameras_involved": cand["cameras_involved"],
        }
        if cand.get("activity"):
            verification["activity"] = cand["activity"]
        if cand.get("event_ids"):
            verification["event_ids"] = cand["event_ids"]
        if cand.get("cluster_ids"):
            verification["cluster_ids"] = cand["cluster_ids"]
        # Issue 10: Key frames for visual verification
        if cand.get("key_frames"):
            verification["key_frames"] = cand["key_frames"]

        debug_info: Dict[str, Any] = {
            "subtype": cand["subtype"],
            "correct_count": correct,
            "cameras_involved": cand["cameras_involved"],
            "cross_camera": cand["cross_camera"],
            "candidate_score": round(cand["_score"], 2),
            "num_candidates_total": len(all_candidates),
            "slot": sg.slot,
        }
        if cand.get("activity"):
            debug_info["activity"] = cand["activity"]
            debug_info["activity_human"] = humanize_activity(cand["activity"])

        # Collect clip_files from the events referenced by this candidate
        event_map = {e.event_id: e for e in sg.events}
        clip_files = set()
        for eid in cand.get("event_ids", []):
            evt = event_map.get(eid)
            if evt and evt.video_file:
                cf = evt.video_file.replace(".avi", ".mp4")
                clip_files.add(cf)
        if clip_files:
            debug_info["clip_files"] = sorted(clip_files)

        # Issue 9: Build deterministic reasoning (not GPT-generated)
        reasoning = _build_reasoning(cand)

        qa = {
            "question_id": f"v8_numerical_{idx + 1:03d}",
            "category": "numerical",
            "difficulty": difficulty,
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
            "reasoning": reasoning,  # Issue 9: pre-built, GPT polishes only grammar
            "requires_cameras": requires_cams,
            "requires_multi_camera": len(requires_cams) > 1,
            "verification": verification,
            "debug_info": debug_info,
        }
        qa_pairs.append(qa)

    if verbose:
        subtypes = Counter(q["debug_info"]["subtype"] for q in qa_pairs)
        print(f"  Numerical: {len(qa_pairs)} questions "
              f"({', '.join(f'{s}={n}' for s, n in subtypes.items())})")

    return qa_pairs

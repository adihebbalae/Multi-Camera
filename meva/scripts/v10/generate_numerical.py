"""
FINAL generate_numerical.py — Numerical/counting questions across cameras.

Tests a model's ability to count activities or entities from a
multi-camera scene.  Two counting subtypes:

1. **activity_counting**: "How many times does [activity] occur across all cameras?"
   → Count event instances of that activity type from sg.events.
2. **entity_counting**: "How many distinct people are visible across all cameras?"
   → Count unique entity clusters from the resolved graph.

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

def _activity_counting_candidates(sg: SceneGraph) -> List[Dict]:
    """
    For each activity type, count total event instances across all cameras.
    """
    activity_counts: Dict[str, int] = Counter(e.activity for e in sg.events)
    activity_cameras: Dict[str, Set[str]] = defaultdict(set)
    activity_events: Dict[str, List[str]] = defaultdict(list)

    for e in sg.events:
        activity_cameras[e.activity].add(e.camera_id)
        activity_events[e.activity].append(e.event_id)

    candidates = []
    for act, cnt in activity_counts.items():
        if cnt < MIN_COUNT or cnt > MAX_COUNT:
            continue
        candidates.append({
            "subtype": "activity_counting",
            "activity": act,
            "correct_count": cnt,
            "cameras_involved": sorted(activity_cameras[act]),
            "event_ids": activity_events[act],
            "cross_camera": len(activity_cameras[act]) >= 2,
        })
    return candidates


def _entity_counting_candidates(
    sg: SceneGraph, resolved: ResolvedGraph
) -> List[Dict]:
    """
    Count distinct entity clusters (unique people/entities) across all cameras.
    """
    # Person clusters only (most natural counting question)
    person_clusters = []
    for cluster in resolved.entity_clusters:
        # Check if cluster contains at least one person entity
        for eid in cluster.entities:
            ent = sg.entities.get(eid)
            if ent and ent.entity_type == "person":
                person_clusters.append(cluster)
                break

    cnt = len(person_clusters)
    if cnt < MIN_COUNT or cnt > MAX_COUNT:
        return []

    all_cameras: Set[str] = set()
    cluster_ids = []
    for cl in person_clusters:
        cluster_ids.append(cl.cluster_id)
        for cam in cl.cameras:
            all_cameras.add(cam)

    return [{
        "subtype": "entity_counting",
        "activity": None,
        "correct_count": cnt,
        "cameras_involved": sorted(all_cameras),
        "event_ids": [],
        "cluster_ids": cluster_ids,
        "cross_camera": len(all_cameras) >= 2,
    }]


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

    # Subtype preference: entity counting is most "scene-level"
    if cand["subtype"] == "entity_counting":
        score += 1.0
    elif cand["subtype"] == "activity_counting":
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

    if subtype == "entity_counting":
        return (
            "How many distinct people are visible across all cameras in this slot?"
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
    all_candidates.extend(_entity_counting_candidates(sg, resolved))

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
        print(f"  Numerical: {len(all_candidates)} candidates "
              f"(act={sum(1 for c in all_candidates if c['subtype'] == 'activity_counting')}, "
              f"ent={sum(1 for c in all_candidates if c['subtype'] == 'entity_counting')})")

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

        qa = {
            "question_id": f"v8_numerical_{idx + 1:03d}",
            "category": "numerical",
            "difficulty": difficulty,
            "question_template": question,
            "options": options,
            "correct_answer_index": correct_idx,
            "correct_answer": options[correct_idx],
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

"""
V8 generate_event_ordering.py — Event ordering questions (chronological arrangement).

Unlike temporal questions which ask "which of 2 events happened first?" (binary),
event ordering asks "arrange 3-4 events in chronological order" (combinatorial).
This is significantly harder since the answer space grows factorially.

Event selection prefers:
  - Cross-camera events (multi-camera ordering is harder than single-camera)
  - Clear temporal gaps (>3s) so ordering is unambiguous
  - Related activities (causal / co-occurring chains)
  - Events involving entities with MEVID descriptions (for visual grounding)

Output: MCQ with scrambled event labels (I, II, III, IV) and 4 permutation options.
"""

import itertools
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from .parse_annotations import Event
from .build_scene_graph import SceneGraph, Entity
from .entity_resolution import ResolvedGraph
from .activity_hierarchy import (
    humanize_activity, humanize_activity_gerund, are_related, get_relationship,
    CAUSAL_RELATIONSHIPS, CO_OCCURRING,
)
from .utils.mevid import find_mevid_persons_for_slot

# Issue 6: Import cross-camera event dedup from temporal generator
from .generate_temporal import _is_likely_duplicate_event

# ============================================================================
# Constants
# ============================================================================

MIN_GAP_SEC = 2.0            # Minimum gap between consecutive events (unambiguous)
MAX_GAP_SEC = 10.0           # Maximum gap — events must be close enough to be related
DEFAULT_FPS = 30.0
MIN_EVENTS = 3               # Minimum events per ordering question
MAX_EVENTS = 4               # Maximum events per ordering question
ROMAN = ["I", "II", "III", "IV"]


# ============================================================================
# Event Description Helpers
# ============================================================================

def _get_event_description(event: Event, sg: SceneGraph,
                           entity_descs: Dict[str, str],
                           fallback_eids: Optional[Set[str]] = None) -> str:
    """
    Build a human-readable event description using MEVID entity descriptions
    and gerund activity forms.

    Prefers visual (non-fallback) descriptions. Falls back to generic
    'Someone {activity}' if only fallback descriptions are available.

    Example: "A person wearing a gray top, opening a facility door on camera G421"
    """
    from .person_descriptions import is_visual_description
    # Try to find a visual (non-fallback) description for an actor in this event
    desc = None
    fallback_desc = None
    for eid, entity in sg.entities.items():
        if entity.camera_id == event.camera_id:
            for actor in event.actors:
                if actor["actor_id"] == entity.actor_id:
                    d = entity_descs.get(eid)
                    if d:
                        if fallback_eids and eid in fallback_eids:
                            if fallback_desc is None:
                                fallback_desc = d
                        else:
                            desc = d
                            break
            if desc:
                break

    activity_text = humanize_activity_gerund(event.activity)

    if desc:
        # Guard: avoid duplication when desc already contains the activity
        act_check = humanize_activity(event.activity).lower()
        if act_check in desc.lower() or activity_text.lower() in desc.lower():
            return desc
        return f"{desc}, {activity_text.lower()}"
    
    # Use fallback description if available (still better than "Someone")
    if fallback_desc:
        act_check = humanize_activity(event.activity).lower()
        if act_check in fallback_desc.lower() or activity_text.lower() in fallback_desc.lower():
            return fallback_desc
        return f"{fallback_desc}, {activity_text.lower()}"
    
    # Last resort: use geom description directly from entity_descs
    for eid, entity in sg.entities.items():
        if entity.camera_id == event.camera_id:
            for actor in event.actors:
                if actor["actor_id"] == entity.actor_id:
                    d = entity_descs.get(eid)
                    if d and d not in ("a person", "a vehicle", "someone"):
                        if activity_text.lower() in d.lower():
                            return d
                        return f"{d}, {activity_text.lower()}"
    
    return f"A person {activity_text.lower()}"


# ============================================================================
# Group Scoring
# ============================================================================

def _count_related_pairs(events: List[Event]) -> int:
    """Count how many event pairs in the group have causal/co-occurring relationships."""
    count = 0
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            if are_related(events[i].activity, events[j].activity):
                count += 1
    return count


def _count_described_events(events: List[Event], sg: SceneGraph,
                            entity_descs: Dict[str, str]) -> int:
    """Count events that have visual (non-fallback) entity descriptions."""
    from .person_descriptions import is_visual_description
    count = 0
    for event in events:
        for eid, entity in sg.entities.items():
            if entity.camera_id == event.camera_id:
                for actor in event.actors:
                    if actor["actor_id"] == entity.actor_id:
                        desc = entity_descs.get(eid, "")
                        if desc and is_visual_description(desc):
                            count += 1
                            break
                else:
                    continue
                break
    return count


def _score_group(events: List[Event], sg: SceneGraph,
                 entity_descs: Dict[str, str],
                 mevid_person_cameras: Dict[int, Set[str]]) -> float:
    """
    Score a candidate event group.  Higher = better.

    Criteria:
      +2.0 per unique camera (cross-camera diversity)
      +1.5 per related-activity pair
      +1.0 per event with an entity description
      +1.0 if any MEVID-validated camera pair exists in the group
      +0.5 for 4-event groups (more challenging)
    """
    cameras = set(e.camera_id for e in events)
    score = len(cameras) * 2.0

    score += _count_related_pairs(events) * 1.5
    score += _count_described_events(events, sg, entity_descs) * 1.0

    # MEVID cross-camera bonus
    cam_list = list(cameras)
    for pid, pcams in mevid_person_cameras.items():
        if len(pcams & cameras) >= 2:
            score += 1.0
            break

    if len(events) == MAX_EVENTS:
        score += 0.5

    return score


# ============================================================================
# Candidate Group Discovery
# ============================================================================

def _find_ordering_groups(events: List[Event], sg: SceneGraph,
                          entity_descs: Dict[str, str],
                          mevid_person_cameras: Dict[int, Set[str]],
                          rng: random.Random,
                          target_count: int = 6) -> List[List[Event]]:
    """
    Find candidate groups of 3-4 events suitable for ordering questions.

    Requirements per group:
      - Events sorted chronologically with consecutive gaps > MIN_GAP_SEC
      - At least 2 distinct cameras
      - No overlapping events (start of next > end of previous + MIN_GAP_SEC)

    Returns groups sorted by score (best first), capped at target_count.
    """
    if len(events) < MIN_EVENTS:
        return []

    # Sort all events by start_sec for chronological processing
    sorted_events = sorted(events, key=lambda e: e.start_sec)

    # De-duplicate: keep one event per (camera, activity, ~time bucket)
    # to avoid near-identical events cluttering groups
    seen_keys: Set[Tuple[str, str, int]] = set()
    unique_events: List[Event] = []
    for e in sorted_events:
        bucket = int(e.start_sec // 5)  # 5-second buckets
        key = (e.camera_id, e.activity, bucket)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_events.append(e)

    if len(unique_events) < MIN_EVENTS:
        return []

    # Issue 5: Keep only the FIRST (earliest) instance per (activity, camera).
    # Continuous activities must use their first instance to avoid misleading
    # temporal comparisons (e.g., talking from t=10 to t=200 should use t=10).
    first_instance: Dict[Tuple[str, str], Event] = {}
    for evt in unique_events:
        key = (evt.activity, evt.camera_id)
        if key not in first_instance or evt.start_sec < first_instance[key].start_sec:
            first_instance[key] = evt
    # Skip events in the first 5 seconds (camera stabilization period)
    unique_events = [e for e in first_instance.values() if e.start_sec >= 5.0]
    unique_events.sort(key=lambda e: e.start_sec)

    if len(unique_events) < MIN_EVENTS:
        return []

    # Build candidate groups using a sliding-window + greedy approach
    # For each starting event, try to build a chain of 3-4 events with gaps
    groups: List[Tuple[float, List[Event]]] = []  # (score, events)
    seen_group_keys: Set[Tuple[str, ...]] = set()

    for start_idx in range(len(unique_events)):
        # Try to build chains of length 3 and 4
        for chain_len in (MAX_EVENTS, MIN_EVENTS):
            chain = [unique_events[start_idx]]

            for next_idx in range(start_idx + 1, len(unique_events)):
                if len(chain) >= chain_len:
                    break
                candidate = unique_events[next_idx]
                last = chain[-1]

                # Must have clear temporal gap (2-10 seconds)
                gap = candidate.start_sec - last.end_sec
                if gap < MIN_GAP_SEC:
                    continue
                if gap > MAX_GAP_SEC:
                    continue

                # Prefer cross-camera: skip same-camera if we already have
                # an event on that camera AND we haven't reached min cameras
                chain_cameras = set(e.camera_id for e in chain)
                if (candidate.camera_id in chain_cameras
                        and len(chain_cameras) < 2
                        and len(chain) >= 2):
                    continue

                # Issue 6: Cross-camera event dedup — skip if candidate
                # is a likely duplicate of any event already in the chain
                if any(_is_likely_duplicate_event(candidate, existing, sg)
                       for existing in chain):
                    continue

                chain.append(candidate)

            if len(chain) < chain_len:
                continue

            # Require at least 2 cameras
            chain_cameras = set(e.camera_id for e in chain)
            if len(chain_cameras) < 2:
                continue

            # Require activity diversity: at most 1 repeated activity
            chain_activities = set(e.activity for e in chain)
            if len(chain_activities) < len(chain) - 1:
                continue

            # De-duplicate by group key (sorted event_ids)
            gkey = tuple(sorted(e.event_id for e in chain))
            if gkey in seen_group_keys:
                continue
            seen_group_keys.add(gkey)

            score = _score_group(chain, sg, entity_descs, mevid_person_cameras)
            groups.append((score, chain))

    # Sort by score descending
    groups.sort(key=lambda g: -g[0])

    return [g[1] for g in groups[:target_count]]


# ============================================================================
# Distractor Permutation Generation
# ============================================================================

def _generate_permutation_label(order: List[int]) -> str:
    """
    Convert an index-based ordering to a Roman-numeral arrow string.

    Example: [1, 3, 0, 2] → "II -> IV -> I -> III"
    """
    return " -> ".join(ROMAN[i] for i in order)


def _generate_distractor_permutations(n: int, correct_order: List[int],
                                      rng: random.Random) -> List[List[int]]:
    """
    Generate 3 distinct distractor permutations for n events.

    Strategies (in priority order):
      1. Reverse of correct order
      2. Swap two adjacent elements
      3. Swap first and last elements
      4. Random permutation (fallback)

    All distractors are guaranteed distinct from each other and from the
    correct order.
    """
    correct_tuple = tuple(correct_order)
    distractors: List[List[int]] = []
    seen: Set[Tuple[int, ...]] = {correct_tuple}

    def _try_add(perm: List[int]) -> bool:
        t = tuple(perm)
        if t not in seen:
            seen.add(t)
            distractors.append(perm)
            return True
        return False

    # Strategy 1: full reverse
    _try_add(list(reversed(correct_order)))

    # Strategy 2: swap adjacent pairs
    for i in range(n - 1):
        if len(distractors) >= 3:
            break
        swapped = list(correct_order)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        _try_add(swapped)

    # Strategy 3: swap first and last
    if len(distractors) < 3:
        swapped = list(correct_order)
        swapped[0], swapped[-1] = swapped[-1], swapped[0]
        _try_add(swapped)

    # Strategy 3.5: rotate by 1 position
    if len(distractors) < 3:
        rotated = correct_order[1:] + correct_order[:1]
        _try_add(rotated)

    # Strategy 4: random permutations as fallback
    all_perms = list(itertools.permutations(range(n)))
    rng.shuffle(all_perms)
    for perm in all_perms:
        if len(distractors) >= 3:
            break
        _try_add(list(perm))

    return distractors[:3]


# ============================================================================
# Question Construction
# ============================================================================

def _build_question_text(descriptions: List[str]) -> str:
    """
    Build the question text with numbered event descriptions.

    Returns multi-line text like:
        Identify the correct chronological order of the following events
        observed across the cameras:
        I. A person wearing a gray top, opening a facility door on camera G421
        II. A person entering a scene through a structure on camera G330
        ...
        Which is the correct chronological order?
    """
    lines = [
        "Identify the correct chronological order of the following events "
        "observed across the cameras:"
    ]
    for i, desc in enumerate(descriptions):
        lines.append(f"{ROMAN[i]}. {desc}")
    lines.append("Which is the correct chronological order?")
    return "\n".join(lines)


def _build_debug_info(event: Event, sg: SceneGraph,
                      entity_descs: Dict[str, str]) -> Dict:
    """Build debug info dict for one event."""
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


# ============================================================================
# Public API
# ============================================================================

def generate_event_ordering_qa(sg: SceneGraph, resolved: ResolvedGraph,
                               entity_descs: Dict[str, str],
                               rng: random.Random, count: int = 2,
                               verbose: bool = False,
                               fallback_eids: Optional[Set[str]] = None) -> List[Dict]:
    """
    Generate event-ordering cross-camera questions.

    Each question presents 3-4 events in scrambled order and asks the
    viewer to identify the correct chronological sequence.  Significantly
    harder than binary temporal questions because the answer space is
    combinatorial (3! = 6 or 4! = 24 permutations).

    Args:
        sg:           Scene graph with events, entities, cameras.
        resolved:     Resolved graph with entity clusters.
        entity_descs: entity_id → human-readable description string.
        rng:          Seeded RNG for reproducibility.
        count:        Target number of questions (default 2).
        verbose:      Print progress info.

    Returns:
        List of question dicts in V8 MCQ format.
    """
    slot_cameras = list(sg.cameras.keys())
    mevid_person_cameras = find_mevid_persons_for_slot(sg.slot, slot_cameras)

    # Step 1: Find candidate groups
    groups = _find_ordering_groups(
        sg.events, sg, entity_descs, mevid_person_cameras, rng,
        target_count=count * 3,  # over-generate for diversity
    )

    if verbose:
        print(f"  Event ordering: {len(groups)} candidate groups")

    if not groups:
        return []

    # Filter out groups where ANY event uses fallback (non-visual) descriptions
    # All events appear in the question text, so all must have visual descriptions
    if fallback_eids:
        def _all_events_visual(group):
            for event in group:
                has_visual = False
                for eid, entity in sg.entities.items():
                    if entity.camera_id == event.camera_id:
                        for actor in event.actors:
                            if actor["actor_id"] == entity.actor_id:
                                if eid not in fallback_eids:
                                    has_visual = True
                                    break
                        if has_visual:
                            break
                if not has_visual:
                    return False
            return True

        visual_groups = [g for g in groups if _all_events_visual(g)]
        if verbose and len(visual_groups) < len(groups):
            print(f"    Filtered {len(groups) - len(visual_groups)} "
                  f"fallback-only groups → {len(visual_groups)} remaining")
        groups = visual_groups

    if not groups:
        return []

    # Step 2: Select diverse groups (avoid reusing same camera sets)
    selected: List[List[Event]] = []
    used_camera_sets: Set[Tuple[str, ...]] = set()
    used_event_ids: Set[str] = set()

    for group in groups:
        if len(selected) >= count:
            break

        cam_set = tuple(sorted(set(e.camera_id for e in group)))
        group_eids = set(e.event_id for e in group)

        # Prefer distinct camera combinations and non-overlapping events
        overlap = group_eids & used_event_ids
        if overlap:
            continue
        if cam_set in used_camera_sets and len(selected) > 0:
            continue

        selected.append(group)
        used_camera_sets.add(cam_set)
        used_event_ids.update(group_eids)

    # If we couldn't fill enough with strict diversity, relax constraints
    if len(selected) < count:
        for group in groups:
            if len(selected) >= count:
                break
            if group not in selected:
                selected.append(group)

    # Step 3: Generate QA for each selected group
    qa_pairs: List[Dict] = []

    for idx, group in enumerate(selected[:count]):
        n = len(group)

        # Events are already in chronological order from _find_ordering_groups
        chronological = sorted(group, key=lambda e: e.start_sec)

        # Build descriptions in chronological order
        descriptions_chrono = [
            _get_event_description(e, sg, entity_descs, fallback_eids) for e in chronological
        ]

        # Scramble presentation order:  correct_order[i] = chronological
        # position of the event presented as Roman numeral (i+1)
        presentation_indices = list(range(n))
        rng.shuffle(presentation_indices)

        # presentation_indices[i] tells which chronological event goes to
        # presentation slot i.  So the event presented as "I" is
        # chronological[presentation_indices[0]], etc.
        presented_descriptions = [descriptions_chrono[pi] for pi in presentation_indices]

        # The correct answer is the permutation that re-sorts presentation
        # back to chronological.  If presentation_indices = [2, 0, 3, 1],
        # then chronological order in presentation labels is determined by
        # argsort(presentation_indices).
        # argsort: for each chrono position k, which presentation slot has it?
        chrono_to_presentation = [0] * n
        for pres_slot, chrono_pos in enumerate(presentation_indices):
            chrono_to_presentation[chrono_pos] = pres_slot

        # correct_order = the presentation slots in chronological sequence
        correct_order = chrono_to_presentation  # e.g. [1, 3, 0, 2]

        # Generate distractor permutations
        distractor_perms = _generate_distractor_permutations(n, correct_order, rng)

        # Build options: correct answer + 3 distractors, then shuffle
        correct_label = _generate_permutation_label(correct_order)
        distractor_labels = [_generate_permutation_label(d) for d in distractor_perms]

        options = [correct_label] + distractor_labels
        correct_answer_index = 0

        # Shuffle options so correct isn't always first
        option_pairs = list(enumerate(options))
        rng.shuffle(option_pairs)
        shuffled_options = [label for _, label in option_pairs]
        correct_answer_index = next(
            i for i, (orig_idx, _) in enumerate(option_pairs) if orig_idx == 0
        )

        # Cameras involved
        all_cameras = sorted(set(e.camera_id for e in chronological))

        # Compute minimum gap between consecutive chronological events
        gaps = []
        for i in range(len(chronological) - 1):
            gap = chronological[i + 1].start_sec - chronological[i].end_sec
            gaps.append(round(gap, 2))
        min_gap = min(gaps) if gaps else 0.0

        # Difficulty: 4 events = hard, 3 events = medium-hard
        difficulty = "hard" if n == 4 else "medium-hard"

        # Build question text
        question_text = _build_question_text(presented_descriptions)

        # Verification: ordered events with timing
        ordered_events = []
        for i, event in enumerate(chronological):
            ordered_events.append({
                "activity": event.activity,
                "camera": event.camera_id,
                "start_sec": round(event.start_sec, 2),
                "description": descriptions_chrono[i],
            })

        # Debug info per event
        debug_events = [_build_debug_info(e, sg, entity_descs) for e in chronological]

        # Check MEVID validation
        mevid_validated = False
        for pid, pcams in mevid_person_cameras.items():
            if len(pcams & set(all_cameras)) >= 2:
                mevid_validated = True
                break

        qa = {
            "question_id": f"v8_event_ordering_{idx + 1:03d}",
            "category": "event_ordering",
            "difficulty": difficulty,
            "question_template": question_text,
            "options": shuffled_options,
            "correct_answer_index": correct_answer_index,
            "correct_answer": shuffled_options[correct_answer_index],
            "requires_cameras": all_cameras,
            "requires_multi_camera": len(all_cameras) >= 2,
            "verification": {
                "ordered_events": ordered_events,
                "min_gap_sec": min_gap,
                "num_events": n,
                "gaps_sec": gaps,
            },
            "debug_info": {
                "events": debug_events,
                "presentation_order": presentation_indices,
                "correct_permutation": correct_order,
                "mevid_validated": mevid_validated,
                "group_score": _score_group(
                    chronological, sg, entity_descs, mevid_person_cameras
                ),
                "related_pairs": _count_related_pairs(chronological),
                "described_events": _count_described_events(
                    chronological, sg, entity_descs
                ),
            },
        }
        qa_pairs.append(qa)

    if verbose:
        print(f"  Event ordering: {len(qa_pairs)} questions generated "
              f"({sum(1 for q in qa_pairs if q['difficulty'] == 'hard')} hard, "
              f"{sum(1 for q in qa_pairs if q['difficulty'] == 'medium-hard')} medium-hard)")

    return qa_pairs

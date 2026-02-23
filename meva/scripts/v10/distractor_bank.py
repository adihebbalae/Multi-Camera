"""
V6 distractor_bank.py — Wrong answer generation from activity pool.

Generates plausible distractor activities for multiple-choice questions.
Distractors are same entity type (person/vehicle) as the correct answer,
preferring activities present in the current slot for plausibility.
"""

import random
from typing import List, Set


# All 37 MEVA activity names grouped by entity type
PERSON_ACTIVITIES = [
    "person_opens_trunk", "person_closes_trunk",
    "person_opens_vehicle_door", "person_closes_vehicle_door",
    "person_opens_facility_door", "person_closes_facility_door",
    "person_enters_vehicle", "person_exits_vehicle",
    "person_unloads_vehicle", "person_loads_vehicle",
    "person_picks_up_object", "person_puts_down_object",
    "person_carries_heavy_object", "person_transfers_object",
    "person_talks_to_person", "person_embraces_person",
    "person_enters_scene_through_structure",
    "person_exits_scene_through_structure",
    "person_sits_down", "person_stands_up",
    "person_talks_on_phone", "person_texts_on_phone",
    "person_reads_document", "person_interacts_with_laptop",
    "person_purchases", "person_rides_bicycle",
]

VEHICLE_ACTIVITIES = [
    "vehicle_starts", "vehicle_stops",
    "vehicle_turns_left", "vehicle_turns_right",
    "vehicle_makes_u_turn", "vehicle_reverses",
    "vehicle_drops_off_person", "vehicle_picks_up_person",
]

ALL_ACTIVITIES = PERSON_ACTIVITIES + VEHICLE_ACTIVITIES


def get_distractors(correct_activity: str, slot_activities: Set[str],
                    rng: random.Random, n: int = 3) -> List[str]:
    """
    Pick n distractor activities that are:
    1. Same entity type (person/vehicle) as correct answer
    2. NOT the correct activity
    3. Prefer activities present in THIS slot (more plausible)
    4. Fall back to global pool if needed
    
    Args:
        correct_activity: The correct answer activity
        slot_activities: Set of all activities in the current slot
        rng: Random number generator (for reproducibility)
        n: Number of distractors to generate
    
    Returns:
        List of n distractor activity names
    """
    entity = "vehicle" if correct_activity.startswith("vehicle_") else "person"
    pool = VEHICLE_ACTIVITIES if entity == "vehicle" else PERSON_ACTIVITIES

    # Prefer activities in this slot (more believable distractors)
    in_slot = [a for a in slot_activities if a != correct_activity and a in pool]
    out_slot = [a for a in pool if a != correct_activity and a not in in_slot]

    rng.shuffle(in_slot)
    rng.shuffle(out_slot)

    distractors = in_slot[:n]
    if len(distractors) < n:
        distractors += out_slot[:n - len(distractors)]
    return distractors[:n]


def get_camera_distractors(correct_cameras: List[str], all_cameras: List[str],
                           rng: random.Random, n: int = 3) -> List[str]:
    """
    Pick n distractor camera IDs.
    
    Args:
        correct_cameras: Camera(s) that are the correct answer
        all_cameras: All cameras in the slot
        rng: Random number generator
        n: Number of distractors
    
    Returns:
        List of distractor camera IDs
    """
    pool = [c for c in all_cameras if c not in correct_cameras]
    rng.shuffle(pool)
    result = pool[:n]
    
    # If not enough distractors from slot cameras, pad from global MEVA pool
    if len(result) < n:
        global_pool = [
            "G299", "G300", "G328", "G330", "G336", "G339", "G341", "G419",
            "G420", "G421", "G423", "G424", "G436", "G638", "G639",
            "G503", "G504", "G505", "G506", "G507", "G508", "G509",
        ]
        extras = [c for c in global_pool 
                  if c not in correct_cameras and c not in result and c not in all_cameras]
        rng.shuffle(extras)
        result.extend(extras[:n - len(result)])
    
    return result[:n]

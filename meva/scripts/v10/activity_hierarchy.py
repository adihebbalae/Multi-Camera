"""
V7 activity_hierarchy.py — Activity relationship definitions for connected event pairing.

Defines causal, sequential, and co-occurring activity relationships from the
37 MEVA ActEV activities. Used by generate_temporal.py to prefer event pairs
that have meaningful scene connections.
"""

from typing import Dict, List, Set, Tuple

# ============================================================================
# Activity Relationships
# ============================================================================

# Causal/sequential relationships: act_a often leads to act_b
CAUSAL_RELATIONSHIPS: Dict[str, List[str]] = {
    # Object manipulation chains
    "person_picks_up_object": [
        "person_carries_heavy_object",
        "person_puts_down_object",
        "person_transfers_object",
    ],
    "person_puts_down_object": [
        "person_abandons_package",
    ],

    # Trunk/vehicle loading
    "person_opens_trunk": [
        "person_loads_vehicle",
        "person_unloads_vehicle",
        "person_closes_trunk",
    ],
    "person_loads_vehicle": ["person_closes_trunk"],
    "person_unloads_vehicle": ["person_closes_trunk", "person_carries_heavy_object"],

    # Vehicle door sequences
    "person_opens_vehicle_door": [
        "person_enters_vehicle",
        "person_exits_vehicle",
        "person_closes_vehicle_door",
    ],
    "person_enters_vehicle": ["person_closes_vehicle_door", "vehicle_starts"],
    "person_exits_vehicle": ["person_closes_vehicle_door"],

    # Facility door sequences
    "person_opens_facility_door": [
        "person_enters_scene_through_structure",
        "person_exits_scene_through_structure",
        "person_closes_facility_door",
    ],
    "person_enters_scene_through_structure": ["person_closes_facility_door"],
    "person_exits_scene_through_structure": ["person_closes_facility_door"],

    # Vehicle sequences
    "vehicle_stops": ["person_exits_vehicle", "vehicle_starts"],
    "vehicle_starts": ["vehicle_turns_left", "vehicle_turns_right", "vehicle_makes_u_turn"],

    # Sit/stand
    "person_sits_down": ["person_stands_up"],

    # Drop-off/pick-up
    "vehicle_drops_off_person": ["person_enters_scene_through_structure"],
    "vehicle_picks_up_person": ["vehicle_starts"],
}

# Symmetric co-occurring relationships (activities that often happen together)
CO_OCCURRING: List[Tuple[str, str]] = [
    ("person_talks_to_person", "person_embraces_person"),
    ("person_talks_on_phone", "person_texts_on_phone"),
    ("person_reads_document", "person_interacts_with_laptop"),
    ("person_enters_vehicle", "person_exits_vehicle"),
    ("person_opens_vehicle_door", "person_closes_vehicle_door"),
    ("person_opens_trunk", "person_closes_trunk"),
    ("person_opens_facility_door", "person_closes_facility_door"),
    ("person_picks_up_object", "person_puts_down_object"),
    ("person_loads_vehicle", "person_unloads_vehicle"),
    ("vehicle_starts", "vehicle_stops"),
]

# Build fast lookup sets
_CAUSAL_PAIRS: Set[Tuple[str, str]] = set()
for parent, children in CAUSAL_RELATIONSHIPS.items():
    for child in children:
        _CAUSAL_PAIRS.add((parent, child))

_CO_OCCURRING_PAIRS: Set[Tuple[str, str]] = set()
for a, b in CO_OCCURRING:
    _CO_OCCURRING_PAIRS.add((a, b))
    _CO_OCCURRING_PAIRS.add((b, a))


# ============================================================================
# Public API
# ============================================================================

def are_related(act_a: str, act_b: str) -> bool:
    """Check if two activities have any relationship (causal or co-occurring)."""
    return get_relationship(act_a, act_b) is not None


def get_relationship(act_a: str, act_b: str) -> str:
    """
    Get relationship type between two activities.
    
    Returns:
        "causal" if act_a causally leads to act_b
        "reverse_causal" if act_b causally leads to act_a  
        "co_occurring" if they commonly co-occur
        None if no relationship
    """
    if (act_a, act_b) in _CAUSAL_PAIRS:
        return "causal"
    if (act_b, act_a) in _CAUSAL_PAIRS:
        return "reverse_causal"
    if (act_a, act_b) in _CO_OCCURRING_PAIRS:
        return "co_occurring"
    return None


def get_relationship_strength(act_a: str, act_b: str) -> float:
    """
    Get relationship strength (0.0 = none, 1.0 = strong).
    
    causal: 1.0
    reverse_causal: 0.9 (order matters but both are related)
    co_occurring: 0.7
    same_entity_type: 0.3 (both person_ or both vehicle_)
    none: 0.0
    """
    rel = get_relationship(act_a, act_b)
    if rel == "causal":
        return 1.0
    if rel == "reverse_causal":
        return 0.9
    if rel == "co_occurring":
        return 0.7
    # Same entity type (weak connection)
    if act_a.split("_")[0] == act_b.split("_")[0]:
        return 0.3
    return 0.0


def get_activity_entity_type(activity: str) -> str:
    """Get entity type from activity name."""
    if activity.startswith("vehicle_"):
        return "vehicle"
    return "person"


def humanize_activity(activity: str) -> str:
    """Convert activity name to short human-readable form."""
    # Special-case enter/exit scene to clarify MEVA camera-view semantics
    _SPECIAL = {
        "person_enters_scene_through_structure":
            "enters the camera's view through a doorway",
        "person_exits_scene_through_structure":
            "leaves the camera's view through a doorway",
    }
    if activity in _SPECIAL:
        return _SPECIAL[activity]
    # Remove entity prefix and replace underscores with spaces
    for prefix in ("person_", "vehicle_", "hand_"):
        if activity.startswith(prefix):
            activity = activity[len(prefix):]
            break
    return activity.replace("_", " ")


# Verb → gerund mappings for natural sentence construction
_GERUND_MAP = {
    "opens": "opening", "closes": "closing", "enters": "entering",
    "exits": "exiting", "reads": "reading", "carries": "carrying",
    "picks": "picking", "puts": "putting", "sets": "setting",
    "rides": "riding", "loads": "loading", "unloads": "unloading",
    "talks": "talking", "stands": "standing", "walks": "walking",
    "runs": "running", "sits": "sitting", "texts": "texting",
    "pulls": "pulling", "pushes": "pushing", "interacts": "interacting",
    "drops": "dropping", "embraces": "embracing", "uses": "using",
    "makes": "making", "steals": "stealing", "starts": "starting",
    "stops": "stopping", "turns": "turning", "transfers": "transferring",
    "reverses": "reversing", "abandons": "abandoning",
    "leaves": "leaving", "purchases": "purchasing",
}


def _conjugate_gerund(verb: str) -> str:
    """Smart fallback: conjugate an unknown verb to its -ing form.

    Handles common English patterns:
      leaves → leaving, purchases → purchasing, transfers → transferring,
      runs → running, sits → sitting, walks → walking
    """
    if verb in _GERUND_MAP:
        return _GERUND_MAP[verb]
    # Strip third-person 's'/'es' to get base form
    if verb.endswith("es"):
        base = verb[:-2]   # "leaves" → "leav", "purchases" → "purchas"
    elif verb.endswith("s") and not verb.endswith("ss"):
        base = verb[:-1]   # "abandons" → "abandon"
    else:
        base = verb
    # Apply standard English gerund rules on base form
    if base.endswith("ie"):       # "die" → "dying"
        return base[:-2] + "ying"
    if base.endswith("ee"):       # "see" → "seeing"
        return base + "ing"
    if base.endswith("e"):        # "leave" → "leaving", "make" → "making"
        return base[:-1] + "ing"
    # CVC doubling: short base ending in consonant-vowel-consonant
    if (len(base) >= 3
            and base[-1] not in "aeiouwxy"
            and base[-2] in "aeiou"
            and base[-3] not in "aeiou"):
        return base + base[-1] + "ing"   # "run" → "running", "sit" → "sitting"
    return base + "ing"


def humanize_activity_gerund(activity: str) -> str:
    """
    Convert activity to gerund form for sentence construction.
    e.g. 'person_opens_facility_door' → 'Opening a facility door'
    """
    base = humanize_activity(activity)  # e.g. 'opens facility door'
    words = base.split()
    if words:
        first = words[0]
        gerund = _conjugate_gerund(first)
        rest = " ".join(words[1:])
        # Only add article if rest starts with a noun-like word
        # Skip if rest starts with preposition, adverb, article, or particle
        _no_article = {"up", "down", "on", "off", "out", "in", "to", "from",
                       "through", "with", "around", "right", "left", "a",
                       "an", "the", "into", "onto", "over", "away"}
        if rest:
            first_rest = rest.split()[0]
            if first_rest not in _no_article:
                rest = "a " + rest
        result = f"{gerund} {rest}".strip() if rest else gerund
        return result.capitalize()
    return base.capitalize()

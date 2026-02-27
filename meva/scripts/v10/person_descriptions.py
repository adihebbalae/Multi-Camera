"""
V8 person_descriptions.py — MEVID person description loading and entity enrichment.

Loads the YOLO+GPT person database and injects natural-language person
descriptions into scene graph entities. This is the key V8 addition over V7.

Description priority:
  1. GPT description (richest, from person_database_yolo.json) — simplified
  2. YOLO color summary (structured fallback)
  3. Activity-verb description (V7 style, no MEVID)
"""

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .build_scene_graph import SceneGraph, Entity
from .activity_hierarchy import humanize_activity

# ============================================================================
# Paths
# ============================================================================

# Repo-relative data directory (meva/data/) — works for any clone location
_REPO_DATA = Path(__file__).resolve().parent.parent.parent / "data"
# User output directory — override with MEVA_OUTPUT_DIR env var
_OUTPUT = Path(os.environ.get("OUTPUT_DIR") or os.environ.get("MEVA_OUTPUT_DIR") or str(Path.home() / "data"))
# Entity descriptions directory — override with MEVA_ENTITY_DESC_DIR env var
_ENTITY_DESC_DIR = Path(os.environ.get("MEVA_ENTITY_DESC_DIR") or "/nas/mars/dataset/MEVA/entity_descriptions")

PERSON_DB_PATH = _REPO_DATA / "person_database_yolo.json"
PERSON_DB_ORIG_PATH = _REPO_DATA / "person_database.json"
MEVID_SLOTS_PATH = _REPO_DATA / "mevid_supported_slots.json"


# ============================================================================
# Person Database Loading
# ============================================================================

_person_db_cache: Optional[Dict] = None
_person_db_orig_cache: Optional[Dict] = None

def load_person_database() -> Dict:
    """Load the YOLO+GPT person attribute database (cached)."""
    global _person_db_cache
    if _person_db_cache is not None:
        return _person_db_cache
    
    if not PERSON_DB_PATH.exists():
        return {"persons": {}, "metadata": {}}
    
    with open(PERSON_DB_PATH) as f:
        _person_db_cache = json.load(f)
    return _person_db_cache


def _load_person_db_orig() -> Dict:
    """Load the original person database with camera mappings (cached)."""
    global _person_db_orig_cache
    if _person_db_orig_cache is not None:
        return _person_db_orig_cache
    
    if not PERSON_DB_ORIG_PATH.exists():
        return {"persons": {}}
    
    with open(PERSON_DB_ORIG_PATH) as f:
        _person_db_orig_cache = json.load(f)
    return _person_db_orig_cache


# ============================================================================
# Description Simplification — Clean GPT's verbose formal language
# ============================================================================

# Garment type simplifications: "upper body garment" → "top", etc.
_GARMENT_SUBS = [
    (r"upper body garment with a hood", "hoodie"),
    (r"upper body garment", "top"),
    (r"lower body clothing", "pants"),
    (r"lower body garment", "pants"),
    (r"lower body pants", "pants"),
    (r"lower body shorts", "shorts"),
    (r"hooded jacket", "hoodie"),
]

# Posture/action context to strip (not useful for identification)
_STRIP_PATTERNS = [
    r",?\s*sitting on a chair[^,\.]*",
    r",?\s*with their back turned[^,\.]*",
    r",?\s*while ascending a staircase[^,\.]*",
    r",?\s*while holding a clipboard or some papers",
    r",?\s*and they appear to have\s+",
    r",?\s*sitting\b[^,\.]*",
]

def simplify_description(desc: str) -> str:
    """
    Simplify a GPT description into natural short form.
    
    "wearing a blue upper body garment and blue lower body clothing, 
     with a black hoodie featuring a graphic design on the back."
    →
    "wearing a blue top and blue pants, with a black hoodie featuring 
     a graphic design on the back"
    """
    if not desc:
        return desc
    
    # 1. Remove trailing period
    desc = desc.rstrip(". ")
    
    # 2. Simplify garment types
    for pattern, replacement in _GARMENT_SUBS:
        desc = re.sub(pattern, replacement, desc, flags=re.IGNORECASE)
    
    # 3. Strip posture/action context
    for pattern in _STRIP_PATTERNS:
        desc = re.sub(pattern, "", desc, flags=re.IGNORECASE)
    
    # 4. Clean up orphaned commas / double spaces
    desc = re.sub(r",\s*,", ",", desc)
    desc = re.sub(r"\s{2,}", " ", desc)
    desc = desc.strip(", ")
    
    # 5. Remove trailing period again (after stripping may leave one)
    desc = desc.rstrip(".")
    
    return desc


def get_person_description(person_id: str, outfit_id: str = None) -> str:
    """
    Get the best available description for a MEVID person.
    
    Priority:
      1. GPT description (natural language, simplified)
      2. YOLO color summary
      3. Generic "a person"
    
    Args:
        person_id: MEVID person ID (e.g., "0008")
        outfit_id: Optional outfit ID for outfit-specific colors
    
    Returns:
        Description string like "a person in a dark hoodie and dark pants"
    """
    db = load_person_database()
    persons = db.get("persons", {})
    
    if person_id not in persons:
        return "a person"
    
    person = persons[person_id]
    
    # Priority 1: GPT description (simplified)
    gpt_desc = person.get("gpt_description")
    if gpt_desc:
        desc = gpt_desc.strip()
        # Remove leading boilerplate
        for prefix in ["The person is ", "the person is ", "The person ", "the person "]:
            if desc.startswith(prefix):
                desc = desc[len(prefix):]
                break
        # Simplify formal language
        desc = simplify_description(desc)
        return f"a person {desc}"
    
    # Priority 2: YOLO color summary
    upper = person.get("primary_upper_color", "unknown")
    lower = person.get("primary_lower_color", "unknown")
    
    # Try outfit-specific colors if specified
    if outfit_id and outfit_id in person.get("outfits", {}):
        outfit = person["outfits"][outfit_id]
        upper = outfit.get("upper_body_color", upper)
        lower = outfit.get("lower_body_color", lower)
    
    if upper != "unknown" or lower != "unknown":
        parts = []
        if upper != "unknown":
            parts.append(f"{upper} top")
        if lower != "unknown":
            parts.append(f"{lower} pants")
        return f"a person in {' and '.join(parts)}"
    
    # Priority 3: Generic
    return "a person"


# ============================================================================
# Visual Description Detection
# ============================================================================

_VISUAL_KEYWORDS = frozenset([
    "top", "pants", "wearing", "shirt", "jacket", "hoodie", "shorts",
    "color", "blue", "red", "black", "white", "green", "gray", "grey",
    "yellow", "brown", "orange", "purple", "pink", "navy", "dark", "light",
    "backpack", "bag", "suitcase", "hat", "cap", "dress", "coat",
])


def is_visual_description(desc: str) -> bool:
    """Check if a description is based on visual appearance (clothing/color).

    Returns True for MEVID and geom-color descriptions like:
      - "a person in a blue top and black pants"
      - "a person wearing a dark hoodie"

    Returns False for activity-verb fallbacks like:
      - "a person opens facility door"
      - "a person carries object"
      - "a person"
      - "a vehicle"
      - "someone walking"
    """
    if not desc:
        return False
    desc_lower = desc.lower()
    if desc_lower in ("a person", "a vehicle", "someone"):
        return False
    return any(kw in desc_lower for kw in _VISUAL_KEYWORDS)


def get_person_short_label(person_id: str) -> str:
    """
    Get a short label for a person (for options / distractor text).
    
    Returns things like "person in blue jacket" (shorter than full GPT description).
    """
    db = load_person_database()
    persons = db.get("persons", {})
    
    if person_id not in persons:
        return f"Person #{person_id}"
    
    person = persons[person_id]
    upper = person.get("primary_upper_color", "unknown")
    lower = person.get("primary_lower_color", "unknown")
    
    objects = person.get("all_carried_objects", [])
    
    parts = []
    if upper != "unknown":
        parts.append(f"{upper} top")
    if lower != "unknown":
        parts.append(f"{lower} bottom")
    if objects:
        parts.append(f"carrying {objects[0]}")
    
    if parts:
        return f"person with {', '.join(parts)}"
    return f"Person #{person_id}"


# ============================================================================
# Slot Filtering — Only MEVID-Supported Slots
# ============================================================================

_mevid_slots_cache: Optional[Dict] = None

def load_mevid_slots() -> Dict:
    """Load the MEVID-supported slots data (cached)."""
    global _mevid_slots_cache
    if _mevid_slots_cache is not None:
        return _mevid_slots_cache
    
    if not MEVID_SLOTS_PATH.exists():
        return {"slots": {}}
    
    with open(MEVID_SLOTS_PATH) as f:
        _mevid_slots_cache = json.load(f)
    return _mevid_slots_cache


def _resolve_mevid_slot(slot: str) -> Optional[str]:
    """Resolve a slot name to its key in mevid_supported_slots.json.

    The MEVID index uses HH-MM-SS format while the pipeline uses HH-MM.
    This bridges the gap by trying both forms.
    """
    data = load_mevid_slots()
    slots = data.get("slots", {})
    if slot in slots:
        return slot
    # Try appending -00 to get HH-MM-SS from HH-MM
    parts = slot.split(".")
    if len(parts) >= 2:
        time_part = parts[1]
        if len(time_part) == 5:  # HH-MM
            expanded = f"{parts[0]}.{time_part}-00.{'.' .join(parts[2:])}"
            if expanded in slots:
                return expanded
    return None


def _resolve_all_mevid_slots(slot: str) -> List[str]:
    """Return all MEVID slot keys matching a canonical HH-MM slot.

    A canonical HH-MM slot may map to multiple HH-MM-SS raw slots in the
    MEVID index (e.g., 2018-03-11.11-25.school → 11-25-00, 11-25-01, etc.).
    """
    data = load_mevid_slots()
    slots = data.get("slots", {})
    if slot in slots:
        return [slot]
    parts = slot.split(".")
    if len(parts) < 3 or len(parts[1]) != 5:
        return []
    prefix = f"{parts[0]}.{parts[1]}"
    site = parts[2]
    return [k for k in slots if k.startswith(prefix) and k.endswith(f".{site}")]


def is_mevid_supported(slot: str) -> bool:
    """Check if a slot has MEVID person support."""
    return len(_resolve_all_mevid_slots(slot)) > 0


def get_mevid_persons_for_slot(slot: str) -> List[str]:
    """
    Get list of MEVID person IDs available for a slot.

    Reads from mevid_supported_slots.json which maps each slot to its
    MEVID persons (built by aggregate_mevid_slots.py).
    Merges across all matching raw slots for canonical HH-MM lookups.
    """
    data = load_mevid_slots()
    slots = data.get("slots", {})
    matching = _resolve_all_mevid_slots(slot)
    all_persons = set()
    for m in matching:
        slot_info = slots.get(m, {})
        all_persons.update(slot_info.get("mevid_persons", []))
    return sorted(all_persons)


def get_mevid_persons_with_cameras(slot: str) -> Dict[str, List[str]]:
    """
    Get MEVID person IDs mapped to their cameras for this specific slot.

    Cross-references:
      - mevid_supported_slots.json → which persons and cameras are in this slot
      - person_database.json → which cameras each person globally appears on

    Merges across all matching raw slots for canonical HH-MM lookups.
    Returns: {person_id: [camera_ids_in_this_slot]}
    """
    # Get slot info (merge across all matching raw slots)
    slot_data = load_mevid_slots()
    slots = slot_data.get("slots", {})
    matching = _resolve_all_mevid_slots(slot)

    mevid_persons = set()
    mevid_cameras = set()
    for m in matching:
        slot_info = slots.get(m, {})
        mevid_persons.update(slot_info.get("mevid_persons", []))
        mevid_cameras.update(slot_info.get("mevid_cameras", []))
    
    if not mevid_persons or not mevid_cameras:
        return {}
    
    # Get per-person camera lists from original person database
    orig_db = _load_person_db_orig()
    orig_persons = orig_db.get("persons", {})
    
    result = {}
    for pid in sorted(mevid_persons):
        person_data = orig_persons.get(pid, {})
        person_cameras = set(person_data.get("cameras", {}).keys())
        # Intersect with this slot's MEVID cameras
        cameras_in_slot = sorted(person_cameras & mevid_cameras)
        if cameras_in_slot:
            result[pid] = cameras_in_slot
    
    return result


# ============================================================================
# Entity Enrichment — Inject MEVID Descriptions into Scene Graph
# ============================================================================

# Geom-extracted description bank directory
_GEOM_DESC_DIR = _ENTITY_DESC_DIR


def _load_geom_descriptions(slot: str) -> Dict[str, str]:
    """
    Load pre-extracted visual descriptions from extract_entity_descriptions.py.
    These are HSV color-based descriptions from raw AVI + geom.yml bounding boxes.
    Returns Dict[entity_id → description], e.g. "G330_actor_123" → "a person in a blue top and black pants"
    """
    desc_path = _GEOM_DESC_DIR / f"{slot}.json"
    if not desc_path.exists():
        return {}
    try:
        with open(desc_path) as f:
            data = json.load(f)
        return {eid: info["description"] for eid, info in data.get("actors", {}).items()
                if info.get("description") and info["description"] != "a person"}
    except (json.JSONDecodeError, KeyError):
        return {}


def enrich_entities(sg: SceneGraph, verbose: bool = False) -> Dict[str, str]:
    """
    Enrich scene graph entities with visual descriptions.
    
    Priority:
      1. MEVID descriptions (GPT/YOLO from MEVID crops — highest quality)
      2. Geom-extracted descriptions (HSV color from raw AVI + geom.yml bbox)
      3. Activity-verb fallback ("a person walking")
    
    The geom-extracted layer covers ALL annotated actors (not just MEVID's ~10%),
    giving every entity a color-based description for disambiguation.
    
    Args:
        sg: Scene graph to enrich
        verbose: Print enrichment details
    
    Returns:
        Dict mapping entity_id → description string
    """
    slot = sg.slot
    person_cameras = get_mevid_persons_with_cameras(slot)
    geom_descs = _load_geom_descriptions(slot)
    
    # Build reverse map: camera_id → [person_ids on this camera]
    camera_persons: Dict[str, List[str]] = {}
    for pid, cams in person_cameras.items():
        for cam in cams:
            if cam not in camera_persons:
                camera_persons[cam] = []
            camera_persons[cam].append(pid)
    
    entity_descriptions: Dict[str, str] = {}
    mevid_count = 0
    geom_count = 0
    fallback_count = 0
    
    # Track which MEVID persons have been assigned to avoid reuse
    assigned_persons: Dict[str, Set[str]] = {}  # camera → set of used person_ids
    
    for eid, entity in sg.entities.items():
        # Determine effective entity type: if entity is tagged as "vehicle"
        # but participates ONLY in person_* activities, treat it as a person.
        # This fixes annotation artifacts where types.yml misclassifies actors.
        effective_type = entity.entity_type
        if effective_type == "vehicle":
            # Check if ALL activities for this entity start with 'person_'
            entity_activities = []
            for evt in sg.events:
                if evt.camera_id == entity.camera_id:
                    for actor in evt.actors:
                        if actor["actor_id"] == entity.actor_id:
                            entity_activities.append(evt.activity)
            # If entity has person activities, treat as person
            has_person_acts = any(a.startswith("person_") for a in entity_activities)
            has_vehicle_acts = any(a.startswith("vehicle_") for a in entity_activities)
            if has_person_acts and not has_vehicle_acts:
                effective_type = "person"  # reclassify
        
        if effective_type != "person":
            entity_descriptions[eid] = "a vehicle"
            continue
        
        cam = entity.camera_id
        available_persons = camera_persons.get(cam, [])
        
        # Priority 1: MEVID person description
        if available_persons:
            used = assigned_persons.get(cam, set())
            unused = [p for p in available_persons if p not in used]
            
            if unused:
                # Assign next available person
                pid = unused[0]
                desc = get_person_description(pid)
                entity_descriptions[eid] = desc
                
                if cam not in assigned_persons:
                    assigned_persons[cam] = set()
                assigned_persons[cam].add(pid)
                
                # Also store the MEVID person_id on the entity for re-ID questions
                entity._mevid_person_id = pid
                mevid_count += 1
                
                if verbose:
                    print(f"    {eid}: MEVID → {desc[:60]}...")
                continue
        
        # Priority 2: Geom-extracted color description (from raw AVI + bbox)
        if eid in geom_descs:
            desc = geom_descs[eid]
            entity_descriptions[eid] = desc
            geom_count += 1
            if verbose:
                print(f"    {eid}: geom → {desc}")
            continue
        
        # Priority 3: Activity-verb fallback (V7 style)
        primary_activity = None
        for evt in sg.events:
            if evt.camera_id == entity.camera_id:
                for actor in evt.actors:
                    if actor["actor_id"] == entity.actor_id:
                        primary_activity = evt.activity
                        break
                if primary_activity:
                    break
        
        if primary_activity:
            short_act = humanize_activity(primary_activity)
            desc = f"a person {short_act}"
        else:
            desc = "a person"
        
        entity_descriptions[eid] = desc
        fallback_count += 1
    
    if verbose:
        total = mevid_count + geom_count + fallback_count
        print(f"  Entity enrichment: {mevid_count} MEVID, {geom_count} geom-color, "
              f"{fallback_count} fallback ({total} total)")
    
    # Build set of PERSON entity IDs that got fallback (non-visual) descriptions.
    # Non-person entities (vehicles, objects) always get generic descriptions like
    # "a vehicle" — that's correct and complete, not a quality failure.
    # Only person entities with generic fallbacks ("a person", "someone walking")
    # degrade question quality, so only they are flagged here.
    fallback_eids = set()
    for eid, desc in entity_descriptions.items():
        entity = sg.entities.get(eid)
        if entity and entity.entity_type != "person":
            continue  # vehicles/objects: "a vehicle" is acceptable, not fallback
        if not is_visual_description(desc):
            fallback_eids.add(eid)

    return (entity_descriptions,
            {"mevid": mevid_count, "geom": geom_count, "fallback": fallback_count},
            fallback_eids)

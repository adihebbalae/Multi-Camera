"""
V8 person_descriptions.py — MEVID person description loading and entity enrichment.

Loads the YOLO+GPT person database and injects natural-language person
descriptions into scene graph entities. This is the key V8 addition over V7.

Description priority:
  1. GPT description (richest, from person_database_yolo.json) — simplified
  2. YOLO color summary (structured fallback)
  3. Activity-verb description (V7 style, no MEVID)
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .build_scene_graph import SceneGraph, Entity
from .activity_hierarchy import humanize_activity

# ============================================================================
# Paths
# ============================================================================

PERSON_DB_PATH = Path("/home/ah66742/data/person_database_yolo.json")
MEVID_SLOTS_PATH = Path("/home/ah66742/data/mevid_supported_slots.json")


# ============================================================================
# Person Database Loading
# ============================================================================

_person_db_cache: Optional[Dict] = None

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


def is_mevid_supported(slot: str) -> bool:
    """Check if a slot has MEVID person support."""
    data = load_mevid_slots()
    slots = data.get("slots", {})
    return slot in slots


def get_mevid_persons_for_slot(slot: str) -> List[str]:
    """
    Get list of MEVID person IDs available for a slot.
    
    Cross-references the person database (23 persons with descriptions)
    against the slot's camera set.
    """
    db = load_person_database()
    persons = db.get("persons", {})
    
    result = []
    for pid, pdata in persons.items():
        for slot_info in pdata.get("slots", []):
            if slot_info.get("slot") == slot:
                result.append(pid)
                break
    
    return sorted(result)


def get_mevid_persons_with_cameras(slot: str) -> Dict[str, List[str]]:
    """
    Get MEVID person IDs mapped to their cameras for this specific slot.
    
    Returns: {person_id: [camera_ids]}
    """
    db = load_person_database()
    persons = db.get("persons", {})
    
    result = {}
    for pid, pdata in persons.items():
        for slot_info in pdata.get("slots", []):
            if slot_info.get("slot") == slot:
                result[pid] = slot_info.get("cameras", [])
                break
    
    return result


# ============================================================================
# Entity Enrichment — Inject MEVID Descriptions into Scene Graph
# ============================================================================

# Geom-extracted description bank directory
_GEOM_DESC_DIR = Path("/home/ah66742/data/entity_descriptions")


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
        if entity.entity_type != "person":
            entity_descriptions[eid] = f"a vehicle on camera {entity.camera_id}"
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
            desc = f"a person on camera {cam}"
        
        entity_descriptions[eid] = desc
        fallback_count += 1
    
    if verbose:
        total = mevid_count + geom_count + fallback_count
        print(f"  Entity enrichment: {mevid_count} MEVID, {geom_count} geom-color, "
              f"{fallback_count} fallback ({total} total)")
    
    return entity_descriptions, {"mevid": mevid_count, "geom": geom_count, "fallback": fallback_count}

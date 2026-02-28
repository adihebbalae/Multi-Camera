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
# VLM description directory (InternVL2.5-8B output)
_VLM_DESC_DIR = _ENTITY_DESC_DIR / "vlm"

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


# ============================================================================
# Color Match Scoring — Cross-validate MEVID ↔ geom-extracted colors (Issue 1)
# ============================================================================

# Color similarity groups: colors in the same group are considered "matching"
_COLOR_GROUPS = {
    "black": {"black", "charcoal", "dark"},
    "dark_gray": {"charcoal", "dark gray", "dark"},
    "gray": {"gray", "grey", "silver", "dark gray"},
    "white": {"white", "ivory", "light"},
    "red": {"red", "crimson", "maroon", "rust"},
    "orange": {"orange", "rust"},
    "yellow": {"yellow", "gold", "khaki"},
    "green": {"green", "olive", "teal"},
    "blue": {"blue", "navy", "indigo", "teal"},
    "purple": {"purple", "plum", "indigo", "mauve"},
    "pink": {"pink", "mauve"},
    "brown": {"brown", "khaki", "beige"},
    "beige": {"beige", "khaki", "ivory"},
    "navy": {"navy", "blue", "dark blue", "indigo"},
    "olive": {"olive", "green", "khaki"},
}

def _normalize_color(color: str) -> str:
    """Normalize a color string for matching (lowercase, strip prefixes)."""
    if not color:
        return ""
    color = color.lower().strip()
    # Strip 'dark ' / 'light ' prefixes for fuzzy matching
    for prefix in ("dark ", "light ", "bright "):
        if color.startswith(prefix):
            return color[len(prefix):]
    return color

def _colors_similar(color_a: str, color_b: str) -> bool:
    """Check if two color names are semantically similar."""
    a = _normalize_color(color_a)
    b = _normalize_color(color_b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Check if they share a color group
    for group_colors in _COLOR_GROUPS.values():
        if a in group_colors and b in group_colors:
            return True
    return False

def _color_match_score(mevid_person_data: Dict, geom_desc: str) -> float:
    """
    Score how well a MEVID person's colors match a geom-extracted description.

    Returns 0.0-2.0:
      - 1.0 per matching region (upper, lower)
      - 0.0 for unknown/missing data
      - -1.0 penalty for clear mismatch (known colors that don't match)
    
    Args:
        mevid_person_data: Dict with primary_upper_color, primary_lower_color
        geom_desc: e.g. "a person in a blue top and black pants"
    """
    if not geom_desc or geom_desc in ("a person", "a vehicle", "someone"):
        return 0.0  # No geom data to compare — neutral score

    mevid_upper = mevid_person_data.get("primary_upper_color", "unknown")
    mevid_lower = mevid_person_data.get("primary_lower_color", "unknown")

    # Parse geom description for colors: "a person in a {color} top and {color} pants"
    geom_upper = ""
    geom_lower = ""
    desc_lower = geom_desc.lower()
    
    # Extract upper color: look for "{color} top" or "wearing a {color} top"
    import re as _re_color
    upper_match = _re_color.search(r'(?:in\s+(?:a\s+)?|wearing\s+(?:a\s+)?)(\w+)\s+top', desc_lower)
    if upper_match:
        geom_upper = upper_match.group(1)
    
    # Extract lower color: look for "{color} pants/shorts"
    lower_match = _re_color.search(r'(\w+)\s+(?:pants|shorts|skirt)', desc_lower)
    if lower_match:
        geom_lower = lower_match.group(1)

    score = 0.0
    
    # Score upper body
    if mevid_upper != "unknown" and geom_upper:
        if _colors_similar(mevid_upper, geom_upper):
            score += 1.0
        else:
            score -= 1.0  # Clear mismatch penalty
    
    # Score lower body
    if mevid_lower != "unknown" and geom_lower:
        if _colors_similar(mevid_lower, geom_lower):
            score += 1.0
        else:
            score -= 1.0  # Clear mismatch penalty
    
    return score


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
        descs = {eid: info["description"] for eid, info in data.get("actors", {}).items()
                if info.get("description") and info["description"] != "a person"}
        # Clean up geom descriptions: remove "unknown" qualifiers, fix articles
        return {eid: _clean_geom_description(desc) for eid, desc in descs.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def _clean_geom_description(desc: str) -> str:
    """Clean a pre-formatted geom description string.

    Removes "unknown" qualifier words and fixes article agreement (a→an
    before vowels).  Does NOT consolidate colors — specific colors like
    indigo, navy, teal are kept for better entity differentiation.
    """
    import re
    result = desc
    # Remove "unknown" qualifier
    result = re.sub(r'\bunknown\s+', '', result, flags=re.IGNORECASE)
    # Fix article agreement: "a indigo" → "an indigo", "a olive" → "an olive"
    result = re.sub(r'\ba\s+([aeiou])', r'an \1', result, flags=re.IGNORECASE)
    return result


def _load_vlm_descriptions(slot: str) -> Dict[str, str]:
    """
    Load VLM-generated descriptions (InternVL2.5-8B) for a slot.
    These are rich natural-language descriptions from video crops.
    Returns Dict[entity_id → description], e.g. "G330_actor_123" → "a man in a dark blue jacket..."
    """
    vlm_path = _VLM_DESC_DIR / f"{slot}.vlm.json"
    if not vlm_path.exists():
        return {}
    try:
        with open(vlm_path) as f:
            data = json.load(f)
        descs = data.get("descriptions", {})
        # Filter out empty/generic descriptions
        return {eid: desc for eid, desc in descs.items()
                if desc and len(desc) > 10 and desc.lower() != "a person"}
    except (json.JSONDecodeError, KeyError):
        return {}


def enrich_entities(sg: SceneGraph, verbose: bool = False) -> Dict[str, str]:
    """
    Enrich scene graph entities with visual descriptions.
    
    Priority:
      1. MEVID descriptions (GPT/YOLO from MEVID crops — highest quality)
      2. VLM descriptions (InternVL2.5-8B from video crops — rich NL)
      3. Geom-extracted descriptions (SegFormer color from raw AVI + bbox)
      4. Activity-verb fallback ("a person walking")
    
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
    vlm_descs = _load_vlm_descriptions(slot)
    
    # Build reverse map: camera_id → [person_ids on this camera]
    camera_persons: Dict[str, List[str]] = {}
    for pid, cams in person_cameras.items():
        for cam in cams:
            if cam not in camera_persons:
                camera_persons[cam] = []
            camera_persons[cam].append(pid)
    
    entity_descriptions: Dict[str, str] = {}
    mevid_count = 0
    vlm_count = 0
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
        
        # Priority 1: MEVID person description (color-matched, Issue 1)
        if available_persons:
            used = assigned_persons.get(cam, set())
            unused = [p for p in available_persons if p not in used]
            
            if unused:
                # Color-match: score all unused MEVID persons against geom-extracted colors
                # Pick the best match instead of sequential first-come-first-served
                geom_desc = geom_descs.get(eid, "")
                db = load_person_database()
                persons_db = db.get("persons", {})
                
                best_pid = None
                best_score = -999.0
                for candidate_pid in unused:
                    pdata = persons_db.get(candidate_pid, {})
                    score = _color_match_score(pdata, geom_desc)
                    if score > best_score:
                        best_score = score
                        best_pid = candidate_pid
                
                # Use best match if score is acceptable (>= -0.5 threshold),
                # otherwise fall back to geom description directly
                if best_pid is not None and best_score >= -0.5:
                    pid = best_pid
                    desc = get_person_description(pid)
                    entity_descriptions[eid] = desc
                    
                    if cam not in assigned_persons:
                        assigned_persons[cam] = set()
                    assigned_persons[cam].add(pid)
                    
                    # Also store the MEVID person_id on the entity for re-ID questions
                    entity._mevid_person_id = pid
                    mevid_count += 1
                    
                    if verbose:
                        print(f"    {eid}: MEVID → {desc[:60]}... (color_score={best_score:.1f})")
                    continue
                elif verbose and best_pid:
                    print(f"    {eid}: MEVID rejected (color_score={best_score:.1f} < -0.5, using geom)")
        
        # Priority 2: VLM description (InternVL2.5-8B from video crops)
        if eid in vlm_descs:
            desc = vlm_descs[eid]
            entity_descriptions[eid] = desc
            vlm_count += 1
            if verbose:
                print(f"    {eid}: VLM → {desc[:60]}...")
            continue

        # Priority 3: Geom-extracted color description (SegFormer + bbox)
        if eid in geom_descs:
            desc = geom_descs[eid]
            entity_descriptions[eid] = desc
            geom_count += 1
            if verbose:
                print(f"    {eid}: geom → {desc}")
            continue
        
        # Priority 4: Activity-verb fallback (V7 style)
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
        total = mevid_count + vlm_count + geom_count + fallback_count
        print(f"  Entity enrichment: {mevid_count} MEVID, {vlm_count} VLM, "
              f"{geom_count} geom-color, {fallback_count} fallback ({total} total)")
    
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
            {"mevid": mevid_count, "vlm": vlm_count, "geom": geom_count, "fallback": fallback_count},
            fallback_eids)


# ============================================================================
# Cross-Camera Clustering — Unify descriptions for cross-camera entities
# ============================================================================

# Standard color palette: map exotic HSV names → standard ~12 colors
_COLOR_CONSOLIDATION = {
    "navy": "dark blue", "indigo": "dark blue",
    "teal": "teal", "olive": "olive",
    "charcoal": "dark gray", "rust": "brown",
    "plum": "purple", "mauve": "pink",
    "gold": "yellow", "khaki": "tan",
    "ivory": "white", "beige": "tan",
    "crimson": "red", "maroon": "dark red",
    "silver": "gray",
}


def _consolidate_color(color: str) -> str:
    """Map exotic color names to standard palette for better matching."""
    if not color or color == "unknown":
        return color
    return _COLOR_CONSOLIDATION.get(color.lower(), color.lower())


def _height_category(avg_crop_height: float) -> str:
    """Categorize entity height from average crop pixel height.

    Height thresholds calibrated for MEVA surveillance cameras:
      - Tall: > 200px (close to camera or genuinely tall)
      - Short: < 100px (far from camera or genuinely short)
      - Average: in between (majority)

    Returns empty string if height doesn't meaningfully differentiate.
    """
    if avg_crop_height >= 200:
        return "tall"
    elif avg_crop_height <= 80:
        return "short"
    return ""


def _majority_vote_attr(values: List[str]) -> str:
    """Return the most common non-unknown value, or 'unknown'.
    
    For color attributes, groups similar colors (e.g., navy/dark blue/indigo)
    before voting, but returns the RAW most-common color (not consolidated)
    to preserve display richness.
    """
    valid = [v for v in values if v and v != "unknown"]
    if not valid:
        return "unknown"
    from collections import Counter
    
    # Group by consolidated color for voting strength, but return raw winner
    consolidated_groups = {}  # consolidated_color → [raw_colors]
    for v in valid:
        c = _consolidate_color(v)
        if c not in consolidated_groups:
            consolidated_groups[c] = []
        consolidated_groups[c].append(v)
    
    # Find the consolidated group with most votes
    best_group = max(consolidated_groups.values(), key=len)
    # Return the most common raw color within that group
    return Counter(best_group).most_common(1)[0][0]


def _merge_accessories(acc_lists: List[List[str]]) -> List[str]:
    """Merge accessory lists — keep items appearing in 2+ sources."""
    from collections import Counter
    all_items = Counter()
    for acc in acc_lists:
        for item in set(acc):  # deduplicate within each source
            all_items[item] += 1
    # Keep items appearing in at least 1 source (any evidence is useful)
    return sorted(set(all_items.keys()))


def _build_description(attrs: dict) -> str:
    """Build a natural description string from merged attributes.

    Uses the same format as extract_entity_descriptions.py for consistency.
    Includes 68b fields: texture (striped/patterned) and brightness (dark/light).
    """
    parts = []

    # Hair
    hair = attrs.get("hair_color", "unknown")
    if hair != "unknown":
        parts.append(f"with {hair} hair")

    # Clothing — keep specific colors (indigo, navy, teal etc.) for differentiation
    upper = attrs.get("upper_color", "unknown")
    lower = attrs.get("lower_color", "unknown")
    lower_type = attrs.get("lower_type", "pants")
    upper_brightness = attrs.get("upper_brightness", "")
    lower_brightness = attrs.get("lower_brightness", "")
    upper_texture = attrs.get("upper_texture", "")
    lower_texture = attrs.get("lower_texture", "")

    clothing = []
    if upper != "unknown":
        # Build qualifier: "dark patterned navy" or just "navy"
        upper_quals = []
        # Skip brightness qualifier if the consolidated color already includes it
        # (e.g., "dark blue" already implies "dark", so don't say "dark dark blue")
        if (upper_brightness and upper_brightness not in ("", "medium", "unknown")
                and not upper.startswith(upper_brightness)):
            upper_quals.append(upper_brightness)
        if upper_texture and upper_texture not in ("", "solid", "unknown"):
            upper_quals.append(upper_texture)
        qualifier = " ".join(upper_quals)
        if qualifier:
            article = "an" if qualifier[0].lower() in "aeiou" else "a"
            clothing.append(f"{article} {qualifier} {upper} top")
        else:
            article = "an" if upper[0].lower() in "aeiou" else "a"
            clothing.append(f"{article} {upper} top")
    if lower != "unknown":
        lower_quals = []
        if (lower_brightness and lower_brightness not in ("", "medium", "unknown")
                and not lower.startswith(lower_brightness)):
            lower_quals.append(lower_brightness)
        if lower_texture and lower_texture not in ("", "solid", "unknown"):
            lower_quals.append(lower_texture)
        qualifier = " ".join(lower_quals)
        if qualifier:
            clothing.append(f"{qualifier} {lower} {lower_type}")
        else:
            clothing.append(f"{lower} {lower_type}")

    if clothing:
        parts.append("wearing " + " and ".join(clothing))

    # Shoes
    shoe = attrs.get("shoe_color", "unknown")
    if shoe != "unknown":
        parts.append(f"{shoe} shoes")

    # Accessories
    accessories = attrs.get("accessories", [])
    if accessories:
        parts.append("with " + ", ".join(accessories))

    if not parts:
        return "a person"

    return "a person " + ", ".join(parts)


def merge_cross_camera_descriptions(
    entity_descs: Dict[str, str],
    resolved,  # ResolvedGraph
    sg,       # SceneGraph
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Post-processing: unify descriptions for cross-camera entity clusters.

    For each entity cluster (same person seen on multiple cameras):
      1. Collect raw SegFormer attributes from all entities in the cluster
      2. Majority-vote on each attribute across cameras
      3. Merge accessories/carried objects (union)
      4. Build a single unified description
      5. Assign it to ALL entities in the cluster

    Also adds height hints to differentiate same-description entities
    within a single camera.

    Args:
        entity_descs: Current entity_id → description mapping
        resolved: ResolvedGraph from entity_resolution
        sg: SceneGraph with entity data
        verbose: Print progress

    Returns:
        Updated entity_descs with unified cross-camera descriptions
    """
    if not resolved.entity_clusters:
        if verbose:
            print("  Cross-camera clustering: no clusters to merge")
        return entity_descs

    # Load raw SegFormer actor data for attribute-level merging
    desc_path = _ENTITY_DESC_DIR / f"{sg.slot}.json"
    raw_actors = {}
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                raw_data = json.load(f)
            raw_actors = raw_data.get("actors", {})
        except (json.JSONDecodeError, KeyError):
            pass

    merged_count = 0
    enriched_count = 0

    for cluster in resolved.entity_clusters:
        entity_ids = cluster.entities
        if len(entity_ids) < 2:
            continue

        # Collect attributes from all entities in cluster
        hair_colors = []
        upper_colors = []
        lower_colors = []
        lower_types = []
        shoe_colors = []
        upper_textures = []
        lower_textures = []
        upper_brightnesses = []
        lower_brightnesses = []
        all_accessories = []
        all_carried = []
        heights = []

        for eid in entity_ids:
            actor_data = raw_actors.get(eid, {})
            if not actor_data:
                continue

            hair_colors.append(actor_data.get("hair_color", "unknown"))
            # Use RAW colors for majority vote (display) — NOT consolidated
            # Color consolidation is only for cross-camera matching similarity
            upper_colors.append(actor_data.get("upper_color", "unknown"))
            lower_colors.append(actor_data.get("lower_color", "unknown"))
            lower_types.append(actor_data.get("lower_type", "pants"))
            shoe_colors.append(actor_data.get("shoe_color", "unknown"))
            upper_textures.append(actor_data.get("upper_texture", ""))
            lower_textures.append(actor_data.get("lower_texture", ""))
            upper_brightnesses.append(actor_data.get("upper_brightness", ""))
            lower_brightnesses.append(actor_data.get("lower_brightness", ""))
            all_accessories.append(actor_data.get("accessories", []))
            all_carried.append(actor_data.get("carried_objects", []))
            h = actor_data.get("avg_crop_height", 0)
            if h > 0:
                heights.append(h)

        if not upper_colors:
            continue  # No raw data available for this cluster

        # Majority vote on each attribute
        merged_attrs = {
            "hair_color": _majority_vote_attr(hair_colors),
            "upper_color": _majority_vote_attr(upper_colors),
            "lower_color": _majority_vote_attr(lower_colors),
            "lower_type": _majority_vote_attr(lower_types),
            "shoe_color": _majority_vote_attr(shoe_colors),
            "upper_texture": _majority_vote_attr(upper_textures),
            "lower_texture": _majority_vote_attr(lower_textures),
            "upper_brightness": _majority_vote_attr(upper_brightnesses),
            "lower_brightness": _majority_vote_attr(lower_brightnesses),
            "accessories": _merge_accessories(all_accessories + all_carried),
        }

        # Build unified description
        unified = _build_description(merged_attrs)

        # Count how many attributes the unified version has vs individual ones
        old_descs = {eid: entity_descs.get(eid, "a person") for eid in entity_ids}

        # Assign to all entities in cluster
        for eid in entity_ids:
            old = entity_descs.get(eid, "a person")
            # Only upgrade — don't replace a richer MEVID description with a simpler one
            if unified != "a person" and (
                not is_visual_description(old) or
                len(unified) >= len(old)
            ):
                entity_descs[eid] = unified
                if unified != old:
                    enriched_count += 1

        merged_count += 1

        if verbose and merged_count <= 3:
            print(f"    Cluster {cluster.cluster_id}: {len(entity_ids)} entities "
                  f"across {cluster.cameras}")
            for eid in entity_ids[:2]:
                print(f"      {eid}: {old_descs.get(eid, '?')[:50]} → {unified[:50]}")

    if verbose:
        print(f"  Cross-camera clustering: {merged_count} clusters merged, "
              f"{enriched_count} descriptions unified")

    return entity_descs


def differentiate_within_camera(
    entity_descs: Dict[str, str],
    sg,  # SceneGraph
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Add differentiating attributes for entities with identical descriptions
    on the same camera.

    Strategy:
      - Group entities by (camera, description)
      - For groups with >1 entity, add height category if available
      - This makes "a person wearing a navy top and black pants" into
        "a tall person wearing a navy top and black pants"

    Args:
        entity_descs: entity_id → description mapping
        sg: SceneGraph with entity data
        verbose: Print stats

    Returns:
        Updated entity_descs with differentiated descriptions
    """
    # Load raw actor data for height info
    desc_path = _ENTITY_DESC_DIR / f"{sg.slot}.json"
    raw_actors = {}
    if desc_path.exists():
        try:
            with open(desc_path) as f:
                raw_data = json.load(f)
            raw_actors = raw_data.get("actors", {})
        except (json.JSONDecodeError, KeyError):
            pass

    if not raw_actors:
        return entity_descs

    # Group entities by (camera, description)
    from collections import defaultdict
    cam_desc_groups: Dict[tuple, list] = defaultdict(list)
    for eid, desc in entity_descs.items():
        entity = sg.entities.get(eid)
        if not entity or entity.entity_type != "person":
            continue
        cam_desc_groups[(entity.camera_id, desc)].append(eid)

    differentiated = 0
    for (cam, desc), eids in cam_desc_groups.items():
        if len(eids) < 2:
            continue

        # Get heights for entities in this group
        eid_heights = {}
        for eid in eids:
            actor_data = raw_actors.get(eid, {})
            h = actor_data.get("avg_crop_height", 0)
            if h > 0:
                eid_heights[eid] = h

        if not eid_heights:
            continue

        # Compute relative height categories within this group
        heights = sorted(eid_heights.values())
        if len(heights) < 2:
            continue

        median_h = heights[len(heights) // 2]
        spread = max(heights) - min(heights)

        # Only differentiate if there's meaningful height spread (>30% of median)
        if spread < median_h * 0.3:
            continue

        for eid in eids:
            h = eid_heights.get(eid)
            if h is None:
                continue

            # Relative categorization within the group
            if h > median_h * 1.2:
                prefix = "tall"
            elif h < median_h * 0.8:
                prefix = "short"
            else:
                continue  # Near median — don't label

            old_desc = entity_descs[eid]
            # Insert height after "a " — "a person..." → "a tall person..."
            if old_desc.startswith("a person"):
                new_desc = f"a {prefix} person" + old_desc[len("a person"):]
                entity_descs[eid] = new_desc
                differentiated += 1

    if verbose:
        print(f"  Height differentiation: {differentiated} entities labeled tall/short")

    return entity_descs
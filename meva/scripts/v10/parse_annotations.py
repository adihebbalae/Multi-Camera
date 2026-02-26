"""
V6 parse_annotations.py — Step 1: Parse Kitware YAML annotations into raw events.

Input: Slot name (e.g., "2018-03-11.11-25.school")
Output: List of Event dicts with activity, camera, frame range, actors.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field

# ============================================================================
# Paths
# ============================================================================

MEVA_ROOT = Path("/nas/mars/dataset/MEVA")
ANNOTATION_BASE = MEVA_ROOT / "meva-data-repo" / "annotation" / "DIVA-phase-2" / "MEVA"
KITWARE_ROOT = ANNOTATION_BASE / "kitware"
KITWARE_TRAINING_ROOT = ANNOTATION_BASE / "kitware-meva-training"
# Repo-relative data directory (meva/data/) — works for any clone location
_REPO_DATA = Path(__file__).resolve().parent.parent.parent / "data"
SLOT_INDEX_PATH = _REPO_DATA / "slot_index.json"

DEFAULT_FRAMERATE = 30.0


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Event:
    """A single annotated activity event."""
    event_id: str
    activity: str
    camera_id: str
    site: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    actors: List[Dict[str, Any]]  # [{actor_id, entity_type}]
    video_file: str
    annotation_source: str

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# YAML Parsing
# ============================================================================

def _load_yaml_fast(path: Path) -> list:
    """Load YAML using CSafeLoader for speed."""
    import yaml
    try:
        Loader = yaml.CSafeLoader
    except AttributeError:
        Loader = yaml.SafeLoader
    with open(path) as f:
        return yaml.load(f, Loader=Loader) or []


def _parse_types_yml(path: Path) -> Dict[int, str]:
    """Parse types.yml to get actor_id → entity_type mapping."""
    if not path.exists():
        return {}
    type_map = {}
    for entry in _load_yaml_fast(path):
        t = entry.get("types", {})
        if t:
            aid = t.get("id1")
            cset = t.get("cset3", {})
            etype = next(iter(cset.keys()), "unknown") if cset else "unknown"
            if aid is not None:
                type_map[aid] = etype
    return type_map


def _parse_activities_yml(path: Path, camera_id: str, site: str,
                          framerate: float, source: str) -> List[Event]:
    """Parse a Kitware activities.yml file into Event objects."""
    if not path.exists():
        return []
    entries = _load_yaml_fast(path)
    events = []

    # Load types.yml for actor type resolution
    types_path = path.with_name(path.name.replace(".activities.yml", ".types.yml"))
    type_map = _parse_types_yml(types_path)

    for entry in entries:
        act = entry.get("act", {})
        if not act:
            continue
        act2 = act.get("act2", {})
        activity_name = next(iter(act2.keys()), "unknown")
        activity_id = act.get("id2", -1)
        timespan = act.get("timespan", [])
        if not timespan:
            continue
        tsr = timespan[0].get("tsr0", [])
        if len(tsr) < 2:
            continue
        start_frame, end_frame = int(tsr[0]), int(tsr[1])
        start_sec = round(start_frame / framerate, 2)
        end_sec = round(end_frame / framerate, 2)

        actors = []
        for actor_entry in act.get("actors", []):
            aid = actor_entry.get("id1")
            if aid is not None:
                actors.append({
                    "actor_id": aid,
                    "entity_type": type_map.get(aid, "unknown"),
                })

        clip_name = path.stem.replace(".activities", "")
        event_id = f"{camera_id}_evt_{activity_id}"
        events.append(Event(
            event_id=event_id,
            activity=activity_name,
            camera_id=camera_id,
            site=site,
            start_frame=start_frame,
            end_frame=end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=round(end_sec - start_sec, 2),
            actors=actors,
            video_file=f"{clip_name}.r13.avi",
            annotation_source=source,
        ))

    return events


# ============================================================================
# Slot-Level Annotation Discovery
# ============================================================================

CANONICAL_SLOTS_PATH = _REPO_DATA / "canonical_slots.json"

def _resolve_to_raw_slots(slot: str) -> List[str]:
    """Resolve a slot name to raw slot names.
    
    With HH-MM slot format (no seconds), each slot is already canonical.
    No indirection needed.
    """
    return [slot]


def find_clips_for_slot(slot: str) -> List[Dict]:
    """
    Find all annotation clips for a given slot using slot_index.json.
    
    Handles both raw slots and canonical slots (auto-resolves via canonical_slots.json).
    Returns list of clip metadata dicts with paths to activities.yml files.
    Priority: kitware > kitware-training (skip camera if already found in higher-priority source).
    """
    if not SLOT_INDEX_PATH.exists():
        raise FileNotFoundError(
            "slot_index.json not found. Run: python3 scripts/extract_logic_tuples.py --build-index"
        )

    with open(SLOT_INDEX_PATH) as f:
        index = json.load(f)

    # Resolve canonical → raw slots if needed
    if slot not in index:
        raw_slots = _resolve_to_raw_slots(slot)
        if not any(rs in index for rs in raw_slots):
            raise ValueError(f"Slot '{slot}' not found in index ({len(index)} total slots)")
    else:
        raw_slots = [slot]

    # Merge clips from all raw slots, deduplicating by camera
    clips = []
    cameras_seen = set()

    for raw_slot in raw_slots:
        if raw_slot not in index:
            continue
        info = index[raw_slot]

        # Parse date/time/site from slot name: "2018-03-11.11-25.school"
        slot_parts = raw_slot.split(".")
        date = slot_parts[0]
        slot_time = slot_parts[1]  # HH-MM (no seconds)
        site = slot_parts[2] if len(slot_parts) > 2 else "school"
        hour = slot_time[:2]

        # Priority order
        source_dirs = {
            "kitware": KITWARE_ROOT,
            "kitware-training": KITWARE_TRAINING_ROOT,
        }

        for source_name, source_dir in source_dirs.items():
            if source_name not in info.get("sources", {}):
                continue
            ann_dir = source_dir / date / hour
            if not ann_dir.exists():
                continue

            for cam_id in sorted(info["cameras"]):
                if cam_id in cameras_seen:
                    continue
                if cam_id not in info["sources"].get(source_name, []):
                    continue

                # Find matching activities.yml (minute-level match)
                pattern = f"{date}.{slot_time}*{cam_id}*.activities.yml"
                matches = sorted(ann_dir.glob(pattern))
                if matches:
                    act_file = matches[0]
                    clip_name = act_file.stem.replace(".activities", "")
                    cameras_seen.add(cam_id)
                    clips.append({
                        "clip_name": clip_name,
                        "camera_id": cam_id,
                        "site": site,
                        "annotation_dir": str(ann_dir),
                        "annotation_source": source_name,
                        "framerate": DEFAULT_FRAMERATE,
                        "activities_file": str(act_file),
                    })

    return clips


def parse_slot_events(slot: str, verbose: bool = False) -> List[Event]:
    """
    Parse all annotation events for a slot.
    
    Args:
        slot: Slot name e.g. "2018-03-11.11-25.school"
        verbose: Print progress info
    
    Returns:
        Sorted list of Event objects (chronological).
    """
    clips = find_clips_for_slot(slot)
    if verbose:
        print(f"  Found {len(clips)} clips: {[c['camera_id'] for c in clips]}")

    all_events = []
    for clip in clips:
        events = _parse_activities_yml(
            Path(clip["activities_file"]),
            clip["camera_id"],
            clip["site"],
            clip["framerate"],
            clip["annotation_source"],
        )
        all_events.extend(events)
        if verbose:
            print(f"    {clip['camera_id']}: {len(events)} events ({clip['annotation_source']})")

    # Sort chronologically
    all_events.sort(key=lambda e: (e.start_sec, e.camera_id))
    return all_events

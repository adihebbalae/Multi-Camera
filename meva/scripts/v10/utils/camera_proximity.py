"""
Camera proximity detection — identifies spatial relationships between cameras.

Unlike camera_overlap.py (which flags duplicate-event risk for dedup), this module
identifies cameras that cover ADJACENT areas — ideal for multi-camera temporal
questions because events on nearby but non-overlapping cameras are likely part
of the same scene narrative.

Combines:
  1. KRTD-derived camera positions + viewing directions (outdoor cameras)
  2. Camera-set metadata from the MEVA clip table (groups cameras by shared FOV)
  3. Manual indoor camera knowledge (admin, school hallways, bus indoor)

Camera proximity tiers:
  - "same_area": Cameras that likely observe the same physical space
                 (overlapping FOV, <15m apart, or same camera-set)
  - "adjacent": Cameras covering adjacent areas — a person walking between
                them is plausible in 10-120s (15-80m apart, same site)
  - "same_site": Same site but far apart (>80m or different indoor/outdoor zones)
  - "different_site": Different sites (should never be paired)
"""

import numpy as np
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .krtd import KRTD_DIR, INDOOR_CAMERAS, load_camera_model

# ============================================================================
# Camera-Set Groups (from MEVA clip table metadata)
# ============================================================================

# Camera sets indicate cameras that share sufficient FOV for frame-level sync.
# Source: meva-data-repo/metadata/meva-clip-camera-and-time-table.txt
CAMERA_SETS: Dict[str, List[str]] = {
    # School
    "3-300": ["G300"],
    "3-328": ["G328"],
    "3-330": ["G299", "G330"],       # Indoor gym pair — both see basketball court
    "3-336": ["G336"],
    "3-420": ["G419", "G420"],       # Indoor hallway close-up cameras
    "3-421": ["G421"],               # Indoor close-up
    "3-423": ["G423"],               # Indoor close-up
    "3-424": ["G424"],
    "skip":  ["G339"],               # PTZ / not yet assigned
    "IR-school": ["G474"],           # Infrared
    "3-638": ["G638"],
    "3-639": ["G639"],
    # Bus
    "3-331": ["G331"],               # Indoor
    "3-340": ["G340"],
    "3-508": ["G508", "G509"],       # G508 indoor, G509 outdoor — adjacent
    "IR-bus": ["G475"],              # Infrared
    "3-505": ["G505"],
    "3-506": ["G506"],
    # Hospital
    "3-301": ["G301"],
    "3-341": ["G341"],
    "3-436": ["G436"],
    "3-476": ["G476"],
    "3-479": ["G479"],
    # Admin
    "admin": ["G326", "G329"],       # Both point at same building entrance
}

# Which site each camera belongs to
CAMERA_SITE: Dict[str, str] = {
    # School
    "G299": "school", "G300": "school", "G328": "school", "G330": "school",
    "G336": "school", "G339": "school", "G419": "school", "G420": "school",
    "G421": "school", "G423": "school", "G424": "school", "G474": "school",
    "G638": "school", "G639": "school",
    # Bus
    "G331": "bus", "G340": "bus", "G475": "bus", "G505": "bus",
    "G506": "bus", "G508": "bus", "G509": "bus",
    # Hospital
    "G301": "hospital", "G341": "hospital", "G436": "hospital",
    "G476": "hospital", "G479": "hospital",
    # Admin
    "G326": "admin", "G329": "admin",
}

# Interior vs exterior classification
# Source: meva-data-repo/metadata/meva-camera-daily-status.txt
INDOOR_CAMERAS_FULL: Set[str] = {
    "G299", "G326", "G329", "G330", "G331",
    "G419", "G420", "G421", "G423", "G508",
}

OUTDOOR_CAMERAS: Set[str] = {
    "G300", "G301", "G328", "G336", "G339", "G340", "G341",
    "G424", "G436", "G474", "G475", "G476", "G479",
    "G505", "G506", "G509", "G638", "G639",
}

# ============================================================================
# Manual Indoor Camera Adjacency
# ============================================================================

# Indoor cameras without KRTD: manually specify which outdoor cameras are
# adjacent (a person exiting the indoor area would appear on these cameras).
# Based on camera-set analysis, site layout, and annotation patterns.

INDOOR_ADJACENT_OUTDOOR: Dict[str, List[str]] = {
    # School gym cameras → nearby school outdoor cameras
    "G299": ["G300", "G328", "G336", "G424", "G638", "G639"],
    "G330": ["G300", "G328", "G336", "G424", "G638", "G639"],
    # School hallway close-up cameras → nearby outdoor + gym
    "G419": ["G300", "G328", "G424", "G299", "G330"],
    "G420": ["G300", "G328", "G424", "G299", "G330"],
    "G421": ["G300", "G328", "G424"],
    "G423": ["G300", "G328", "G424"],
    # Admin indoor → both see same entrance (overlap, not adjacent)
    "G326": ["G329"],
    "G329": ["G326"],
    # Bus indoor cameras → bus outdoor cameras
    "G331": ["G340", "G505", "G506", "G509", "G508"],
    "G508": ["G340", "G505", "G506", "G509", "G331"],
}


# ============================================================================
# KRTD-based Distance Cache
# ============================================================================

_DISTANCE_CACHE: Optional[Dict[FrozenSet[str], float]] = None


def _compute_krtd_distances() -> Dict[FrozenSet[str], float]:
    """Compute pairwise distances between all cameras with KRTD calibration."""
    distances: Dict[FrozenSet[str], float] = {}

    # Parse all KRTD files for camera centers
    positions: Dict[str, np.ndarray] = {}
    for f in KRTD_DIR.glob("*.krtd"):
        cam_id = f.stem.split(".")[-1]
        if cam_id in INDOOR_CAMERAS:
            continue
        try:
            lines = [l.strip() for l in open(f) if l.strip()]
            R = np.array([[float(x) for x in lines[i].split()] for i in range(3, 6)])
            T = np.array([float(x) for x in lines[6].split()])
            C = -R.T @ T
            positions[cam_id] = C
        except Exception:
            continue

    cam_list = sorted(positions.keys())
    for i in range(len(cam_list)):
        for j in range(i + 1, len(cam_list)):
            key = frozenset({cam_list[i], cam_list[j]})
            dist = float(np.linalg.norm(positions[cam_list[i]] - positions[cam_list[j]]))
            distances[key] = dist

    return distances


def _get_distances() -> Dict[FrozenSet[str], float]:
    global _DISTANCE_CACHE
    if _DISTANCE_CACHE is None:
        _DISTANCE_CACHE = _compute_krtd_distances()
    return _DISTANCE_CACHE


# ============================================================================
# Public API
# ============================================================================

def get_camera_distance(cam_a: str, cam_b: str) -> Optional[float]:
    """Get distance in meters between two cameras (KRTD-based). None if unavailable."""
    if cam_a == cam_b:
        return 0.0
    key = frozenset({cam_a, cam_b})
    return _get_distances().get(key)


def are_cameras_same_set(cam_a: str, cam_b: str) -> bool:
    """Check if two cameras belong to the same camera-set (shared FOV)."""
    for cameras in CAMERA_SETS.values():
        if cam_a in cameras and cam_b in cameras:
            return True
    return False


def get_camera_site(cam_id: str) -> Optional[str]:
    """Get the site a camera belongs to."""
    return CAMERA_SITE.get(cam_id)


def get_proximity_tier(cam_a: str, cam_b: str) -> str:
    """
    Classify the spatial relationship between two cameras.

    Returns one of:
      "same_area"     — Overlapping FOV / same camera-set
      "adjacent"      — Nearby, non-overlapping (ideal for temporal Q)
      "same_site"     — Same site, but far apart
      "different_site" — Different sites
    """
    if cam_a == cam_b:
        return "same_area"

    site_a = CAMERA_SITE.get(cam_a)
    site_b = CAMERA_SITE.get(cam_b)

    # Different sites → never pair
    if site_a and site_b and site_a != site_b:
        return "different_site"

    # Same camera-set → same area (overlapping, use for dedup not questions)
    if are_cameras_same_set(cam_a, cam_b):
        return "same_area"

    # Check indoor adjacency map
    if cam_a in INDOOR_ADJACENT_OUTDOOR and cam_b in INDOOR_ADJACENT_OUTDOOR[cam_a]:
        return "adjacent"
    if cam_b in INDOOR_ADJACENT_OUTDOOR and cam_a in INDOOR_ADJACENT_OUTDOOR[cam_b]:
        return "adjacent"

    # Use KRTD distance if available
    dist = get_camera_distance(cam_a, cam_b)
    if dist is not None:
        if dist < 15.0:
            return "same_area"
        elif dist < 80.0:
            return "adjacent"
        else:
            return "same_site"

    # Fallback: both at same site but no distance info
    if site_a == site_b:
        # Indoor+outdoor at same site = adjacent (person can walk between)
        if (cam_a in INDOOR_CAMERAS_FULL) != (cam_b in INDOOR_CAMERAS_FULL):
            return "adjacent"
        return "same_site"

    return "different_site"


def score_camera_pair_for_temporal(cam_a: str, cam_b: str) -> float:
    """
    Score how good a camera pair is for temporal cross-camera questions.

    Higher = better. Ideal pairs are adjacent cameras at the same site
    (close enough that events are related, far enough that they require
    multi-camera reasoning).

    Returns:
      3.0 — adjacent cameras (best: events are likely related)
      2.0 — same_site but far (ok: events could be part of same scene)
      0.5 — same_area / overlapping (bad: same event seen twice)
      0.0 — different_site (invalid)
    """
    tier = get_proximity_tier(cam_a, cam_b)
    if tier == "adjacent":
        return 3.0
    elif tier == "same_site":
        return 2.0
    elif tier == "same_area":
        return 0.5
    else:
        return 0.0

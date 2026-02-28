"""
Camera overlap detection — identifies camera pairs with significant FOV overlap.

Used by dedup logic to avoid double-counting events seen by overlapping cameras.
Combines:
  1. KRTD-derived camera positions + viewing directions (bus, hospital, school)
  2. Manual/hardcoded overlap for sites without KRTD (admin)

An overlapping pair means the same physical event could appear in both cameras,
so cross-camera dedup should be MORE aggressive (wider time window, no 3D req).
"""

import numpy as np
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Set, Tuple

from .krtd import KRTD_DIR, INDOOR_CAMERAS

# ============================================================================
# Hardcoded overlaps for sites without KRTD calibration
# ============================================================================

# Admin site: G326 and G329 point at the same building entrance.
# No KRTD models exist for admin. Confirmed via visual inspection.
_MANUAL_OVERLAPS: Dict[str, Set[FrozenSet[str]]] = {
    "admin": {frozenset({"G326", "G329"})},
}

# ============================================================================
# KRTD-based overlap computation
# ============================================================================

# Thresholds for automatic overlap detection
_CLOSE_DIST_M = 15.0       # cameras within 15m are "close"
_MEDIUM_DIST_M = 50.0      # cameras within 50m might overlap
_HIGH_COS_SIM = 0.5        # cosine similarity of viewing dirs for "similar view"
_MEDIUM_COS_SIM = 0.3      # lower threshold for medium-distance cameras


def _compute_krtd_overlaps() -> Set[FrozenSet[str]]:
    """Compute overlapping camera pairs from KRTD calibration files.

    Two cameras are considered overlapping if:
      - Within 15m of each other (any viewing direction), OR
      - Within 50m AND viewing directions have cosine similarity > 0.3, OR
      - Within 30m AND viewing directions have cosine similarity > 0.5
    """
    overlaps: Set[FrozenSet[str]] = set()

    # Parse all KRTD files
    cameras: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # cam_id -> (position, view_dir)
    for f in KRTD_DIR.glob("*.krtd"):
        cam_id = f.stem.split(".")[-1]
        if cam_id in INDOOR_CAMERAS:
            continue
        try:
            lines = [l.strip() for l in open(f) if l.strip()]
            K = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
            R = np.array([[float(x) for x in lines[i].split()] for i in range(3, 6)])
            T = np.array([float(x) for x in lines[6].split()])
            C = -R.T @ T
            view_dir = R.T @ np.array([0.0, 0.0, 1.0])
            view_dir = view_dir / np.linalg.norm(view_dir)
            cameras[cam_id] = (C, view_dir)
        except Exception:
            continue

    cam_list = sorted(cameras.keys())
    for i in range(len(cam_list)):
        for j in range(i + 1, len(cam_list)):
            ca, cb = cam_list[i], cam_list[j]
            pos_a, dir_a = cameras[ca]
            pos_b, dir_b = cameras[cb]
            dist = float(np.linalg.norm(pos_a - pos_b))
            cos_sim = float(np.dot(dir_a, dir_b))

            if dist < _CLOSE_DIST_M:
                overlaps.add(frozenset({ca, cb}))
            elif dist < 30.0 and cos_sim > _HIGH_COS_SIM:
                overlaps.add(frozenset({ca, cb}))
            elif dist < _MEDIUM_DIST_M and cos_sim > _MEDIUM_COS_SIM:
                overlaps.add(frozenset({ca, cb}))

    return overlaps


# ============================================================================
# Cached overlap map
# ============================================================================

_OVERLAP_CACHE: Optional[Set[FrozenSet[str]]] = None


def _get_all_overlaps() -> Set[FrozenSet[str]]:
    """Get all overlapping camera pairs (cached)."""
    global _OVERLAP_CACHE
    if _OVERLAP_CACHE is None:
        _OVERLAP_CACHE = _compute_krtd_overlaps()
        for pairs in _MANUAL_OVERLAPS.values():
            _OVERLAP_CACHE |= pairs
    return _OVERLAP_CACHE


def cameras_overlap(cam_a: str, cam_b: str) -> bool:
    """Check if two cameras have significant FOV overlap.

    When True, events on these cameras with similar timing are likely
    the same real-world event and should be deduped aggressively.
    """
    if cam_a == cam_b:
        return True
    return frozenset({cam_a, cam_b}) in _get_all_overlaps()


def get_overlapping_cameras(camera_id: str) -> Set[str]:
    """Get all cameras that overlap with the given camera."""
    result = set()
    for pair in _get_all_overlaps():
        if camera_id in pair:
            result |= pair
    result.discard(camera_id)
    return result


def get_overlap_pairs_for_cameras(camera_ids: list) -> list:
    """Get all overlap pairs among a set of cameras.

    Returns list of (cam_a, cam_b) tuples.
    """
    cam_set = set(camera_ids)
    result = []
    for pair in _get_all_overlaps():
        if pair <= cam_set:
            a, b = sorted(pair)
            result.append((a, b))
    return sorted(result)

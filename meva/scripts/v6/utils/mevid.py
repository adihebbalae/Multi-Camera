"""
V6 utils/mevid.py — MEVID tracklet parsing and cross-camera person mapping.

Parses MEVID annotation data to find:
1. All (person_id, camera_id) pairs in the dataset
2. Cross-camera person links (same person on 2+ cameras)
3. MEVID-supported slots (slots where cross-camera persons exist)
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

MEVID_DATA_DIR = Path("/nas/mars/dataset/MEVA/mevid_data/mevid-v1-annotation-data")
MEVID_URLS = Path("/nas/mars/dataset/MEVA/mevid_data/mevid-v1-video-URLS.txt")

# Regex for MEVID image filename: {PersonID}O{OutfitID}C{CameraID}T{TrackletID}F{Frame}.jpg
MEVID_NAME_RE = re.compile(r'^(\d{4})O(\d{3})C(\d+)T(\d{3})F(\d{5})\.jpg$')


def parse_mevid_person_cameras() -> Dict[int, Set[str]]:
    """
    Parse MEVID train_name.txt and test_name.txt to extract all (person_id, camera_id) pairs.
    
    Returns:
        {person_id: {camera_id, ...}, ...}
        Camera IDs are in MEVA format: "G424" (not "C424").
    """
    person_cameras: Dict[int, Set[str]] = defaultdict(set)
    
    for fname in ["train_name.txt", "test_name.txt"]:
        fpath = MEVID_DATA_DIR / fname
        if not fpath.exists():
            continue
        # Use set to avoid re-parsing the same (person, camera) pair
        seen_keys = set()
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = MEVID_NAME_RE.match(line)
                if not m:
                    continue
                person_id = int(m.group(1))
                camera_id = f"G{m.group(3)}"
                key = (person_id, camera_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    person_cameras[person_id].add(camera_id)
    
    return dict(person_cameras)


def parse_mevid_clips() -> List[Dict[str, str]]:
    """
    Parse MEVID video URLs to get the list of MEVA clips used by MEVID.
    
    Returns list of dicts with keys: clip_name, date, start_time, end_time, site, camera_id
    """
    if not MEVID_URLS.exists():
        return []
    
    clips = []
    clip_re = re.compile(
        r'(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\w+)\.(G\d+)'
    )
    
    with open(MEVID_URLS) as f:
        for line in f:
            line = line.strip()
            m = clip_re.search(line)
            if m:
                clips.append({
                    "clip_name": f"{m.group(1)}.{m.group(2)}.{m.group(3)}.{m.group(4)}.{m.group(5)}",
                    "date": m.group(1),
                    "start_time": m.group(2),
                    "end_time": m.group(3),
                    "site": m.group(4),
                    "camera_id": m.group(5),
                })
    return clips


def find_mevid_persons_for_slot(slot: str, slot_cameras: List[str]) -> Dict[int, Set[str]]:
    """
    Find MEVID persons who appear on 2+ cameras within this slot's camera set.
    
    This is an approximation: MEVID tells us person X appears on camera Y globally,
    but we can't confirm the specific time slot without extracted tracklet images.
    For March dates within a single session day, this mapping is reliable.
    
    Args:
        slot: Slot name e.g. "2018-03-11.11-25-00.school"
        slot_cameras: List of camera IDs in this slot
    
    Returns:
        {person_id: {camera_ids in this slot}, ...}
        Only persons appearing on 2+ slot cameras are included.
    """
    all_person_cameras = parse_mevid_person_cameras()
    slot_camera_set = set(slot_cameras)
    
    result = {}
    for person_id, cameras in all_person_cameras.items():
        overlap = cameras & slot_camera_set
        if len(overlap) >= 2:
            result[person_id] = overlap
    
    return result


def find_mevid_supported_slots(slot_index: Dict) -> List[Dict]:
    """
    Find all slots in the index that have MEVID cross-camera person coverage.
    
    Returns list of {slot, cameras, mevid_persons_count, mevid_camera_overlap}.
    """
    all_person_cameras = parse_mevid_person_cameras()
    
    supported = []
    for slot, info in slot_index.items():
        # Only March dates are on disk
        if not slot.startswith("2018-03"):
            continue
        
        slot_cameras = set(info.get("cameras", []))
        cross_persons = 0
        for person_id, cameras in all_person_cameras.items():
            if len(cameras & slot_cameras) >= 2:
                cross_persons += 1
        
        if cross_persons > 0:
            supported.append({
                "slot": slot,
                "cameras": info["cameras"],
                "mevid_persons_count": cross_persons,
                "mevid_camera_overlap": len(slot_cameras),
                "sources": list(info.get("sources", {}).keys()),
            })
    
    supported.sort(key=lambda x: x["mevid_persons_count"], reverse=True)
    return supported


def get_mevid_stats() -> Dict:
    """Get summary statistics of MEVID data."""
    person_cameras = parse_mevid_person_cameras()
    clips = parse_mevid_clips()
    
    cross_camera = sum(1 for cams in person_cameras.values() if len(cams) >= 2)
    all_cameras = set()
    for cams in person_cameras.values():
        all_cameras.update(cams)
    
    march_clips = [c for c in clips if c["date"].startswith("2018-03")]
    
    return {
        "total_persons": len(person_cameras),
        "cross_camera_persons": cross_camera,
        "total_cameras": len(all_cameras),
        "cameras": sorted(all_cameras),
        "total_clips": len(clips),
        "march_clips": len(march_clips),
    }

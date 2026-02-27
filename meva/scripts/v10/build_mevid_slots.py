"""
build_mevid_slots.py — Rebuild mevid_supported_slots.json from authoritative sources.

DATA SOURCES:
  - mevid-v1-annotation-data/{train,test}_name.txt  → person-camera associations
  - mevid-v1-video-URLS.txt                          → which MEVA clips are in MEVID
  - data/slot_index.json                             → all cameras per canonical slot

LOGIC:
  For each March MEVID clip, determine the canonical slot (date.HH-MM.site).
  For each slot, collect:
    - mevid_cameras: cameras that have MEVID clips in this 5-min window
    - mevid_persons: persons from annotation files with 2+ cameras in mevid_cameras
  Write to data/mevid_supported_slots.json.

USAGE:
  cd /path/to/meva
  python3 -m scripts.v10.build_mevid_slots
  python3 -m scripts.v10.build_mevid_slots --include-may   # also include May dates
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Paths
_SCRIPT_DIR = Path(__file__).parent
_REPO_DATA = _SCRIPT_DIR.parent.parent / "data"

MEVID_DATA_DIR = Path(os.environ.get("MEVA_MEVID_DATA_DIR") or "/nas/mars/dataset/MEVA/mevid_data/mevid-v1-annotation-data")
MEVID_URLS = Path(os.environ.get("MEVA_MEVID_URLS") or "/nas/mars/dataset/MEVA/mevid_data/mevid-v1-video-URLS.txt")

# Regexes
_RE_NAME = re.compile(r'^(\d{4})O\d{3}C(\d+)T')
_RE_CLIP = re.compile(
    r'(\d{4}-\d{2}-\d{2})\.(\d{2})-(\d{2})-\d{2}\.\d{2}-\d{2}-\d{2}\.(\w+)\.(G\d+)'
)


def parse_person_cameras() -> dict[str, set[str]]:
    """
    Parse train/test name files → {person_id_str: {camera_ids}}.
    Camera IDs are in MEVA format: "G424" (not "C424").
    Person IDs are zero-padded strings: "0041".
    """
    person_cams: dict[str, set[str]] = defaultdict(set)
    for fname in ("train_name.txt", "test_name.txt"):
        fpath = MEVID_DATA_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found", file=sys.stderr)
            continue
        seen = set()
        with open(fpath) as f:
            for line in f:
                m = _RE_NAME.match(line.strip())
                if not m:
                    continue
                pid = m.group(1)             # e.g. "0041"
                cam = f"G{m.group(2)}"       # e.g. "G424"
                key = (pid, cam)
                if key not in seen:
                    seen.add(key)
                    person_cams[pid].add(cam)
    return dict(person_cams)


def parse_slot_cameras(include_may: bool = False) -> dict[str, set[str]]:
    """
    Parse video URLs → {slot: {cameras with MEVID clips}}.
    slot format: "2018-03-11.11-25.school"
    """
    if not MEVID_URLS.exists():
        print(f"  ERROR: {MEVID_URLS} not found", file=sys.stderr)
        return {}

    slot_cams: dict[str, set[str]] = defaultdict(set)
    with open(MEVID_URLS) as f:
        for line in f:
            m = _RE_CLIP.search(line.strip())
            if not m:
                continue
            date, hh, mm, site, cam = m.groups()
            if not include_may and date.startswith("2018-05"):
                continue
            slot = f"{date}.{hh}-{mm}.{site}"
            slot_cams[slot].add(cam)
    return dict(slot_cams)


def load_slot_index() -> dict:
    path = _REPO_DATA / "slot_index.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def build(include_may: bool = False, verbose: bool = True) -> dict:
    if verbose:
        print("=== Building mevid_supported_slots.json ===")

    # 1. Person → cameras from annotation files
    person_cams = parse_person_cameras()
    if verbose:
        multi = {p: c for p, c in person_cams.items() if len(c) >= 2}
        print(f"  Persons in annotation files: {len(person_cams)}")
        print(f"  Persons with 2+ cameras (global): {len(multi)}")

    # 2. Slot → MEVID cameras from video URLs
    slot_cameras = parse_slot_cameras(include_may=include_may)
    if verbose:
        dates = set(s.split(".")[0] for s in slot_cameras)
        print(f"  Slots with MEVID clips: {len(slot_cameras)} across {len(dates)} dates")

    # 3. slot_index.json for all_cameras
    slot_index = load_slot_index()

    # 4. For each slot, find cross-camera persons
    result_slots: dict[str, dict] = {}
    for slot in sorted(slot_cameras):
        mevid_cams = sorted(slot_cameras[slot])
        mevid_cam_set = set(mevid_cams)

        cross_persons = sorted(
            pid for pid, cams in person_cams.items()
            if len(cams & mevid_cam_set) >= 2
        )

        if not cross_persons:
            continue

        # Parse slot components
        parts = slot.split(".")
        date = parts[0]
        time_part = parts[1] if len(parts) > 1 else ""
        site = parts[2] if len(parts) > 2 else ""

        # Look up all_cameras from slot_index (best effort)
        all_cams_count = len(slot_cameras[slot])
        if slot in slot_index:
            all_cams_count = len(slot_index[slot].get("cameras", []))

        result_slots[slot] = {
            "date": date,
            "time": f"{time_part}-00",
            "site": site,
            "all_cameras": all_cams_count,
            "mevid_cameras": mevid_cams,
            "mevid_persons": cross_persons,
        }

    # 5. Summary
    all_persons = set()
    for v in result_slots.values():
        all_persons.update(v["mevid_persons"])

    if verbose:
        print(f"\n  Result: {len(result_slots)} slots with cross-camera MEVID persons")
        print(f"  Unique persons: {len(all_persons)}")
        print(f"  Person IDs: {sorted(all_persons)}")

    return {"slots": result_slots}


def main():
    parser = argparse.ArgumentParser(description="Rebuild mevid_supported_slots.json")
    parser.add_argument("--include-may", action="store_true",
                        help="Include May 2018 MEVID data (not on disk, for future use)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print results without writing the file")
    args = parser.parse_args()

    data = build(include_may=args.include_may, verbose=True)

    out_path = _REPO_DATA / "mevid_supported_slots.json"
    if args.dry_run:
        print(f"\nDry run — would write to {out_path}")
    else:
        out_path.write_text(json.dumps(data, indent=2))
        print(f"\nWritten to {out_path}")
        print(f"({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

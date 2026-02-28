#!/usr/bin/env python3
"""
Re-extract entity descriptions for slots with missing cameras.

The original batch extraction missed cameras from kitware-meva-training/.
This script identifies affected slots and re-extracts them with the current
find_slot_files() which searches BOTH kitware/ and kitware-meva-training/.

Usage:
    python3 scripts/v10/reextract_missing_cameras.py --dry-run   # Show affected slots
    python3 scripts/v10/reextract_missing_cameras.py -v          # Re-extract all
    python3 scripts/v10/reextract_missing_cameras.py --slot "2018-03-09.10-40.bus" -v  # Single slot
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Use venv python for subprocess calls — prefer activated venv, else system python3
VENV_PYTHON = shutil.which("python3") or "python3"

KITWARE_BASE = Path("/nas/mars/dataset/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware")
KITWARE_TRAINING = Path("/nas/mars/dataset/MEVA/meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware-meva-training")
DESC_DIR = Path("/nas/mars/dataset/MEVA/entity_descriptions")
SLOT_INDEX = Path(__file__).resolve().parent.parent.parent / "data" / "slot_index.json"


def find_geom_cameras(slot: str) -> set:
    """Find all cameras with geom.yml files for a slot."""
    parts = slot.split(".")
    date, time_part, site = parts[0], parts[1], parts[2]
    hour = time_part.split("-")[0]
    prefix = f"{date}.{time_part}"

    cameras = set()
    for base in [KITWARE_BASE, KITWARE_TRAINING]:
        d = base / date / hour
        if d.is_dir():
            for gf in d.glob(f"{prefix}*.{site}.*.geom.yml"):
                m = re.search(rf'\.{site}\.(G\d+)\.geom\.yml$', gf.name)
                if m:
                    cameras.add(m.group(1))
    return cameras


def find_desc_cameras(slot: str) -> set:
    """Get cameras already in the entity description file."""
    desc_file = DESC_DIR / f"{slot}.json"
    if not desc_file.exists():
        return set()
    with open(desc_file) as f:
        data = json.load(f)
    return set(data.get("cameras", {}).keys())


def find_affected_slots() -> list:
    """Find all slots with cameras missing from entity descriptions."""
    with open(SLOT_INDEX) as f:
        slot_index = json.load(f)

    affected = []
    for slot in sorted(slot_index.keys()):
        geom_cams = find_geom_cameras(slot)
        desc_cams = find_desc_cameras(slot)
        missing = geom_cams - desc_cams

        if missing:
            # Also check slots with no desc file at all but geom exists
            affected.append({
                "slot": slot,
                "missing_cameras": sorted(missing),
                "existing_cameras": sorted(desc_cams),
                "all_geom_cameras": sorted(geom_cams),
            })

    # Also find slots with no desc file but geom cameras exist
    for slot in sorted(slot_index.keys()):
        desc_file = DESC_DIR / f"{slot}.json"
        if not desc_file.exists():
            geom_cams = find_geom_cameras(slot)
            if geom_cams:
                affected.append({
                    "slot": slot,
                    "missing_cameras": sorted(geom_cams),
                    "existing_cameras": [],
                    "all_geom_cameras": sorted(geom_cams),
                })

    # Deduplicate
    seen = set()
    deduped = []
    for a in affected:
        if a["slot"] not in seen:
            seen.add(a["slot"])
            deduped.append(a)

    return deduped


def reextract_slot(slot: str, verbose: bool = False) -> dict:
    """Re-extract entity descriptions for a slot (overwrites existing file)."""
    import subprocess

    output_path = DESC_DIR / f"{slot}.json"
    cmd = [
        VENV_PYTHON,
        "-m", "scripts.v10.extract_entity_descriptions",
        "--slot", slot,
        "--output", str(output_path),
        "--method", "segformer",
    ]
    if verbose:
        cmd.append("-v")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )

        if result.returncode != 0:
            return {
                "success": False,
                "entities": 0,
                "error": f"Exit {result.returncode}: {result.stderr[:300]}",
            }

        if output_path.exists():
            with open(output_path) as f:
                data = json.load(f)
            return {
                "success": True,
                "entities": len(data.get("actors", {})),
                "cameras": list(data.get("cameras", {}).keys()),
            }
        return {"success": False, "entities": 0, "error": "No output file"}

    except subprocess.TimeoutExpired:
        return {"success": False, "entities": 0, "error": "Timeout >10min"}
    except Exception as e:
        return {"success": False, "entities": 0, "error": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser(description="Re-extract slots with missing cameras")
    parser.add_argument("--dry-run", action="store_true", help="Show affected slots without processing")
    parser.add_argument("--slot", type=str, help="Process a single slot")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.slot:
        # Single slot mode
        geom_cams = find_geom_cameras(args.slot)
        desc_cams = find_desc_cameras(args.slot)
        missing = geom_cams - desc_cams
        print(f"Slot: {args.slot}")
        print(f"  Geom cameras: {sorted(geom_cams)}")
        print(f"  Desc cameras: {sorted(desc_cams)}")
        print(f"  Missing: {sorted(missing)}")

        if not missing and desc_cams == geom_cams:
            print("  All cameras already extracted!")
            return

        if args.dry_run:
            return

        print(f"\n  Re-extracting...")
        result = reextract_slot(args.slot, verbose=args.verbose)
        print(f"  Result: {result}")
        return

    # Batch mode
    affected = find_affected_slots()
    print(f"\nFound {len(affected)} slots with missing camera descriptions")
    total_missing = sum(len(a["missing_cameras"]) for a in affected)
    print(f"Total missing cameras: {total_missing}")

    if args.dry_run:
        print("\n--- Affected Slots ---")
        for a in affected:
            print(f"  {a['slot']}: missing {a['missing_cameras']} (has {a['existing_cameras']})")
        return

    # Process all affected slots
    print(f"\nProcessing {len(affected)} slots...")
    success = 0
    failed = 0
    total_new_actors = 0

    for i, a in enumerate(affected):
        slot = a["slot"]
        print(f"\n[{i+1}/{len(affected)}] {slot} (missing: {a['missing_cameras']})")

        # Get actor count before
        desc_file = DESC_DIR / f"{slot}.json"
        before_actors = 0
        if desc_file.exists():
            with open(desc_file) as f:
                before_actors = len(json.load(f).get("actors", {}))

        result = reextract_slot(slot, verbose=args.verbose)

        if result["success"]:
            after_actors = result["entities"]
            delta = after_actors - before_actors
            total_new_actors += max(0, delta)
            print(f"  OK: {before_actors} → {after_actors} actors (+{delta}), cameras: {result.get('cameras', [])}")
            success += 1
        else:
            print(f"  FAILED: {result['error']}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"DONE: {success} succeeded, {failed} failed")
    print(f"Total new actors: +{total_new_actors}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

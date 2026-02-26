#!/usr/bin/env python3
"""
Build Geom Slot Index — Scan all canonical slots for geom file availability.

This creates a filtered slot index containing only slots that have at least
one camera with geom.yml bounding box data available.

Checks both:
  - /nas/mars/.../kitware/
  - /nas/mars/.../kitware-meva-training/

Output: data/geom_slot_index.json
Time: ~15-20 minutes for 929 slots
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

SLOT_INDEX_PATH = Path("/home/ah66742/data_back/slot_index.json")
OUTPUT_PATH = Path("/home/ah66742/data_back/geom_slot_index.json")
EXTRACTION_SCRIPT = Path("/home/ah66742/scripts/final/extract_entity_descriptions.py")


def main():
    print("="*60, flush=True)
    print("Building Geom Slot Index", flush=True)
    print("="*60, flush=True)
    print(f"Input: {SLOT_INDEX_PATH}", flush=True)
    print(f"Output: {OUTPUT_PATH}", flush=True)
    print(flush=True)
    
    # Load all slots
    if not SLOT_INDEX_PATH.exists():
        print(f"ERROR: Slot index not found: {SLOT_INDEX_PATH}", flush=True)
        sys.exit(1)
    
    with open(SLOT_INDEX_PATH) as f:
        all_slots = json.load(f)
    
    total_slots = len(all_slots)
    print(f"Scanning {total_slots} canonical slots...", flush=True)
    print("This will take ~15-20 minutes.\n", flush=True)
    
    geom_slots = {}
    stats = {
        "total_cameras": 0,
        "total_usable_actors": 0,
        "kitware_count": 0,
        "training_count": 0,
    }
    
    for i, (slot, slot_data) in enumerate(sorted(all_slots.items()), 1):
        # Run dry-run to check for geom files
        try:
            result = subprocess.run(
                ["python3", str(EXTRACTION_SCRIPT), "--slot", slot, "--dry-run"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            
            # Parse output
            lines = result.stdout.split("\n")
            cameras_line = [l for l in lines if "Cameras with geom:" in l]
            
            if not cameras_line:
                continue
            
            n_cameras = int(cameras_line[0].split(":")[1].strip())
            
            if n_cameras > 0:
                # Extract usable actor count
                usable_actors = 0
                for line in lines:
                    if " usable," in line:
                        parts = line.split(",")
                        for p in parts:
                            if " usable" in p:
                                usable_actors += int(p.split()[0])
                
                geom_slots[slot] = {
                    **slot_data,
                    "geom_cameras": n_cameras,
                    "geom_usable_actors": usable_actors,
                }
                
                stats["total_cameras"] += n_cameras
                stats["total_usable_actors"] += usable_actors
        
        except subprocess.TimeoutExpired:
            print(f"  [{i:4d}/{total_slots}] {slot}: TIMEOUT (skipped)", flush=True)
            continue
        except Exception as e:
            print(f"  [{i:4d}/{total_slots}] {slot}: ERROR {e}", flush=True)
            continue
        
        # Progress every 50 slots
        if i % 50 == 0:
            coverage = len(geom_slots) * 100 / i
            print(f"  [{i:4d}/{total_slots}] Progress: {len(geom_slots)} slots with geom ({coverage:.1f}%), "
                  f"{stats['total_usable_actors']:,} actors", flush=True)
    
    # Final stats
    print(flush=True)
    print("="*60, flush=True)
    print("SCAN COMPLETE", flush=True)
    print("="*60, flush=True)
    print(f"Total canonical slots: {total_slots}", flush=True)
    print(f"Slots with geom files: {len(geom_slots)} ({len(geom_slots)*100/total_slots:.1f}%)", flush=True)
    print(f"Total cameras with geom: {stats['total_cameras']}", flush=True)
    print(f"Total usable actors: {stats['total_usable_actors']:,}", flush=True)
    print(f"Avg actors/slot: {stats['total_usable_actors']/max(len(geom_slots),1):.0f}", flush=True)
    print(flush=True)
    
    # Save
    output_data = {
        "slots": geom_slots,
        "stats": {
            "total_canonical_slots": total_slots,
            "slots_with_geom": len(geom_slots),
            "coverage_percent": round(len(geom_slots) * 100 / total_slots, 1),
            "total_cameras": stats["total_cameras"],
            "total_usable_actors": stats["total_usable_actors"],
            "avg_actors_per_slot": round(stats["total_usable_actors"] / max(len(geom_slots), 1)),
        }
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved: {OUTPUT_PATH}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()

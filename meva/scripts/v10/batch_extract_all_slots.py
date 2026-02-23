#!/usr/bin/env python3
"""
Batch Entity Description Extractor — Process all canonical slots overnight.

Extracts visual descriptions (YOLO + HSV color) from bounding boxes for ALL
annotated entities across the full MEVA dataset.

This solves the low-visual-description problem:
- Before: ~7% visual coverage (MEVID only)
- After: ~95% visual coverage (all entities with geom bboxes)

Usage:
    # Dry-run: show what will be processed
    python3 scripts/final/batch_extract_all_slots.py --dry-run

    # Full extraction (overnight, ~20 hours for 390 slots)
    python3 scripts/final/batch_extract_all_slots.py -v

    # Resume from interruption
    python3 scripts/final/batch_extract_all_slots.py -v --resume

Cost: $0 (local YOLO, no API calls)
Time: ~3-4 min per slot × 390 slots = ~20 hours

Output: /home/ah66742/data/entity_descriptions/{canonical_slot}.json
Progress: /home/ah66742/output/extraction_logs/batch_progress.json
Logs: /home/ah66742/output/extraction_logs/batch_extraction_TIMESTAMP.log
"""

import argparse
import json
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional

# ============================================================================
# Paths
# ============================================================================

SLOT_INDEX_PATH = Path("/home/ah66742/data/geom_slot_index.json")
OUTPUT_DIR = Path("/home/ah66742/data/entity_descriptions")
LOG_DIR = Path("/home/ah66742/output/extraction_logs")
PROGRESS_FILE = LOG_DIR / "batch_progress.json"
EXTRACTION_SCRIPT = Path("/home/ah66742/scripts/final/extract_entity_descriptions.py")

# ============================================================================
# Progress Tracking
# ============================================================================

def load_progress() -> Dict:
    """Load progress state from disk (for resume capability)."""
    if not PROGRESS_FILE.exists():
        return {
            "started_at": None,
            "last_updated": None,
            "completed_slots": [],
            "failed_slots": [],
            "skipped_slots": [],
            "total_slots": 0,
            "total_entities_extracted": 0,
        }
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def save_progress(progress: Dict):
    """Save progress state to disk."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def init_progress(total_slots: int) -> Dict:
    """Initialize fresh progress state."""
    return {
        "started_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "completed_slots": [],
        "failed_slots": [],
        "skipped_slots": [],
        "total_slots": total_slots,
        "total_entities_extracted": 0,
    }


# ============================================================================
# Slot Processing
# ============================================================================

def find_raw_slots_for_canonical(canonical_slot: str, slot_index: Dict) -> List[str]:
    """
    Resolve canonical slot to its raw variants.
    
    Canonical: 2018-03-11.16-20-00.school
    Raw variants: [2018-03-11.16-20-00.school, 2018-03-11.16-20-01.school, ...]
    """
    if canonical_slot not in slot_index:
        return []
    
    slot_data = slot_index[canonical_slot]
    raw_slots = slot_data.get("raw_slot_variants", [canonical_slot])
    return raw_slots


def run_extraction(slot: str, verbose: bool = False, dry_run: bool = False) -> Dict:
    """
    Run extract_entity_descriptions.py on a single slot.
    
    Returns: {"success": bool, "entities": int, "error": str or None}
    """
    output_path = OUTPUT_DIR / f"{slot}.json"
    
    # Check if already exists and has data
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            entity_count = len([k for k in data.keys() if "_actor_" in k])
            if entity_count > 0:
                return {
                    "success": True,
                    "entities": entity_count,
                    "error": None,
                    "skipped": True,
                }
        except Exception:
            pass  # corrupted file, re-extract
    
    if dry_run:
        return {"success": True, "entities": 0, "error": None, "dry_run": True}
    
    # Run extraction
    cmd = [
        "python3",
        str(EXTRACTION_SCRIPT),
        "--slot", slot,
        "--output", str(output_path),
    ]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per slot
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "entities": 0,
                "error": f"Exit code {result.returncode}: {result.stderr[:200]}",
            }
        
        # Parse output JSON to count entities
        if output_path.exists():
            with open(output_path) as f:
                data = json.load(f)
            entity_count = len(data.get("actors", {}))
            return {
                "success": True,
                "entities": entity_count,
                "error": None,
            }
        else:
            return {
                "success": False,
                "entities": 0,
                "error": "Output file not created",
            }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "entities": 0,
            "error": "Timeout (>10 min)",
        }
    except Exception as e:
        return {
            "success": False,
            "entities": 0,
            "error": str(e)[:200],
        }


# ============================================================================
# Batch Processing
# ============================================================================

def process_all_slots(dry_run: bool = False, verbose: bool = False,
                      resume: bool = False) -> Dict:
    """
    Process all canonical slots from slot_index.json.
    
    Returns summary dict with stats.
    """
    # Load geom slot index (only slots with extractable geom data)
    if not SLOT_INDEX_PATH.exists():
        print(f"ERROR: Geom slot index not found: {SLOT_INDEX_PATH}")
        sys.exit(1)
    
    with open(SLOT_INDEX_PATH) as f:
        geom_index = json.load(f)
    
    # Extract slots from nested structure
    slot_index = geom_index['slots']
    canonical_slots = sorted(slot_index.keys())
    
    print(f"\nLoaded {len(canonical_slots)} slots with geom data (filtered from {geom_index['stats']['total_canonical_slots']} total)")
    print(f"Coverage: {geom_index['stats']['coverage_percent']:.1f}% of canonical slots have extractable geom")
    print(f"Expected entities: ~{geom_index['stats']['total_usable_actors']:,} actors\n")
    
    # Load or init progress
    if resume:
        progress = load_progress()
        print(f"\nResuming from previous run:")
        print(f"  Completed: {len(progress['completed_slots'])}")
        print(f"  Failed: {len(progress['failed_slots'])}")
        print(f"  Skipped: {len(progress['skipped_slots'])}")
    else:
        progress = init_progress(len(canonical_slots))
    
    completed_set = set(progress["completed_slots"])
    failed_set = set(progress["failed_slots"])
    skipped_set = set(progress["skipped_slots"])
    
    # Setup logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"batch_extraction_{timestamp}.log"
    
    log_file = open(log_path, "w", buffering=1)  # line buffered
    
    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")
    
    # Process each canonical slot
    log(f"\n{'='*60}")
    log(f"Batch Entity Description Extraction")
    log(f"{'='*60}")
    log(f"Mode: {'DRY-RUN' if dry_run else 'FULL EXTRACTION'}")
    log(f"Canonical slots: {len(canonical_slots)}")
    log(f"Resume: {resume}")
    log(f"Log: {log_path}")
    log(f"Progress: {PROGRESS_FILE}")
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}\n")
    
    start_time = time.time()
    total_entities = progress.get("total_entities_extracted", 0)
    
    for i, canonical_slot in enumerate(canonical_slots, 1):
        # Skip if already completed
        if canonical_slot in completed_set:
            continue
        
        # Find raw variants for this canonical slot
        raw_slots = find_raw_slots_for_canonical(canonical_slot, slot_index)
        
        if not raw_slots:
            log(f"[{i:3d}/{len(canonical_slots)}] {canonical_slot}: NO RAW VARIANTS")
            skipped_set.add(canonical_slot)
            continue
        
        # Use first raw variant (they share same geom files)
        raw_slot = raw_slots[0]
        
        log(f"[{i:3d}/{len(canonical_slots)}] {canonical_slot} → {raw_slot}")
        
        result = run_extraction(raw_slot, verbose=verbose, dry_run=dry_run)
        
        if result.get("skipped"):
            log(f"  ✓ SKIPPED (already exists, {result['entities']} entities)")
            skipped_set.add(canonical_slot)
        elif result.get("dry_run"):
            log(f"  ✓ DRY-RUN OK")
        elif result["success"]:
            n = result["entities"]
            total_entities += n
            log(f"  ✓ SUCCESS ({n} entities)")
            completed_set.add(canonical_slot)
        else:
            log(f"  ✗ FAILED: {result['error']}")
            failed_set.add(canonical_slot)
        
        # Update progress every 5 slots
        if i % 5 == 0 or not result["success"]:
            progress["completed_slots"] = sorted(completed_set)
            progress["failed_slots"] = sorted(failed_set)
            progress["skipped_slots"] = sorted(skipped_set)
            progress["total_entities_extracted"] = total_entities
            save_progress(progress)
        
        # ETA calculation
        if i > 0 and not dry_run:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(canonical_slots) - i) * avg_time
            eta_hours = remaining / 3600
            log(f"  Progress: {i}/{len(canonical_slots)} ({i*100//len(canonical_slots)}%), "
                f"ETA: {eta_hours:.1f}h\n")
    
    # Final stats
    elapsed = time.time() - start_time
    
    log(f"\n{'='*60}")
    log(f"BATCH EXTRACTION COMPLETE")
    log(f"{'='*60}")
    log(f"Total time: {elapsed/3600:.1f} hours")
    log(f"Completed: {len(completed_set)}")
    log(f"Skipped (already exist): {len(skipped_set)}")
    log(f"Failed: {len(failed_set)}")
    log(f"Total entities: {total_entities}")
    log(f"Avg entities/slot: {total_entities/max(len(completed_set),1):.1f}")
    
    if failed_set:
        log(f"\nFailed slots ({len(failed_set)}):")
        for slot in sorted(failed_set):
            log(f"  - {slot}")
    
    log(f"\nOutput directory: {OUTPUT_DIR}")
    log(f"Log file: {log_path}")
    log(f"{'='*60}\n")
    
    log_file.close()
    
    # Final progress save
    progress["completed_slots"] = sorted(completed_set)
    progress["failed_slots"] = sorted(failed_set)
    progress["skipped_slots"] = sorted(skipped_set)
    progress["total_entities_extracted"] = total_entities
    save_progress(progress)
    
    return {
        "total_slots": len(canonical_slots),
        "completed": len(completed_set),
        "skipped": len(skipped_set),
        "failed": len(failed_set),
        "total_entities": total_entities,
        "elapsed_hours": elapsed / 3600,
        "log_path": str(log_path),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch extract entity descriptions for all slots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry-run", action="store_true",
        help="Show what will be processed without extracting")
    parser.add_argument("--resume", action="store_true",
        help="Resume from previous run (skip completed slots)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Verbose output from extraction script")
    
    args = parser.parse_args()
    
    summary = process_all_slots(
        dry_run=args.dry_run,
        verbose=args.verbose,
        resume=args.resume,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total slots: {summary['total_slots']}")
    print(f"Completed: {summary['completed']}")
    print(f"Skipped (exist): {summary['skipped']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total entities: {summary['total_entities']}")
    print(f"Time: {summary['elapsed_hours']:.1f} hours")
    print(f"Log: {summary['log_path']}")
    print("="*60)


if __name__ == "__main__":
    main()

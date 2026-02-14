#!/usr/bin/env python3
"""
V8 run_pipeline.py — Main orchestrator for V8 QA generation pipeline.

V8 CHANGES from V7:
- MEVID person descriptions injected into all question categories
- New categories: re_identification, scene_summary
- Attribute verification added to perception
- Only processes slots with MEVID support (filterable)
- 5 categories: temporal(2) + spatial(2) + perception(2) + re_id(2) + scene_summary(1) = ~9 Qs
- version: "v8"

Usage:
    python3 -m scripts.v8.run_pipeline --slot "2018-03-11.11-25-00.school" -v
    python3 scripts/v8/run_pipeline.py --slot "2018-03-11.11-25-00.school" -v
    python3 scripts/v8/run_pipeline.py --list-mevid-slots
"""

import json
import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Handle both direct execution and module execution
try:
    from .parse_annotations import parse_slot_events, find_clips_for_slot
    from .build_scene_graph import build_scene_graph
    from .entity_resolution import resolve_entities
    from .person_descriptions import (
        enrich_entities, is_mevid_supported, get_mevid_persons_for_slot,
        get_mevid_persons_with_cameras, load_person_database,
    )
    from .generate_temporal import generate_temporal_qa
    from .generate_spatial import generate_spatial_qa
    from .generate_perception import generate_perception_qa
    from .generate_reidentification import generate_reidentification_qa
    from .generate_scene_summary import generate_scene_summary_qa
    from .utils.mevid import find_mevid_persons_for_slot as mevid_persons_check
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.v8.parse_annotations import parse_slot_events, find_clips_for_slot
    from scripts.v8.build_scene_graph import build_scene_graph
    from scripts.v8.entity_resolution import resolve_entities
    from scripts.v8.person_descriptions import (
        enrich_entities, is_mevid_supported, get_mevid_persons_for_slot,
        get_mevid_persons_with_cameras, load_person_database,
    )
    from scripts.v8.generate_temporal import generate_temporal_qa
    from scripts.v8.generate_spatial import generate_spatial_qa
    from scripts.v8.generate_perception import generate_perception_qa
    from scripts.v8.generate_reidentification import generate_reidentification_qa
    from scripts.v8.generate_scene_summary import generate_scene_summary_qa
    from scripts.v8.utils.mevid import find_mevid_persons_for_slot as mevid_persons_check


# ============================================================================
# Constants
# ============================================================================

OUTPUT_DIR = Path("/home/ah66742/data/qa_pairs")
RANDOM_SEED = 42

# V8: 5 categories, weighted question distribution
TARGET_TEMPORAL = 2
TARGET_SPATIAL = 2
TARGET_PERCEPTION = 2      # includes 1 attribute_verification if MEVID
TARGET_REIDENTIFICATION = 2
TARGET_SCENE_SUMMARY = 1
# Total target: ~9 questions per slot


# ============================================================================
# Deduplication
# ============================================================================

def is_duplicate_within_slot(new_q: dict, existing_qs: list) -> bool:
    """Prevent asking similar questions within the same slot."""
    for eq in existing_qs:
        if eq["category"] != new_q["category"]:
            continue
        v_new = new_q.get("verification", {})
        v_old = eq.get("verification", {})
        
        cat = new_q["category"]
        
        if cat == "temporal":
            ea_new = v_new.get("event_a", {}).get("activity")
            eb_new = v_new.get("event_b", {}).get("activity")
            ea_old = v_old.get("event_a", {}).get("activity")
            eb_old = v_old.get("event_b", {}).get("activity")
            if ea_new and eb_new and ea_old and eb_old:
                if ea_new == ea_old and eb_new == eb_old:
                    return True
        
        elif cat == "spatial":
            if (v_new.get("entity_a") and
                v_new.get("entity_a") == v_old.get("entity_a") and
                v_new.get("entity_b") == v_old.get("entity_b")):
                return True
        
        elif cat == "perception":
            if v_new.get("question_type") == v_old.get("question_type"):
                qt = v_new.get("question_type")
                if qt == "which_camera" and v_new.get("activity") == v_old.get("activity"):
                    return True
                elif qt == "activity_identification" and v_new.get("camera") == v_old.get("camera"):
                    return True
                elif qt == "multi_camera_confirmation" and v_new.get("activity") == v_old.get("activity"):
                    return True
                elif qt == "attribute_verification" and v_new.get("mevid_person_id") == v_old.get("mevid_person_id"):
                    return True
        
        elif cat == "re_identification":
            if (v_new.get("mevid_person_id") == v_old.get("mevid_person_id") and
                v_new.get("question_type") == v_old.get("question_type")):
                return True
        
        elif cat == "scene_summary":
            if v_new.get("question_type") == v_old.get("question_type"):
                return True
    
    return False


# ============================================================================
# Validation
# ============================================================================

def validate_temporal(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    ea = v.get("event_a", {})
    eb = v.get("event_b", {})
    
    if not ea or not eb:
        errors.append("Missing event_a or event_b in verification")
        return errors
    
    if ea.get("camera") == eb.get("camera"):
        errors.append(f"Same camera: {ea.get('camera')}")
    
    gap = v.get("gap_sec", 0)
    if gap < 3.0:
        errors.append(f"Gap too small: {gap}s (min 3s)")
    if gap > 20.0:
        errors.append(f"Gap too large: {gap}s (max 20s)")
    
    if ea.get("start_sec", 0) >= eb.get("start_sec", 0):
        errors.append("Event A does not precede Event B")
    
    return errors


def validate_spatial(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    d = v.get("distance_meters")
    proximity = v.get("proximity")
    
    if d is None:
        errors.append("Missing distance_meters")
        return errors
    
    if proximity == "near" and d > 5.0:
        errors.append(f"Near but distance={d}m (should be <=5m)")
    elif proximity == "moderate" and (d <= 5.0 or d > 15.0):
        errors.append(f"Moderate but distance={d}m (should be 5-15m)")
    elif proximity == "far" and d <= 15.0:
        errors.append(f"Far but distance={d}m (should be >15m)")
    
    return errors


def validate_perception(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    if not v.get("question_type"):
        errors.append("Missing question_type in verification")
    return errors


def validate_reidentification(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    if not v.get("mevid_person_id"):
        errors.append("Missing mevid_person_id")
    if not v.get("all_person_cameras"):
        errors.append("Missing all_person_cameras list")
    return errors


def validate_scene_summary(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    if not v.get("question_type"):
        errors.append("Missing question_type")
    return errors


def validate_all(qa_pairs: List[dict]) -> Dict[str, List[str]]:
    validators = {
        "temporal": validate_temporal,
        "spatial": validate_spatial,
        "perception": validate_perception,
        "re_identification": validate_reidentification,
        "scene_summary": validate_scene_summary,
    }
    issues = {}
    for q in qa_pairs:
        cat = q.get("category", "")
        if cat in validators:
            errors = validators[cat](q)
            if errors:
                issues[q["question_id"]] = errors
    return issues


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(slot: str, verbose: bool = False, 
                 seed: int = RANDOM_SEED,
                 require_mevid: bool = True) -> Dict[str, Any]:
    """
    Run the complete V8 QA generation pipeline on one slot.
    
    Steps:
        1. Parse annotations (Kitware YAML)
        2. Build entity-based scene graph
        3. Resolve cross-camera entities (MEVID + heuristic)
        4. Enrich entities with MEVID person descriptions (V8 NEW)
        5. Generate temporal questions (with MEVID descriptions)
        6. Generate spatial questions (with MEVID descriptions)
        7. Generate perception questions (+ attribute verification)
        8. Generate re-identification questions (V8 NEW)
        9. Generate scene summary questions (V8 NEW)
       10. Validate and output
    """
    t0 = time.time()
    rng = random.Random(seed)
    
    if verbose:
        print(f"{'=' * 60}")
        print(f"V8 Pipeline: {slot}")
        print(f"{'=' * 60}")
    
    # Check MEVID support
    mevid_supported = is_mevid_supported(slot)
    mevid_persons = get_mevid_persons_for_slot(slot)
    
    if require_mevid and not mevid_supported and not mevid_persons:
        if verbose:
            print(f"  WARNING: Slot {slot} has no MEVID support")
            print(f"  Running anyway (descriptions will fallback to activity-verb)")
    
    if verbose:
        print(f"  MEVID persons in slot: {len(mevid_persons)}")
    
    # Step 1: Parse annotations
    if verbose:
        print(f"\nStep 1: Parsing annotations...")
    events = parse_slot_events(slot, verbose=verbose)
    if not events:
        raise ValueError(f"No events found for slot {slot}")
    
    # Step 2: Build scene graph
    if verbose:
        print(f"\nStep 2: Building scene graph...")
    sg = build_scene_graph(slot, events, verbose=verbose)
    
    # Step 3: Entity resolution
    if verbose:
        print(f"\nStep 3: Resolving entities...")
    resolved = resolve_entities(sg, verbose=verbose)
    
    # Step 4 (V8 NEW): Enrich entities with visual descriptions
    if verbose:
        print(f"\nStep 4: Enriching entities with visual descriptions...")
    entity_descs, desc_counts = enrich_entities(sg, verbose=verbose)
    
    mevid_desc_count = desc_counts["mevid"]
    geom_desc_count = desc_counts["geom"]
    fallback_count = desc_counts["fallback"]
    
    if verbose:
        print(f"  {mevid_desc_count} MEVID + {geom_desc_count} geom-color + "
              f"{fallback_count} fallback / {len(entity_descs)} total")
    
    # Step 5-9: Generate QA pairs (all categories)
    if verbose:
        print(f"\nStep 5-9: Generating questions (5 categories)...")
    
    temporal_qa = generate_temporal_qa(sg, resolved, entity_descs, rng,
                                       count=TARGET_TEMPORAL, verbose=verbose)
    spatial_qa = generate_spatial_qa(sg, resolved, entity_descs, rng,
                                     count=TARGET_SPATIAL, verbose=verbose)
    perception_qa = generate_perception_qa(sg, resolved, entity_descs, rng,
                                           count=TARGET_PERCEPTION, verbose=verbose)
    reid_qa = generate_reidentification_qa(sg, resolved, entity_descs, rng,
                                            count=TARGET_REIDENTIFICATION, verbose=verbose)
    summary_qa = generate_scene_summary_qa(sg, resolved, entity_descs, rng,
                                            count=TARGET_SCENE_SUMMARY, verbose=verbose)
    
    all_qa = temporal_qa + spatial_qa + perception_qa + reid_qa + summary_qa
    
    # Deduplication
    unique_qa = []
    for q in all_qa:
        if not is_duplicate_within_slot(q, unique_qa):
            unique_qa.append(q)
    
    # Renumber question IDs sequentially
    for i, q in enumerate(unique_qa):
        q["question_id"] = f"v8_{q['category']}_{i+1:03d}"
    
    # Step 10: Validation
    if verbose:
        print(f"\nStep 10: Validating...")
    issues = validate_all(unique_qa)
    if verbose:
        if issues:
            print(f"  Validation issues:")
            for qid, errs in issues.items():
                for e in errs:
                    print(f"    {qid}: {e}")
        else:
            print(f"  All questions passed validation")
    
    # Build output
    cameras_in_slot = sorted(sg.cameras.keys())
    person_cameras = get_mevid_persons_with_cameras(slot)
    
    # Category counts
    cat_counts = {}
    for q in unique_qa:
        cat = q["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    # V8 stats
    desc_used = sum(
        1 for q in unique_qa
        if any("description" in str(v) for k, v in q.get("verification", {}).items()
               if isinstance(v, str) and "person" in v.lower())
    )
    attr_verification = sum(
        1 for q in unique_qa
        if q.get("subcategory") == "attribute_verification"
    )
    
    output = {
        "slot": slot,
        "version": "v8",
        "annotation_source": "kitware",
        "entity_resolution_source": "mevid+heuristic",
        "description_source": "mevid_yolo_gpt",
        "generator": "v8_pipeline",
        "seed": seed,
        "cameras": cameras_in_slot,
        "mevid_supported": mevid_supported or len(mevid_persons) > 0,
        "mevid_persons_in_slot": len(mevid_persons),
        "mevid_person_ids": sorted(mevid_persons),
        "mevid_person_cameras": {
            pid: sorted(cams) for pid, cams in person_cameras.items()
        },
        "total_events": len(events),
        "total_entities": len(sg.entities),
        "entities_with_mevid_descriptions": mevid_desc_count,
        "cross_camera_clusters": len(resolved.entity_clusters),
        "total_questions": len(unique_qa),
        "category_counts": cat_counts,
        "v8_stats": {
            "entities_with_mevid_descriptions": mevid_desc_count,
            "entities_with_geom_descriptions": geom_desc_count,
            "entities_with_fallback_descriptions": fallback_count,
            "attribute_verification_questions": attr_verification,
            "reid_questions": cat_counts.get("re_identification", 0),
            "scene_summary_questions": cat_counts.get("scene_summary", 0),
            "questions_with_debug_info": sum(1 for q in unique_qa if "debug_info" in q),
        },
        "validation_issues": len(issues),
        "generation_time_sec": round(time.time() - t0, 2),
        "qa_pairs": unique_qa,
    }
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"DONE: {len(unique_qa)} questions generated in {output['generation_time_sec']}s")
        for cat, cnt in sorted(cat_counts.items()):
            print(f"  {cat:25s}: {cnt}")
        print(f"  ---")
        print(f"  Cameras:    {cameras_in_slot}")
        print(f"  Events:     {len(events)}")
        print(f"  Entities:   {len(sg.entities)} ({mevid_desc_count} with MEVID descriptions)")
        print(f"  MEVID persons: {sorted(mevid_persons)}")
        print(f"  Cross-cam clusters: {len(resolved.entity_clusters)}")
        print(f"  Attribute verification: {attr_verification}")
        print(f"  Validation issues: {len(issues)}")
        print(f"{'=' * 60}")
    
    return output


def list_mevid_slots():
    """List all slots with MEVID person support."""
    db = load_person_database()
    persons = db.get("persons", {})
    
    slot_persons: Dict[str, List[str]] = {}
    for pid, pdata in persons.items():
        for slot_info in pdata.get("slots", []):
            slot = slot_info.get("slot", "")
            if slot:
                if slot not in slot_persons:
                    slot_persons[slot] = []
                slot_persons[slot].append(pid)
    
    print(f"Slots with MEVID person support: {len(slot_persons)}")
    print(f"{'Slot':40s} {'Persons':>8s} {'Person IDs'}")
    print("-" * 80)
    
    for slot in sorted(slot_persons.keys()):
        pids = sorted(slot_persons[slot])
        print(f"{slot:40s} {len(pids):8d}   {', '.join(pids[:5])}"
              + (f" +{len(pids)-5} more" if len(pids) > 5 else ""))


def main():
    parser = argparse.ArgumentParser(description="V8 QA Pipeline")
    parser.add_argument("--slot", help="Slot name (e.g., 2018-03-11.11-25-00.school)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--output", help="Output file path (default: data/qa_pairs/{slot}.v8.json)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file, just print")
    parser.add_argument("--list-mevid-slots", action="store_true",
                       help="List all slots with MEVID support")
    parser.add_argument("--no-require-mevid", action="store_true",
                       help="Process slot even without MEVID support")
    args = parser.parse_args()
    
    if args.list_mevid_slots:
        list_mevid_slots()
        return
    
    if not args.slot:
        parser.error("--slot is required (or use --list-mevid-slots)")
    
    output = run_pipeline(args.slot, verbose=args.verbose, seed=args.seed,
                          require_mevid=not args.no_require_mevid)
    
    # Save output
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.slot}.v8.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()

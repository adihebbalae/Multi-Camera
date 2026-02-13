#!/usr/bin/env python3
"""
V6 run_pipeline.py — Main orchestrator for the V6 QA generation pipeline.

Usage:
    python3 -m scripts.v6.run_pipeline --slot "2018-03-11.11-25-00.school" -v
    python3 scripts/v6/run_pipeline.py --slot "2018-03-11.11-25-00.school" -v

Output:
    data/qa_pairs/{slot}.v6.json
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
    from .generate_temporal import generate_temporal_qa
    from .generate_spatial import generate_spatial_qa
    from .generate_perception import generate_perception_qa
    from .generate_counting import generate_counting_qa
    from .utils.mevid import find_mevid_persons_for_slot, get_mevid_stats
except ImportError:
    # Direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.v6.parse_annotations import parse_slot_events, find_clips_for_slot
    from scripts.v6.build_scene_graph import build_scene_graph
    from scripts.v6.entity_resolution import resolve_entities
    from scripts.v6.generate_temporal import generate_temporal_qa
    from scripts.v6.generate_spatial import generate_spatial_qa
    from scripts.v6.generate_perception import generate_perception_qa
    from scripts.v6.generate_counting import generate_counting_qa
    from scripts.v6.utils.mevid import find_mevid_persons_for_slot, get_mevid_stats


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("/nas/neurosymbolic/multi-cam-dataset/meva/qa_pairs")
RANDOM_SEED = 42
TARGET_PER_CATEGORY = 3  # 3 temporal + 3 spatial + 3 perception + 3 counting = 12 total


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
            # Same activity pair = duplicate
            ea_new = v_new.get("event_a", {}).get("activity")
            eb_new = v_new.get("event_b", {}).get("activity")
            ea_old = v_old.get("event_a", {}).get("activity")
            eb_old = v_old.get("event_b", {}).get("activity")
            if ea_new and eb_new and ea_old and eb_old:
                if ea_new == ea_old and eb_new == eb_old:
                    return True
        
        elif cat == "spatial":
            # Same entity pair = duplicate
            if (v_new.get("entity_a") and
                v_new.get("entity_a") == v_old.get("entity_a") and
                v_new.get("entity_b") == v_old.get("entity_b")):
                return True
        
        elif cat == "perception":
            # Same question type + same key = duplicate
            if v_new.get("question_type") == v_old.get("question_type"):
                qt = v_new.get("question_type")
                if qt == "which_camera" and v_new.get("activity") == v_old.get("activity"):
                    return True
                elif qt == "activity_identification" and v_new.get("camera") == v_old.get("camera"):
                    return True
                elif qt == "multi_camera_confirmation" and v_new.get("activity") == v_old.get("activity"):
                    return True
        
        elif cat == "counting":
            # Same activity = duplicate
            if v_new.get("activity") == v_old.get("activity"):
                return True
    
    return False


# ============================================================================
# Validation
# ============================================================================

def validate_temporal(q: dict) -> List[str]:
    """Validate a temporal question."""
    errors = []
    v = q.get("verification", {})
    
    ea = v.get("event_a", {})
    eb = v.get("event_b", {})
    
    if not ea or not eb:
        errors.append("Missing event_a or event_b in verification")
        return errors
    
    # Must be cross-camera
    if ea.get("camera") == eb.get("camera"):
        errors.append(f"Same camera: {ea.get('camera')}")
    
    # Gap check
    gap = v.get("gap_sec", 0)
    if gap < 3.0:
        errors.append(f"Gap too small: {gap}s (min 3s)")
    if gap > 20.0:
        errors.append(f"Gap too large: {gap}s (max 20s)")
    
    # A should be before B
    if ea.get("start_sec", 0) >= eb.get("start_sec", 0):
        errors.append("Event A does not precede Event B")
    
    return errors


def validate_spatial(q: dict) -> List[str]:
    """Validate a spatial question."""
    errors = []
    v = q.get("verification", {})
    
    d = v.get("distance_meters")
    proximity = v.get("proximity")
    
    if d is None:
        errors.append("Missing distance_meters")
        return errors
    
    if proximity == "near" and d > 5.0:
        errors.append(f"Near but distance={d}m (should be ≤5m)")
    elif proximity == "moderate" and (d <= 5.0 or d > 15.0):
        errors.append(f"Moderate but distance={d}m (should be 5-15m)")
    elif proximity == "far" and d <= 15.0:
        errors.append(f"Far but distance={d}m (should be >15m)")
    
    return errors


def validate_perception(q: dict) -> List[str]:
    """Validate a perception question."""
    errors = []
    v = q.get("verification", {})
    
    qtype = v.get("question_type")
    if not qtype:
        errors.append("Missing question_type in verification")
    
    return errors


def validate_counting(q: dict) -> List[str]:
    """Validate a counting question."""
    errors = []
    v = q.get("verification", {})
    
    correct_count = v.get("correct_count")
    activity = v.get("activity")
    
    if correct_count is None:
        errors.append("Missing correct_count in verification")
    elif correct_count < 0:
        errors.append(f"Invalid correct_count: {correct_count} (must be >= 0)")
    
    if not activity:
        errors.append("Missing activity in verification")
    
    # Check that correct answer matches correct_count
    correct_answer = q.get("correct_answer")
    if correct_answer and str(correct_count) != correct_answer:
        errors.append(f"correct_answer ({correct_answer}) doesn't match correct_count ({correct_count})")
    
    return errors


def validate_all(qa_pairs: List[dict]) -> Dict[str, List[str]]:
    """
    Validate all QA pairs and return errors by question_id.
    """
    validators = {
        "temporal": validate_temporal,
        "spatial": validate_spatial,
        "perception": validate_perception,
        "counting": validate_counting,
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
                 seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Run the complete V6 QA generation pipeline on one slot.
    
    Steps:
        1. Parse annotations (Kitware YAML)
        2. Build entity-based scene graph
        3. Resolve cross-camera entities (MEVID + heuristic)
        4. Generate temporal questions (3 per slot)
        5. Generate spatial questions (3 per slot)
        6. Generate perception questions (3 per slot)
        7. Generate counting questions (3 per slot)
        8. Validate and output
    
    Args:
        slot: Slot name e.g. "2018-03-11.11-25-00.school"
        verbose: Print progress
        seed: Random seed for reproducibility
    
    Returns:
        Output dict matching V6 schema (Section 9 of v6_todo.md)
    """
    t0 = time.time()
    rng = random.Random(seed)
    
    if verbose:
        print(f"=" * 60)
        print(f"V6 Pipeline: {slot}")
        print(f"=" * 60)
    
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
    
    # Step 4-7: Generate QA pairs
    if verbose:
        print(f"\nStep 4-7: Generating questions...")
    
    temporal_qa = generate_temporal_qa(sg, resolved, rng, 
                                        count=TARGET_PER_CATEGORY, verbose=verbose)
    spatial_qa = generate_spatial_qa(sg, resolved, rng,
                                     count=TARGET_PER_CATEGORY, verbose=verbose)
    perception_qa = generate_perception_qa(sg, resolved, rng,
                                            count=TARGET_PER_CATEGORY, verbose=verbose)
    counting_qa = generate_counting_qa(sg, resolved, rng,
                                       count=TARGET_PER_CATEGORY, verbose=verbose)
    
    all_qa = temporal_qa + spatial_qa + perception_qa + counting_qa
    
    # Deduplication
    unique_qa = []
    for q in all_qa:
        if not is_duplicate_within_slot(q, unique_qa):
            unique_qa.append(q)
    
    # Validation
    if verbose:
        print(f"\nStep 8: Validating...")
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
    
    # MEVID stats
    mevid_persons = find_mevid_persons_for_slot(slot, cameras_in_slot)
    
    output = {
        "slot": slot,
        "version": "v6",
        "annotation_source": "kitware",
        "entity_resolution_source": "mevid+heuristic",
        "generator": "v6_pipeline",
        "seed": seed,
        "difficulty": "easy",
        "cameras": cameras_in_slot,
        "mevid_persons_in_slot": len(mevid_persons),
        "total_events": len(events),
        "total_entities": len(sg.entities),
        "cross_camera_clusters": len(resolved.entity_clusters),
        "total_questions": len(unique_qa),
        "category_counts": {
            "temporal": sum(1 for q in unique_qa if q["category"] == "temporal"),
            "spatial": sum(1 for q in unique_qa if q["category"] == "spatial"),
            "perception": sum(1 for q in unique_qa if q["category"] == "perception"),
            "counting": sum(1 for q in unique_qa if q["category"] == "counting"),
        },
        "validation_issues": len(issues),
        "generation_time_sec": round(time.time() - t0, 2),
        "qa_pairs": unique_qa,
    }
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"DONE: {len(unique_qa)} questions generated in {output['generation_time_sec']}s")
        print(f"  Temporal:   {output['category_counts']['temporal']}")
        print(f"  Spatial:    {output['category_counts']['spatial']}")
        print(f"  Perception: {output['category_counts']['perception']}")
        print(f"  Counting:   {output['category_counts']['counting']}")
        print(f"  Cameras:    {cameras_in_slot}")
        print(f"  Events:     {len(events)}")
        print(f"  Entities:   {len(sg.entities)}")
        print(f"  MEVID persons: {len(mevid_persons)}")
        print(f"  Cross-cam clusters: {len(resolved.entity_clusters)}")
        print(f"  Validation issues: {len(issues)}")
        print(f"{'=' * 60}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="V6 QA Pipeline")
    parser.add_argument("--slot", required=True, help="Slot name (e.g., 2018-03-11.11-25-00.school)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--output", help="Output file path (default: data/qa_pairs/{slot}.v6.json)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file, just print")
    args = parser.parse_args()
    
    output = run_pipeline(args.slot, verbose=args.verbose, seed=args.seed)
    
    # Save output
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.slot}.v6.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FINAL run_pipeline.py — Main orchestrator for FINAL QA generation pipeline.

6 categories (matching paper taxonomy):
  temporal(2) + event_ordering(2) + spatial(3)
  + summarization(1) + counting(1) + best_camera(3) = ~12 Qs/slot

REMOVED from V9: re_identification, causality
REMOVED: perception (killed — not useful for benchmark)
ADDED: best_camera (Camera Transition Logic)

Setup (run from the meva/ directory inside the repo):
    cd /path/to/repo/meva
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    export OPENAI_API_KEY=sk-...          # required only for naturalization step
    export OUTPUT_DIR=~/data                # optional: where QA output is saved (default: ~/data)

Usage:
    # Step 1 — generate raw QA (free, ~5s/slot)
    python3 -m scripts.v10.run_pipeline --slot "2018-03-11.11-25.school" -v

    # Step 2 — naturalize with GPT (costs tokens, requires OPENAI_API_KEY)
    python3 -m scripts.v10.naturalize \\
        --input $MEVA_OUTPUT_DIR/qa_pairs/2018-03-11.11-25.school/2018-03-11.11-25.school.final.raw.json \\
        -v --yes

    # Step 3 — export to multi-cam-dataset repo format
    python3 -m scripts.v10.export_to_multicam_format --slot "2018-03-11.11-25.school"

    # List all available slots
    python3 -m scripts.v10.run_pipeline --list-slots
"""

import json
import argparse
import os
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
    # from .generate_perception import generate_perception_qa  # KILLED: perception category removed
    from .generate_scene_summary import generate_scene_summary_qa
    from .generate_event_ordering import generate_event_ordering_qa
    from .generate_numerical import generate_numerical_qa
    from .generate_best_camera import generate_best_camera_qa
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parse_annotations import parse_slot_events, find_clips_for_slot
    from build_scene_graph import build_scene_graph
    from entity_resolution import resolve_entities
    from person_descriptions import (
        enrich_entities, is_mevid_supported, get_mevid_persons_for_slot,
        get_mevid_persons_with_cameras, load_person_database,
    )
    from generate_temporal import generate_temporal_qa
    from generate_spatial import generate_spatial_qa
    # from generate_perception import generate_perception_qa  # KILLED: perception category removed
    from generate_scene_summary import generate_scene_summary_qa
    from generate_event_ordering import generate_event_ordering_qa
    from generate_numerical import generate_numerical_qa
    from generate_best_camera import generate_best_camera_qa


# ============================================================================
# Constants
# ============================================================================

# Repo-relative data directory (meva/data/) — works for any clone location
_REPO_DATA = Path(__file__).resolve().parent.parent.parent / "data"
# User output directory — override with MEVA_OUTPUT_DIR env var
_OUTPUT = Path(os.environ.get("OUTPUT_DIR") or os.environ.get("MEVA_OUTPUT_DIR") or str(Path.home() / "data"))

OUTPUT_DIR = _OUTPUT / "qa_pairs" / "raw"
MEVA_MP4_BASE = Path("/nas/mars/dataset/MEVA/mp4s")
# Entity descriptions directory — override with MEVA_ENTITY_DESC_DIR env var
_ENTITY_DESC_DIR = Path(os.environ.get("MEVA_ENTITY_DESC_DIR") or "/nas/mars/dataset/MEVA/entity_descriptions")
CANONICAL_SLOTS_PATH = _REPO_DATA / "canonical_slots.json"
RANDOM_SEED = 42

# 6 categories — max question counts per slot (soft ceilings, not rigid targets)
# Generators produce all valid candidates and cap at MAX.
MAX_TEMPORAL = 2
MAX_EVENT_ORDERING = 2
# MAX_PERCEPTION = 2           # KILLED: perception category removed
MAX_SPATIAL = 3              # ~70% slot hit rate requires 3/slot for 500 total
MAX_SUMMARIZATION = 1        # scene_summary (renamed for paper alignment)
MAX_COUNTING = 1             # activity-counting only (entity-counting removed)
MAX_BEST_CAMERA = 3          # Camera Transition Logic, ~70% hit rate
# Maximum total: ~14 questions per slot (actual count may be lower)


# ============================================================================
# Text Post-Processing
# ============================================================================

import re as _re

def fix_articles(text: str) -> str:
    """Fix 'a' → 'an' before vowel sounds (a orange → an orange)."""
    return _re.sub(r'\ba ([aeiouAEIOU])', r'an \1', text)


# ============================================================================
# Video Path Construction
# ============================================================================

def _build_video_paths(q: dict, slot: str) -> List[str]:
    """Build absolute MP4 paths for each camera in a QA pair.
    
    Uses slot-grouped MP4 directory structure:
      slot = "2018-03-11.11-25.school"
      clip_file = "2018-03-11.11-25-00.11-30-00.school.G330.r13.mp4"
      → /nas/mars/dataset/MEVA/mp4s/2018-03-11/11/2018-03-11.11-25.school/
            2018-03-11.11-25-00.11-30-00.school.G330.r13.mp4
    """
    date = slot.split(".")[0]  # "2018-03-11"
    hour = slot.split(".")[1].split("-")[0]  # "11"
    slot_dir = MEVA_MP4_BASE / date / hour / slot
    
    # Prefer including all videos for the slot, if available.
    if slot_dir.exists():
        return sorted(str(p) for p in slot_dir.glob("*.mp4"))

    paths = []
    seen = set()
    
    def _add_clip(clip: str):
        """Normalize a clip filename and add its MP4 path (deduped)."""
        if not clip or clip in seen:
            return
        seen.add(clip)
        # Normalize: strip any extension, ensure single .r13.mp4
        stem = clip
        for ext in (".mp4", ".avi"):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
        # Strip trailing .r13 if present (avoid double .r13)
        if stem.endswith(".r13"):
            stem = stem[:-4]
        mp4_name = f"{stem}.r13.mp4"
        mp4_path = str(slot_dir / mp4_name)
        if mp4_path not in paths:
            paths.append(mp4_path)
    
    # Collect clip files from debug_info
    debug = q.get("debug_info", {})
    
    # 1. Direct clip_files list (counting, best_camera, summarization, spatial)
    for cf in debug.get("clip_files", []):
        _add_clip(cf)
    
    # 2. temporal: event_a, event_b (dict with clip_file)
    for key in ["event_a", "event_b"]:
        info = debug.get(key, {})
        if isinstance(info, dict) and info.get("clip_file"):
            _add_clip(info["clip_file"])
    
    # 3. perception / best_camera: representative_event (dict with clip_file)
    rep = debug.get("representative_event", {})
    if isinstance(rep, dict) and rep.get("clip_file"):
        _add_clip(rep["clip_file"])
    
    # 4. perception multi-cam: camera_1_event, camera_2_event, etc.
    for key in debug:
        if key.startswith("camera_") and key.endswith("_event"):
            info = debug[key]
            if isinstance(info, dict) and info.get("clip_file"):
                _add_clip(info["clip_file"])
    
    # 5. event_ordering: events (list of dicts with clip_file)
    events_list = debug.get("events", [])
    if isinstance(events_list, list):
        for ev in events_list:
            if isinstance(ev, dict) and ev.get("clip_file"):
                _add_clip(ev["clip_file"])
    
    # 6. Spatial: entity_a, entity_b (dict with clip_file)
    for key in ["entity_a", "entity_b"]:
        info = debug.get(key, {})
        if isinstance(info, dict) and info.get("clip_file"):
            _add_clip(info["clip_file"])
    
    # Fallback: match only this slot's cameras in the slot directory
    if not paths:
        cameras = q.get("requires_cameras", [])
        if slot_dir.exists() and cameras:
            for cam in cameras:
                for f in sorted(slot_dir.glob(f"*{cam}*.r13.mp4")):
                    p = str(f)
                    if p not in paths:
                        paths.append(p)
    
    return sorted(set(paths))


# ============================================================================
# Canonical Slot Resolution
# ============================================================================

def resolve_canonical_slot(slot: str) -> List[str]:
    """Given a slot name, return the list of slots to process.
    
    With HH-MM slot format (no seconds), each slot name is already canonical.
    The old canonical_slots.json indirection is no longer needed.
    """
    return [slot]


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
        
        elif cat == "summarization":
            if v_new.get("question_type") == v_old.get("question_type"):
                return True
        
        elif cat == "event_ordering":
            new_events = v_new.get("ordered_events", [])
            old_events = v_old.get("ordered_events", [])
            if new_events and old_events:
                new_acts = {e.get("activity") for e in new_events}
                old_acts = {e.get("activity") for e in old_events}
                if len(new_acts & old_acts) >= 3:
                    return True
        
        elif cat == "counting":
            if (v_new.get("question_type") == v_old.get("question_type") and
                v_new.get("activity") == v_old.get("activity")):
                return True
        
        elif cat == "best_camera":
            if (v_new.get("cluster_id") == v_old.get("cluster_id") and
                v_new.get("question_type") == v_old.get("question_type")):
                return True
            # V10: Also dedup by question text / entity description
            if (v_new.get("question_type") == v_old.get("question_type") and
                v_new.get("entity_description") == v_old.get("entity_description")):
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
    d = v.get("min_distance_meters")
    proximity = v.get("proximity")
    if d is None:
        errors.append("Missing min_distance_meters")
        return errors
    if proximity == "near" and d > 5.0:
        errors.append(f"Near but distance={d}m (should be <=5m)")
    elif proximity == "moderate" and (d <= 5.0 or d > 15.0):
        errors.append(f"Moderate but distance={d}m (should be 5-15m)")
    elif proximity == "far" and d <= 15.0:
        errors.append(f"Far but distance={d}m (should be >15m)")
    # Validate cross-paths flag consistency
    crosses = v.get("crosses_paths", False)
    if crosses and d > 2.0:
        errors.append(f"crosses_paths=True but distance={d}m (should be <=2m)")
    return errors


def validate_perception(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    if not v.get("question_type"):
        errors.append("Missing question_type in verification")
    return errors


def validate_summarization(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    if not v.get("question_type"):
        errors.append("Missing question_type")
    return errors


def validate_event_ordering(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    ordered = v.get("ordered_events", [])
    if len(ordered) < 3:
        errors.append(f"Too few events: {len(ordered)} (min 3)")
    min_gap = v.get("min_gap_sec", 0)
    if min_gap < 3.0:
        errors.append(f"Min gap too small: {min_gap}s (min 3s)")
    for i in range(len(ordered) - 1):
        if ordered[i].get("start_sec", 0) >= ordered[i+1].get("start_sec", 0):
            errors.append(f"Events not in chronological order at position {i}")
    return errors


def validate_counting(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    if not v.get("question_type"):
        errors.append("Missing question_type")
    cnt = v.get("correct_count")
    if cnt is not None and (cnt < 2 or cnt > 20):
        errors.append(f"Count out of range: {cnt} (should be 2-20)")
    return errors


def validate_best_camera(q: dict) -> List[str]:
    errors = []
    v = q.get("verification", {})
    if not v.get("question_type"):
        errors.append("Missing question_type")
    if not v.get("correct_camera"):
        errors.append("Missing correct_camera")
    return errors


def validate_all(qa_pairs: List[dict]) -> Dict[str, List[str]]:
    validators = {
        "temporal": validate_temporal,
        "spatial": validate_spatial,
        # "perception": validate_perception,  # KILLED
        "summarization": validate_summarization,
        "event_ordering": validate_event_ordering,
        "counting": validate_counting,
        "best_camera": validate_best_camera,
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
# Category Renaming (paper alignment)
# ============================================================================

def _rename_category(q: dict) -> dict:
    """Rename internal category names to match paper taxonomy.
    
    scene_summary → summarization
    numerical     → counting
    """
    cat = q.get("category", "")
    if cat == "scene_summary":
        q["category"] = "summarization"
    elif cat == "numerical":
        q["category"] = "counting"
    return q


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(slot: str, verbose: bool = False, 
                 seed: int = RANDOM_SEED,
                 require_mevid: bool = True) -> Dict[str, Any]:
    """
    Run the complete FINAL QA generation pipeline on one slot.
    
    Steps:
        1. Parse annotations (Kitware YAML)
        2. Build entity-based scene graph
        3. Resolve cross-camera entities (MEVID + heuristic)
        3.5. Auto-extract entity descriptions if not yet done
        4. Enrich entities with visual descriptions
        5. Generate temporal questions (2)
        6. Generate event ordering questions (2)
        7. Generate perception questions (2)
        8. Generate spatial questions (2)
        9. Generate summarization questions (1)
       10. Generate counting questions (1)
       11. Generate best_camera questions (2)
       12. Validate and output
    """
    t0 = time.time()
    rng = random.Random(seed)
    
    if verbose:
        print(f"{'=' * 60}")
        print(f"FINAL Pipeline: {slot}")
        print(f"{'=' * 60}")
    
    # Resolve canonical slot → raw slots
    raw_slots = resolve_canonical_slot(slot)
    if verbose and len(raw_slots) > 1:
        print(f"  Canonical slot → {len(raw_slots)} raw variants: {raw_slots}")
    
    # Check MEVID support
    mevid_supported = is_mevid_supported(slot)
    mevid_persons = get_mevid_persons_for_slot(slot)
    
    if require_mevid and not mevid_supported and not mevid_persons:
        if verbose:
            print(f"  WARNING: Slot {slot} has no MEVID support")
            print(f"  Running anyway (descriptions will fallback to geom-color or activity-verb)")
    
    if verbose:
        print(f"  MEVID persons in slot: {len(mevid_persons)}")
    
    # Step 1: Parse annotations (merge across raw slots if canonical)
    if verbose:
        print(f"\nStep 1: Parsing annotations...")
    events = []
    for raw_slot in raw_slots:
        slot_events = parse_slot_events(raw_slot, verbose=verbose)
        events.extend(slot_events)
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
    
    # Step 3.5: Auto-extract entity descriptions if not yet done
    desc_path = _ENTITY_DESC_DIR / f"{slot}.json"
    if not desc_path.exists():
        if verbose:
            print(f"\nStep 3.5: Extracting entity descriptions (YOLO+HSV)...")
        try:
            from scripts.final.extract_entity_descriptions import process_slot as extract_descs
            result = extract_descs(slot, use_yolo=True, verbose=verbose)
            desc_path.parent.mkdir(parents=True, exist_ok=True)
            with open(desc_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            if verbose:
                n_actors = len(result.get("actors", {}))
                n_described = sum(1 for a in result.get("actors", {}).values()
                                  if a.get("description", "a person") != "a person")
                print(f"  Extracted {n_described}/{n_actors} entity descriptions → {desc_path.name}")
        except Exception as e:
            if verbose:
                print(f"  WARNING: Entity description extraction failed: {e}")
                print(f"  Continuing with MEVID + activity-verb fallback only")
    elif verbose:
        print(f"\nStep 3.5: Entity descriptions already exist → {desc_path.name}")
    
    # Step 4: Enrich entities with visual descriptions
    if verbose:
        print(f"\nStep 4: Enriching entities with visual descriptions...")
    entity_descs, desc_counts, fallback_eids = enrich_entities(sg, verbose=verbose)
    
    mevid_cnt = desc_counts["mevid"]
    geom_cnt = desc_counts["geom"]
    fallback_cnt = desc_counts["fallback"]
    
    if verbose:
        print(f"  {mevid_cnt} MEVID + {geom_cnt} geom-color + "
              f"{fallback_cnt} fallback / {len(entity_descs)} total")
    
    # Step 5-11: Generate QA pairs (6 categories)
    if verbose:
        print(f"\nStep 5-11: Generating questions (6 categories)...")
    
    temporal_qa = generate_temporal_qa(sg, resolved, entity_descs, rng,
                                       count=MAX_TEMPORAL, verbose=verbose,
                                       fallback_eids=fallback_eids)
    ordering_qa = generate_event_ordering_qa(sg, resolved, entity_descs, rng,
                                              count=MAX_EVENT_ORDERING, verbose=verbose,
                                              fallback_eids=fallback_eids)
    # KILLED: perception category removed
    # perception_qa = generate_perception_qa(sg, resolved, entity_descs, rng,
    #                                        count=MAX_PERCEPTION, verbose=verbose)
    perception_qa = []
    spatial_qa = generate_spatial_qa(sg, resolved, entity_descs, rng,
                                     count=MAX_SPATIAL, verbose=verbose,
                                     fallback_eids=fallback_eids)
    summary_qa = generate_scene_summary_qa(sg, resolved, entity_descs, rng,
                                            count=MAX_SUMMARIZATION, verbose=verbose)
    counting_qa = generate_numerical_qa(sg, resolved, entity_descs, rng,
                                         count=MAX_COUNTING, verbose=verbose)
    best_camera_qa = generate_best_camera_qa(sg, resolved, entity_descs, rng,
                                              count=MAX_BEST_CAMERA, verbose=verbose)
    
    # Rename categories for paper alignment
    for q in summary_qa:
        _rename_category(q)
    for q in counting_qa:
        _rename_category(q)
    
    all_qa = (temporal_qa + ordering_qa + perception_qa + spatial_qa 
              + summary_qa + counting_qa + best_camera_qa)
    
    # Deduplication
    unique_qa = []
    for q in all_qa:
        if not is_duplicate_within_slot(q, unique_qa):
            unique_qa.append(q)
    
    # Renumber question IDs sequentially
    for i, q in enumerate(unique_qa):
        q["question_id"] = f"final_{q['category']}_{i+1:03d}"

    # NOTE: article fixes (a→an) and text polish happen in naturalization.py,
    # not here. Generators produce raw mechanical text; naturalization owns all
    # wording/grammar changes.

    # Add video paths to each QA pair
    for q in unique_qa:
        q["video_paths"] = _build_video_paths(q, slot)
    
    # Step 12: Validation
    if verbose:
        print(f"\nStep 12: Validating...")
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
    
    # Stats
    attr_verification = sum(
        1 for q in unique_qa
        if q.get("subcategory") == "attribute_verification"
    )
    
    output = {
        "slot": slot,
        "version": "final",
        "annotation_source": "kitware",
        "entity_resolution_source": "mevid+heuristic",
        "description_source": "mevid_yolo+geom_color",
        "generator": "final_pipeline",
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
        "cross_camera_clusters": len(resolved.entity_clusters),
        "total_questions": len(unique_qa),
        "category_counts": cat_counts,
        "stats": {
            "entities_with_mevid_descriptions": mevid_cnt,
            "entities_with_geom_descriptions": geom_cnt,
            "entities_with_fallback_descriptions": fallback_cnt,
            "attribute_verification_questions": attr_verification,
            "best_camera_questions": cat_counts.get("best_camera", 0),
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
        print(f"  Entities:   {len(sg.entities)} ({mevid_cnt} MEVID + {geom_cnt} geom)")
        print(f"  MEVID persons: {sorted(mevid_persons)}")
        print(f"  Cross-cam clusters: {len(resolved.entity_clusters)}")
        print(f"  Validation issues: {len(issues)}")
        print(f"{'=' * 60}")
    
    return output


def list_canonical_slots():
    """List all canonical slots from canonical_slots.json."""
    if not CANONICAL_SLOTS_PATH.exists():
        print(f"ERROR: {CANONICAL_SLOTS_PATH} not found. Run slot audit first.")
        sys.exit(1)
    
    with open(CANONICAL_SLOTS_PATH) as f:
        canonical = json.load(f)
    
    multi = sum(1 for v in canonical.values() if v.get("multi_camera"))
    cam_counts = [len(v.get("cameras", [])) for v in canonical.values()]
    
    print(f"Canonical slots: {len(canonical)} ({multi} multi-camera)")
    print(f"Cameras: min={min(cam_counts)}, max={max(cam_counts)}, avg={sum(cam_counts)/len(cam_counts):.1f}")
    print(f"\n{'Slot':45s} {'Cams':>5s} {'Clips':>6s} {'Variants':>9s}")
    print("-" * 70)
    
    for slot in sorted(canonical.keys()):
        info = canonical[slot]
        n_cams = len(info.get("cameras", []))
        n_clips = info.get("clip_count", 0)
        n_vars = len(info.get("raw_slots", []))
        cam_str = ",".join(info.get("cameras", []))
        print(f"{slot:45s} {n_cams:5d} {n_clips:6d} {n_vars:9d}   [{cam_str}]")


def main():
    parser = argparse.ArgumentParser(description="FINAL QA Pipeline (6 categories)")
    parser.add_argument("--slot", help="Slot name (e.g., 2018-03-11.11-25.school)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--output", help="Output file path (default: data/qa_pairs/raw/{slot}.raw.json)")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file, just print")
    parser.add_argument("--list-slots", action="store_true",
                       help="List all canonical slots")
    parser.add_argument("--no-require-mevid", action="store_true",
                       help="Process slot even without MEVID support")
    args = parser.parse_args()
    
    if args.list_slots:
        list_canonical_slots()
        return
    
    if not args.slot:
        parser.error("--slot is required (or use --list-slots)")
    
    output = run_pipeline(args.slot, verbose=args.verbose, seed=args.seed,
                          require_mevid=not args.no_require_mevid)
    
    # Save output
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{args.slot}.raw.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")
    else:
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()

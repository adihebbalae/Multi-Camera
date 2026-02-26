#!/usr/bin/env python3
"""
export_to_multicam_format.py — Transform FINAL naturalized QA to multi-cam-dataset schema.

Reads:  /home/ah66742/data/qa_pairs/{slot}.final.naturalized.json
Writes: /nas/neurosymbolic/multi-cam-dataset/meva/qa_pairs/{slot}.json

Target schema matches agibot / ego-exo4d format:
  {
    "slot": "...",
    "question_type": "temporal",
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "answer": "A",
    "reasoning": "...",
    "video_paths": [...],
    "metadata": { site, cameras, camera_names, difficulty, verification }
  }

Usage:
    python3 scripts/final/export_to_multicam_format.py --slot "2018-03-11.11-25-00.school"
    python3 scripts/final/export_to_multicam_format.py --all
    python3 scripts/final/export_to_multicam_format.py --slot "..." --dry-run
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Paths
# User output directory — override with MEVA_OUTPUT_DIR env var
_OUTPUT = Path(os.environ.get("OUTPUT_DIR") or os.environ.get("MEVA_OUTPUT_DIR") or str(Path.home() / "data"))
INPUT_DIR = _OUTPUT / "qa_pairs"
OUTPUT_DIR = Path("/nas/neurosymbolic/multi-cam-dataset/meva/qa_pairs")

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}


def _options_to_dict(options: List[str]) -> Dict[str, str]:
    """Convert list of options to {A: ..., B: ..., C: ..., D: ...} dict."""
    return {LETTER_MAP[i]: opt for i, opt in enumerate(options) if i in LETTER_MAP}


def _index_to_letter(idx: int) -> str:
    """Convert 0-based index to letter answer."""
    return LETTER_MAP.get(idx, "A")


def _extract_site(slot: str) -> str:
    """Extract site from slot name: '2018-03-11.11-25-00.school' → 'school'."""
    parts = slot.split(".")
    return parts[-1] if len(parts) >= 3 else "unknown"


def _build_camera_names(cameras: List[str]) -> Dict[str, str]:
    """Build camera_names dict: {'G421': 'Camera G421', ...}."""
    return {cam: f"Camera {cam}" for cam in cameras}


def _transform_question(q: dict, slot: str, envelope: dict) -> dict:
    """Transform a single V9 naturalized QA item to multi-cam-dataset format."""
    # Pick the best available question text
    question_text = (
        q.get("naturalized_question")
        or q.get("question_template")
        or q.get("question", "")
    )

    # Pick the best available options list
    options_list = (
        q.get("naturalized_options")
        or q.get("options")
        or []
    )

    # Build the camera list for this question
    q_cameras = q.get("requires_cameras", [])
    if not q_cameras:
        # Fall back to envelope-level cameras
        q_cameras = envelope.get("cameras", [])

    # Core answer index
    answer_idx = q.get("correct_answer_index", 0)

    # Verification data (strip internal fields, keep useful ones)
    verification = q.get("verification", {})

    # Build grounding from verification events (match ego-exo4d ordered_events style)
    grounding = _build_grounding(q)

    # Assemble metadata
    site = _extract_site(slot)
    metadata: Dict[str, Any] = {
        "site": site,
        "slot": slot,
        "cameras": q_cameras,
        "camera_names": _build_camera_names(q_cameras),
        "difficulty": q.get("difficulty", "medium"),
    }
    if grounding:
        metadata["grounding"] = grounding
    if verification:
        metadata["verification"] = verification

    # Assemble the output question
    out: Dict[str, Any] = {
        "slot": slot,
        "question_type": q.get("category", "unknown"),
        "question": question_text,
        "options": _options_to_dict(options_list),
        "answer": _index_to_letter(answer_idx),
        "reasoning": q.get("reasoning", ""),
        "video_paths": q.get("video_paths", []),
        "metadata": metadata,
    }

    return out


def _build_grounding(q: dict) -> List[Dict[str, Any]]:
    """Build a grounding/ordered_events list from verification data.
    
    Matches ego-exo4d style:
      [{"activity": "...", "camera": "G421", "start_timestamp": 0.33}, ...]
    """
    verification = q.get("verification", {})
    category = q.get("category", "")
    events = []

    if category in ("temporal", "spatial"):
        for key in ["event_a", "event_b"]:
            ev = verification.get(key, {})
            if ev and ev.get("activity"):
                entry: Dict[str, Any] = {
                    "activity": ev["activity"],
                    "camera": ev.get("camera", ""),
                }
                if "start_sec" in ev:
                    entry["start_timestamp"] = ev["start_sec"]
                if "end_sec" in ev:
                    entry["end_timestamp"] = ev["end_sec"]
                events.append(entry)

    elif category == "event_ordering":
        ordered = verification.get("ordered_events", [])
        for ev in ordered:
            if ev.get("activity"):
                entry = {
                    "activity": ev["activity"],
                    "camera": ev.get("camera", ""),
                }
                if "start_sec" in ev:
                    entry["start_timestamp"] = ev["start_sec"]
                events.append(entry)

    elif category == "best_camera":
        ev = verification
        if ev.get("activity"):
            entry = {
                "activity": ev["activity"],
                "camera": ev.get("correct_camera", ""),
            }
            if "entrance_time_sec" in ev:
                entry["start_timestamp"] = ev["entrance_time_sec"]
            events.append(entry)

    elif category in ("perception", "summarization", "counting"):
        ev = verification.get("target_event", verification)
        if ev.get("activity"):
            entry = {
                "activity": ev["activity"],
                "camera": ev.get("camera", ev.get("correct_camera", "")),
            }
            if "start_sec" in ev:
                entry["start_timestamp"] = ev["start_sec"]
            events.append(entry)

    return events


def export_slot(slot: str, dry_run: bool = False, verbose: bool = False) -> Optional[List[dict]]:
    """Export one slot from V9 naturalized format to multi-cam-dataset format.
    
    Returns the exported list of questions, or None on failure.
    """
    input_file = INPUT_DIR / f"{slot}.final.naturalized.json"
    if not input_file.exists():
        # Try v9 format as fallback
        input_file = INPUT_DIR / f"{slot}.v9.naturalized.json"
    if not input_file.exists():
        print(f"  ERROR: Input not found: {input_file}", file=sys.stderr)
        return None

    with open(input_file) as f:
        envelope = json.load(f)

    qa_pairs = envelope.get("qa_pairs", [])
    if not qa_pairs:
        print(f"  WARNING: No qa_pairs in {input_file}", file=sys.stderr)
        return []

    exported = []
    for q in qa_pairs:
        out = _transform_question(q, slot, envelope)
        exported.append(out)

    if verbose:
        print(f"  Transformed {len(exported)} questions for {slot}")
        # Category breakdown
        cats = {}
        for q in exported:
            cats[q["question_type"]] = cats.get(q["question_type"], 0) + 1
        for cat, count in sorted(cats.items()):
            print(f"    {cat}: {count}")

    if dry_run:
        print(json.dumps(exported[:2], indent=2))
        print(f"  ... ({len(exported)} total, showing first 2)")
        return exported

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{slot}.json"
    with open(output_file, "w") as f:
        json.dump(exported, f, indent=2)
    
    if verbose:
        print(f"  Written to {output_file}")

    return exported


def find_all_naturalized_slots() -> List[str]:
    """Find all slots that have .final.naturalized.json files."""
    slots = []
    for f in sorted(INPUT_DIR.glob("*.final.naturalized.json")):
        slot = f.name.replace(".final.naturalized.json", "")
        slots.append(slot)
    # Also check v9 format as fallback
    for f in sorted(INPUT_DIR.glob("*.v9.naturalized.json")):
        slot = f.name.replace(".v9.naturalized.json", "")
        if slot not in slots:
            slots.append(slot)
    return sorted(slots)


def main():
    parser = argparse.ArgumentParser(description="Export V9 QA to multi-cam-dataset format")
    parser.add_argument("--slot", type=str, help="Slot name to export")
    parser.add_argument("--all", action="store_true", help="Export all available naturalized slots")
    parser.add_argument("--dry-run", action="store_true", help="Print sample output without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not args.slot and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.all:
        slots = find_all_naturalized_slots()
        if not slots:
            print("No .v9.naturalized.json files found.", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(slots)} slot(s) to export:")
        total = 0
        for slot in slots:
            print(f"\n  [{slot}]")
            result = export_slot(slot, dry_run=args.dry_run, verbose=args.verbose)
            if result is not None:
                total += len(result)
        print(f"\nTotal: {total} questions exported across {len(slots)} slot(s)")
    else:
        result = export_slot(args.slot, dry_run=args.dry_run, verbose=args.verbose)
        if result is None:
            sys.exit(1)
        print(f"Exported {len(result)} questions for {args.slot}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
batch_verify.py — Automated quality verification for batch-generated QA.

Validates raw QA JSONs from batch_run_all_slots.py WITHOUT watching videos.
Runs structural, temporal-consistency, uniqueness, and statistical checks.

Usage:
    # Verify all raw outputs
    python3 -m scripts.v10.batch_verify

    # Just temporal category
    python3 -m scripts.v10.batch_verify --category temporal

    # Verbose per-slot details
    python3 -m scripts.v10.batch_verify -v

    # Export report to file
    python3 -m scripts.v10.batch_verify --report report.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_OUTPUT = Path(
    os.environ.get("MEVA_OUTPUT_DIR") or
    os.environ.get("OUTPUT_DIR") or
    "/nas/neurosymbolic/multi-cam-dataset/meva/data"
)
RAW_DIR = _OUTPUT / "qa_pairs" / "raw"


# ============================================================================
# Structural Checks (per question)
# ============================================================================

REQUIRED_FIELDS = {
    "question_id", "category", "question_template", "options",
    "correct_answer_index", "correct_answer", "requires_cameras",
}

CATEGORIES = {"temporal", "event_ordering", "spatial", "summarization",
              "counting", "best_camera"}


def _check_structure(q: Dict) -> List[str]:
    """Return list of structural issues for one question."""
    issues = []
    missing = REQUIRED_FIELDS - set(q.keys())
    if missing:
        issues.append(f"missing fields: {missing}")

    opts = q.get("options", [])
    if not isinstance(opts, list) or len(opts) < 2:
        issues.append(f"options invalid (got {type(opts).__name__} len={len(opts) if isinstance(opts, list) else '?'})")

    idx = q.get("correct_answer_index")
    if isinstance(idx, int) and isinstance(opts, list):
        if idx < 0 or idx >= len(opts):
            issues.append(f"correct_answer_index={idx} OOB (options len={len(opts)})")
        elif q.get("correct_answer") != opts[idx]:
            issues.append(f"correct_answer mismatch: '{q.get('correct_answer')[:40]}' vs opts[{idx}]='{opts[idx][:40]}'")

    cat = q.get("category", "")
    if cat not in CATEGORIES:
        issues.append(f"unknown category: {cat}")

    cams = q.get("requires_cameras", [])
    if not cams:
        issues.append("no cameras listed")

    return issues


# ============================================================================
# Temporal-Specific Checks
# ============================================================================

def _check_temporal(q: Dict) -> List[str]:
    """Check temporal ordering correctness from verification block."""
    issues = []
    v = q.get("verification", {})
    if not v:
        issues.append("no verification block")
        return issues

    ea = v.get("event_a", {})
    eb = v.get("event_b", {})
    gap = v.get("gap_sec")

    if not ea or not eb:
        issues.append("missing event_a/event_b in verification")
        return issues

    # Event A should end before Event B starts
    a_end = ea.get("end_sec")
    b_start = eb.get("start_sec")
    if a_end is not None and b_start is not None:
        if a_end > b_start:
            issues.append(f"temporal overlap: event_a ends at {a_end:.1f}s but event_b starts at {b_start:.1f}s")

    # Gap should be positive and ≤15s (FALLBACK_MAX_GAP)
    if gap is not None:
        if gap <= 0:
            issues.append(f"non-positive gap: {gap}s")
        elif gap > 15:
            issues.append(f"gap exceeds max: {gap}s > 15s")

    # Events should be on different cameras
    if ea.get("camera") == eb.get("camera"):
        issues.append(f"same camera: {ea.get('camera')}")

    # Descriptions should be different
    if ea.get("description") and ea.get("description") == eb.get("description"):
        issues.append(f"identical descriptions: '{ea['description'][:50]}'")

    return issues


# ============================================================================
# Event Ordering Checks
# ============================================================================

def _check_event_ordering(q: Dict) -> List[str]:
    """Check event ordering chain consistency."""
    issues = []
    v = q.get("verification", {})
    chain = v.get("event_chain", v.get("chain", []))
    if not chain:
        return issues

    # Chain should be chronologically ordered
    for i in range(len(chain) - 1):
        t_curr = chain[i].get("end_sec") or chain[i].get("start_sec", 0)
        t_next = chain[i + 1].get("start_sec", 0)
        if t_curr > t_next + 1.0:  # 1s tolerance
            issues.append(f"chain out of order at step {i}: {t_curr:.1f}s > {t_next:.1f}s")

    return issues


# ============================================================================
# Description Uniqueness (cross-check)
# ============================================================================

def _check_uniqueness(q: Dict) -> List[str]:
    """Check that descriptions in options are sufficiently distinct."""
    issues = []
    opts = q.get("options", [])
    if len(opts) != len(set(opts)):
        dupes = [o for o in opts if opts.count(o) > 1]
        issues.append(f"duplicate options: {dupes[:2]}")

    # For temporal: check event descriptions aren't identical
    v = q.get("verification", {})
    ea_desc = v.get("event_a", {}).get("description", "")
    eb_desc = v.get("event_b", {}).get("description", "")
    if ea_desc and eb_desc and ea_desc == eb_desc:
        issues.append(f"event descriptions identical: '{ea_desc[:50]}'")

    return issues


# ============================================================================
# Aggregate Statistics
# ============================================================================

def verify_all(raw_dir: Path, category_filter: Optional[str] = None,
               verbose: bool = False) -> Dict[str, Any]:
    """Run all verification checks on raw QA outputs.
    
    Returns a report dict with counts, issues, and statistics.
    """
    files = sorted(raw_dir.glob("*.raw.json"))
    if not files:
        print(f"No raw QA files found in {raw_dir}")
        return {"error": "no files found"}

    total_qs = 0
    total_issues = 0
    total_slots = len(files)
    slots_with_issues = 0
    category_counts = Counter()
    category_issues = defaultdict(int)
    issue_types = Counter()
    questions_per_slot = []
    slots_zero_qs = []
    all_issues_detail = []
    temporal_gaps = []
    temporal_formats = Counter()

    for fpath in files:
        slot = fpath.stem.replace(".raw", "")
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            if verbose:
                print(f"  ERROR reading {fpath.name}: {e}")
            slots_with_issues += 1
            continue

        qa_pairs = data.get("qa_pairs", [])
        if category_filter:
            qa_pairs = [q for q in qa_pairs if q.get("category") == category_filter]

        questions_per_slot.append(len(qa_pairs))
        if len(qa_pairs) == 0:
            slots_zero_qs.append(slot)

        slot_issues = []
        for q in qa_pairs:
            total_qs += 1
            cat = q.get("category", "unknown")
            category_counts[cat] += 1

            issues = _check_structure(q)
            issues += _check_uniqueness(q)

            if cat == "temporal":
                issues += _check_temporal(q)
                # Collect temporal stats
                v = q.get("verification", {})
                gap = v.get("gap_sec")
                if gap is not None:
                    temporal_gaps.append(gap)
                fmt = q.get("debug_info", {}).get("question_format", "unknown")
                temporal_formats[fmt] += 1

            elif cat == "event_ordering":
                issues += _check_event_ordering(q)

            if issues:
                total_issues += len(issues)
                category_issues[cat] += len(issues)
                for iss in issues:
                    issue_types[iss.split(":")[0]] += 1
                slot_issues.append({
                    "question_id": q.get("question_id", "?"),
                    "category": cat,
                    "issues": issues,
                })

        if slot_issues:
            slots_with_issues += 1
            all_issues_detail.append({"slot": slot, "issues": slot_issues})
            if verbose:
                print(f"  {slot}: {len(slot_issues)} questions with issues")
                for si in slot_issues:
                    for iss in si["issues"]:
                        print(f"    {si['question_id']}: {iss}")

    # Statistics
    avg_qs = sum(questions_per_slot) / len(questions_per_slot) if questions_per_slot else 0
    temporal_avg_gap = sum(temporal_gaps) / len(temporal_gaps) if temporal_gaps else 0

    report = {
        "summary": {
            "total_slots": total_slots,
            "total_questions": total_qs,
            "avg_questions_per_slot": round(avg_qs, 1),
            "slots_with_zero_questions": len(slots_zero_qs),
            "slots_with_issues": slots_with_issues,
            "total_issues": total_issues,
            "pass_rate_pct": round(
                (total_qs - sum(1 for d in all_issues_detail for q in d["issues"]))
                / total_qs * 100 if total_qs else 0, 1
            ),
        },
        "category_counts": dict(category_counts),
        "category_issues": dict(category_issues),
        "issue_types": dict(issue_types.most_common(20)),
        "temporal_stats": {
            "count": category_counts.get("temporal", 0),
            "avg_gap_sec": round(temporal_avg_gap, 2),
            "min_gap_sec": round(min(temporal_gaps), 2) if temporal_gaps else None,
            "max_gap_sec": round(max(temporal_gaps), 2) if temporal_gaps else None,
            "formats": dict(temporal_formats),
        },
        "zero_question_slots": slots_zero_qs[:20],  # first 20
        "zero_question_count": len(slots_zero_qs),
        "issues_detail": all_issues_detail[:50],  # first 50 slots with issues
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Verify batch QA output quality")
    parser.add_argument("--dir", type=str, default=str(RAW_DIR),
                        help=f"Directory with raw JSON files (default: {RAW_DIR})")
    parser.add_argument("--category", choices=list(CATEGORIES),
                        help="Only verify questions of this category")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--report", type=str,
                        help="Save report JSON to this path")
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    print(f"Verifying raw QA in: {raw_dir}")
    print(f"Category filter: {args.category or 'all'}\n")

    report = verify_all(raw_dir, category_filter=args.category, verbose=args.verbose)

    if "error" in report:
        sys.exit(1)

    # Print summary
    s = report["summary"]
    print(f"\n{'=' * 60}")
    print(f"VERIFICATION REPORT")
    print(f"{'=' * 60}")
    print(f"Slots:       {s['total_slots']}")
    print(f"Questions:   {s['total_questions']} ({s['avg_questions_per_slot']} avg/slot)")
    print(f"Zero-Q slots:{s['slots_with_zero_questions']}")
    print(f"Pass rate:   {s['pass_rate_pct']}%")
    print(f"Issues:      {s['total_issues']} across {s['slots_with_issues']} slots")

    print(f"\nCategory breakdown:")
    for cat, cnt in sorted(report["category_counts"].items()):
        iss = report["category_issues"].get(cat, 0)
        print(f"  {cat:25s}: {cnt:4d} questions, {iss} issues")

    ts = report.get("temporal_stats", {})
    if ts.get("count"):
        print(f"\nTemporal stats:")
        print(f"  Count:    {ts['count']}")
        print(f"  Gap:      {ts['avg_gap_sec']}s avg, {ts['min_gap_sec']}-{ts['max_gap_sec']}s range")
        print(f"  Formats:  {ts['formats']}")

    if report["issue_types"]:
        print(f"\nTop issue types:")
        for iss, cnt in list(report["issue_types"].items())[:10]:
            print(f"  {cnt:4d}x  {iss}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved: {args.report}")


if __name__ == "__main__":
    main()

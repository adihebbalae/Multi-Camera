#!/usr/bin/env python3
"""
Interactive QA Review CLI — manual verification of generated questions.

For each question in a slot, displays:
  - Category, cameras, question text, answer, options
  - Debug info summary (frame ranges, connection strength, etc.)
  - Path to validation video (for playback)
  - Checklist of things to verify

Usage:
    # Review a slot (auto-finds raw JSON)
    python3 -m scripts.final.review_qa --slot 2018-03-07.17-05.school

    # Review naturalized version
    python3 -m scripts.final.review_qa --slot 2018-03-07.17-05.school --natural

    # Review specific QA file
    python3 -m scripts.final.review_qa --qa-file data/qa_pairs/SLOT.final.raw.json

    # Generate audit report without interactive prompts
    python3 -m scripts.final.review_qa --slot 2018-03-07.17-05.school --report
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from datetime import datetime

QA_DIR = Path("/home/ah66742/data/qa_pairs")
VIDEO_DIR = Path("/home/ah66742/output/validation_videos")
AUDIT_DIR = Path("/home/ah66742/output/qa_audits")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ANSI colors
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_GREEN  = "\033[32m"
C_RED    = "\033[31m"
C_YELLOW = "\033[33m"
C_CYAN   = "\033[36m"
C_DIM    = "\033[2m"

CATEGORY_ORDER = ["temporal", "event_ordering", "perception", "spatial",
                   "summarization", "counting", "best_camera"]

CHECKLIST = [
    "Answer observable in video",
    "Cameras minimal & sufficient",
    "Geom boxes track actors",
    "Question clear & unambiguous",
    "Frame range covers activity",
]


def fmt_debug(q: dict) -> str:
    """Format debug_info into a readable summary."""
    debug = q.get("debug_info", {})
    cat = q.get("category", "")
    lines = []

    # Temporal-specific
    if cat == "temporal":
        lines.append(f"  Connection: {debug.get('connection_strength', '?')} "
                      f"(score={debug.get('connection_score', '?')})")
        lines.append(f"  Gap: {debug.get('gap_sec', '?')}s | "
                      f"MEVID: {debug.get('mevid_validated', False)}")
        for key in ["event_a", "event_b"]:
            ev = debug.get(key, {})
            if ev:
                lines.append(f"  {key}: {ev.get('activity','?')} on {ev.get('camera','?')} "
                              f"frames={ev.get('frame_range','?')}")

    # Event ordering
    elif cat == "event_ordering":
        events = debug.get("events", [])
        lines.append(f"  Events: {len(events)} | "
                      f"Group score: {debug.get('group_score', '?')} | "
                      f"MEVID: {debug.get('mevid_validated', False)}")
        for i, ev in enumerate(events):
            lines.append(f"  [{i+1}] {ev.get('activity','?')} on {ev.get('camera','?')} "
                          f"frames={ev.get('frame_range','?')}")

    # Perception
    elif cat == "perception":
        lines.append(f"  Type: {debug.get('question_type', '?')}")
        rep = debug.get("representative_event", {})
        if rep:
            lines.append(f"  Event: {rep.get('activity','?')} on {rep.get('camera','?')} "
                          f"frames={rep.get('frame_range','?')}")

    # Spatial
    elif cat == "spatial":
        lines.append(f"  Distance: {debug.get('distance_meters', '?')}m "
                      f"({debug.get('proximity', '?')})")
        lines.append(f"  Method: {debug.get('projection_method', '?')}")
        for key in ["entity_a", "entity_b"]:
            ent = debug.get(key, {})
            if ent:
                lines.append(f"  {key}: actor={ent.get('actor_ids','?')} on "
                              f"{ent.get('camera','?')} frames={ent.get('frame_range','?')}")

    # Summarization
    elif cat == "summarization":
        lines.append(f"  Type: {debug.get('question_type', '?')}")
        lines.append(f"  Scene: {debug.get('scene_type', '?')}")
        cf = debug.get("clip_files", [])
        lines.append(f"  Clip files: {len(cf)}")

    # Counting
    elif cat == "counting":
        lines.append(f"  Subtype: {debug.get('subtype', '?')}")
        lines.append(f"  Correct count: {debug.get('correct_count', '?')}")
        lines.append(f"  Cross-camera: {debug.get('cross_camera', '?')}")
        lines.append(f"  Cameras: {debug.get('cameras_involved', '?')}")

    # Best camera
    elif cat == "best_camera":
        rep = debug.get("representative_event", {})
        if rep:
            lines.append(f"  Event: {rep.get('activity','?')} on {rep.get('camera','?')} "
                          f"frames={rep.get('frame_range','?')}")

    return "\n".join(lines) if lines else "  (no debug info)"


def display_question(q: dict, idx: int, total: int, slot: str, show_video_path: bool = True):
    """Display a single question for review."""
    cat = q.get("category", "unknown")
    cameras = q.get("requires_cameras", [])
    q_text = q.get("naturalized_question") or q.get("question_template", "")
    answer = q.get("correct_answer") or q.get("answer", "N/A")
    correct_idx = q.get("correct_answer_index")
    options = q.get("options", [])

    print(f"\n{C_BOLD}{'='*72}{C_RESET}")
    print(f"{C_BOLD}Q{idx}/{total-1}  [{cat.upper()}]  Cameras: {', '.join(cameras)}{C_RESET}")
    print(f"{'='*72}")

    # Question text (wrapped)
    print(f"\n{C_CYAN}Question:{C_RESET}")
    for line in textwrap.wrap(q_text, width=70):
        print(f"  {line}")

    # Options
    if options:
        print(f"\n{C_CYAN}Options:{C_RESET}")
        for i, opt in enumerate(options):
            is_correct = (i == correct_idx) if correct_idx is not None else (opt.strip() == str(answer).strip())
            marker = f" {C_GREEN}← ANSWER{C_RESET}" if is_correct else ""
            print(f"  {chr(65+i)}) {opt[:80]}{marker}")
    else:
        print(f"\n{C_CYAN}Answer:{C_RESET} {answer}")

    # Debug summary
    print(f"\n{C_CYAN}Debug:{C_RESET}")
    print(fmt_debug(q))

    # Video path
    if show_video_path:
        video = VIDEO_DIR / f"{slot}_q{idx}_{cat}.mp4"
        if video.exists():
            sz = video.stat().st_size / 1024
            print(f"\n{C_DIM}Video: {video} ({sz:.0f}KB){C_RESET}")
        else:
            print(f"\n{C_YELLOW}Video: NOT RENDERED{C_RESET}")


def interactive_review(qa_data: dict):
    """Interactive review loop — walk through each question."""
    slot = qa_data.get("slot", "unknown")
    qa_pairs = qa_data.get("qa_pairs", [])
    total = len(qa_pairs)

    verdicts = {}  # idx -> {"verdict": "pass"/"fail"/"skip", "notes": "..."}

    print(f"\n{C_BOLD}╔══════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BOLD}║  Interactive QA Review — {slot:<24s}║{C_RESET}")
    print(f"{C_BOLD}║  {total} questions to review{' '*26}║{C_RESET}")
    print(f"{C_BOLD}╚══════════════════════════════════════════════════╝{C_RESET}")

    idx = 0
    while idx < total:
        q = qa_pairs[idx]
        display_question(q, idx, total, slot)

        # Checklist
        print(f"\n{C_CYAN}Checklist:{C_RESET}")
        for item in CHECKLIST:
            print(f"  □ {item}")

        # Prompt
        print(f"\n{C_BOLD}Actions:{C_RESET}")
        print(f"  {C_GREEN}[p]ass{C_RESET}  {C_RED}[f]ail{C_RESET}  "
              f"[s]kip  [n]otes  [v]ideo  [b]ack  [q]uit  [j]ump #")
        try:
            resp = input(f"\n  Q{idx} verdict > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted.")
            break

        if resp == 'q':
            break
        elif resp == 'b':
            idx = max(0, idx - 1)
            continue
        elif resp.startswith('j'):
            try:
                target = int(resp[1:].strip())
                if 0 <= target < total:
                    idx = target
                else:
                    print(f"  Invalid index (0-{total-1})")
            except ValueError:
                print(f"  Usage: j5 (jump to Q5)")
            continue
        elif resp == 'v':
            video = VIDEO_DIR / f"{slot}_q{idx}_{q['category']}.mp4"
            if video.exists():
                print(f"  Opening: {video}")
                os.system(f"nohup xdg-open '{video}' >/dev/null 2>&1 &")
            else:
                print(f"  {C_RED}Video not found. Run render_question_validation.py --slot {slot} --all{C_RESET}")
            continue
        elif resp == 'n':
            notes = input("  Notes > ").strip()
            verdicts.setdefault(idx, {})["notes"] = notes
            print(f"  {C_DIM}Notes saved for Q{idx}{C_RESET}")
            continue
        elif resp in ('p', 'pass', ''):
            verdicts[idx] = {**verdicts.get(idx, {}), "verdict": "pass"}
            print(f"  {C_GREEN}✓ Q{idx} PASS{C_RESET}")
        elif resp in ('f', 'fail'):
            notes = input("  Failure reason > ").strip()
            verdicts[idx] = {"verdict": "fail", "notes": notes}
            print(f"  {C_RED}✗ Q{idx} FAIL: {notes}{C_RESET}")
        elif resp == 's':
            verdicts[idx] = {**verdicts.get(idx, {}), "verdict": "skip"}
            print(f"  {C_YELLOW}— Q{idx} SKIPPED{C_RESET}")
        else:
            print(f"  Unknown command: {resp}")
            continue

        idx += 1

    # Summary
    return print_summary(qa_data, verdicts)


def print_summary(qa_data: dict, verdicts: dict) -> dict:
    """Print review summary and return audit dict."""
    slot = qa_data.get("slot", "unknown")
    qa_pairs = qa_data.get("qa_pairs", [])
    total = len(qa_pairs)

    passes = sum(1 for v in verdicts.values() if v.get("verdict") == "pass")
    fails = sum(1 for v in verdicts.values() if v.get("verdict") == "fail")
    skips = sum(1 for v in verdicts.values() if v.get("verdict") == "skip")
    unreviewed = total - len(verdicts)

    print(f"\n{C_BOLD}{'='*72}{C_RESET}")
    print(f"{C_BOLD}REVIEW SUMMARY — {slot}{C_RESET}")
    print(f"{'='*72}")
    print(f"  {C_GREEN}Pass: {passes}{C_RESET}  |  {C_RED}Fail: {fails}{C_RESET}  |  "
          f"{C_YELLOW}Skip: {skips}{C_RESET}  |  Unreviewed: {unreviewed}")
    pct = (passes / total * 100) if total else 0
    print(f"  Pass rate: {pct:.0f}% ({passes}/{total})")

    if pct >= 80:
        print(f"\n  {C_GREEN}✓ READY FOR BATCH PROCESSING{C_RESET}")
    elif pct >= 50:
        print(f"\n  {C_YELLOW}⚠ NEEDS TARGETED FIXES{C_RESET}")
    else:
        print(f"\n  {C_RED}✗ SIGNIFICANT ISSUES — DO NOT BATCH{C_RESET}")

    # Show failures
    if fails:
        print(f"\n{C_RED}FAILURES:{C_RESET}")
        for idx, v in sorted(verdicts.items()):
            if v.get("verdict") == "fail":
                cat = qa_pairs[idx]["category"]
                print(f"  Q{idx} [{cat}]: {v.get('notes', 'no reason')}")

    # Per-category breakdown
    print(f"\n{C_CYAN}Per-Category:{C_RESET}")
    cat_stats = {}
    for idx, q in enumerate(qa_pairs):
        cat = q["category"]
        v = verdicts.get(idx, {}).get("verdict", "unreviewed")
        cat_stats.setdefault(cat, []).append(v)
    for cat in CATEGORY_ORDER:
        if cat in cat_stats:
            results = cat_stats[cat]
            p = results.count("pass")
            f = results.count("fail")
            icon = C_GREEN + "✓" if f == 0 else C_RED + "✗"
            print(f"  {icon} {cat:<18s}{C_RESET}  pass={p} fail={f} total={len(results)}")

    # Build audit record
    audit = {
        "slot": slot,
        "date": datetime.now().isoformat(),
        "total": total,
        "pass": passes,
        "fail": fails,
        "skip": skips,
        "pass_rate": round(pct, 1),
        "verdicts": {str(k): v for k, v in verdicts.items()},
        "version": qa_data.get("version", "unknown"),
    }
    return audit


def generate_report(qa_data: dict) -> dict:
    """Non-interactive report — auto-checks for common issues."""
    slot = qa_data.get("slot", "unknown")
    qa_pairs = qa_data.get("qa_pairs", [])
    total = len(qa_pairs)

    print(f"\n{C_BOLD}AUTO-REPORT — {slot}{C_RESET}")
    print(f"{'='*72}")

    issues = []
    verdicts = {}

    for idx, q in enumerate(qa_pairs):
        cat = q.get("category", "")
        debug = q.get("debug_info", {})
        cameras = q.get("requires_cameras", [])
        q_text = q.get("question_template", "")
        options = q.get("options", [])
        q_issues = []

        correct_answer = q.get("correct_answer") or q.get("answer", "")
        correct_idx = q.get("correct_answer_index")

        # Check 1: Empty answer
        if not correct_answer and correct_idx is None:
            q_issues.append("EMPTY ANSWER")

        # Check 2: Answer not in options
        if options and correct_answer and correct_answer not in options:
            # Also check by index
            if correct_idx is None or correct_idx >= len(options):
                q_issues.append(f"ANSWER NOT IN OPTIONS")

        # Check 3: Too many cameras for simple categories
        # best_camera inherently needs all cameras (it picks the best one)
        if cat == "perception" and len(cameras) > 10:
            q_issues.append(f"TOO MANY CAMERAS ({len(cameras)})")

        # Check 4: No debug_info
        if not debug:
            q_issues.append("NO DEBUG INFO")

        # Check 5: Missing frame range (for event-based categories)
        if cat == "temporal":
            for key in ["event_a", "event_b"]:
                ev = debug.get(key, {})
                if ev and not ev.get("frame_range"):
                    q_issues.append(f"MISSING FRAME RANGE in {key}")
        # Spatial entities use camera-center distance; frame_range is optional
        # (entity may only have actor_ids + camera, no specific frames)

        # Check 6: Temporal — weak connection
        if cat == "temporal":
            strength = debug.get("connection_strength", "")
            if strength == "weak":
                q_issues.append(f"WEAK CONNECTION (low quality)")

        # Check 7: No clip_file in events
        if cat in ("temporal", "spatial"):
            for key in ["event_a", "event_b", "entity_a", "entity_b"]:
                ev = debug.get(key, {})
                if ev and not ev.get("clip_file"):
                    q_issues.append(f"NO CLIP FILE in {key}")

        # Check 8: Video file exists for rendering
        video = VIDEO_DIR / f"{slot}_q{idx}_{cat}.mp4"
        has_video = video.exists()

        # Verdict
        if q_issues:
            verdict = "fail"
            icon = C_RED + "✗"
        else:
            verdict = "pass"
            icon = C_GREEN + "✓"

        verdicts[idx] = {"verdict": verdict, "notes": "; ".join(q_issues) if q_issues else ""}

        vid_icon = "📹" if has_video else "  "
        print(f"  {icon} Q{idx:<2d} [{cat:<16s}]{C_RESET} {vid_icon} "
              f"cams={len(cameras):<3d} "
              f"{'  '.join(q_issues) if q_issues else 'OK'}")
        issues.extend([(idx, cat, iss) for iss in q_issues])

    audit = print_summary(qa_data, verdicts)
    audit["mode"] = "auto-report"
    return audit


def main():
    parser = argparse.ArgumentParser(description="Interactive QA Review CLI")
    parser.add_argument("--slot", help="Slot name")
    parser.add_argument("--qa-file", help="QA JSON file")
    parser.add_argument("--natural", action="store_true",
                        help="Use naturalized version")
    parser.add_argument("--report", action="store_true",
                        help="Auto-report (non-interactive)")
    parser.add_argument("--save", action="store_true",
                        help="Save audit to file")
    args = parser.parse_args()

    # Resolve input file
    if args.qa_file:
        qa_path = Path(args.qa_file)
    elif args.slot:
        suffix = ".final.naturalized.json" if args.natural else ".final.raw.json"
        qa_path = QA_DIR / f"{args.slot}{suffix}"
        if not qa_path.exists() and args.natural:
            qa_path = QA_DIR / f"{args.slot}.final.raw.json"
            print(f"  (naturalized not found, using raw)")
    else:
        parser.error("Must provide --slot or --qa-file")

    if not qa_path.exists():
        print(f"ERROR: {qa_path} not found")
        return

    with open(qa_path) as f:
        qa_data = json.load(f)

    print(f"Loaded: {qa_path.name}")

    if args.report:
        audit = generate_report(qa_data)
    else:
        audit = interactive_review(qa_data)

    # Save audit
    if args.save or args.report:
        slot = qa_data.get("slot", "unknown")
        audit_path = AUDIT_DIR / f"{slot}.audit.json"
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"\n  Audit saved: {audit_path}")


if __name__ == "__main__":
    main()

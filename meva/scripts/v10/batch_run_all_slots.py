#!/usr/bin/env python3
"""
batch_run_all_slots.py — Generate raw QA for all (or a subset of) annotated slots.

Uses multiprocessing for speed.  Skips slots already done.  Resumes cleanly
after interruption.

Usage:
    # All 381 slots, 8 parallel workers
    python3 -m scripts.v10.batch_run_all_slots -v

    # Specific site only
    python3 -m scripts.v10.batch_run_all_slots --site school -v

    # Custom worker count
    python3 -m scripts.v10.batch_run_all_slots -w 4 -v

    # Dry-run: list slots that would be processed
    python3 -m scripts.v10.batch_run_all_slots --dry-run

    # Re-process even if output already exists
    python3 -m scripts.v10.batch_run_all_slots --overwrite
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from .run_pipeline import run_pipeline
    from .parse_annotations import find_clips_for_slot
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_pipeline import run_pipeline
    from parse_annotations import find_clips_for_slot

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent          # meva/
_DATA_DIR = _REPO_ROOT / "data"

_OUTPUT = Path(
    os.environ.get("MEVA_OUTPUT_DIR") or
    os.environ.get("OUTPUT_DIR") or
    "/nas/neurosymbolic/multi-cam-dataset/meva/data"
)
RAW_OUTPUT_DIR = _OUTPUT / "qa_pairs" / "raw"

SLOT_INDEX = _DATA_DIR / "slot_index.json"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _process_slot(args):
    """Worker function: generate QA for one slot and save to disk. Returns result dict."""
    slot, seed, verbose, output_dir = args
    start = time.time()
    out_dir = Path(output_dir)
    out_path = out_dir / f"{slot}.raw.json"
    try:
        result = run_pipeline(slot, seed=seed, verbose=False)
        elapsed = time.time() - start
        n_questions = len(result.get("qa_pairs", []))
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return {
            "slot": slot,
            "status": "ok",
            "questions": n_questions,
            "elapsed": round(elapsed, 1),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "slot": slot,
            "status": "error",
            "error": str(e),
            "elapsed": round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate raw QA for all annotated MEVA slots"
    )
    parser.add_argument("--site", choices=["admin", "bus", "hospital", "school"],
                        help="Only process slots for this site")
    parser.add_argument("--date", help="Only process slots for this date (e.g. 2018-03-11)")
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process slots even if output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="List slots without running")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Load slot index
    if not SLOT_INDEX.exists():
        print(f"ERROR: slot index not found: {SLOT_INDEX}")
        print("  Run: python3 -m scripts.v10.extract_logic_tuples --build-index")
        sys.exit(1)

    with open(SLOT_INDEX) as f:
        all_slots = list(json.load(f).keys())

    # Filter
    if args.site:
        all_slots = [s for s in all_slots if s.endswith(f".{args.site}")]
    if args.date:
        all_slots = [s for s in all_slots if s.startswith(args.date)]

    all_slots.sort()

    # Skip already-done
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        todo = [s for s in all_slots if not (RAW_OUTPUT_DIR / f"{s}.raw.json").exists()]
        done_count = len(all_slots) - len(todo)
    else:
        todo = list(all_slots)
        done_count = 0

    from collections import Counter
    site_counts = Counter(s.split(".")[-1] for s in todo)

    print(f"Slots total:   {len(all_slots)}")
    print(f"Already done:  {done_count}  (use --overwrite to redo)")
    print(f"To process:    {len(todo)}")
    print(f"By site:       {dict(site_counts)}")
    print(f"Workers:       {args.workers}")
    print(f"Output:        {RAW_OUTPUT_DIR}")
    print()

    if args.dry_run:
        for s in todo:
            print(f"  {s}")
        return

    if not todo:
        print("Nothing to do.")
        return

    # Run
    start_all = time.time()
    ok, errors, total = 0, 0, len(todo)
    error_list = []

    worker_args = [(slot, args.seed, args.verbose, str(RAW_OUTPUT_DIR)) for slot in todo]

    print(f"Starting at {datetime.now().strftime('%H:%M:%S')} ...\n")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_slot, wa): wa[0] for wa in worker_args}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res["status"] == "ok":
                ok += 1
                status_str = f"✓ {res['questions']:2d}q  {res['elapsed']:5.1f}s"
            else:
                errors += 1
                error_list.append(res)
                status_str = f"✗ ERROR: {res['error'][:60]}"

            pct = i / total * 100
            elapsed_total = time.time() - start_all
            eta = (elapsed_total / i) * (total - i) if i > 0 else 0
            print(f"  [{i:3d}/{total}  {pct:5.1f}%  ETA {eta/60:.1f}min]  "
                  f"{res['slot']:45s}  {status_str}")

    elapsed_total = time.time() - start_all
    print(f"\n{'='*70}")
    print(f"DONE  |  {ok} ok, {errors} errors  |  {elapsed_total/60:.1f} min total")
    print(f"Output: {RAW_OUTPUT_DIR}")

    if error_list:
        print(f"\nFailed slots ({errors}):")
        for r in error_list:
            print(f"  {r['slot']}: {r['error']}")

        # Write error log
        err_log = _OUTPUT / "qa_pairs" / "batch_errors.json"
        with open(err_log, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "errors": error_list,
            }, f, indent=2)
        print(f"\nError log: {err_log}")


if __name__ == "__main__":
    main()

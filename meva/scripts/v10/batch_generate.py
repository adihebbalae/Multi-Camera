#!/usr/bin/env python3
"""
batch_generate.py — Iterative generate→validate→fix→regenerate loop for MEVA QA.

Orchestrates the full pipeline:
  1. Generate raw QA (run_pipeline)
  2. Validate (validate_qa)
  3. Auto-fix structural issues (grammar, raw leaks, duplicates)
  4. Regenerate only if unfixable issues remain
  5. Repeat until quality score ≥ threshold or max iterations

Usage:
    # Run 5 iterations on one slot
    python3 -m scripts.v10.batch_generate --slot 2018-03-11.11-25.school --rounds 5 -v

    # Run 5 iterations on multiple slots
    python3 -m scripts.v10.batch_generate --slots 2018-03-11.11-25.school 2018-03-11.16-20.school --rounds 5

    # Run with custom quality threshold
    python3 -m scripts.v10.batch_generate --slot 2018-03-11.11-25.school --rounds 5 --min-score 90

Output:
    $MEVA_OUTPUT_DIR/qa_pairs/raw/{slot}.raw.json      — best raw QA
    $MEVA_OUTPUT_DIR/qa_pairs/batch_logs/{slot}.log.json — per-iteration scores
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from .run_pipeline import run_pipeline
    from .validate_qa import validate, Issue, KITWARE_ACTIVITY_IDS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_pipeline import run_pipeline
    from validate_qa import validate, Issue, KITWARE_ACTIVITY_IDS


# ============================================================================
# Constants
# ============================================================================

_OUTPUT = Path(os.environ.get("OUTPUT_DIR") or os.environ.get("MEVA_OUTPUT_DIR") or str(Path.home() / "data"))
RAW_OUTPUT_DIR = _OUTPUT / "qa_pairs" / "raw"
BATCH_LOG_DIR = _OUTPUT / "qa_pairs" / "batch_logs"

DEFAULT_ROUNDS = 5
DEFAULT_MIN_SCORE = 90
DEFAULT_SEED = 42


# ============================================================================
# Auto-Fix: Grammar
# ============================================================================

# Exceptions for a→an (words starting with vowel letter but consonant sound)
_AN_EXCEPTIONS = {"uniform", "university", "unique", "united", "union",
                  "european", "user", "useful", "usual", "utility", "one", "once"}


def fix_grammar(text: str) -> str:
    """Fix common grammar issues in generated text."""
    if not text:
        return text
    
    # Fix a [vowel] → an [vowel] (with exceptions)
    def _fix_article(m):
        word = m.group(1)
        if word.lower() in _AN_EXCEPTIONS:
            return m.group(0)
        return f"an {word}"
    
    text = re.sub(r'\ba ([aeiouAEIOU]\w*)\b', _fix_article, text)
    
    # Fix double articles
    text = re.sub(r'\b(a|an|the)\s+\1\b', r'\1', text, flags=re.I)
    
    return text


def fix_raw_leaks(text: str) -> str:
    """Remove raw pipeline tokens from naturalized text."""
    if not text:
        return text
    
    # Remove raw timestamps: "127.13-133.43s", "at 127s", "42 seconds"
    text = re.sub(r'\b\d+\.\d+-\d+\.\d+s\b', '', text)
    text = re.sub(r'\bat\s+\d+s\b', '', text)
    text = re.sub(r'\b\d{2,3}\s*seconds?\b', '', text)
    
    # Remove raw activity IDs
    for act in KITWARE_ACTIVITY_IDS:
        text = text.replace(act, act.split("_", 1)[-1].replace("_", " "))
    
    # Remove template artifacts
    text = re.sub(r'\{[a-z_]+\}', '', text)
    text = re.sub(r'__\w+__', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    return text


# ============================================================================
# Auto-Fix: Apply Fixes to QA Data
# ============================================================================

def auto_fix(data: Dict[str, Any], report: Dict[str, Any],
             verbose: bool = False) -> Dict[str, Any]:
    """
    Apply automated fixes to QA data based on validation report.
    
    Fixes (no LLM needed):
      - Grammar: a→an, double articles
      - Raw leaks: strip timestamps, activity IDs, template artifacts  
      - Generic descriptions: replace "someone" → "a person"
      - Duplicates: drop lower-quality duplicate
    
    Returns: fixed data (new copy, original unchanged)
    """
    fixed = copy.deepcopy(data)
    qa_pairs = fixed["qa_pairs"]
    
    issues_by_qid = {}
    for issue in report.get("issues", []):
        qid = issue["question_id"]
        if qid not in issues_by_qid:
            issues_by_qid[qid] = []
        issues_by_qid[qid].append(issue)
    
    fixes_applied = 0
    
    for q in qa_pairs:
        qid = q.get("question_id", "")
        q_issues = issues_by_qid.get(qid, [])
        
        for issue in q_issues:
            check = issue["check_name"]
            
            # --- Grammar fixes ---
            if check == "grammar":
                for field in ("question_template", "correct_answer"):
                    if field in q:
                        q[field] = fix_grammar(q[field])
                if "options" in q:
                    q["options"] = [fix_grammar(o) for o in q["options"]]
                if "naturalized_question" in q:
                    q["naturalized_question"] = fix_grammar(q["naturalized_question"])
                if "naturalized_options" in q:
                    q["naturalized_options"] = [fix_grammar(o) for o in q["naturalized_options"]]
                fixes_applied += 1
            
            # --- Raw leak fixes ---
            elif check == "raw_token_leak":
                for field in ("question_template", "correct_answer"):
                    if field in q:
                        q[field] = fix_raw_leaks(q[field])
                if "options" in q:
                    q["options"] = [fix_raw_leaks(o) for o in q["options"]]
                if "naturalized_question" in q:
                    q["naturalized_question"] = fix_raw_leaks(q["naturalized_question"])
                if "naturalized_options" in q:
                    q["naturalized_options"] = [fix_raw_leaks(o) for o in q["naturalized_options"]]
                fixes_applied += 1
            
            # --- Generic description fixes ---
            elif check == "generic_description":
                # Replace "someone" with "a person" in text fields
                for field in ("question_template", "correct_answer"):
                    if field in q and q[field]:
                        q[field] = re.sub(r'\bsomeone\b', 'a person', q[field], flags=re.I)
                        q[field] = re.sub(r'\bSomeone\b', 'A person', q[field])
                if "options" in q:
                    q["options"] = [
                        re.sub(r'\bsomeone\b', 'a person', o, flags=re.I) for o in q["options"]
                    ]
                fixes_applied += 1
    
    # --- Duplicate removal ---
    # Find pairs flagged as duplicates
    dup_qids_to_remove = set()
    for issue in report.get("issues", []):
        if issue["check_name"] in ("duplicate", "near_duplicate"):
            # The flagged question_id is one of the duplicates
            # Keep the first (lower index), remove the second if possible
            msg = issue.get("message", "")
            # Extract the other QID from the message
            other_match = re.search(r'of (final_\w+_\d+)', msg)
            if other_match:
                other_qid = other_match.group(1)
                # Remove the one with higher index (comes later)
                qid = issue["question_id"]
                # Compare by index number
                try:
                    idx_a = int(re.search(r'(\d+)$', qid).group(1))
                    idx_b = int(re.search(r'(\d+)$', other_qid).group(1))
                    dup_qids_to_remove.add(qid if idx_a > idx_b else other_qid)
                except (AttributeError, ValueError):
                    dup_qids_to_remove.add(qid)
    
    if dup_qids_to_remove:
        original_count = len(qa_pairs)
        qa_pairs = [q for q in qa_pairs if q.get("question_id") not in dup_qids_to_remove]
        fixed["qa_pairs"] = qa_pairs
        fixed["total_questions"] = len(qa_pairs)
        fixes_applied += len(dup_qids_to_remove)
        if verbose:
            print(f"    Removed {original_count - len(qa_pairs)} duplicate questions")
    
    if verbose:
        print(f"    Auto-fixes applied: {fixes_applied}")
    
    return fixed


# ============================================================================
# Iterative Loop
# ============================================================================

def run_iterative(
    slot: str,
    rounds: int = DEFAULT_ROUNDS,
    min_score: int = DEFAULT_MIN_SCORE,
    verbose: bool = False,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Run the iterative generate→validate→fix→regenerate loop.
    
    Args:
        slot: Slot name (e.g., "2018-03-11.11-25.school")
        rounds: Maximum number of iterations
        min_score: Target quality score (0-100)
        verbose: Print detailed output
        seed: Random seed
    
    Returns:
        {
            "slot": str,
            "final_score": int,
            "iterations": int,
            "scores": [int, ...],
            "issues_per_round": [int, ...],
            "best_round": int,
            "output_path": str,
        }
    """
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    iteration_log = {
        "slot": slot,
        "started_at": datetime.now().isoformat(),
        "rounds_requested": rounds,
        "min_score": min_score,
        "seed": seed,
        "iterations": [],
    }
    
    best_score = -1
    best_data = None
    best_round = 0
    scores = []
    issue_counts = []
    
    for round_num in range(1, rounds + 1):
        t0 = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{rounds} — Slot: {slot}")
            print(f"{'='*60}")
        
        # Step 1: Generate (vary seed per round for diversity)
        round_seed = seed + round_num - 1
        try:
            data = run_pipeline(slot, verbose=verbose, seed=round_seed,
                               require_mevid=False)
        except Exception as e:
            if verbose:
                print(f"  Generation FAILED: {e}")
            iteration_log["iterations"].append({
                "round": round_num,
                "status": "generation_failed",
                "error": str(e),
                "time_sec": round(time.time() - t0, 2),
            })
            scores.append(0)
            issue_counts.append(999)
            continue
        
        # Step 2: Validate
        report = validate(data, verbose=verbose)
        score = report["quality_score"]
        total_issues = report["total_issues"]
        
        if verbose:
            print(f"\n  Round {round_num} score: {score}/100 ({total_issues} issues)")
        
        # Step 3: Auto-fix
        if total_issues > 0:
            if verbose:
                print(f"  Applying auto-fixes...")
            data = auto_fix(data, report, verbose=verbose)
            
            # Re-validate after fixes
            report_after = validate(data, verbose=False)
            score_after = report_after["quality_score"]
            issues_after = report_after["total_issues"]
            
            if verbose:
                print(f"  Post-fix score: {score_after}/100 ({issues_after} issues)")
            
            score = score_after
            total_issues = issues_after
        
        scores.append(score)
        issue_counts.append(total_issues)
        
        # Track best
        if score > best_score:
            best_score = score
            best_data = data
            best_round = round_num
        
        # Log iteration
        elapsed = round(time.time() - t0, 2)
        iteration_log["iterations"].append({
            "round": round_num,
            "status": "ok",
            "seed": round_seed,
            "score_before_fix": report["quality_score"],
            "score_after_fix": score,
            "issues_before_fix": report["total_issues"],
            "issues_after_fix": total_issues,
            "total_questions": data.get("total_questions", 0),
            "time_sec": elapsed,
        })
        
        # Early exit if target reached
        if score >= min_score:
            if verbose:
                print(f"\n  Target score {min_score} reached in round {round_num}!")
            break
    
    # Save best result
    if best_data:
        out_path = RAW_OUTPUT_DIR / f"{slot}.raw.json"
        with open(out_path, "w") as f:
            json.dump(best_data, f, indent=2, default=str)
        if verbose:
            print(f"\n  Saved best result (round {best_round}, score {best_score}): {out_path}")
    
    # Save batch log
    iteration_log["completed_at"] = datetime.now().isoformat()
    iteration_log["best_round"] = best_round
    iteration_log["best_score"] = best_score
    iteration_log["scores"] = scores
    iteration_log["issue_counts"] = issue_counts
    
    log_path = BATCH_LOG_DIR / f"{slot}.log.json"
    with open(log_path, "w") as f:
        json.dump(iteration_log, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE — {slot}")
        print(f"{'='*60}")
        print(f"  Rounds: {len(scores)}")
        print(f"  Scores: {scores}")
        print(f"  Best: round {best_round} → {best_score}/100")
        print(f"  Log: {log_path}")
    
    return {
        "slot": slot,
        "final_score": best_score,
        "iterations": len(scores),
        "scores": scores,
        "issues_per_round": issue_counts,
        "best_round": best_round,
        "output_path": str(RAW_OUTPUT_DIR / f"{slot}.raw.json"),
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Iterative QA generation + validation loop"
    )
    parser.add_argument("--slot", help="Single slot to process")
    parser.add_argument("--slots", nargs="+", help="Multiple slots to process")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"Max iterations per slot (default: {DEFAULT_ROUNDS})")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                        help=f"Target quality score (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base random seed")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    slots = []
    if args.slot:
        slots.append(args.slot)
    if args.slots:
        slots.extend(args.slots)
    
    if not slots:
        parser.error("Provide --slot or --slots")
    
    results = []
    for slot in slots:
        result = run_iterative(
            slot,
            rounds=args.rounds,
            min_score=args.min_score,
            verbose=args.verbose,
            seed=args.seed,
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"BATCH SUMMARY")
    print(f"{'='*70}")
    print(f"{'Slot':45s} {'Rounds':>6s} {'Best':>5s} {'Final':>6s}")
    print("-" * 70)
    for r in results:
        print(f"{r['slot']:45s} {r['iterations']:6d} "
              f"{r['best_round']:5d} {r['final_score']:6d}/100")
    
    avg_score = sum(r["final_score"] for r in results) / len(results) if results else 0
    print(f"\nAverage final score: {avg_score:.1f}/100")


if __name__ == "__main__":
    main()

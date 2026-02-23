#!/usr/bin/env python3
"""
V8 Naturalization V3 — Single-Pass Naturalizer (V9 Pipeline)

Merges the 2-pass naturalize + grammar-check pipeline from V2 into a single
GPT call per question, halving API calls from 18 to 9 per slot.

Key changes from V2:
1. SYSTEM_PROMPT_UNIFIED replaces SYSTEM_PROMPT_V3 + GRAMMAR_CHECKER_PROMPT
2. Single _naturalize_and_polish_one() replaces _naturalize_one() + _grammar_check_one()
3. Unified temperature 0.8 (midpoint of V2's 0.7 and V3's 0.95)
4. V3's creative variety + temporal camera-ref stripping is the default behavior
5. Output suffix: .v8.natural.v4.json, version tag: v8_natural_v4

Pre-processing functions are imported from naturalize_v8_qa_v2 (no duplication).

Usage:
    # Pre-process only (free):
    python3 scripts/v8/naturalize_v8_qa_v3.py --input data/qa_pairs/SLOT.v8.json --preprocess-only

    # Full pipeline (pre-process + single-pass GPT):
    python3 scripts/v8/naturalize_v8_qa_v3.py --input data/qa_pairs/SLOT.v8.json

    # Dry-run (show what would be sent to GPT):
    python3 scripts/v8/naturalize_v8_qa_v3.py --input data/qa_pairs/SLOT.v8.json --dry-run
"""

import json
import time
import re
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Import pre-processing functions from V2 (no duplication)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from naturalize_v8_qa_v2 import (
    preprocess_all,
    simplify_description,
    CATEGORY_EXAMPLES_V3,
)

# ============================================================================
# Paths & Constants
# ============================================================================

QA_DIR = Path("/home/ah66742/data/qa_pairs")
LOG_DIR = Path("/home/ah66742/data/gpt_logs")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.8
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ============================================================================
# Unified System Prompt (merged naturalizer + grammar checker)
# ============================================================================

SYSTEM_PROMPT_UNIFIED = """\
You are a skilled question writer AND meticulous copy editor for a multi-camera \
surveillance video QA benchmark.

Your task: rewrite each template question into varied, natural English with \
perfect grammar, punctuation, and phrasing — all in a single step. Each \
question should sound like a DIFFERENT person wrote it.

IMPORTANT: You rewrite ONLY the question text and provide a reasoning sentence. \
You do NOT rewrite the answer options — those are deterministically generated \
and must not be changed.

## Creative Rephrasing Rules
1. VARY your phrasing — use different sentence openings, structures, and \
vocabulary each time. Avoid formulaic patterns like always starting with \
"In this scene..." or "Looking at the cameras..." or "Two events are observed..."
2. Preserve ALL factual content: person descriptions (clothing colors, carried \
objects), activities, spatial terms (near, moderate, far, meters), \
"simultaneously", and "cannot be determined"
3. NEVER include raw camera identifiers (e.g., G421, G330) in the question text. \
Camera references like "Camera G421" are acceptable ONLY in perception and \
re-identification questions where cameras are inherent to the question.
4. Do NOT add information not in the template
5. Do NOT reorder or modify the answer options in any way
6. For PERCEPTION questions with the format "What activity is occurring on camera X?" \
or "Which camera captures a ... event?", preserve this exact phrasing structure. \
Do NOT rephrase into "Can you identify..." or "Identify the camera that..." — \
keep the direct "What activity..." / "Which camera..." wording.

## Ontology Translation
6. Translate robotic activity labels and rigid bounding-box descriptions into \
natural human prose. For example, change "enters scene through structure" to \
"walks into the building", "person_opens_facility_door" to "opens a door", \
and smooth out awkward clothing lists into natural descriptions. \
Do NOT invent new details — only rephrase what is given.

## Grammar & Polish Rules (apply simultaneously)
7. Fix ALL grammatical errors, run-on sentences, and punctuation mistakes
8. Ensure proper capitalization and sentence structure
9. Eliminate awkward phrasing, redundancy, and unclear references
10. Be conservative with meaning — only fix form, never alter facts

## Reasoning
11. Add a 1-sentence "reasoning" for why the correct answer is right

Phrasing variety examples (do NOT copy these verbatim — invent your own):
- "A man in a gray hoodie appears near the entrance..."
- "Which of these events took place first?"
- "Based on the footage, what happened after..."
- "The woman carrying a red backpack was seen..."
- Direct question without preamble: "Who was spotted on more than one camera?"
- "After reviewing the video, can you determine..."
- Vary active/passive voice, question-first vs. description-first
- Sometimes be brief and direct, sometimes more descriptive

Output format — respond with ONLY a JSON object:
{
  "question": "The creatively rephrased and grammar-polished question",
  "reasoning": "Brief explanation of why the correct answer is right"
}
"""

# ============================================================================
# GPT Client
# ============================================================================

def _create_client():
    """Create OpenAI client."""
    import openai
    return openai.OpenAI()


# ============================================================================
# Post-processing helpers
# ============================================================================

_LETTER_PREFIX_RE = re.compile(r'^[A-Da-d]\)\s*')

def _strip_letter_prefixes(options: list) -> list:
    """Remove GPT-baked letter prefixes like 'A) ', 'B) ' from option text."""
    return [_LETTER_PREFIX_RE.sub('', opt) for opt in options]


# ============================================================================
# Single-Pass Naturalize + Polish
# ============================================================================

def _naturalize_and_polish_one(
    client,
    question: Dict,
    model: str,
    temperature: float,
) -> Optional[Dict]:
    """
    Single GPT call that rephrases the question text and generates reasoning.

    GPT only touches question + reasoning. Options are frozen (deterministic).
    """
    category = question["category"]
    template = question["question_template"]
    options = question["options"]
    verification = question.get("verification", {})

    # Select category-specific few-shot examples
    lookup_cat = question.get("subcategory", category)
    cat_info = CATEGORY_EXAMPLES_V3.get(
        lookup_cat, CATEGORY_EXAMPLES_V3.get(category, {})
    )
    hint = cat_info.get("hint", "Rephrase this question naturally with perfect grammar.")
    example_in = cat_info.get("example_input", "")
    example_out = cat_info.get("example_output", "")

    # Build user message
    user_message = f"Category: {category}\n{hint}\n\n"

    if example_in and example_out:
        user_message += (
            f"Example:\n  Input: {example_in}\n  Output: {example_out}\n\n"
        )

    user_message += (
        f"Now rewrite ONLY the question text (do NOT rewrite the options):\n\n"
        f"Template: {template}\n\nOptions (for context only — do NOT modify these):\n"
    )
    for i, opt in enumerate(options):
        user_message += f"  {chr(65 + i)}) {opt}\n"

    # Add verification context for reasoning
    if category == "temporal" and "gap_sec" in verification:
        user_message += (
            f"\nContext: The gap between events is {verification['gap_sec']}s.\n"
        )
    elif category == "spatial" and "distance_meters" in verification:
        user_message += (
            f"\nContext: Distance is {verification['distance_meters']}m.\n"
        )
    elif category == "best_camera":
        correct_cam = verification.get("correct_camera", "")
        entrance_time = verification.get("entrance_time_sec", 0)
        user_message += (
            f"\nContext: Camera transition logic — first entrance on {correct_cam} "
            f"at {entrance_time}s.\n"
        )

    user_message += (
        "\nRespond with ONLY a JSON object: "
        "{\"question\": \"...\", \"reasoning\": \"...\"}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_UNIFIED},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=400,
            )

            result = json.loads(response.choices[0].message.content)

            if "question" not in result:
                print(f"    WARNING: Missing 'question' field, retry {attempt + 1}")
                continue

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            # Options are frozen — pass through from pre-processed input unchanged
            return {
                "naturalized_question": result["question"],
                "naturalized_options": options,  # frozen, no GPT rewriting
                "reasoning": result.get("reasoning", ""),
                "usage": usage,
            }

        except json.JSONDecodeError:
            print(f"    WARNING: Invalid JSON response, retry {attempt + 1}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    WARNING: API error: {e}, retry {attempt + 1}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    return None


# ============================================================================
# Batch Processing
# ============================================================================

def naturalize_batch(
    input_data: Dict,
    model: str,
    temperature: float,
    verbose: bool = False,
) -> Dict:
    """Pre-process + single-pass GPT naturalize all QA pairs."""

    # Step 1: Pre-process (free) — uses V3 mode (strips camera refs from temporal Qs)
    preprocessed = preprocess_all(input_data, verbose=verbose, version="v3")

    # Step 2: GPT naturalize + polish (single pass)
    client = _create_client()
    qa_pairs = preprocessed["qa_pairs"]
    total = len(qa_pairs)

    print(f"\n  Naturalizing {total} questions with {model} "
          f"(single-pass, temp={temperature})...")

    naturalized_pairs = []
    total_tokens = 0
    failures = 0

    for i, q in enumerate(qa_pairs):
        if verbose:
            print(f"  [{i + 1}/{total}] {q['category']}: "
                  f"{q['question_template'][:60]}...")

        result = _naturalize_and_polish_one(client, q, model, temperature)

        nat_q = q.copy()

        if result is None:
            failures += 1
            nat_q["naturalized_question"] = q["question_template"]
            nat_q["naturalized_options"] = q["options"]
            nat_q["reasoning"] = ""
            nat_q["naturalization_failed"] = True
        else:
            nat_q["naturalized_question"] = result["naturalized_question"]
            nat_q["naturalized_options"] = result["naturalized_options"]
            nat_q["reasoning"] = result["reasoning"]
            total_tokens += result["usage"]["total_tokens"]

        naturalized_pairs.append(nat_q)

        if (i + 1) % 5 == 0:
            print(f"    Progress: {i + 1}/{total} ({total_tokens} tokens)")

    output = {
        "slot": input_data["slot"],
        "version": "final_naturalized",
        "generator": "naturalize_final.py",
        "preprocessor": "v3_preprocess",
        "model": model,
        "temperature": temperature,
        "total_tokens": total_tokens,
        "total_questions": len(naturalized_pairs),
        "failures": failures,
        "cameras": input_data.get("cameras", []),
        "mevid_supported": input_data.get("mevid_supported", False),
        "mevid_persons_in_slot": input_data.get("mevid_persons_in_slot", 0),
        "category_counts": input_data.get("category_counts", {}),
        "stats": input_data.get("stats", input_data.get("v8_stats", {})),
        "qa_pairs": naturalized_pairs,
    }

    return output


# ============================================================================
# Dry Run
# ============================================================================

def dry_run(input_data: Dict):
    """Show pre-processed templates and what would be sent to GPT."""
    preprocessed = preprocess_all(input_data, verbose=True, version="v3")
    qa_pairs = preprocessed["qa_pairs"]

    print(f"\n  === DRY RUN — {len(qa_pairs)} pre-processed questions ===\n")

    for q in qa_pairs:
        cat = q["category"]
        subcat = q.get("subcategory", "")
        original = q.get("original_template", q["question_template"])

        print(f"  [{cat}{' / ' + subcat if subcat else ''}]")

        if "original_template" in q:
            print(f"    BEFORE:  {original[:100]}...")
            print(f"    AFTER:   {q['question_template'][:100]}...")
        else:
            print(f"    (no change): {q['question_template'][:100]}...")

        # Show options comparison
        original_opts = q.get("original_options", q["options"])
        if original_opts != q["options"]:
            print(f"    OPTIONS BEFORE: {original_opts[0][:60]}...")
            print(f"    OPTIONS AFTER:  {q['options'][0][:60]}...")

        for i, opt in enumerate(q["options"]):
            marker = " *" if i == q.get("correct_answer_index") else ""
            print(f"      {chr(65 + i)}) {opt}{marker}")
        print()

    # Cost estimate — single pass, so 1x tokens
    est_tokens = len(qa_pairs) * 450  # slightly larger prompt (merged instructions)
    est_cost_mini = est_tokens * 0.4e-6
    est_cost_4o = est_tokens * 6e-6

    print(f"  === Cost Estimate (single-pass GPT) ===")
    print(f"  Questions: {len(qa_pairs)}")
    print(f"  API calls: {len(qa_pairs)} (single-pass, down from {len(qa_pairs) * 2} in V2)")
    print(f"  Est. tokens: ~{est_tokens}")
    print(f"  gpt-4o-mini: ~${est_cost_mini:.4f}")
    print(f"  gpt-4o:      ~${est_cost_4o:.4f}")
    print()
    print(f"  TIP: Use --preprocess-only to get the pre-processed output for free.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V8 Naturalization V3 — Single-Pass Naturalizer (V9 Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True,
        help="Path to V8 QA JSON file")
    parser.add_argument("--output", "-o",
        help="Output path (default: auto-generated .v9.naturalized.json)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
        help=f"GPT model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", "-t", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--preprocess-only", action="store_true",
        help="Only pre-process templates (no GPT call, FREE)")
    parser.add_argument("--dry-run", action="store_true",
        help="Show pre-processed templates and cost estimate")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true",
        help="Skip confirmation prompt")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        return

    print(f"Loading: {input_path} (single-pass, temp: {args.temperature})")
    with open(input_path) as f:
        input_data = json.load(f)

    total = len(input_data.get("qa_pairs", []))
    print(f"  Slot: {input_data.get('slot', 'N/A')}")
    print(f"  Version: {input_data.get('version', 'N/A')}")
    print(f"  Questions: {total}")

    # Mode 1: Pre-process only (free)
    if args.preprocess_only:
        result = preprocess_all(input_data, verbose=True, version="v3")

        out_path = args.output or str(input_path).replace(
            ".v8.json", ".v8.preprocessed.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n  Pre-processed output -> {out_path}")

        # Show before/after for each question
        print(f"\n  === Before -> After ===")
        for q in result["qa_pairs"]:
            if "original_template" in q:
                print(f"\n  [{q['category']}]")
                print(f"    BEFORE: {q['original_template'][:120]}")
                print(f"    AFTER:  {q['question_template'][:120]}")
                if "original_options" in q:
                    print(f"    OPTS BEFORE: {q['original_options'][0][:80]}")
                    print(f"    OPTS AFTER:  {q['options'][0][:80]}")

        return

    # Mode 2: Dry run
    if args.dry_run:
        dry_run(input_data)
        return

    # Mode 3: Full pipeline (pre-process + single-pass GPT)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        print("  TIP: Use --preprocess-only for free pre-processing without GPT.")
        return

    if not args.yes:
        print(f"\n  Will pre-process + naturalize {total} questions "
              f"with {args.model} (single-pass, temp={args.temperature})")
        print(f"  API calls: {total} (down from {total * 2} in V2)")
        resp = input("  Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            return

    result = naturalize_batch(
        input_data, args.model, args.temperature, verbose=args.verbose
    )

    print(f"\n  === Results ===")
    print(f"  Naturalized: {total - result['failures']}/{total}")
    print(f"  Failures: {result['failures']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  API calls: {total} (single-pass)")

    # Derive output path: .final.raw.json -> .final.naturalized.json
    if args.output:
        out_path = args.output
    elif ".final.raw.json" in str(input_path):
        out_path = str(input_path).replace(".final.raw.json", ".final.naturalized.json")
    elif ".v9.raw.json" in str(input_path):
        out_path = str(input_path).replace(".v9.raw.json", ".v9.naturalized.json")
    else:
        out_path = str(input_path).replace(".json", ".naturalized.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Output: {out_path}")

    # Save GPT log
    slot = input_data.get("slot", "unknown")
    log_dir = LOG_DIR / slot
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"naturalize_v8_v4_{args.model}.json"
    with open(log_path, "w") as f:
        json.dump({
            "model": args.model,
            "temperature": args.temperature,
            "total_tokens": result["total_tokens"],
            "questions_processed": total,
            "failures": result["failures"],
            "api_calls": total,
            "pipeline": "single-pass (v3)",
        }, f, indent=2)
    print(f"  Log: {log_path}")

    # Show samples
    print(f"\n  === Sample Naturalized Questions ===")
    seen_cats = set()
    for q in result["qa_pairs"]:
        if q["category"] in seen_cats:
            continue
        seen_cats.add(q["category"])

        print(f"\n  [{q['category']}]")
        orig = q.get("original_template", q.get("question_template", ""))
        print(f"    RAW:     {orig[:80]}...")
        print(f"    PREPROC: {q.get('question_template', '')[:80]}...")
        print(f"    NATURAL: {q.get('naturalized_question', '')[:80]}...")
        if q.get("reasoning"):
            print(f"    REASON:  {q['reasoning'][:80]}...")


if __name__ == "__main__":
    main()

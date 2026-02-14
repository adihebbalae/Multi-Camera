#!/usr/bin/env python3
"""
V8 GPT Naturalization Wrapper — Converts V8 template QA pairs into natural language.

V8 CHANGES from V7:
- New category examples: re_identification, scene_summary, attribute_verification
- Updated system prompt for MEVID person descriptions
- Updated cost estimates for 5 categories (~9 Qs/slot)
- Output version: v8_natural

Usage:
    python3 scripts/v8/naturalize_v8_qa.py --input data/qa_pairs/SLOT.v8.json --dry-run
    python3 scripts/v8/naturalize_v8_qa.py --input data/qa_pairs/SLOT.v8.json
"""

import json
import time
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# ============================================================================
# Paths & Constants
# ============================================================================

QA_DIR = Path("/home/ah66742/data/qa_pairs")
LOG_DIR = Path("/home/ah66742/data/gpt_logs")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# ============================================================================
# System Prompt — V8 with MEVID person descriptions
# ============================================================================

SYSTEM_PROMPT = """\
You are a question naturalizer for a multi-camera surveillance video understanding benchmark.

Your task is to rephrase template questions and options into fluent, natural English suitable
for a Video Question Answering (VQA) evaluation. The questions are about activities observed
across multiple synchronized surveillance cameras at the same location.

IMPORTANT for V8: Questions now include person descriptions derived from visual analysis
(e.g., "the person in a dark hooded jacket and dark pants"). Preserve these descriptions
naturally — they are key identifiers that replace debug markers from earlier versions.

Rules:
1. Rephrase the question to sound natural and conversational
2. Rephrase each option to sound natural, keeping the SAME meaning and order
3. REMOVE any remaining debug markers: strip actor IDs (#NNN), timestamps (@ Ns), frame refs
4. KEEP person descriptions (e.g., "the person in a blue jacket") — these are meaningful
5. Keep camera identifiers (e.g., "camera G299") — these are meaningful
6. Keep spatial terms (near, moderate, far, meters) unchanged
7. Keep "simultaneously" and "cannot be determined" as-is
8. Do NOT add information not present in the template
9. Do NOT reorder the options
10. For perception questions: keep camera IDs as-is in options
11. For re-identification questions: preserve person appearance descriptions precisely
12. For scene summary questions: keep activity type names and camera references
13. Add a brief 1-sentence "reasoning" explaining why the correct answer is right
    (based ONLY on the information given, not external knowledge)

Output format — respond with ONLY a JSON object:
{
  "question": "The naturalized question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "reasoning": "Brief explanation of why the answer is correct"
}
"""

# ============================================================================
# Category-specific prompt examples (few-shot)
# ============================================================================

CATEGORY_EXAMPLES = {
    "temporal": {
        "hint": "This is a temporal ordering question about which event happened first across different cameras. The entities may have appearance descriptions — preserve those naturally.",
        "example_input": 'In the scene, the person in a dark hooded jacket opening the trunk on camera G328 and the person in blue jeans exiting the vehicle on camera G421 -- which occurred first?',
        "example_output": '{"question": "Between the person in a dark hooded jacket opening the trunk on camera G328 and the person in blue jeans exiting the vehicle on camera G421, which event happened first?", "options": ["The person in the dark hooded jacket opening the trunk occurred first", "The person in blue jeans exiting the vehicle occurred first", "They occurred simultaneously", "Cannot be determined"], "reasoning": "The trunk opening event started earlier, with a 7-second gap before the vehicle exit."}',
    },
    "spatial": {
        "hint": "This is a spatial distance question about how far apart two people are. Person descriptions should be preserved naturally.",
        "example_input": 'In the scene, how far apart are the person in a blue jacket on camera G328 and the person in dark pants on camera G421?',
        "example_output": '{"question": "How far apart are the person in a blue jacket visible on camera G328 and the person in dark pants seen on camera G421?", "options": ["They are near each other (within a few meters)", "They are at a moderate distance (5-15 meters)", "They are far apart (more than 15 meters)", "They are at the same location"], "reasoning": "Based on their positions in the scene, these two individuals are approximately 8 meters apart."}',
    },
    "perception": {
        "hint": "This is a perception question about activities or attributes. For attribute verification, the question asks about a person's clothing or carried objects.",
        "example_input": 'A person is visible on camera G328. What color is their upper body clothing?',
        "example_output": '{"question": "What color clothing is the person on camera G328 wearing on their upper body?", "options": ["Blue", "Black", "Red", "White"], "reasoning": "Visual analysis of the person on camera G328 shows they are wearing blue upper body clothing."}',
    },
    "re_identification": {
        "hint": "This is a person re-identification question asking whether the same person appears on multiple cameras. Preserve appearance descriptions precisely — they are the key evidence.",
        "example_input": 'A person described as \"wearing a dark hooded jacket and dark pants\" appears on camera G328. Which other camera also shows this person?',
        "example_output": '{"question": "The person wearing a dark hooded jacket and dark pants is seen on camera G328. Which other camera also captures this same person?", "options": ["G421", "G299", "G330", "G336"], "reasoning": "The same individual in the dark hooded jacket and dark pants is identified on both camera G328 and camera G421 based on appearance matching."}',
    },
    "scene_summary": {
        "hint": "This is a scene-level summary question about overall activity patterns or camera utilization. Keep statistical terms and camera references.",
        "example_input": 'What best characterizes the overall scene activity across all cameras during this time period?',
        "example_output": '{"question": "How would you best characterize the overall activity patterns observed across all cameras during this time period?", "options": ["Predominantly pedestrian activity", "Mixed pedestrian and vehicle activity", "Predominantly vehicle activity", "Minimal observable activity"], "reasoning": "The majority of annotated events involve pedestrian activities across the cameras, with only a small fraction being vehicle-related."}',
    },
}


# ============================================================================
# GPT Client
# ============================================================================

def _create_client():
    """Create OpenAI client."""
    import openai
    return openai.OpenAI()


def _naturalize_one(client, question: Dict, model: str,
                    temperature: float) -> Optional[Dict]:
    """
    Send one question to GPT for naturalization.
    Returns {naturalized_question, naturalized_options, reasoning, usage} or None.
    """
    category = question["category"]
    template = question["question_template"]
    options = question["options"]
    verification = question.get("verification", {})
    
    # Use subcategory if available (e.g., attribute_verification)
    lookup_cat = question.get("subcategory", category)
    cat_info = CATEGORY_EXAMPLES.get(lookup_cat, CATEGORY_EXAMPLES.get(category, {}))
    hint = cat_info.get("hint", "Rephrase this question naturally.")
    example_in = cat_info.get("example_input", "")
    example_out = cat_info.get("example_output", "")

    user_message = f"""Category: {category}
{hint}

"""
    if example_in and example_out:
        user_message += f"""Example:
  Input: {example_in}
  Output: {example_out}

"""

    user_message += f"""Now naturalize this question:

Template: {template}

Options:
"""
    for i, opt in enumerate(options):
        user_message += f"  {chr(65+i)}) {opt}\n"
    
    # Add verification context for reasoning
    if category == "temporal" and "gap_sec" in verification:
        user_message += f"\nContext: The gap between events is {verification['gap_sec']}s.\n"
    elif category == "spatial" and "distance_meters" in verification:
        user_message += f"\nContext: Distance is {verification['distance_meters']}m.\n"
    elif category == "re_identification":
        user_message += f"\nContext: Person identified via MEVID cross-camera matching.\n"

    user_message += "\nRespond with ONLY the JSON object."

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=600,
            )

            result = json.loads(response.choices[0].message.content)

            if "question" not in result or "options" not in result:
                print(f"    WARNING: Missing fields in GPT response, retry {attempt+1}")
                continue

            if len(result["options"]) != len(options):
                print(f"    WARNING: Option count mismatch "
                      f"({len(result['options'])} vs {len(options)}), retry {attempt+1}")
                continue

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return {
                "naturalized_question": result["question"],
                "naturalized_options": result["options"],
                "reasoning": result.get("reasoning", ""),
                "usage": usage,
            }

        except json.JSONDecodeError:
            print(f"    WARNING: Invalid JSON response, retry {attempt+1}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"    WARNING: API error: {e}, retry {attempt+1}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    return None


# ============================================================================
# Batch Processing
# ============================================================================

def naturalize_batch(input_data: Dict, model: str, temperature: float,
                     verbose: bool = False) -> Dict:
    """Naturalize all QA pairs in the input data."""
    client = _create_client()
    qa_pairs = input_data["qa_pairs"]
    total = len(qa_pairs)

    print(f"\n  Naturalizing {total} questions with {model}...")

    naturalized_pairs = []
    total_tokens = 0
    failures = 0

    for i, q in enumerate(qa_pairs):
        if verbose:
            print(f"  [{i+1}/{total}] {q['category']}: {q['question_template'][:60]}...")

        result = _naturalize_one(client, q, model, temperature)

        if result is None:
            failures += 1
            nat_q = q.copy()
            nat_q["naturalized_question"] = q["question_template"]
            nat_q["naturalized_options"] = q["options"]
            nat_q["reasoning"] = ""
            nat_q["naturalization_failed"] = True
            naturalized_pairs.append(nat_q)
            continue

        nat_q = q.copy()
        nat_q["naturalized_question"] = result["naturalized_question"]
        nat_q["naturalized_options"] = result["naturalized_options"]
        nat_q["reasoning"] = result["reasoning"]
        nat_q["original_template"] = q["question_template"]
        nat_q["original_options"] = q["options"]
        naturalized_pairs.append(nat_q)

        total_tokens += result["usage"]["total_tokens"]

        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{total} ({total_tokens} tokens)")

    output = {
        "slot": input_data["slot"],
        "version": "v8_natural",
        "generator": "naturalize_v8_qa.py",
        "model": model,
        "temperature": temperature,
        "total_tokens": total_tokens,
        "total_questions": len(naturalized_pairs),
        "failures": failures,
        "cameras": input_data.get("cameras", []),
        "mevid_supported": input_data.get("mevid_supported", False),
        "mevid_persons_in_slot": input_data.get("mevid_persons_in_slot", 0),
        "category_counts": input_data.get("category_counts", {}),
        "v8_stats": input_data.get("v8_stats", {}),
        "qa_pairs": naturalized_pairs,
    }

    return output


# ============================================================================
# Strip Debug Markers
# ============================================================================

def strip_debug_markers(input_data: Dict) -> Dict:
    """Remove debug_info blocks from QA pairs (for VQA model input)."""
    output = input_data.copy()
    cleaned_pairs = []
    for q in input_data["qa_pairs"]:
        cleaned = {k: v for k, v in q.items() if k != "debug_info"}
        cleaned_pairs.append(cleaned)
    output["qa_pairs"] = cleaned_pairs
    return output


# ============================================================================
# Dry Run
# ============================================================================

def dry_run(input_data: Dict):
    """Show what would be sent to GPT without making API calls."""
    qa_pairs = input_data["qa_pairs"]
    
    print(f"\n  === DRY RUN — {len(qa_pairs)} questions ===\n")

    categories = {}
    for q in qa_pairs:
        categories.setdefault(q["category"], []).append(q)

    for cat, qs in sorted(categories.items()):
        q = qs[0]
        cat_info = CATEGORY_EXAMPLES.get(q.get("subcategory", cat), 
                                          CATEGORY_EXAMPLES.get(cat, {}))
        
        print(f"  [{cat}] ({len(qs)} questions)")
        print(f"    TEMPLATE: {q['question_template'][:100]}...")
        for i, opt in enumerate(q["options"]):
            marker = " *" if i == q.get("correct_answer_index") else ""
            print(f"      {chr(65+i)}) {opt}{marker}")
        
        if q.get("subcategory"):
            print(f"    SUBCATEGORY: {q['subcategory']}")
        print(f"    DIFFICULTY: {q.get('difficulty', 'N/A')}")
        print(f"    HINT: {cat_info.get('hint', 'N/A')[:80]}...")
        print()

    # Cost estimate (V8: ~9 Qs/slot)
    est_tokens = len(qa_pairs) * 350  # slightly more with person descriptions
    est_cost_mini = est_tokens * 0.4e-6
    est_cost_4o = est_tokens * 6e-6

    print(f"  === Cost Estimate ===")
    print(f"  Questions: {len(qa_pairs)}")
    print(f"  Est. tokens: ~{est_tokens}")
    print(f"  gpt-4o-mini: ~${est_cost_mini:.4f}")
    print(f"  gpt-4o:      ~${est_cost_4o:.4f}")
    
    # V8 stats
    mevid = input_data.get("mevid_persons_in_slot", 0)
    print(f"\n  === V8 Stats ===")
    print(f"  MEVID persons: {mevid}")
    print(f"  Categories: {sorted(categories.keys())}")
    v8_stats = input_data.get("v8_stats", {})
    if v8_stats:
        print(f"  MEVID descriptions: {v8_stats.get('entities_with_mevid_descriptions', 0)}")
        print(f"  Attr verification Qs: {v8_stats.get('attribute_verification_questions', 0)}")
        print(f"  Re-ID Qs: {v8_stats.get('reid_questions', 0)}")
        print(f"  Scene summary Qs: {v8_stats.get('scene_summary_questions', 0)}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V8 GPT Naturalization Wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True,
        help="Path to V8 QA JSON file")
    parser.add_argument("--output", "-o",
        help="Output path (default: {input}.natural.json)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
        help=f"GPT model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", "-t", type=float,
        default=DEFAULT_TEMPERATURE)
    parser.add_argument("--dry-run", action="store_true",
        help="Show prompts without making API calls")
    parser.add_argument("--strip-debug", action="store_true",
        help="Only strip debug_info blocks (no GPT call, free)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true",
        help="Skip confirmation prompt")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        return

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        input_data = json.load(f)

    total = len(input_data.get("qa_pairs", []))
    print(f"  Slot: {input_data.get('slot', 'N/A')}")
    print(f"  Version: {input_data.get('version', 'N/A')}")
    print(f"  Questions: {total}")
    print(f"  MEVID supported: {input_data.get('mevid_supported', False)}")

    if args.strip_debug:
        result = strip_debug_markers(input_data)
        out_path = args.output or str(input_path).replace(".v8.json", ".v8.clean.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Stripped debug_info → {out_path}")
        return

    if args.dry_run:
        dry_run(input_data)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return

    if not args.yes:
        print(f"\n  Will send {total} questions to {args.model}")
        resp = input("  Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            return

    result = naturalize_batch(input_data, args.model, args.temperature,
                              verbose=args.verbose)

    print(f"\n  === Results ===")
    print(f"  Naturalized: {total - result['failures']}/{total}")
    print(f"  Failures: {result['failures']}")
    print(f"  Total tokens: {result['total_tokens']}")

    out_path = args.output or str(input_path).replace(".v8.json", ".v8.natural.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Output: {out_path}")

    # Save GPT log
    slot = input_data.get("slot", "unknown")
    log_dir = LOG_DIR / slot
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"naturalize_v8_{args.model}.json"
    with open(log_path, "w") as f:
        json.dump({
            "model": args.model,
            "temperature": args.temperature,
            "total_tokens": result["total_tokens"],
            "questions_processed": total,
            "failures": result["failures"],
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
        print(f"    TEMPLATE:  {q.get('original_template', q['question_template'])[:80]}...")
        print(f"    NATURAL:   {q['naturalized_question'][:80]}...")
        if q.get("reasoning"):
            print(f"    REASONING: {q['reasoning'][:80]}...")


if __name__ == "__main__":
    main()

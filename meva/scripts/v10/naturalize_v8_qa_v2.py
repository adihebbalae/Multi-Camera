#!/usr/bin/env python3
"""
V8 Naturalization V2 — Pre-processes and naturalizes V8 template QA pairs.

Key improvements over V1:
1. Pre-processing layer (FREE, no API call):
   - Description simplification: "blue upper body garment" → "blue top"
   - Camera reference removal from temporal/spatial question text
   - Temporal anchors for disambiguation ("about 6 seconds in")
   - Activity de-duplication ("enters scene enters scene" → "enters scene")
   - Grammar fixes: capitalization, mid-sentence periods
   - Event descriptions reconstructed from structured verification data

2. Updated GPT prompts (optional GPT naturalization):
   - Better few-shot examples reflecting cleaned templates
   - Category-specific format guidance
   - Post-processing validation

3. Three modes:
   --preprocess-only: Just pre-process templates (free, instant)
   --dry-run: Show what would be sent to GPT
   (default): Pre-process + GPT naturalization

Usage:
    # Pre-process only (free):
    python3 scripts/v8/naturalize_v8_qa_v2.py --input data/qa_pairs/SLOT.v8.json --preprocess-only

    # Full pipeline (pre-process + GPT):
    python3 scripts/v8/naturalize_v8_qa_v2.py --input data/qa_pairs/SLOT.v8.json

    # Dry-run:
    python3 scripts/v8/naturalize_v8_qa_v2.py --input data/qa_pairs/SLOT.v8.json --dry-run
"""

import json
import re
import time
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ============================================================================
# Paths & Constants
# ============================================================================

QA_DIR = Path("/home/ah66742/data/qa_pairs")
LOG_DIR = Path("/home/ah66742/data/gpt_logs")

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CLIP_DURATION = 300.0  # 5-minute clips


# ============================================================================
# Description Simplification (standalone, mirrors person_descriptions.py)
# ============================================================================

# Garment type simplifications
_GARMENT_SUBS = [
    (r"upper body garment with a hood", "hoodie"),
    (r"upper body garment", "top"),
    (r"lower body clothing", "pants"),
    (r"lower body garment", "pants"),
    (r"lower body pants", "pants"),
    (r"lower body shorts", "shorts"),
    (r"hooded jacket", "hoodie"),
]

# Phrasing cleanups
_PHRASING_SUBS = [
    (r",\s*and they are\s+", ", "),
    (r",\s*and is\s+", ", "),
    (r"\.\s*The person appears to be [^\.]+\.?", ""),
    (r",?\s*and they may be carrying personal belongings", ""),
    (r",?\s*and they appear to have [^,\.]+", ""),
    (r"\.\s*Their hair appears short[^\.]*\.?", ""),
]

# Posture/action context to strip (not useful for identification)
_STRIP_PATTERNS = [
    r",?\s*sitting on a chair[^,\.]*",
    r",?\s*with their back turned[^,\.]*",
    r",?\s*while ascending a staircase[^,\.]*",
    r",?\s*while holding a clipboard or some papers",
    r",?\s*sitting\b[^,\.]*",
    r",?\s*items on the table[^,\.]*",
    r",?\s*appears to be looking[^,\.]*",
]


def simplify_description(desc: str) -> str:
    """
    Simplify verbose GPT descriptions into natural short form.

    "wearing a blue upper body garment and blue lower body clothing,
     with a black hoodie featuring a graphic design on the back."
    →
    "wearing a blue top and blue pants, with a black hoodie featuring
     a graphic design on the back"
    """
    if not desc:
        return desc

    desc = desc.rstrip(". ")

    for pattern, replacement in _GARMENT_SUBS:
        desc = re.sub(pattern, replacement, desc, flags=re.IGNORECASE)

    for pattern, replacement in _PHRASING_SUBS:
        desc = re.sub(pattern, replacement, desc, flags=re.IGNORECASE)

    for pattern in _STRIP_PATTERNS:
        desc = re.sub(pattern, "", desc, flags=re.IGNORECASE)

    # Clean up orphaned commas / double spaces
    desc = re.sub(r",\s*,", ",", desc)
    desc = re.sub(r"\s{2,}", " ", desc)
    desc = desc.strip(", ")
    desc = desc.rstrip(".")

    return desc


# ============================================================================
# Activity Humanization (standalone, mirrors activity_hierarchy.py)
# ============================================================================

_GERUND_MAP = {
    "opens": "opening", "closes": "closing", "enters": "entering",
    "exits": "exiting", "reads": "reading", "carries": "carrying",
    "picks": "picking", "puts": "putting", "sets": "setting",
    "rides": "riding", "loads": "loading", "unloads": "unloading",
    "talks": "talking", "stands": "standing", "walks": "walking",
    "runs": "running", "sits": "sitting", "texts": "texting",
    "pulls": "pulling", "pushes": "pushing", "interacts": "interacting",
    "drops": "dropping", "embraces": "embracing", "uses": "using",
    "makes": "making", "steals": "stealing", "starts": "starting",
    "stops": "stopping", "turns": "turning",
}

_NO_ARTICLE = frozenset({
    "up", "down", "on", "off", "out", "in", "to", "from",
    "through", "with", "around", "right", "left", "a",
    "an", "the", "into", "onto", "over", "away",
})


def _humanize_activity(activity: str) -> str:
    """person_opens_facility_door → opens facility door"""
    for prefix in ("person_", "vehicle_", "hand_"):
        if activity.startswith(prefix):
            activity = activity[len(prefix):]
            break
    return activity.replace("_", " ")


def _humanize_gerund(activity: str) -> str:
    """person_opens_facility_door → Opening a facility door"""
    base = _humanize_activity(activity)
    words = base.split()
    if not words:
        return base.capitalize()

    first = words[0]
    gerund = _GERUND_MAP.get(first, first + "ing")
    rest = " ".join(words[1:])

    if rest:
        first_rest = rest.split()[0]
        if first_rest not in _NO_ARTICLE:
            rest = "a " + rest

    result = f"{gerund} {rest}".strip() if rest else gerund

    # Fix missing articles in common phrases
    result = re.sub(r'\bwith person\b', 'with a person', result)
    result = re.sub(r'\bto person\b', 'to a person', result)
    result = re.sub(r'\bthrough structure\b', 'through a structure', result)
    result = re.sub(r'\bthrough door\b', 'through a door', result)
    result = re.sub(r'\bin vehicle\b', 'in a vehicle', result)
    result = re.sub(r'\bwith object\b', 'with an object', result)
    result = re.sub(r'\bon phone\b', 'on a phone', result)

    return result.capitalize()


def _short_activity_label(activity: str) -> str:
    """Short gerund label for options: person_opens_facility_door → opening a facility door"""
    return _humanize_gerund(activity).lower()


# ============================================================================
# Temporal Anchoring
# ============================================================================

def _temporal_anchor(sec: float, clip_duration: float = CLIP_DURATION) -> str:
    """Generate a temporal anchor for event disambiguation within a 5-min clip."""
    if sec < 5:
        return "at the very start"
    elif sec < 30:
        return f"about {int(round(sec))} seconds in"
    elif sec < 60:
        return f"roughly {int(round(sec))} seconds in"
    elif sec < 120:
        return f"around the {int(round(sec))}-second mark"
    elif sec < 180:
        return f"around {int(round(sec / 10)) * 10} seconds in"
    elif sec < 250:
        return f"around {int(round(sec / 10)) * 10} seconds in"
    else:
        return "near the end of the clip"


# ============================================================================
# Appearance Detection
# ============================================================================

_APPEARANCE_WORDS = frozenset({
    "wearing", "shirt", "top", "pants", "jacket", "hoodie", "backpack",
    "blue", "red", "green", "gray", "grey", "black", "white", "dark", "light",
    "brown", "orange", "yellow", "purple", "pink", "navy", "beige", "tan",
    "khaki", "camouflage", "striped", "patterned", "logo", "graphic",
    "garment", "clothing", "jeans", "shorts", "dress", "skirt", "hat", "cap",
    "glasses", "sunglasses", "scarf", "vest", "coat", "boots", "sneakers",
    "bottle", "bag", "purse", "suitcase", "umbrella",
})


def _has_appearance_info(desc: str) -> bool:
    """Check if description contains visual appearance info (clothing/colors)."""
    if not desc:
        return False
    words = set(desc.lower().split())
    return len(words & _APPEARANCE_WORDS) >= 2


# ============================================================================
# Person Description Extraction
# ============================================================================

def _extract_person_desc(entity_description: str, activity: str = "") -> str:
    """
    Extract just the person appearance, stripping any embedded activity text.

    Input: "the person wearing a gray upper body garment... enters scene"
    Output: "a person wearing a gray top and green pants, carrying a black backpack"

    Input: "a person interacts with person" (fallback, no appearance)
    Output: "a person"
    """
    if not entity_description:
        return "a person"

    desc = entity_description.strip()

    # Check if this is a real appearance description vs. activity fallback
    if not _has_appearance_info(desc):
        return "a person"

    # Remove embedded activity text after the description
    # Pattern: description ends with period, then activity follows
    # e.g. "...backpack. enters scene through structure on camera G421"
    activity_verbs = {
        "enters", "exits", "opens", "closes", "picks", "puts", "carries",
        "talks", "sits", "stands", "reads", "texts", "interacts", "embraces",
        "rides", "loads", "unloads", "transfers", "drops", "pulls", "pushes",
        "walks", "runs", "stops", "turns", "starts",
    }

    # Split on periods; keep appearance parts, drop activity parts
    parts = desc.split(".")
    appearance_parts = []
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue
        first_word = part_stripped.split()[0].lower() if part_stripped.split() else ""
        if first_word in activity_verbs:
            break  # Rest is activity description
        appearance_parts.append(part_stripped)

    if appearance_parts:
        desc = ". ".join(appearance_parts)

    # Strip "on camera GXXX"
    desc = re.sub(r"\s*on camera G\d+", "", desc)

    # Simplify garment terminology and strip clutter
    desc = simplify_description(desc)

    # Normalize prefix: "the person" → "a person"
    for prefix in ["The person ", "the person "]:
        if desc.startswith(prefix):
            desc = "a person " + desc[len(prefix):]
            break

    # Ensure starts with "a person" or "A person"
    if not desc.lower().startswith(("a ", "the ")):
        desc = "a person " + desc

    return desc


# ============================================================================
# Per-Category Pre-processing
# ============================================================================

def _preprocess_temporal(qa: Dict, strip_camera_refs: bool = False) -> Dict:
    """
    Reconstruct temporal question from structured verification data.

    Fixes: camera refs in question, description verbosity, activity doubling,
    temporal ambiguity (adds timestamp anchors), capitalization.

    If strip_camera_refs=True (V3 mode), camera IDs are removed from the question
    text — person descriptions (clothing colors, carried objects) serve as the
    primary disambiguator instead. Camera refs are still kept in answer options.
    """
    result = {k: v for k, v in qa.items()}
    v = qa.get("verification", {})
    d = qa.get("debug_info", {})

    if not v or "event_a" not in v or "event_b" not in v:
        return result

    ev_a = v["event_a"]
    ev_b = v["event_b"]
    da = d.get("event_a", {})
    db_ = d.get("event_b", {})

    # Get person descriptions from debug_info (enriched with MEVID)
    desc_a = _extract_person_desc(
        da.get("entity_description", ""), ev_a.get("activity", "")
    )
    desc_b = _extract_person_desc(
        db_.get("entity_description", ""), ev_b.get("activity", "")
    )

    # Get activities in gerund form
    act_a = _short_activity_label(ev_a.get("activity", ""))
    act_b = _short_activity_label(ev_b.get("activity", ""))

    cam_a = ev_a.get("camera", da.get("camera", ""))
    cam_b = ev_b.get("camera", db_.get("camera", ""))

    # Build clean event descriptions
    # V3 mode: strip camera refs from question text (person descriptions disambiguate)
    # V2 mode: keep camera refs in question text
    include_cam = not strip_camera_refs

    def _fmt_event(desc, act, cam, with_camera=True):
        d = desc.rstrip('.')
        cam_ref = f" on camera {cam}" if cam and with_camera else ""
        if d.lower() in ("a person", "someone"):
            return f"{d} {act}{cam_ref}"
        return f"{d}, {act}{cam_ref}"

    clean_a = _fmt_event(desc_a, act_a, cam_a, with_camera=include_cam)
    clean_b = _fmt_event(desc_b, act_b, cam_b, with_camera=include_cam)

    # Capitalize first letter
    clean_a = clean_a[0].upper() + clean_a[1:]
    clean_b = clean_b[0].upper() + clean_b[1:]

    # Determine mention order (preserve original answer randomization)
    ci = qa["correct_answer_index"]
    if ci == 0:
        # Event A mentioned first AND occurred first → option 0 correct
        desc_1, desc_2 = clean_a, clean_b
        ev_1, ev_2 = ev_a, ev_b
    else:
        # Event B mentioned first but Event A occurred first → option 1 correct
        desc_1, desc_2 = clean_b, clean_a
        ev_1, ev_2 = ev_b, ev_a

    # Build clean question
    question = (
        f"Consider two events in this multi-camera scene: "
        f"(1) {desc_1}. (2) {desc_2}. "
        f"Which event occurred first?"
    )

    # Build options using camera + activity labels
    act_1 = _humanize_gerund(ev_1.get("activity", "event"))
    act_2 = _humanize_gerund(ev_2.get("activity", "event"))
    cam_1 = ev_1.get("camera", "")
    cam_2 = ev_2.get("camera", "")

    # Use person descriptions to disambiguate events (no camera IDs in options)
    # desc_1 / desc_2 are person appearance descriptions + activity
    def _option_label(desc, act):
        """Build a concise option label from event description."""
        d = desc.strip().rstrip('.')
        if d.startswith("A "):
            d = "The " + d[2:]
        elif d.startswith("a "):
            d = "The " + d[2:]
        elif d.startswith("Someone "):
            d = d  # keep as-is
        return d

    opt_1 = _option_label(desc_1, act_1)
    opt_2 = _option_label(desc_2, act_2)

    options = [
        f"{opt_1} occurred first",
        f"{opt_2} occurred first",
        "They occurred simultaneously",
        "Cannot be determined",
    ]

    # Fix article agreement (a → an before vowels)
    question = re.sub(r'\ba ([aeiouAEIOU])', r'an \1', question)
    options = [re.sub(r'\ba ([aeiouAEIOU])', r'an \1', o) for o in options]

    result["question_template"] = question
    result["options"] = options
    result["correct_answer"] = options[ci]

    return result


def _preprocess_spatial(qa: Dict) -> Dict:
    """Clean spatial question: simplify descriptions, remove camera refs."""
    result = {k: v for k, v in qa.items()}
    v = qa.get("verification", {})

    if not v:
        return result

    # Get and simplify entity descriptions
    desc_a = simplify_description(v.get("entity_a_desc", "a person"))
    desc_b = simplify_description(v.get("entity_b_desc", "another person"))

    # Normalize prefixes
    for prefix in ["The person ", "the person "]:
        if desc_a.startswith(prefix):
            desc_a = "the person " + desc_a[len(prefix):]
        if desc_b.startswith(prefix):
            desc_b = "the person " + desc_b[len(prefix):]

    desc_a = desc_a.rstrip(".")
    desc_b = desc_b.rstrip(".")

    # Build question without camera reference
    question = f"How close are {desc_a} and {desc_b} in the scene?"
    question = question[0].upper() + question[1:]
    # Fix article agreement (a → an before vowels)
    question = re.sub(r'\ba ([aeiouAEIOU])', r'an \1', question)

    result["question_template"] = question
    return result


def _preprocess_perception(qa: Dict) -> Dict:
    """
    Clean perception question: simplify any embedded descriptions, fix grammar.
    Camera refs kept since they're inherent to perception question types.
    """
    result = {k: v for k, v in qa.items()}
    v = qa.get("verification", {})
    q_type = v.get("question_type", "")

    template = qa.get("question_template", "")

    if q_type == "attribute_verification":
        # "A person is visible on camera G423. What color are they wearing..."
        # Simplify any embedded person description
        person_desc = v.get("person_description", "")
        if person_desc:
            simplified = simplify_description(person_desc).rstrip(".")
            # Template is already clean for this type — just ensure capitalization
    elif q_type == "which_camera":
        # "Which camera captures a carries heavy object event?"
        # Make activity name more natural
        alias = v.get("activity_alias", "")
        if alias:
            gerund = _humanize_gerund(v.get("activity", alias))
            template = f"Which camera captures {gerund.lower().lstrip('a ')}?"
            template = template[0].upper() + template[1:]

    # General cleanup
    template = simplify_description(template)
    if template:
        template = template[0].upper() + template[1:]
        # Re-add question mark if simplification stripped it
        if not template.endswith("?"):
            template += "?"

    result["question_template"] = template
    return result


def _preprocess_reid(qa: Dict) -> Dict:
    """
    Clean re-ID question: simplify descriptions, fix grammar.
    Camera refs kept since they're fundamental to re-identification.
    """
    result = {k: v for k, v in qa.items()}
    v = qa.get("verification", {})
    d = qa.get("debug_info", {})

    if not v:
        return result

    q_type = v.get("question_type", "")
    desc = simplify_description(v.get("person_description", "a person"))
    desc = desc.rstrip(".")

    # Normalize prefix
    for prefix in ["the person ", "The person "]:
        if desc.startswith(prefix):
            desc = "a person " + desc[len(prefix):]
            break

    if q_type == "which_camera_reid":
        cam = v.get("source_camera", "")
        question = (
            f"On camera {cam}, {desc} is visible. "
            f"Which other camera also shows this same person?"
        )
    elif q_type == "same_person_confirmation":
        cam_a = v.get("camera_a", d.get("camera_a", ""))
        cam_b = v.get("camera_b", d.get("camera_b", ""))
        capitalized_desc = desc[0].upper() + desc[1:] if desc else desc
        question = (
            f"{capitalized_desc} is observed on camera {cam_a}. "
            f"Is this the same person visible on camera {cam_b}?"
        )
    else:
        # Fallback: simplify in place
        question = simplify_description(qa.get("question_template", ""))
        if question:
            question = question[0].upper() + question[1:]
        if question and not question.endswith("?"):
            question += "?"

    result["question_template"] = question
    return result


def _preprocess_scene_summary(qa: Dict) -> Dict:
    """Clean scene summary question: minimal changes."""
    result = {k: v for k, v in qa.items()}
    template = qa.get("question_template", "")
    if template:
        template = template[0].upper() + template[1:]
    result["question_template"] = template
    return result


# ============================================================================
# Main Pre-processing Pipeline
# ============================================================================

def preprocess_all(input_data: Dict, verbose: bool = False,
                   version: str = "v2") -> Dict:
    """
    Pre-process all QA pairs: simplify descriptions, remove camera refs,
    add temporal anchors, fix grammar. FREE (no API call).

    version='v3' strips camera refs from temporal question text.
    """
    output = {k: v for k, v in input_data.items() if k != "qa_pairs"}
    output["version"] = "v8_preprocessed"
    output["preprocessor"] = "naturalize_v8_qa_v2.py"

    preprocessed = []
    changes = {"temporal": 0, "spatial": 0, "perception": 0,
               "re_identification": 0, "scene_summary": 0}

    for qa in input_data.get("qa_pairs", []):
        cat = qa.get("category", "")

        if cat == "temporal":
            cleaned = _preprocess_temporal(qa, strip_camera_refs=(version == "v3"))
        elif cat == "spatial":
            cleaned = _preprocess_spatial(qa)
        elif cat == "perception":
            cleaned = _preprocess_perception(qa)
        elif cat == "re_identification":
            cleaned = _preprocess_reid(qa)
        elif cat == "scene_summary":
            cleaned = _preprocess_scene_summary(qa)
        else:
            cleaned = qa.copy()

        # Track original template for comparison
        if cleaned.get("question_template") != qa.get("question_template"):
            cleaned["original_template"] = qa["question_template"]
            changes[cat] = changes.get(cat, 0) + 1
        if cleaned.get("options") != qa.get("options"):
            cleaned["original_options"] = qa["options"]

        preprocessed.append(cleaned)

    # Global fix: article agreement (a → an before vowels) in all text fields
    for qa in preprocessed:
        if "question_template" in qa:
            qa["question_template"] = re.sub(
                r'\ba ([aeiouAEIOU])', r'an \1', qa["question_template"]
            )
        if "options" in qa:
            qa["options"] = [
                re.sub(r'\ba ([aeiouAEIOU])', r'an \1', str(o))
                for o in qa["options"]
            ]
        if "correct_answer" in qa:
            qa["correct_answer"] = re.sub(
                r'\ba ([aeiouAEIOU])', r'an \1', str(qa["correct_answer"])
            )

    output["qa_pairs"] = preprocessed

    if verbose:
        total_changed = sum(changes.values())
        print(f"\n  Pre-processing: {total_changed}/{len(preprocessed)} questions modified")
        for cat, cnt in sorted(changes.items()):
            if cnt > 0:
                print(f"    {cat}: {cnt} changed")

    return output


# ============================================================================
# Updated GPT System Prompt (V2)
# ============================================================================

SYSTEM_PROMPT_V2 = """\
You are a question naturalizer for a multi-camera surveillance video QA benchmark.

Your task is to polish pre-processed template questions into fluent, natural English
suitable for a Video Question Answering (VQA) evaluation. The templates have already
been cleaned up — your job is to make them sound conversational while preserving all
factual content.

Rules:
1. Rephrase the question to sound natural and conversational
2. Rephrase each option to sound natural, keeping the SAME meaning and order
3. Preserve person descriptions precisely (clothing colors, carried objects, distinctive features)
4. Preserve camera identifiers (e.g., "camera G299") when present — they tell the VLM where to look
5. Preserve event numbering (Event 1, Event 2) when present
6. Keep spatial terms unchanged (near, moderate, far, meters)
7. Keep "simultaneously" and "cannot be determined" as-is
8. Do NOT add information not present in the template
9. Do NOT reorder the options
10. Add a brief 1-sentence "reasoning" explaining why the correct answer is right

Output format — respond with ONLY a JSON object:
{
  "question": "The naturalized question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "reasoning": "Brief explanation of why the answer is correct"
}
"""

SYSTEM_PROMPT_V3 = """\
You are a creative question writer for a multi-camera surveillance video QA benchmark.

Your task: rewrite template questions into varied, natural English. Each question should
sound like a DIFFERENT person wrote it. Vary sentence structure, word choice, and phrasing
aggressively — avoid formulaic patterns like always starting with "In this scene..." or
"Looking at the cameras..." or "Two events are observed...".

Rules:
1. VARY your phrasing — use different sentence openings, structures, and vocabulary each time
2. Preserve ALL factual content: person descriptions (clothing colors, carried objects), activities
3. Rephrase options naturally but keep the SAME meaning and order
4. Camera identifiers in answer options should be preserved
5. Keep spatial terms (near, moderate, far, meters) and "simultaneously"/"cannot be determined"
6. Do NOT add information not in the template
7. Do NOT reorder options
8. Add a 1-sentence "reasoning" for why the correct answer is right

Phrasing variety examples (do NOT copy these verbatim — invent your own):
- "Two things happen in view of the cameras..."
- "Watch for these two events..."
- "Based on what the cameras recorded..."
- "Among the people visible..."
- Direct question without preamble: "Which happened first: ..."
- "The footage shows..." / "Can you tell..." / "What do you notice about..."
- Vary active/passive voice, question-first vs. description-first
- Sometimes be brief and direct, sometimes more descriptive

Output format — respond with ONLY a JSON object:
{
  "question": "The creatively rephrased question",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "reasoning": "Brief explanation of why the answer is correct"
}
"""

GRAMMAR_CHECKER_PROMPT = """\
You are a meticulous copy editor. You receive a JSON object containing a VQA question,
options, and reasoning. Your ONLY job is to fix grammar, punctuation, and awkward phrasing.

Rules:
1. Fix grammatical errors, run-on sentences, and punctuation mistakes
2. Do NOT change meaning, add information, or remove details
3. Do NOT reorder options
4. Do NOT change camera IDs, person descriptions, or spatial/temporal terms
5. Keep the same JSON structure
6. If the text is already grammatically correct, return it unchanged
7. Be conservative — only fix clear errors

Output format — respond with ONLY a JSON object:
{
  "question": "The grammar-checked question",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "reasoning": "The grammar-checked reasoning"
}
"""

# ============================================================================
# Category-specific prompt examples (few-shot, V2)
# ============================================================================

CATEGORY_EXAMPLES_V3 = {
    "temporal": {
        "hint": "This is a temporal ordering question about two events. Person descriptions (clothing, objects) identify who is who — there are NO camera references in the question. VARY your phrasing creatively. Return ONLY {question, reasoning}.",
        "example_input": 'Consider two events in this multi-camera scene: (1) A person wearing a gray top and green pants, carrying a black backpack, entering a scene through a structure. (2) A person in a blue top and green pants, interacting with a person. Which event occurred first?',
        "example_output": '{"question": "Which of these happened first: a person in gray with green pants and a black backpack walking in through a structure, or a person in a blue top and green pants interacting with someone?", "reasoning": "The person in gray entered through the structure before the blue-topped person interacted with anyone."}',
    },
    "spatial": {
        "hint": "Spatial distance question about two people. Use their appearance descriptions to identify them. No camera references. VARY phrasing. Return ONLY {question, reasoning}.",
        "example_input": 'How close are a person wearing a blue top and blue pants, with a black hoodie featuring a graphic design on the back and a person wearing a white hoodie with a Puma logo, camouflage pants, and a camouflage cap in the scene?',
        "example_output": '{"question": "How far apart would you say the person in blue with the black graphic hoodie is from the one wearing a white Puma hoodie and camo pants?", "reasoning": "Their positions in the scene place them approximately 6 meters apart."}',
    },
    "perception": {
        "hint": "Perception question about activities or visual attributes. Camera references are part of the question — keep them. IMPORTANT: Preserve 'What activity...' and 'Which camera...' question structures EXACTLY — do NOT rephrase to 'Can you identify...' or 'Identify the camera...'. Only fix grammar and smooth descriptions. Return ONLY {question, reasoning}.",
        "example_input": 'What activity is occurring on camera G423?',
        "example_output": '{"question": "What activity is occurring on Camera G423?", "reasoning": "The activity visible on Camera G423 is a person opening a door."}',
    },
    "re_identification": {
        "hint": "Person re-identification across cameras. Camera refs are essential (use 'Camera Gxxx' format). Appearance descriptions must be precise. VARY phrasing. Return ONLY {question, reasoning}.",
        "example_input": 'On Camera G419, a person wearing a blue top and blue pants, with a black hoodie featuring a graphic design on the back, is visible. Which other camera also shows this same person?',
        "example_output": '{"question": "There is someone on Camera G419 in a blue top, blue pants, and a black hoodie with a graphic on the back. Can you spot the same person on any other camera?", "reasoning": "This person with the distinctive graphic hoodie shows up on both Camera G419 and Camera G423."}',
    },
    "scene_summary": {
        "hint": "Scene-level summary question. Keep camera counts and activity references. VARY phrasing. Return ONLY {question, reasoning}.",
        "example_input": 'Considering all 8 camera feeds in this slot, which description best characterizes the overall scene?',
        "example_output": '{"question": "What would you say best describes what is going on across all 8 camera feeds?", "reasoning": "Pedestrian activities dominate across all 8 feeds, with putting down objects being the most common."}',
    },
    "event_ordering": {
        "hint": "Event ordering question about chronological sequence. Rephrase the event list naturally — vary how you introduce the events and ask for the order. Do NOT add letter prefixes (A, B, C, D) to options. Return ONLY {question, reasoning}.",
        "example_input": "Identify the correct chronological order of the following events: I. Someone opening a door II. A person walking through the courtyard. Which is the correct chronological order?",
        "example_output": '{"question": "Several activities were captured across different camera feeds. Place these events in the order they occurred: I. A door being opened near the entrance II. Someone strolling through the courtyard What is the right sequence?", "reasoning": "The door was opened before the person walked through the courtyard."}',
    },
    "causality": {
        "hint": "Cause-effect reasoning question. Rephrase naturally — for forward causal, vary how you ask 'what happened next'; for backward, vary how you ask 'why did this happen'. Keep the causal logic intact. Return ONLY {question, reasoning}.",
        "example_input": "After a person picks up object, what activity most likely followed?",
        "example_output": '{"question": "Once the individual grabbed an item from the ground, what did they most likely do next?", "reasoning": "Picking up an object is commonly followed by putting it down in another location."}',
    },
    "numerical": {
        "hint": "Counting question about activities, entities, or cameras. Rephrase the counting query naturally — vary sentence structure but preserve the exact scope. Return ONLY {question, reasoning}.",
        "example_input": "How many cameras capture at least one instance of talking to person?",
        "example_output": '{"question": "Across the available camera feeds, on how many of them can you spot at least one conversation taking place?", "reasoning": "Conversations were observed on 5 of the available camera feeds."}',
    },
}

CATEGORY_EXAMPLES_V2 = {
    "temporal": {
        "hint": "This is a temporal ordering question with two numbered events on specific cameras. Preserve the event numbers, person descriptions, and camera references exactly.",
        "example_input": 'Consider two events in this multi-camera scene: (1) A person wearing a gray top and green pants, carrying a black backpack, entering a scene through a structure on camera G421. (2) A person interacting with a person on camera G330. Which event occurred first?',
        "example_output": '{"question": "Two events are observed across the camera feeds: (1) A person in a gray top and green pants, carrying a black backpack, enters through a structure on camera G421. (2) A person interacts with another person on camera G330. Which of these events happened first?", "options": ["Entering a scene through a structure (camera G421) occurred first", "Interacting with a person (camera G330) occurred first", "They occurred simultaneously", "Cannot be determined"], "reasoning": "Based on the video evidence, the scene entry on camera G421 occurred before the interaction on camera G330."}',
    },
    "spatial": {
        "hint": "This is a spatial distance question about how far apart two people are. Person descriptions should be preserved naturally. No camera references in the question.",
        "example_input": 'How close are the person wearing a blue top and blue pants, with a black hoodie featuring a graphic design on the back, and the person wearing a white hoodie with a Puma logo, camouflage pants, and a camouflage cap in the scene?',
        "example_output": '{"question": "In the scene, how far apart are the person in blue clothes with a black graphic hoodie and the person in a white Puma hoodie with camouflage pants and cap?", "options": ["They are near each other (within a few meters)", "They are at a moderate distance (5-15 meters)", "They are far apart (more than 15 meters)", "They are at the same location"], "reasoning": "Based on their projected positions in the scene, these two individuals are approximately 6 meters apart, placing them at a moderate distance."}',
    },
    "perception": {
        "hint": "This is a perception question about activities or visual attributes. Camera references are part of the question structure — preserve them.",
        "example_input": 'A person is visible on camera G423. What color are they wearing on their lower body?',
        "example_output": '{"question": "Looking at camera G423, what color is the visible person wearing on their lower body?", "options": ["Gray", "Navy", "Blue", "Brown"], "reasoning": "The person on camera G423 is wearing blue pants, making Blue the correct answer."}',
    },
    "re_identification": {
        "hint": "This is a person re-identification question. Camera references are essential — preserve them. Preserve appearance descriptions precisely.",
        "example_input": 'On camera G419, a person wearing a blue top and blue pants, with a black hoodie featuring a graphic design on the back, is visible. Which other camera also shows this same person?',
        "example_output": '{"question": "A person in a blue top and blue pants with a black graphic hoodie is visible on camera G419. Which other camera also shows this same person?", "options": ["G423", "G299", "G328", "None of these cameras"], "reasoning": "The person wearing a blue top and pants with the distinctive black graphic hoodie appears on both camera G419 and camera G423."}',
    },
    "scene_summary": {
        "hint": "This is a scene-level summary question. Keep statistical terms, camera counts, and activity references.",
        "example_input": 'Considering all 8 camera feeds in this slot, which description best characterizes the overall scene?',
        "example_output": '{"question": "Looking at all 8 camera feeds together, which description best captures the overall activity in this scene?", "options": ["An empty scene with minimal activity, captured on 5 cameras", "A vehicle-focused scene with mostly parking and driving activity", "A single-camera scene showing only indoor activities", "A pedestrian-dominant scene across 8 cameras, primarily featuring putting down objects"], "reasoning": "The vast majority of events are pedestrian activities observed across all 8 cameras, with putting down objects being the most frequent activity."}',
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
                    temperature: float,
                    system_prompt: str = None,
                    examples: Dict = None) -> Optional[Dict]:
    """Send one pre-processed question to GPT for naturalization."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_V2
    if examples is None:
        examples = CATEGORY_EXAMPLES_V2

    category = question["category"]
    template = question["question_template"]
    options = question["options"]
    verification = question.get("verification", {})

    lookup_cat = question.get("subcategory", category)
    cat_info = examples.get(lookup_cat,
                            examples.get(category, {}))
    hint = cat_info.get("hint", "Rephrase this question naturally.")
    example_in = cat_info.get("example_input", "")
    example_out = cat_info.get("example_output", "")

    user_message = f"Category: {category}\n{hint}\n\n"

    if example_in and example_out:
        user_message += f"Example:\n  Input: {example_in}\n  Output: {example_out}\n\n"

    user_message += f"Now naturalize this question:\n\nTemplate: {template}\n\nOptions:\n"
    for i, opt in enumerate(options):
        user_message += f"  {chr(65+i)}) {opt}\n"

    # Add verification context for reasoning
    if category == "temporal" and "gap_sec" in verification:
        user_message += f"\nContext: The gap between events is {verification['gap_sec']}s.\n"
    elif category == "spatial" and "distance_meters" in verification:
        user_message += f"\nContext: Distance is {verification['distance_meters']}m.\n"
    elif category == "re_identification":
        user_message += "\nContext: Person identified via cross-camera appearance matching.\n"

    user_message += "\nRespond with ONLY the JSON object."

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=600,
            )

            result = json.loads(response.choices[0].message.content)

            if "question" not in result or "options" not in result:
                print(f"    WARNING: Missing fields, retry {attempt+1}")
                continue

            if len(result["options"]) != len(options):
                print(f"    WARNING: Option count mismatch, retry {attempt+1}")
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


def _grammar_check_one(client, naturalized: Dict, model: str) -> Optional[Dict]:
    """Send one naturalized question through grammar checker (pass 2)."""
    user_message = json.dumps({
        "question": naturalized.get("naturalized_question", ""),
        "options": naturalized.get("naturalized_options", []),
        "reasoning": naturalized.get("reasoning", ""),
    }, indent=2)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.3,  # Low temperature for conservative edits
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": GRAMMAR_CHECKER_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=600,
            )

            result = json.loads(response.choices[0].message.content)

            if "question" not in result or "options" not in result:
                break  # Fall back to naturalized version

            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return {
                "question": result["question"],
                "options": result["options"],
                "reasoning": result.get("reasoning", ""),
                "usage": usage,
            }

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            continue

    return None  # Grammar check failed, caller uses naturalized version as-is


# ============================================================================
# Batch Processing
# ============================================================================

def naturalize_batch(input_data: Dict, model: str, temperature: float,
                     verbose: bool = False, version: str = "v2") -> Dict:
    """Pre-process + GPT naturalize all QA pairs."""
    # Step 1: Pre-process (free) — V3 strips camera refs from temporal Qs
    preprocessed = preprocess_all(input_data, verbose=verbose, version=version)

    # Step 2: Select prompts based on version
    if version == "v3":
        sys_prompt = SYSTEM_PROMPT_V3
        cat_examples = CATEGORY_EXAMPLES_V3
    else:
        sys_prompt = SYSTEM_PROMPT_V2
        cat_examples = CATEGORY_EXAMPLES_V2

    # Step 3: GPT naturalize
    client = _create_client()
    qa_pairs = preprocessed["qa_pairs"]
    total = len(qa_pairs)

    print(f"\n  Naturalizing {total} pre-processed questions with {model} ({version})...")

    naturalized_pairs = []
    total_tokens = 0
    failures = 0

    for i, q in enumerate(qa_pairs):
        if verbose:
            print(f"  [{i+1}/{total}] {q['category']}: "
                  f"{q['question_template'][:60]}...")

        # --- Pass 1: Naturalization ---
        result = _naturalize_one(client, q, model, temperature,
                                system_prompt=sys_prompt, examples=cat_examples)

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
        total_tokens += result["usage"]["total_tokens"]

        # --- Pass 2: Grammar check ---
        gc_result = _grammar_check_one(client, nat_q, model)

        if gc_result is not None:
            nat_q["naturalized_question"] = gc_result["question"]
            nat_q["naturalized_options"] = gc_result["options"]
            nat_q["reasoning"] = gc_result["reasoning"]
            nat_q["grammar_checked"] = True
            total_tokens += gc_result["usage"]["total_tokens"]
        else:
            nat_q["grammar_checked"] = False

        naturalized_pairs.append(nat_q)

        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{total} ({total_tokens} tokens)")

    version_tag = "v8_natural_v3" if version == "v3" else "v8_natural_v2"

    output = {
        "slot": input_data["slot"],
        "version": version_tag,
        "generator": "naturalize_v8_qa_v2.py",
        "preprocessor": f"{version}_preprocess",
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
# Dry Run
# ============================================================================

def dry_run(input_data: Dict):
    """Show pre-processed templates and what would be sent to GPT."""
    preprocessed = preprocess_all(input_data, verbose=True)
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
            print(f"      {chr(65+i)}) {opt}{marker}")
        print()

    # Cost estimate
    est_tokens = len(qa_pairs) * 400  # slightly more with examples
    est_cost_mini = est_tokens * 0.4e-6
    est_cost_4o = est_tokens * 6e-6

    print(f"  === Cost Estimate (GPT naturalization) ===")
    print(f"  Questions: {len(qa_pairs)}")
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
        description="V8 Naturalization — Pre-process + GPT naturalize (V2/V3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True,
        help="Path to V8 QA JSON file")
    parser.add_argument("--output", "-o",
        help="Output path (default: auto-generated)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
        help=f"GPT model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", "-t", type=float, default=None,
        help="Temperature (default: 0.7 for V2, 0.95 for V3)")
    parser.add_argument("--v3", action="store_true",
        help="V3 mode: more question variety, strip camera refs from temporal Qs")
    parser.add_argument("--preprocess-only", action="store_true",
        help="Only pre-process templates (no GPT call, FREE)")
    parser.add_argument("--dry-run", action="store_true",
        help="Show pre-processed templates and cost estimate")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--yes", "-y", action="store_true",
        help="Skip confirmation prompt")

    args = parser.parse_args()

    # Resolve version and temperature
    version = "v3" if args.v3 else "v2"
    temperature = args.temperature if args.temperature is not None else (
        0.95 if version == "v3" else DEFAULT_TEMPERATURE
    )

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        return

    print(f"Loading: {input_path} (mode: {version}, temp: {temperature})")
    with open(input_path) as f:
        input_data = json.load(f)

    total = len(input_data.get("qa_pairs", []))
    print(f"  Slot: {input_data.get('slot', 'N/A')}")
    print(f"  Version: {input_data.get('version', 'N/A')}")
    print(f"  Questions: {total}")

    # Mode 1: Pre-process only (free)
    if args.preprocess_only:
        result = preprocess_all(input_data, verbose=True, version=version)

        out_path = args.output or str(input_path).replace(
            ".v8.json", ".v8.preprocessed.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n  Pre-processed output → {out_path}")

        # Show before/after for each question
        print(f"\n  === Before → After ===")
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

    # Mode 3: Full pipeline (pre-process + GPT)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        print("  TIP: Use --preprocess-only for free pre-processing without GPT.")
        return

    if not args.yes:
        print(f"\n  Will pre-process + naturalize {total} questions with {args.model} ({version}, temp={temperature})")
        resp = input("  Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            return

    result = naturalize_batch(input_data, args.model, temperature,
                              verbose=args.verbose, version=version)

    print(f"\n  === Results ===")
    print(f"  Naturalized: {total - result['failures']}/{total}")
    print(f"  Failures: {result['failures']}")
    print(f"  Total tokens: {result['total_tokens']}")

    suffix = ".v8.natural.v3.json" if version == "v3" else ".v8.natural.v2.json"
    out_path = args.output or str(input_path).replace(".v8.json", suffix)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Output: {out_path}")

    # Save GPT log
    slot = input_data.get("slot", "unknown")
    log_dir = LOG_DIR / slot
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"naturalize_v8_{version}_{args.model}.json"
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
        orig = q.get("original_template", q.get("question_template", ""))
        print(f"    RAW:     {orig[:80]}...")
        print(f"    PREPROC: {q.get('question_template', '')[:80]}...")
        print(f"    NATURAL: {q.get('naturalized_question', '')[:80]}...")
        if q.get("reasoning"):
            print(f"    REASON:  {q['reasoning'][:80]}...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Naturalize — Pre-processes and naturalizes template QA pairs via GPT.

Clean 3-stage architecture:
  Stage 1 — Structural pre-processing (Python, FREE):
    - Reconstruct questions from structured verification data
    - Description simplification: "blue upper body garment" -> "blue top"
    - Camera reference removal from temporal/spatial question text
    - MEVA ontology vocabulary normalization
    - NO grammar fixes — all language rewriting delegated to GPT

  Stage 2 — Language rewrite (GPT, 1 API call per question):
    - Receives plain text question + context (never raw JSON)
    - Handles ALL grammar, phrasing, style variation, article agreement
    - Returns rewritten question + reasoning sentence
    - Options are frozen (never sent to GPT for rewriting)

  Stage 3 — JSON assembly (Python):
    - Inserts GPT's returned text back into QA structure
    - Saves output file + GPT log

Modes:
  --preprocess-only: Just run Stage 1 (free, instant)
  --dry-run: Show what would be sent to GPT
  (default): Full pipeline (Stage 1 + 2 + 3)

Usage:
    python3 -m meva.scripts.v10.naturalize --input data/qa_pairs/SLOT.final.raw.json
    python3 -m meva.scripts.v10.naturalize --input data/qa_pairs/SLOT.final.raw.json --preprocess-only
    python3 -m meva.scripts.v10.naturalize --input data/qa_pairs/SLOT.final.raw.json --dry-run
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

# User output directory — override with MEVA_OUTPUT_DIR env var
_OUTPUT = Path(os.environ.get("OUTPUT_DIR") or os.environ.get("MEVA_OUTPUT_DIR") or str(Path.home() / "data"))

QA_DIR = _OUTPUT / "qa_pairs"
LOG_DIR = _OUTPUT / "gpt_logs"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3  # Low: naturalize only, no creative drift
MAX_RETRIES = 3
RETRY_DELAY = 2.0
CLIP_DURATION = 300.0  # 5-minute clips


# ============================================================================
# Description Simplification
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
    ->
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
# Activity Humanization
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
    "stops": "stopping", "turns": "turning", "transfers": "transferring",
    "reverses": "reversing", "abandons": "abandoning",
    "leaves": "leaving", "purchases": "purchasing",
}


def _conjugate_gerund(verb: str) -> str:
    """Smart fallback: conjugate an unknown verb to its -ing form."""
    if verb in _GERUND_MAP:
        return _GERUND_MAP[verb]
    if verb.endswith("es"):
        base = verb[:-2]
    elif verb.endswith("s") and not verb.endswith("ss"):
        base = verb[:-1]
    else:
        base = verb
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("ee"):
        return base + "ing"
    if base.endswith("e"):
        return base[:-1] + "ing"
    if (len(base) >= 3
            and base[-1] not in "aeiouwxy"
            and base[-2] in "aeiou"
            and base[-3] not in "aeiou"):
        return base + base[-1] + "ing"
    return base + "ing"

_NO_ARTICLE = frozenset({
    "up", "down", "on", "off", "out", "in", "to", "from",
    "through", "with", "around", "right", "left", "a",
    "an", "the", "into", "onto", "over", "away",
})


def _humanize_activity(activity: str) -> str:
    """person_opens_facility_door -> opens facility door"""
    for prefix in ("person_", "vehicle_", "hand_"):
        if activity.startswith(prefix):
            activity = activity[len(prefix):]
            break
    return activity.replace("_", " ")


def _humanize_gerund(activity: str) -> str:
    """person_opens_facility_door -> Opening a facility door"""
    base = _humanize_activity(activity)
    words = base.split()
    if not words:
        return base.capitalize()

    first = words[0]
    gerund = _conjugate_gerund(first)
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
    """Short gerund label: person_opens_facility_door -> opening a facility door"""
    result = _humanize_gerund(activity)
    return result[0].lower() + result[1:] if result else result


# ============================================================================
# Ontology Rewrites (module-level, used by preprocess_all)
# ============================================================================

_ONTOLOGY_REWRITES = [
    (re.compile(r'(enter(?:s|ing))(?: a)? scene through structure', re.IGNORECASE),
     lambda m: m.group(1) + " the camera's view through a doorway/gate"),
    (re.compile(r'(exit(?:s|ing)|leav(?:es|ing))(?: a)? scene through structure', re.IGNORECASE),
     lambda m: ('leaving' if m.group(1).lower().startswith(('exit', 'leav')) and m.group(1)[0].islower()
                else 'Leaving' if m.group(1)[0].isupper()
                else m.group(1)) + " the camera's view through a doorway/gate"),
]


def _apply_ontology_rewrites(text: str) -> str:
    """Apply MEVA ontology clarifications via case-insensitive regex."""
    for pattern, repl in _ONTOLOGY_REWRITES:
        text = pattern.sub(repl, text)
    return text


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
    Extract just the entity appearance, stripping any embedded activity text.

    Input: "the person wearing a gray upper body garment... enters scene"
    Output: "a person wearing a gray top and green pants, carrying a black backpack"

    Input: "a person interacts with person" (fallback, no appearance)
    Output: "a person"

    Input: "a vehicle" (vehicle entity)
    Output: "a vehicle"
    """
    if not entity_description:
        # Use activity prefix to determine entity type
        if activity.startswith("vehicle_"):
            return "a vehicle"
        return "a person"

    desc = entity_description.strip()

    # Preserve vehicle descriptions as-is
    if desc.lower().startswith("a vehicle") or desc.lower() == "vehicle":
        return desc if desc.lower().startswith("a ") else "a " + desc

    # Check if this is a real appearance description vs. activity fallback
    if not _has_appearance_info(desc):
        # Use activity prefix to determine entity type
        if activity.startswith("vehicle_"):
            return "a vehicle"
        return "a person"

    # Remove embedded activity text after the description
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

    # Normalize prefix: "the person" -> "a person"
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

def _preprocess_temporal(qa: Dict, strip_camera_refs: bool = True) -> Dict:
    """
    Reconstruct temporal question from structured verification data.

    Fixes: camera refs in question, description verbosity, activity doubling,
    temporal ambiguity (adds timestamp anchors), capitalization.

    Camera IDs are removed from the question text by default — person
    descriptions (clothing colors, carried objects) serve as the primary
    disambiguator instead. Camera refs are still kept in answer options.
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
    include_cam = not strip_camera_refs

    def _fmt_event(desc, act, cam, with_camera=True):
        d = desc.rstrip('.')
        cam_ref = f" on camera {cam}" if cam and with_camera else ""
        if d.lower() in ("a person", "someone"):
            return f"{d} {act}{cam_ref}"
        if d.lower() in ("a vehicle",):
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
        desc_1, desc_2 = clean_a, clean_b
        ev_1, ev_2 = ev_a, ev_b
    else:
        desc_1, desc_2 = clean_b, clean_a
        ev_1, ev_2 = ev_b, ev_a

    # Build clean question
    question = (
        f"Consider two events in this multi-camera scene: "
        f"(1) {desc_1}. (2) {desc_2}. "
        f"Which event occurred first?"
    )

    # Build options using person descriptions (no camera IDs in options)
    def _option_label(desc, act):
        d = desc.strip().rstrip('.')
        if d.startswith("A "):
            d = "The " + d[2:]
        elif d.startswith("a "):
            d = "The " + d[2:]
        return d

    opt_1 = _option_label(desc_1, None)
    opt_2 = _option_label(desc_2, None)

    options = [
        f"{opt_1} occurred first",
        f"{opt_2} occurred first",
        "They occurred simultaneously",
        "Cannot be determined",
    ]

    # Article agreement is applied globally in preprocess_all; skip here
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
        person_desc = v.get("person_description", "")
        if person_desc:
            template = re.sub(
                re.escape(person_desc),
                simplify_description(person_desc).rstrip("."),
                template,
                count=1,
            )
    elif q_type == "which_camera":
        alias = v.get("activity_alias", "")
        if alias:
            gerund = _humanize_gerund(v.get("activity", alias))
            template = f"Which camera captures {gerund.lower().lstrip('a ')}?"
            template = template[0].upper() + template[1:]

    # General cleanup
    template = simplify_description(template)
    if template:
        template = template[0].upper() + template[1:]
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
                   strip_camera_refs: bool = True) -> Dict:
    """
    Pre-process all QA pairs: simplify descriptions, remove camera refs,
    add temporal anchors, fix grammar. FREE (no API call).

    strip_camera_refs: If True (default), strips camera IDs from temporal
    question text. Person descriptions disambiguate instead.
    """
    output = {k: v for k, v in input_data.items() if k != "qa_pairs"}
    output["version"] = "preprocessed"
    output["preprocessor"] = "naturalize.py"

    preprocessed = []
    changes = {"temporal": 0, "spatial": 0, "perception": 0,
               "re_identification": 0, "scene_summary": 0}

    for qa in input_data.get("qa_pairs", []):
        cat = qa.get("category", "")

        if cat == "temporal":
            cleaned = _preprocess_temporal(qa, strip_camera_refs=strip_camera_refs)
        elif cat == "spatial":
            cleaned = _preprocess_spatial(qa)
        elif cat == "perception":
            cleaned = _preprocess_perception(qa)
        elif cat == "re_identification":
            cleaned = _preprocess_reid(qa)
        elif cat in ("scene_summary", "summarization"):
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

    # ---------------------------------------------------------------
    # Global text fixes (applied to ALL categories, all text fields)
    # Ontology vocabulary normalization only — all grammar/article
    # fixes are delegated to GPT to avoid double-transformation
    # ---------------------------------------------------------------
    for qa in preprocessed:
        for field in ("question_template", "correct_answer"):
            if field in qa:
                qa[field] = _apply_ontology_rewrites(str(qa[field]))
        if "options" in qa:
            qa["options"] = [
                _apply_ontology_rewrites(str(o))
                for o in qa["options"]
            ]

    output["qa_pairs"] = preprocessed

    if verbose:
        total_changed = sum(changes.values())
        print(f"\n  Pre-processing: {total_changed}/{len(preprocessed)} questions modified")
        for cat, cnt in sorted(changes.items()):
            if cnt > 0:
                print(f"    {cat}: {cnt} changed")

    return output


# ============================================================================
# GPT System Prompt
# ============================================================================

SYSTEM_PROMPT = """\
You are a skilled question writer AND meticulous copy editor for a multi-camera \
surveillance video QA benchmark.

Your task: rewrite each template question into varied, natural English with \
perfect grammar, punctuation, and phrasing — all in a single step. Each \
question should sound like a DIFFERENT person wrote it.

IMPORTANT: You rewrite ONLY the question text and provide a reasoning sentence. \
You do NOT rewrite the answer options — those are deterministically generated \
and must not be changed.

Priority order (resolve conflicts by rank):
1. Preserve factual meaning exactly — never alter facts, person descriptions \
(clothing colors, carried objects), spatial terms, or answer options
2. Ensure flawless grammar, punctuation, and natural phrasing — FIX any \
garbled verb forms (e.g. "leavesing" → "leaving", "transfersing" → \
"transferring", "purchasesing" → "purchasing"). These are template bugs, \
not intentional text.
3. Apply creative stylistic variation — use different sentence openings, \
structures, and vocabulary each time. Avoid formulaic patterns like always \
starting with "In this scene..." or "Looking at the cameras..."
4. Add one concise reasoning sentence for why the correct answer is right

## Constraints
- Do NOT change the meaning of the question.
- Do NOT add new facts or details not present in the original.
- Do NOT remove constraints or simplify the logical requirement.
- Do NOT alter person descriptions (clothing colors, carried objects) — but \
DO fix obvious grammar errors within them (broken verb conjugations, garbled \
words, missing articles).
- Do NOT change "a vehicle" to "a person" or vice versa — entity type is \
factually significant. Vehicles and persons are different entities.
- Do not change answer options.
- Camera identifiers (e.g., G421) in question text are acceptable ONLY for \
perception and re-identification questions where cameras are inherent.
- For PERCEPTION questions ("What activity..." / "Which camera..."), maintain \
the direct question form but you may vary surrounding wording naturally.
- You MAY rephrase robotic activity descriptions into natural English \
(e.g. "entering scene through structure" → "walking in through a doorway").
- Fix all grammar, spelling, and conjugation errors.

## Ontology Translation
Translate robotic activity labels into natural human prose. Examples:
- "enters scene through structure" → "walks into the building"
- "person_opens_facility_door" → "opens a door"
Smooth out awkward clothing lists into natural descriptions. Only rephrase \
what is given — do not invent new details.

## Grammar & Polish
Fix grammatical errors, run-on sentences, punctuation, capitalization, \
awkward phrasing, redundancy, and unclear references.

Bad → Good transformation example:
- BAD: "Throughout all the cameras in this time frame, how many instances of \
stopping can be observed?"
- GOOD: "Across all cameras during this time period, how many stopping events \
occur?"

Phrasing variety examples (do NOT copy verbatim — invent your own):
- "A man in a gray hoodie appears near the entrance..."
- "Which of these events took place first?"
- "Based on the footage, what happened after..."
- Direct question without preamble: "Who was spotted on more than one camera?"
- Vary active/passive voice, question-first vs. description-first
- Sometimes be brief and direct, sometimes more descriptive

Output format — respond with ONLY a JSON object:
{
  "question": "The creatively rephrased and grammar-polished question",
  "reasoning": "Brief explanation of why the correct answer is right"
}
"""



# ============================================================================
# Category-specific prompt examples (few-shot)
# ============================================================================

CATEGORY_EXAMPLES = {
    "temporal": {
        "hint": "This is a temporal ordering question about two events. CRITICAL: Do NOT mention camera IDs (like G421, G339), timestamps (like 'at 45 seconds'), or locations (like 'near the parking lot'). Person descriptions (clothing, objects) identify who is who. VARY your phrasing creatively. Return ONLY {question, reasoning}.",
        "example_input": 'Consider two events in this multi-camera scene: (1) A person wearing a gray top and green pants, carrying a black backpack, entering a scene through a structure. (2) A person in a blue top and green pants, interacting with a person. Which event occurred first?',
        "example_output": '{"question": "Which of these happened first: a person in gray with green pants and a black backpack walking in through a structure, or a person in a blue top and green pants interacting with someone?", "reasoning": "The person in gray entered through the structure before the blue-topped person interacted with anyone."}',
    },
    "spatial": {
        "hint": "Spatial closest-approach question about how close two people come to each other. CRITICAL: Do NOT mention camera IDs (like G421), timestamps (like 'at 45 seconds', 'around the 2:10 mark'), or raw time references. Use only visual appearance descriptions (clothing, hair, objects) to identify people. VARY phrasing. Return ONLY {question, reasoning}.",
        "example_input": 'How close do a person wearing a blue top and blue pants, with a black hoodie featuring a graphic design on the back and a person wearing a white hoodie with a Puma logo, camouflage pants, and a camouflage cap come to each other in the scene?',
        "example_output": '{"question": "How close do the person in blue with the black graphic hoodie and the one in a white Puma hoodie with camo pants get to each other?", "reasoning": "Their closest approach places them approximately 6 meters apart, keeping them at a moderate distance throughout."}',
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
        "hint": "Event ordering question about chronological sequence. CRITICAL: Do NOT mention camera IDs (like G421, G339), timestamps, or time references. Rephrase the event list naturally using visual descriptions and activities only. Vary how you introduce the events and ask for the order. Do NOT add letter prefixes (A, B, C, D) to options. Return ONLY {question, reasoning}.",
        "example_input": "Identify the correct chronological order of the following events: I. Someone opening a door II. A person walking through the courtyard. Which is the correct chronological order?",
        "example_output": '{"question": "Several activities were captured across different camera feeds. Place these events in the order they occurred: I. A door being opened near the entrance II. Someone strolling through the courtyard What is the right sequence?", "reasoning": "The door was opened before the person walked through the courtyard."}',
    },
    "causality": {
        "hint": "Cause-effect reasoning question. Rephrase naturally — for forward causal, vary how you ask 'what happened next'; for backward, vary how you ask 'why did this happen'. Keep the causal logic intact. Return ONLY {question, reasoning}.",
        "example_input": "After a person picks up object, what activity most likely followed?",
        "example_output": '{"question": "Once the individual grabbed an item from the ground, what did they most likely do next?", "reasoning": "Picking up an object is commonly followed by putting it down in another location."}',
    },
    "numerical": {
        "hint": "Counting question about activities across cameras. CRITICAL: Do NOT mention specific camera IDs (like G421) in the question. Rephrase the counting query naturally — vary sentence structure but preserve the exact scope. IMPORTANT: The reasoning will be provided separately — you ONLY need to return {question, reasoning} where reasoning preserves ALL numbers from the original reasoning EXACTLY. Do NOT change any count or number in the reasoning.",
        "example_input": "How many times does someone perform the action of opening a vehicle door across all cameras in this slot?",
        "example_output": '{"question": "Across the available camera feeds, how many times can you observe someone opening a vehicle door?", "reasoning": "Opening a vehicle door was observed 8 times across 2 cameras."}',
    },

    "best_camera": {
        "hint": "Question about which camera first/last captures a person entering the scene. VARY phrasing. Keep camera identifiers in options. Return ONLY {question, reasoning}.",
        "example_input": "Which camera first captures the entrance of a person in a blue top and gray pants into the scene?",
        "example_output": '{"question": "On which camera does a person wearing a blue top and gray pants first appear?", "reasoning": "Camera G419 is the first to capture this person entering the scene."}',
    },
}


# ============================================================================
# GPT Client & Category Aliases
# ============================================================================

# Alias categories that share the same few-shot examples
_CAT_ALIASES = {"summarization": "scene_summary", "counting": "numerical"}


def _create_client():
    """Create OpenAI client."""
    import openai
    return openai.OpenAI()


# ============================================================================
# GPT Naturalization (1 API call per question)
# ============================================================================

def _naturalize_question(
    client,
    question: Dict,
    model: str,
    temperature: float,
) -> Optional[Dict]:
    """
    Single GPT call: send plain text question + context, get back rewritten
    question + reasoning. Options are never sent to GPT for rewriting.

    Architecture per colleague review:
    - Input: labeled plaintext fields (never raw JSON structure)
    - Output: JSON with 2 fields {question, reasoning}
    - Python handles all JSON assembly
    """
    category = question["category"]
    template = question["question_template"]
    options = question["options"]
    verification = question.get("verification", {})

    # Select category-specific few-shot examples (with aliases)
    lookup_cat = question.get("subcategory", category)
    lookup_cat = _CAT_ALIASES.get(lookup_cat, lookup_cat)
    cat_info = CATEGORY_EXAMPLES.get(
        lookup_cat, CATEGORY_EXAMPLES.get(
            _CAT_ALIASES.get(category, category), {})
    )
    hint = cat_info.get("hint", "Rephrase this question naturally with perfect grammar.")
    example_in = cat_info.get("example_input", "")
    example_out = cat_info.get("example_output", "")

    # Build user message as labeled plaintext (never send raw JSON)
    parts = [f"CATEGORY: {category}", hint, ""]

    if example_in and example_out:
        parts.append(f"EXAMPLE INPUT:\n{example_in}")
        parts.append(f"EXAMPLE OUTPUT:\n{example_out}")
        parts.append("")

    parts.append(f"QUESTION TO REWRITE:\n{template}")
    parts.append("")

    # Options as context only
    opt_lines = [f"  {chr(65 + i)}) {opt}" for i, opt in enumerate(options)]
    parts.append("OPTIONS (context only — do NOT modify):\n" + "\n".join(opt_lines))

    # Verification context for reasoning (as plain English)
    if category == "temporal" and "gap_sec" in verification:
        parts.append(f"\nCONTEXT: The gap between events is {verification['gap_sec']} seconds.")
    elif category == "spatial" and "min_distance_meters" in verification:
        parts.append(f"\nCONTEXT: Closest approach distance between entities is {verification['min_distance_meters']} meters.")
    elif category == "best_camera":
        correct_cam = verification.get("correct_camera", "")
        entrance_time = verification.get("entrance_time_sec", 0)
        if correct_cam:
            parts.append(f"\nCONTEXT: First entrance on {correct_cam} at {entrance_time}s.")

    # For counting questions: pass the deterministic reasoning so GPT preserves numbers
    original_reasoning = question.get("reasoning", "")
    if category in ("numerical", "counting") and original_reasoning:
        parts.append(f"\nORIGINAL REASONING (preserve all numbers exactly): {original_reasoning}")

    user_message = "\n".join(parts)

    for attempt in range(MAX_RETRIES):
        try:
            request_args = {
                "model": model,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            }
            if model.startswith("gpt-5"):
                request_args["max_completion_tokens"] = 400
            else:
                request_args["max_tokens"] = 400
            response = client.chat.completions.create(**request_args)

            result = json.loads(response.choices[0].message.content)

            if "question" not in result:
                print(f"    WARNING: Missing 'question' field, retry {attempt + 1}")
                continue

            return {
                "naturalized_question": result["question"],
                "naturalized_options": options,  # frozen, no GPT rewriting
                "reasoning": result.get("reasoning", ""),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
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
    """Stage 1 (pre-process) + Stage 2 (GPT) + Stage 3 (assemble).

    Architecture: 1 API call per question. Options are never sent to GPT
    for rewriting — only the question text is naturalized.
    """
    # Stage 1: Pre-process (free)
    preprocessed = preprocess_all(input_data, verbose=verbose,
                                  strip_camera_refs=True)

    # Stage 2: GPT naturalize (1 call per question)
    client = _create_client()
    qa_pairs = preprocessed["qa_pairs"]
    total = len(qa_pairs)

    print(f"\n  Naturalizing {total} questions with {model} (temp={temperature})...")

    naturalized_pairs = []
    total_tokens = 0
    failures = 0

    for i, q in enumerate(qa_pairs):
        if verbose:
            print(f"  [{i + 1}/{total}] {q['category']}: "
                  f"{q['question_template'][:60]}...")

        result = _naturalize_question(client, q, model, temperature)

        # Stage 3: JSON assembly
        nat_q = q.copy()

        if result is None:
            failures += 1
            nat_q["naturalized_question"] = q["question_template"]
            nat_q["naturalized_options"] = q["options"]
            nat_q["reasoning"] = q.get("reasoning", "")  # preserve raw reasoning on failure
            nat_q["naturalization_failed"] = True
        else:
            nat_q["naturalized_question"] = result["naturalized_question"]
            nat_q["naturalized_options"] = result["naturalized_options"]
            nat_q["reasoning"] = result["reasoning"]
            total_tokens += result["usage"]["total_tokens"]
            
            # Post-naturalization safety: for counting questions, verify the
            # correct answer number appears in the reasoning. If GPT changed
            # it, fall back to the raw deterministic reasoning.
            if q["category"] in ("numerical", "counting"):
                correct_answer = q.get("correct_answer", "")
                raw_reasoning = q.get("reasoning", "")
                nat_reasoning = nat_q["reasoning"]
                if correct_answer and nat_reasoning:
                    # Check if the correct count appears in naturalized reasoning
                    if correct_answer not in nat_reasoning and raw_reasoning:
                        if verbose:
                            print(f"    WARNING: Counting reasoning corrupted "
                                  f"(correct={correct_answer}, not found in "
                                  f"'{nat_reasoning[:80]}...'). Using raw reasoning.")
                        nat_q["reasoning"] = raw_reasoning
                        nat_q["reasoning_restored_from_raw"] = True

        naturalized_pairs.append(nat_q)

        if (i + 1) % 5 == 0:
            print(f"    Progress: {i + 1}/{total} ({total_tokens} tokens)")

    output = {
        "slot": input_data["slot"],
        "version": "final_naturalized",
        "generator": "naturalize.py",
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
            print(f"      {chr(65 + i)}) {opt}{marker}")
        print()

    # Cost estimate
    calls = len(qa_pairs)
    est_tokens = len(qa_pairs) * 450

    est_cost_mini = est_tokens * 0.4e-6
    est_cost_4o = est_tokens * 6e-6

    print(f"  === Cost Estimate (1 API call per question) ===")
    print(f"  Questions: {len(qa_pairs)}")
    print(f"  API calls: {calls}")
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
        description="Naturalize — Pre-process + GPT naturalize QA pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True,
        help="Path to QA JSON file (e.g., SLOT.final.raw.json)")
    parser.add_argument("--output", "-o",
        help="Output path (default: auto-generated .naturalized.json)")
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
    temperature = args.temperature

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        return

    print(f"Loading: {input_path} (temp: {temperature})")
    with open(input_path) as f:
        input_data = json.load(f)

    total = len(input_data.get("qa_pairs", []))
    print(f"  Slot: {input_data.get('slot', 'N/A')}")
    print(f"  Version: {input_data.get('version', 'N/A')}")
    print(f"  Questions: {total}")

    # Mode 1: Pre-process only (free)
    if args.preprocess_only:
        result = preprocess_all(input_data, verbose=True)

        out_path = args.output or str(input_path).replace(
            ".json", ".preprocessed.json")
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

    # Mode 3: Full pipeline (pre-process + GPT)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        print("  TIP: Use --preprocess-only for free pre-processing without GPT.")
        return

    if not args.yes:
        print(f"\n  Will pre-process + naturalize {total} questions "
              f"with {args.model} (temp={temperature})")
        print(f"  API calls: {total}")
        resp = input("  Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            return

    result = naturalize_batch(
        input_data, args.model, temperature,
        verbose=args.verbose,
    )

    print(f"\n  === Results ===")
    print(f"  Naturalized: {total - result['failures']}/{total}")
    print(f"  Failures: {result['failures']}")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  API calls: {total}")

    # Derive output path
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
    log_path = log_dir / f"naturalize_{args.model}.json"
    with open(log_path, "w") as f:
        json.dump({
            "model": args.model,
            "temperature": temperature,
            "total_tokens": result["total_tokens"],
            "questions_processed": total,
            "failures": result["failures"],
            "api_calls": total,
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

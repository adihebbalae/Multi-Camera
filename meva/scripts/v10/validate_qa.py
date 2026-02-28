#!/usr/bin/env python3
"""
validate_qa.py — Automated QA validator for MEVA V10 pipeline.

Runs 6 structural checks on raw or naturalized QA JSON without watching video:
  1. Reasoning ↔ answer consistency
  2. Raw token leak detection
  3. Duplicate / near-duplicate detection
  4. Generic description detection
  5. Multi-correct-answer ambiguity
  6. Grammar / conjugation checking

Usage:
    python3 -m scripts.v10.validate_qa --input path/to/slot.raw.json
    python3 -m scripts.v10.validate_qa --input path/to/slot.naturalized.json -v
    python3 -m scripts.v10.validate_qa --input path/to/slot.raw.json --json

Output:
    Structured report: {slot, total_questions, total_issues, issues: [...]}
    Each issue: {question_id, check_name, severity, message, suggestion}
    Severity: error (must fix), warning (should fix), info (cosmetic)
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Constants
# ============================================================================

# All 37 MEVA ActEV activity IDs (raw Kitware names)
KITWARE_ACTIVITY_IDS = {
    "hand_interacts_with_person",
    "person_abandons_package",
    "person_carries_heavy_object",
    "person_closes_facility_door",
    "person_closes_trunk",
    "person_closes_vehicle_door",
    "person_embraces_person",
    "person_enters_scene_through_structure",
    "person_enters_vehicle",
    "person_exits_scene_through_structure",
    "person_exits_vehicle",
    "person_interacts_with_laptop",
    "person_loads_vehicle",
    "person_opens_facility_door",
    "person_opens_trunk",
    "person_opens_vehicle_door",
    "person_picks_up_object",
    "person_purchases",
    "person_puts_down_object",
    "person_reads_document",
    "person_rides_bicycle",
    "person_sits_down",
    "person_stands_up",
    "person_steals_object",
    "person_talks_on_phone",
    "person_talks_to_person",
    "person_texts_on_phone",
    "person_transfers_object",
    "person_unloads_vehicle",
    "vehicle_drops_off_person",
    "vehicle_makes_u_turn",
    "vehicle_picks_up_person",
    "vehicle_reverses",
    "vehicle_starts",
    "vehicle_stops",
    "vehicle_turns_left",
    "vehicle_turns_right",
}

# Severity levels
ERROR   = "error"    # Must fix — answer is wrong or question is invalid
WARNING = "warning"  # Should fix — degrades quality
INFO    = "info"     # Cosmetic — minor wording issue


# ============================================================================
# Issue dataclass
# ============================================================================

class Issue:
    """Single validation issue."""
    __slots__ = ("question_id", "check_name", "severity", "message", "suggestion")

    def __init__(self, question_id: str, check_name: str, severity: str,
                 message: str, suggestion: str = ""):
        self.question_id = question_id
        self.check_name = check_name
        self.severity = severity
        self.message = message
        self.suggestion = suggestion

    def to_dict(self) -> dict:
        d = {
            "question_id": self.question_id,
            "check_name": self.check_name,
            "severity": self.severity,
            "message": self.message,
        }
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


# ============================================================================
# Helper: get question text fields (works on both raw and naturalized)
# ============================================================================

def _get_question_text(q: dict) -> str:
    """Get the question text (naturalized preferred, raw fallback)."""
    return q.get("naturalized_question") or q.get("question_template") or ""

def _get_options(q: dict) -> List[str]:
    """Get options list (naturalized preferred, raw fallback)."""
    return q.get("naturalized_options") or q.get("options") or []

def _get_answer_text(q: dict) -> str:
    """Get correct answer text."""
    return q.get("correct_answer") or ""

def _get_reasoning(q: dict) -> str:
    """Get reasoning text if available (naturalized may have it)."""
    return q.get("reasoning") or q.get("naturalized_reasoning") or ""


# ============================================================================
# Check 1: Reasoning ↔ Answer Consistency
# ============================================================================

def check_reasoning_consistency(qa_pairs: List[dict]) -> List[Issue]:
    """
    Parse reasoning text + correct_answer for each question.
    - For temporal: if reasoning says "X happened first" but answer says Y, flag.
    - For counting: extract number from reasoning, compare to correct_answer.
    - For event_ordering: check if reasoning's described sequence matches answer.
    - For perception: check if camera mentioned in reasoning matches answer.
    """
    issues = []

    for q in qa_pairs:
        qid = q.get("question_id", "?")
        cat = q.get("category", "")
        answer = _get_answer_text(q).lower()
        reasoning = _get_reasoning(q).lower()
        verification = q.get("verification", {})
        correct_idx = q.get("correct_answer_index", -1)
        options = _get_options(q)

        if not reasoning:
            # No reasoning field — skip this check (raw files often lack it)
            continue

        # --- Temporal: "first" keyword matching ---
        if cat == "temporal":
            # Check if reasoning mentions which event was first
            first_match = re.search(r'(\w[\w\s]*?)\s+(?:occurred|happened|took place)\s+first', reasoning)
            if first_match and answer:
                reasoning_first = first_match.group(1).strip()
                # Cross-check with answer
                if reasoning_first not in answer and len(reasoning_first) > 5:
                    issues.append(Issue(
                        qid, "reasoning_consistency", ERROR,
                        f"Reasoning says '{reasoning_first}' occurred first, "
                        f"but correct_answer is: '{answer[:80]}'",
                        "Regenerate reasoning with correct event order"
                    ))

        # --- Counting: number extraction ---
        elif cat == "counting":
            # Extract the primary count from reasoning (the one after "observed X times")
            correct_count = verification.get("correct_count")
            if correct_count is not None:
                # Look for the stated count: "observed N time(s)"
                observed_match = re.search(r'observed\s+(\d+)\s+time', reasoning)
                if observed_match:
                    stated_count = int(observed_match.group(1))
                    if stated_count != correct_count:
                        issues.append(Issue(
                            qid, "reasoning_consistency", ERROR,
                            f"Reasoning says 'observed {stated_count} times', "
                            f"but correct_count is {correct_count}",
                            "Regenerate reasoning with correct count"
                        ))

        # --- Event ordering: sequence match ---
        elif cat == "event_ordering":
            ordered_events = verification.get("ordered_events", [])
            if len(ordered_events) >= 3 and correct_idx >= 0 and correct_idx < len(options):
                correct_opt = options[correct_idx].lower()
                # Extract activity verbs from ordered events
                activities_in_order = [e.get("activity", "") for e in ordered_events]
                # Check if the correct option's described sequence contradicts verification
                # Simple check: does the answer option mention the last event first?
                if activities_in_order:
                    last_act = activities_in_order[-1].replace("_", " ")
                    first_act = activities_in_order[0].replace("_", " ")
                    # If answer says last event happened first, that's wrong
                    if (last_act in correct_opt and
                        correct_opt.index(last_act) < correct_opt.index(first_act)
                        if first_act in correct_opt else False):
                        issues.append(Issue(
                            qid, "reasoning_consistency", ERROR,
                            f"Correct answer mentions '{last_act}' before '{first_act}', "
                            f"contradicting event ordering in verification",
                            "Regenerate with correct chronological order"
                        ))

        # --- Perception: camera consistency ---
        elif cat == "perception":
            qt = verification.get("question_type")
            if qt == "which_camera":
                correct_cam = verification.get("correct_camera", "")
                if correct_cam and reasoning:
                    # Check if reasoning mentions a different camera as "correct"
                    cam_mentions = re.findall(r'G\d{3}', reasoning)
                    if cam_mentions:
                        last_cam = cam_mentions[-1]
                        if last_cam != correct_cam and f"camera {correct_cam}" not in reasoning:
                            issues.append(Issue(
                                qid, "reasoning_consistency", WARNING,
                                f"Reasoning mentions camera {last_cam} but correct is {correct_cam}",
                                "Check reasoning references correct camera"
                            ))

        # --- Structural: correct_answer_index vs correct_answer text match ---
        if correct_idx >= 0 and correct_idx < len(options):
            expected = options[correct_idx].lower().strip()
            actual = answer.lower().strip()
            # They should match (or be very close)
            if actual and expected and actual != expected:
                # Check if one is substring of the other (GPT may have rephrased)
                overlap = len(set(actual.split()) & set(expected.split()))
                total = max(len(actual.split()), len(expected.split()))
                if total > 0 and overlap / total < 0.5:
                    issues.append(Issue(
                        qid, "reasoning_consistency", ERROR,
                        f"correct_answer text doesn't match options[correct_answer_index]: "
                        f"'{actual[:60]}' vs '{expected[:60]}'",
                        "Sync correct_answer with options[correct_answer_index]"
                    ))

    return issues


# ============================================================================
# Check 2: Raw Token Leak Detection
# ============================================================================

# Pre-compiled patterns for speed
_RAW_TIMESTAMP_PATTERNS = [
    re.compile(r'\b\d{2,3}\s*seconds?\b', re.I),        # "127 seconds", "42 second"
    re.compile(r'\b\d+\.\d+s\b'),                        # "127.13s"
    re.compile(r'\bat\s+\d+s\b'),                        # "at 127s"
    re.compile(r'\b\d+\.\d+-\d+\.\d+s\b'),               # "127.13-133.43s"
    re.compile(r'\bframe\s*\d+', re.I),                   # "frame 3814"
    re.compile(r'\bframes?\s*\d+-\d+', re.I),             # "frames 3814-4003"
    re.compile(r'\b\d+\.\d{2}\s*-\s*\d+\.\d{2}\b'),      # "127.13 - 133.43" (timestamp ranges)
]

_RAW_FIELD_PATTERNS = [
    re.compile(r'\bstart_sec\b'),
    re.compile(r'\bend_sec\b'),
    re.compile(r'\bframe_range\b'),
    re.compile(r'\bactor_id\b'),
    re.compile(r'\bclip_file\b'),
    re.compile(r'\bentity_\w+\b'),     # entity_description, entity_a, etc.
]

_TEMPLATE_ARTIFACT_PATTERNS = [
    re.compile(r'\{[a-z_]+\}'),         # {variable_name} template placeholders
    re.compile(r'__\w+__'),             # __PLACEHOLDER__
    re.compile(r'\bevt_\w+\b'),         # evt_prefix internal IDs
    re.compile(r'\bcluster_\d+\b'),     # cluster_123 internal IDs
]

# Build regex alternation for activity IDs (match as whole words)
_ACTIVITY_ID_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(a) for a in sorted(KITWARE_ACTIVITY_IDS)) + r')\b'
)


def check_raw_token_leaks(qa_pairs: List[dict]) -> List[Issue]:
    """
    Regex scan question text + options for raw pipeline tokens that should
    have been cleaned/naturalized.
    """
    issues = []

    for q in qa_pairs:
        qid = q.get("question_id", "?")
        question_text = _get_question_text(q)
        options = _get_options(q)
        all_text = question_text + " " + " ".join(options)

        # Only check naturalized text (raw text is allowed to have these)
        is_naturalized = "naturalized_question" in q

        if not is_naturalized:
            # For raw files, only check for the most egregious leaks
            # (template artifacts still shouldn't be in raw text)
            for pat in _TEMPLATE_ARTIFACT_PATTERNS:
                m = pat.search(all_text)
                if m:
                    issues.append(Issue(
                        qid, "raw_token_leak", WARNING,
                        f"Template artifact in raw text: '{m.group()}'",
                        "Check template formatting"
                    ))
            continue

        # --- Naturalized text checks ---

        # Raw timestamps
        for pat in _RAW_TIMESTAMP_PATTERNS:
            m = pat.search(all_text)
            if m:
                issues.append(Issue(
                    qid, "raw_token_leak", WARNING,
                    f"Raw timestamp in naturalized text: '{m.group()}'",
                    "Re-run naturalization to remove timestamps"
                ))
                break  # One timestamp leak per question is enough

        # Raw activity names (Kitware IDs with underscores)
        m = _ACTIVITY_ID_PATTERN.search(all_text)
        if m:
            issues.append(Issue(
                qid, "raw_token_leak", WARNING,
                f"Raw activity ID in naturalized text: '{m.group()}'",
                "Re-run simplify_description() on this question"
            ))

        # Raw field names
        for pat in _RAW_FIELD_PATTERNS:
            m = pat.search(all_text)
            if m:
                issues.append(Issue(
                    qid, "raw_token_leak", INFO,
                    f"Raw field name in naturalized text: '{m.group()}'",
                    "Remove technical field references"
                ))
                break

        # Template artifacts
        for pat in _TEMPLATE_ARTIFACT_PATTERNS:
            m = pat.search(all_text)
            if m:
                issues.append(Issue(
                    qid, "raw_token_leak", WARNING,
                    f"Template artifact in naturalized text: '{m.group()}'",
                    "Re-run naturalization to fill template"
                ))
                break

    return issues


# ============================================================================
# Check 3: Duplicate / Near-Duplicate Detection
# ============================================================================

def _normalize_tokens(text: str) -> Set[str]:
    """Normalize text to lowercase token set for overlap comparison."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return set(text.split())

def _token_overlap_ratio(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Compute Jaccard-like overlap ratio between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def check_duplicates(qa_pairs: List[dict]) -> List[Issue]:
    """
    Detect duplicate/near-duplicate questions within a slot:
    - Exact text match
    - Fuzzy match: >90% token overlap
    - Same entity pair (verification.entity_a + entity_b)
    - Same category + same correct_answer
    """
    issues = []
    n = len(qa_pairs)

    for i in range(n):
        qi = qa_pairs[i]
        qi_id = qi.get("question_id", f"q{i}")
        qi_text = _get_question_text(qi)
        qi_tokens = _normalize_tokens(qi_text)
        qi_cat = qi.get("category", "")
        qi_answer = _get_answer_text(qi)
        qi_ver = qi.get("verification", {})

        for j in range(i + 1, n):
            qj = qa_pairs[j]
            qj_id = qj.get("question_id", f"q{j}")
            qj_text = _get_question_text(qj)
            qj_tokens = _normalize_tokens(qj_text)
            qj_cat = qj.get("category", "")
            qj_answer = _get_answer_text(qj)
            qj_ver = qj.get("verification", {})

            # 1. Exact text match
            if qi_text and qi_text == qj_text:
                issues.append(Issue(
                    qi_id, "duplicate", ERROR,
                    f"Exact duplicate of {qj_id}",
                    f"Remove one of {qi_id} or {qj_id}"
                ))
                continue

            # 2. Fuzzy match: >90% token overlap
            # Skip for categories with structurally similar templates
            # (spatial, best_camera) — these naturally share template words
            # but reference different entities. True duplicates are caught
            # by the entity pair check (#3) instead.
            _TEMPLATE_CATS = {"spatial", "best_camera"}
            if qi_cat not in _TEMPLATE_CATS or qj_cat not in _TEMPLATE_CATS:
                overlap = _token_overlap_ratio(qi_tokens, qj_tokens)
                if overlap > 0.90:
                    issues.append(Issue(
                        qi_id, "near_duplicate", WARNING,
                        f"Near-duplicate of {qj_id} ({overlap:.0%} token overlap)",
                        "Rephrase or replace one of the questions"
                    ))

            # 3. Same entity pair check
            ea_i = qi_ver.get("entity_a", qi_ver.get("event_a", {}).get("description", ""))
            eb_i = qi_ver.get("entity_b", qi_ver.get("event_b", {}).get("description", ""))
            ea_j = qj_ver.get("entity_a", qj_ver.get("event_a", {}).get("description", ""))
            eb_j = qj_ver.get("entity_b", qj_ver.get("event_b", {}).get("description", ""))

            if ea_i and eb_i and ea_j and eb_j:
                if ({ea_i, eb_i} == {ea_j, eb_j}):
                    issues.append(Issue(
                        qi_id, "same_entity_pair", WARNING,
                        f"Same entity pair as {qj_id}: ({ea_i[:40]}, {eb_i[:40]})",
                        "Use different entity pair for one question"
                    ))

            # 4. Same category + same correct answer
            if qi_cat == qj_cat and qi_answer and qi_answer == qj_answer:
                issues.append(Issue(
                    qi_id, "same_category_answer", INFO,
                    f"Same category '{qi_cat}' and identical correct_answer as {qj_id}",
                    "Consider diversifying answers within category"
                ))

    return issues


# ============================================================================
# Check 4: Generic Description Detection
# ============================================================================

def check_generic_descriptions(qa_pairs: List[dict]) -> List[Issue]:
    """
    Flag questions with generic entity references:
    - Description is just "a person" / "someone" / "an individual"
    - Entity description is null
    - Multiple entities on same camera share exact description
    """
    issues = []
    _GENERIC_DESCS = {"a person", "someone", "an individual", "a vehicle",
                      "the person", "the individual", "a man", "a woman"}

    # Build per-camera description frequency map from all questions
    camera_descriptions: Dict[str, List[str]] = defaultdict(list)

    for q in qa_pairs:
        qid = q.get("question_id", "?")
        cat = q.get("category", "")
        verification = q.get("verification", {})
        debug = q.get("debug_info", {})
        question_text = _get_question_text(q)

        # Collect entity descriptions from verification/debug
        descs_in_q = []

        # Temporal/spatial: event_a, event_b
        for key in ("event_a", "event_b", "entity_a", "entity_b"):
            info = verification.get(key, {})
            if isinstance(info, dict):
                desc = info.get("description") or info.get("entity_description", "")
                cam = info.get("camera", "")
                if desc:
                    descs_in_q.append((desc, cam))
                    camera_descriptions[cam].append(desc)

        # Debug entity_description
        for key in ("event_a", "event_b", "entity_a", "entity_b"):
            info = debug.get(key, {})
            if isinstance(info, dict):
                desc = info.get("entity_description", "")
                cam = info.get("camera", "")
                if desc:
                    descs_in_q.append((desc, cam))
                    camera_descriptions[cam].append(desc)

        # Check for generic descriptions in question text
        for generic in _GENERIC_DESCS:
            if generic in question_text.lower():
                # Only flag if it's a truly generic reference (no clothing/appearance follows)
                # Skip if followed by clothing descriptors ("a person in blue", "a person wearing")
                # or comma+clothing ("a person, wearing a ...")
                clothing_pattern = re.compile(
                    r'(?:^|\s)' + re.escape(generic) +
                    r'(?:[,\s]+(?:wearing|in\s+(?:a\s+)?(?:blue|red|green|black|white|gray|grey|dark|light|navy|teal|indigo|brown|beige|olive|pink|purple|plum|maroon|khaki|camo|charcoal)\b|with\s+(?:a\s+)?(?:blue|red|green|black|white|gray|dark|hat|bag|backpack|hoodie|jacket)\b))',
                    re.I
                )
                if clothing_pattern.search(question_text):
                    continue  # Has clothing descriptor — not generic
                
                # Check by looking at surrounding context for activity verbs
                activity_pattern = re.compile(
                    r'(?:^|\s)' + re.escape(generic) + r'(?:\s+(?:who|that|opens|closes|exits|enters|walks|sits|stands|talks|carries|picks|puts|rides|reads|loads|unloads)\b)',
                    re.I
                )
                if activity_pattern.search(question_text):
                    issues.append(Issue(
                        qid, "generic_description", WARNING,
                        f"Generic entity reference '{generic}' used in question text",
                        "Add clothing/visual description to disambiguate"
                    ))
                    break

        # Check entity descriptions in verification
        for desc, cam in descs_in_q:
            desc_lower = desc.strip().lower()
            if desc_lower in _GENERIC_DESCS or not desc_lower:
                issues.append(Issue(
                    qid, "generic_description", WARNING,
                    f"Generic entity description '{desc}' on camera {cam}",
                    "Re-extract visual description or use spatial context"
                ))

    # Cross-question check: same description on same camera
    for cam, descs in camera_descriptions.items():
        if not cam:
            continue
        desc_counts = Counter(descs)
        for desc, count in desc_counts.items():
            if count >= 2 and desc.lower().strip() not in _GENERIC_DESCS:
                # Same non-generic description appears multiple times on same camera
                issues.append(Issue(
                    f"slot-level:{cam}", "shared_description", INFO,
                    f"Description '{desc[:50]}' used {count} times on camera {cam}",
                    "Use spatial position or activity to disambiguate"
                ))

    return issues


# ============================================================================
# Check 5: Multi-Correct-Answer Ambiguity
# ============================================================================

def check_multi_correct_ambiguity(qa_pairs: List[dict]) -> List[Issue]:
    """
    Detect questions where multiple answer options could be correct:
    - which_camera perception: activity visible on >1 camera in options
    - spatial proximity: distance near bucket boundary (borderline classification)
    """
    issues = []

    for q in qa_pairs:
        qid = q.get("question_id", "?")
        cat = q.get("category", "")
        verification = q.get("verification", {})
        options = _get_options(q)

        # --- Perception: which_camera ambiguity ---
        if cat == "perception":
            qt = verification.get("question_type")
            if qt == "which_camera":
                cameras_with = verification.get("cameras_with_activity", [])
                if len(cameras_with) > 1:
                    # Check how many of those cameras are among the options
                    option_cams = set()
                    for opt in options:
                        for cam in cameras_with:
                            if cam in opt:
                                option_cams.add(cam)
                    if len(option_cams) > 1:
                        issues.append(Issue(
                            qid, "multi_correct_ambiguity", ERROR,
                            f"Activity appears on cameras {cameras_with}, "
                            f"and {len(option_cams)} of them are in the options",
                            "Use activity that appears on exactly 1 camera"
                        ))

        # --- Spatial: borderline proximity ---
        elif cat == "spatial":
            distance = verification.get("distance_meters")
            proximity = verification.get("proximity")
            if distance is not None and proximity:
                # Check if distance is within 20% of bucket boundary
                # near/moderate boundary: 5m (check 4-6m)
                # moderate/far boundary: 15m (check 12-18m)
                if proximity == "near" and 4.0 <= distance <= 6.0:
                    issues.append(Issue(
                        qid, "multi_correct_ambiguity", WARNING,
                        f"Borderline near/moderate: distance={distance:.1f}m "
                        f"(boundary at 5m, ±20% = 4-6m)",
                        "Use entity pair with clearer distance separation"
                    ))
                elif proximity == "moderate" and (
                    (4.0 <= distance <= 6.0) or (12.0 <= distance <= 18.0)
                ):
                    issues.append(Issue(
                        qid, "multi_correct_ambiguity", WARNING,
                        f"Borderline proximity: distance={distance:.1f}m "
                        f"(boundaries at 5m and 15m, ±20%)",
                        "Use entity pair with clearer distance separation"
                    ))
                elif proximity == "far" and 12.0 <= distance <= 18.0:
                    issues.append(Issue(
                        qid, "multi_correct_ambiguity", WARNING,
                        f"Borderline moderate/far: distance={distance:.1f}m "
                        f"(boundary at 15m, ±20% = 12-18m)",
                        "Use entity pair with clearer distance separation"
                    ))

    return issues


# ============================================================================
# Check 6: Grammar / Conjugation Checking
# ============================================================================

# Words that start with vowel sound but use "a" (not "an")
_A_EXCEPTIONS = {"uniform", "university", "unique", "united", "union",
                 "european", "euclidean", "eulerian", "user", "useful",
                 "usual", "utensil", "utility", "uranium", "one", "once"}

_GRAMMAR_PATTERNS = [
    # a [vowel] → should be an (with exceptions)
    (re.compile(r'\ba\s+([aeiou]\w*)\b', re.I), "article_a_an"),
    # Double articles
    (re.compile(r'\b(a|an|the)\s+\1\b', re.I), "double_article"),
    # Common gerund errors: *sing pattern (e.g., "leavesing")
    (re.compile(r'\b\w+(?:es|[^s]s)ing\b', re.I), "gerund_error"),
    # Subject-verb: "a person close" (should be "closes")
    (re.compile(r'\ba\s+person\s+(?:close|open|exit|enter|pick|put|sit|stand|walk|talk|read|ride|carry|load|unload|steal|transfer|abandon|embrace|purchase|text)\b', re.I), "subject_verb"),
]


def check_grammar(qa_pairs: List[dict]) -> List[Issue]:
    """
    Regex-based grammar checking for common errors in generated text.
    """
    issues = []

    for q in qa_pairs:
        qid = q.get("question_id", "?")
        question_text = _get_question_text(q)
        options = _get_options(q)
        answer = _get_answer_text(q)
        all_texts = [question_text] + options + [answer]

        for text in all_texts:
            if not text:
                continue

            # Check a/an
            for m in re.finditer(r'\ba\s+([aeiou]\w*)\b', text, re.I):
                word = m.group(1).lower()
                # Check exceptions (words starting with vowel letter but consonant sound)
                if word not in _A_EXCEPTIONS and not word.startswith("uni"):
                    issues.append(Issue(
                        qid, "grammar", INFO,
                        f"'a {word}' should be 'an {word}' in: '...{text[max(0,m.start()-10):m.end()+10]}...'",
                        f"Change 'a {word}' to 'an {word}'"
                    ))
                    break  # One article issue per text segment

            # Double articles
            m = re.search(r'\b(a|an|the)\s+\1\b', text, re.I)
            if m:
                issues.append(Issue(
                    qid, "grammar", WARNING,
                    f"Double article: '{m.group()}' in text",
                    f"Remove duplicate article"
                ))

            # Gerund errors (e.g., "closesing", "opensing", "leavesing")
            for m in re.finditer(r'\b(\w+(?:es|[^s]s)ing)\b', text, re.I):
                word = m.group(1).lower()
                # Skip legitimate words
                legit = {"missing", "passing", "crossing", "dressing", "pressing",
                         "blessing", "guessing", "assessing", "processing",
                         "accessing", "addressing", "expressing", "possessing",
                         "discussing", "bussing", "kissing", "tossing",
                         "fussing", "messing", "stressing", "confessing",
                         "obsessing", "caressing", "reassessing", "progressing",
                         "impressive", "using", "housing", "causing", "pausing",
                         "refusing", "abusing", "amusing", "bruising",
                         "choosing", "closing", "composing", "losing",
                         "opposing", "proposing", "raising", "praising",
                         "rising", "surprising", "advising", "exercising",
                         "promising", "comprising", "disguising", "revising",
                         "supervising", "nursing", "purchasing", "reversing",
                         "licensing", "sensing", "rinsing", "dispensing",
                         "conversing", "rehearsing", "endorsing", "immersing",
                         "coercing", "cursing"}
                if word not in legit:
                    issues.append(Issue(
                        qid, "grammar", WARNING,
                        f"Possible gerund error: '{word}'",
                        f"Check conjugation of '{word}'"
                    ))

            # Subject-verb disagreement
            m = re.search(
                r'\ba\s+person\s+(close|open|exit|enter|pick|put|sit|stand|walk|talk|read|ride|carry|load|unload|text)\b',
                text, re.I
            )
            if m:
                verb = m.group(1).lower()
                issues.append(Issue(
                    qid, "grammar", INFO,
                    f"Subject-verb disagreement: 'a person {verb}' → 'a person {verb}s'",
                    f"Change '{verb}' to '{verb}s'"
                ))

    return issues


# ============================================================================
# Main Validator
# ============================================================================

# ============================================================================
# Check 7: Forbidden Pattern Detection (Issue 12)
# ============================================================================

# Categories where camera IDs, raw timestamps, and location context are FORBIDDEN
_NO_CAMERA_CATEGORIES = {"temporal", "event_ordering", "spatial", "counting"}

# Regex patterns that should NOT appear in question text for restricted categories
_FORBIDDEN_PATTERNS = [
    (re.compile(r'\bcamera\s+G?\d+', re.IGNORECASE), "camera ID reference"),
    (re.compile(r'\bon camera\b', re.IGNORECASE), "camera reference"),
    (re.compile(r'\bat\s+\d+s\b', re.IGNORECASE), "raw timestamp (e.g. 'at 45s')"),
    (re.compile(r'\b\d+s[-–]\d+s\b', re.IGNORECASE), "timestamp range (e.g. '12s-180s')"),
    (re.compile(r'\baround the \d+', re.IGNORECASE), "temporal marker"),
    (re.compile(r'\bat time\s+\d+', re.IGNORECASE), "timestamp reference"),
    (re.compile(r'\b\d+ seconds?\b', re.IGNORECASE), "raw seconds in question"),
]


def check_forbidden_patterns(qa_pairs: List[dict]) -> List[Issue]:
    """
    Issue 12: Enforce no-camera/no-timestamp policy for restricted categories.
    
    temporal, event_ordering, spatial, counting questions must rely ONLY on
    visual appearance + activity verbs. Camera IDs, raw timestamps, and
    spatial/location markers are forbidden in question text and options.
    """
    issues: List[Issue] = []
    
    for q in qa_pairs:
        cat = q.get("category", "")
        if cat not in _NO_CAMERA_CATEGORIES:
            continue
        
        qid = q.get("question_id", "?")
        
        # Check question text
        for field_name in ("question_template", "naturalized_question"):
            text = q.get(field_name, "")
            if not text:
                continue
            for pattern, label in _FORBIDDEN_PATTERNS:
                m = pattern.search(text)
                if m:
                    issues.append(Issue(qid, "forbidden_pattern", ERROR,
                        f"[{cat}] {field_name} contains {label}: '{m.group()}'"
                    ))
        
        # Check options
        for i, opt in enumerate(q.get("options", [])):
            for pattern, label in _FORBIDDEN_PATTERNS:
                m = pattern.search(opt)
                if m:
                    issues.append(Issue(qid, "forbidden_pattern", WARNING,
                        f"[{cat}] option[{i}] contains {label}: '{m.group()}'"
                    ))
    
    return issues


# ============================================================================
# Check 8: Annotation-Based Verification (Issue 11)
# ============================================================================

def check_annotation_verification(qa_pairs: List[dict]) -> List[Issue]:
    """
    Lightweight sanity check: verify numerical counts and temporal ordering
    match the verification data. NOT circular — checks internal consistency
    between question text/answer and verification metadata.

    Catches:
    - Counting: reasoning mentions wrong number vs correct_answer
    - Temporal: event_a start_sec >= event_b start_sec (wrong ordering)
    - Event ordering: events in verification not chronologically sorted
    - All: correct_answer_index out of range or pointing to wrong option
    """
    issues: List[Issue] = []

    for q in qa_pairs:
        qid = q.get("question_id", "unknown")
        cat = q.get("category", "")
        v = q.get("verification", {})
        options = q.get("options", [])
        correct_idx = q.get("correct_answer_index", -1)
        correct_answer = q.get("correct_answer", "")

        # Universal: correct_answer_index in range and matches correct_answer
        if correct_idx < 0 or correct_idx >= len(options):
            issues.append(Issue(qid, "annotation_verify", ERROR,
                f"correct_answer_index={correct_idx} out of range (options has {len(options)} items)",
                "Fix correct_answer_index"))
        elif options[correct_idx] != correct_answer:
            issues.append(Issue(qid, "annotation_verify", ERROR,
                f"correct_answer '{correct_answer}' != options[{correct_idx}] '{options[correct_idx]}'",
                "Sync correct_answer with correct_answer_index"))

        # Counting: verify reasoning numbers match correct_answer
        if cat in ("numerical", "counting"):
            reasoning = q.get("reasoning", "")
            correct_count = v.get("correct_count")
            if correct_count is not None:
                # Check correct_answer matches correct_count
                try:
                    if str(correct_count) != correct_answer:
                        issues.append(Issue(qid, "annotation_verify", ERROR,
                            f"correct_count={correct_count} but correct_answer='{correct_answer}'",
                            "Sync correct_count with correct_answer"))
                except (ValueError, TypeError):
                    pass

                # Check reasoning mentions the correct number
                if reasoning:
                    import re as _re_verify
                    nums_in_reasoning = [int(n) for n in _re_verify.findall(r'\b(\d+)\b', reasoning)]
                    if correct_count not in nums_in_reasoning and str(correct_count) not in reasoning:
                        issues.append(Issue(qid, "annotation_verify", WARNING,
                            f"Reasoning doesn't mention correct count {correct_count}",
                            "Regenerate reasoning with correct count"))

        # Temporal: verify event ordering matches
        elif cat == "temporal":
            ea = v.get("event_a", {})
            eb = v.get("event_b", {})
            ea_start = ea.get("start_sec", 0)
            eb_start = eb.get("start_sec", 0)
            if ea_start and eb_start and ea_start >= eb_start:
                issues.append(Issue(qid, "annotation_verify", ERROR,
                    f"Event A (start={ea_start:.2f}s) does not precede Event B (start={eb_start:.2f}s)",
                    "Fix temporal ordering — event_a must come first"))

        # Event ordering: verify events are chronologically sorted
        elif cat == "event_ordering":
            ordered = v.get("ordered_events", [])
            for i in range(len(ordered) - 1):
                t_curr = ordered[i].get("start_sec", 0)
                t_next = ordered[i + 1].get("start_sec", 0)
                if t_curr >= t_next:
                    issues.append(Issue(qid, "annotation_verify", ERROR,
                        f"Verification events not chronological at position {i}: "
                        f"{t_curr:.2f}s >= {t_next:.2f}s",
                        "Fix event ordering in verification"))

    return issues


def validate(data: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Run all 6 checks on QA data.
    
    Args:
        data: Full QA JSON (with "qa_pairs" key)
        verbose: Print detailed output
    
    Returns:
        Structured report dict.
    """
    qa_pairs = data.get("qa_pairs", [])
    slot = data.get("slot", "unknown")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"QA Validator — {slot}")
        print(f"{'=' * 60}")
        print(f"Total questions: {len(qa_pairs)}")
        is_nat = any("naturalized_question" in q for q in qa_pairs)
        print(f"Format: {'naturalized' if is_nat else 'raw'}")

    all_issues: List[Issue] = []

    # Run all 8 checks
    checks = [
        ("reasoning_consistency", check_reasoning_consistency),
        ("raw_token_leak", check_raw_token_leaks),
        ("duplicate_detection", check_duplicates),
        ("generic_description", check_generic_descriptions),
        ("multi_correct_ambiguity", check_multi_correct_ambiguity),
        ("grammar", check_grammar),
        ("forbidden_pattern", check_forbidden_patterns),
        ("annotation_verify", check_annotation_verification),
    ]

    for check_name, check_fn in checks:
        found = check_fn(qa_pairs)
        all_issues.extend(found)
        if verbose:
            severity_counts = Counter(i.severity for i in found)
            status = "PASS" if not found else f"FAIL ({len(found)} issues)"
            detail = ""
            if found:
                parts = []
                for sev in [ERROR, WARNING, INFO]:
                    if severity_counts[sev]:
                        parts.append(f"{severity_counts[sev]} {sev}")
                detail = f" [{', '.join(parts)}]"
            print(f"  {check_name:30s}: {status}{detail}")

    # Compute quality score
    score = 100
    for issue in all_issues:
        if issue.severity == ERROR:
            score -= 10
        elif issue.severity == WARNING:
            score -= 5
        elif issue.severity == INFO:
            score -= 1
    score = max(0, score)

    # Build report
    severity_counts = Counter(i.severity for i in all_issues)
    report = {
        "slot": slot,
        "total_questions": len(qa_pairs),
        "total_issues": len(all_issues),
        "quality_score": score,
        "severity_counts": {
            "error": severity_counts[ERROR],
            "warning": severity_counts[WARNING],
            "info": severity_counts[INFO],
        },
        "issues": [i.to_dict() for i in all_issues],
    }

    if verbose:
        print(f"\n  Quality Score: {score}/100")
        print(f"  Total Issues: {len(all_issues)} "
              f"({severity_counts[ERROR]} error, "
              f"{severity_counts[WARNING]} warning, "
              f"{severity_counts[INFO]} info)")
        if all_issues:
            print(f"\n  Issues:")
            for issue in all_issues:
                sev_symbol = {"error": "E", "warning": "W", "info": "I"}[issue.severity]
                print(f"    [{sev_symbol}] {issue.question_id}: "
                      f"{issue.check_name} — {issue.message}")
                if issue.suggestion:
                    print(f"        → {issue.suggestion}")

    return report


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated QA validator for MEVA V10 pipeline"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to raw or naturalized QA JSON file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed validation report")
    parser.add_argument("--json", action="store_true",
                        help="Output report as JSON")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Exit with code 1 if quality score < this threshold")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    report = validate(data, verbose=args.verbose)

    if args.json:
        print(json.dumps(report, indent=2))

    if args.min_score and report["quality_score"] < args.min_score:
        print(f"\nFAIL: Quality score {report['quality_score']} < {args.min_score}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

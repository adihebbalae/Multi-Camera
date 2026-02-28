#!/usr/bin/env python3
"""
Convert MEVA naturalized QA JSONs into standardized per-category formats.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_INPUT_DIR = Path(
    "/home/ss99569/code/multi-cam/Multi-Camera/"
    "datasets/multi-cam-dataset/meva/data_all_slots/qa_pairs/raw"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/ss99569/code/multi-cam/Multi-Camera/"
    "datasets/multi-cam-dataset/meva/converted"
)


def _stable_hex_id(*parts: str) -> str:
    h = hashlib.md5()
    h.update("::".join(parts).encode("utf-8"))
    return h.hexdigest()


def _stable_int_id(*parts: str, mod: int = 1_000_000_000) -> int:
    hex_id = _stable_hex_id(*parts)
    return int(hex_id[:12], 16) % mod


def _options_list_to_dict(options: Optional[List[str]]) -> Optional[Dict[str, str]]:
    if not options:
        return None
    letters = ["A", "B", "C", "D"]
    result = {}
    for i, opt in enumerate(options):
        if i >= len(letters):
            break
        result[letters[i]] = opt
    return result


def _answer_letter(correct_idx: Optional[int]) -> Optional[str]:
    if correct_idx is None or correct_idx < 0:
        return None
    return chr(65 + correct_idx)


def _question_text(qa: Dict[str, Any]) -> str:
    return qa.get("naturalized_question") or qa.get("question_template") or ""


def _metadata_base(slot: str, qa: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "slot": slot,
        "question_id": qa.get("question_id"),
        "difficulty": qa.get("difficulty"),
        "requires_cameras": qa.get("requires_cameras"),
        "requires_multi_camera": qa.get("requires_multi_camera"),
        "verification": qa.get("verification"),
        "debug_info": qa.get("debug_info"),
    }


def _convert_event_ordering(slot: str, qa: Dict[str, Any]) -> Dict[str, Any]:
    options = qa.get("options")
    return {
        "task_id": _stable_int_id(slot, qa.get("question_id", ""), "event_ordering"),
        "task_name": slot,
        "episode_id": _stable_int_id(slot, "event_ordering"),
        "question_type": "event_ordering",
        "question": _question_text(qa),
        "options": _options_list_to_dict(options),
        "answer": _answer_letter(qa.get("correct_answer_index")),
        "reasoning": qa.get("reasoning", ""),
        "video_paths": qa.get("video_paths", []),
        "metadata": _metadata_base(slot, qa),
    }


def _convert_spatial(slot: str, qa: Dict[str, Any]) -> Dict[str, Any]:
    options = qa.get("options")
    return {
        "id": _stable_hex_id(slot, qa.get("question_id", ""), "spatial"),
        "video_id": slot,
        "question_type": "spatial",
        "question": _question_text(qa),
        "options": _options_list_to_dict(options),
        "answer": _answer_letter(qa.get("correct_answer_index")),
        "reasoning": qa.get("reasoning", ""),
        "video_paths": qa.get("video_paths", []),
        "metadata": _metadata_base(slot, qa),
    }


def _convert_summarization(slot: str, qa: Dict[str, Any]) -> Dict[str, Any]:
    options = qa.get("options") or []
    correct_idx = qa.get("correct_answer_index")
    answer = options[correct_idx] if isinstance(correct_idx, int) and correct_idx < len(options) else ""
    return {
        "task_id": _stable_int_id(slot, qa.get("question_id", ""), "summarization"),
        "task_name": slot,
        "episode_id": _stable_int_id(slot, "summarization"),
        "question_type": "summarization",
        "question": _question_text(qa),
        "answer": answer,
        "video_paths": qa.get("video_paths", []),
        "metadata": _metadata_base(slot, qa),
    }


def _convert_temporal(slot: str, qa: Dict[str, Any]) -> Dict[str, Any]:
    options = qa.get("options")
    return {
        "task_id": _stable_int_id(slot, qa.get("question_id", ""), "temporal"),
        "task_name": slot,
        "episode_id": _stable_int_id(slot, "temporal"),
        "question_type": "temporal",
        "question": _question_text(qa),
        "options": _options_list_to_dict(options),
        "answer": _answer_letter(qa.get("correct_answer_index")),
        "reasoning": qa.get("reasoning", ""),
        "video_paths": qa.get("video_paths", []),
        "metadata": _metadata_base(slot, qa),
    }


def _convert_counting(slot: str, qa: Dict[str, Any]) -> Dict[str, Any]:
    options = qa.get("options") or []
    correct_idx = qa.get("correct_answer_index")
    answer = options[correct_idx] if isinstance(correct_idx, int) and correct_idx < len(options) else ""
    return {
        "id": _stable_hex_id(slot, qa.get("question_id", ""), "counting"),
        "video_id": slot,
        "question_type": "counting",
        "question": _question_text(qa),
        "options": None,
        "answer": answer,
        "reasoning": qa.get("reasoning", ""),
        "video_paths": qa.get("video_paths", []),
        "metadata": _metadata_base(slot, qa),
    }


def _convert_best_camera(slot: str, qa: Dict[str, Any]) -> Dict[str, Any]:
    options = qa.get("options")
    verification = qa.get("verification") or {}
    metadata = _metadata_base(slot, qa)
    metadata["camera_question_subtype"] = qa.get("subcategory") or "best_camera"
    if verification.get("correct_camera"):
        metadata["best_camera_scene"] = verification.get("correct_camera")
    return {
        "video_id": slot,
        "question_type": "camera",
        "question": _question_text(qa),
        "options": _options_list_to_dict(options),
        "answer": _answer_letter(qa.get("correct_answer_index")),
        "reasoning": qa.get("reasoning", ""),
        "video_paths": qa.get("video_paths", []),
        "metadata": metadata,
    }


CONVERTERS = {
    "event_ordering": _convert_event_ordering,
    "spatial": _convert_spatial,
    "summarization": _convert_summarization,
    "temporal": _convert_temporal,
    "counting": _convert_counting,
    "best_camera": _convert_best_camera,
}


def _iter_naturalized_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.glob("*.raw.naturalized.json"))


def convert_all(input_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    outputs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CONVERTERS}
    for path in _iter_naturalized_files(input_dir):
        with path.open() as f:
            data = json.load(f)
        slot = data.get("slot") or path.stem.replace(".raw.naturalized", "")
        for qa in data.get("qa_pairs", []):
            category = qa.get("category")
            if category not in CONVERTERS:
                continue
            outputs[category].append(CONVERTERS[category](slot, qa))
    return outputs


def write_outputs(outputs: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, items in outputs.items():
        out_path = output_dir / f"qa_{category}.json"
        if out_path.parent != output_dir:
            raise ValueError(f"Refusing to write outside output dir: {out_path}")
        with out_path.open("w") as f:
            json.dump(items, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MEVA naturalized QA to standard formats")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    outputs = convert_all(args.input_dir)
    write_outputs(outputs, args.output_dir)


if __name__ == "__main__":
    main()

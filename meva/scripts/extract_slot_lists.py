#!/usr/bin/env python3
"""
Extract slot list from slot_index.json and write to a file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_slot_index(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return set(data.keys())


def write_list(path: Path, slots: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for slot in sorted(slots):
            handle.write(f"{slot}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_slot_index = repo_root / "meva" / "data" / "slot_index.json"
    default_out_file = default_slot_index.with_name("slot_list_from_slot_index.txt")

    parser = argparse.ArgumentParser(
        description="Extract slot list from slot_index.json."
    )
    parser.add_argument("--slot-index", type=Path, default=default_slot_index)
    parser.add_argument("--out-file", type=Path, default=default_out_file)
    args = parser.parse_args()

    slot_index_slots = read_slot_index(args.slot_index)

    write_list(args.out_file, slot_index_slots)

    print(f"Wrote {len(slot_index_slots)} slots to {args.out_file}")


if __name__ == "__main__":
    main()

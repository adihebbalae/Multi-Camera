#!/usr/bin/env python3
"""
Dry-run test for SpatioTemporalQAGenerator.

Loads real scene graphs and instance annotations, runs each of the 5 category
context builders individually, and prints structured output for inspection.
No GPT calls are made — this only validates data extraction + context building.

Usage:
    python3 nuscenes/datasetbuilder/test_spatio_temporal.py \
        --scene_graphs_dir /home/hg22723/projects/Multi-Camera/outputs/scene_graphs \
        --instance_annotations_dir /home/hg22723/projects/Multi-Camera/outputs/instance_annotations
"""

import argparse
import json
import sys
from pathlib import Path

# Allow import from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nuscenes.datasetbuilder.nuscens_build import SpatioTemporalQAGenerator


def run_test(scene_graphs_dir: str, instance_annotations_dir: str, limit: int = 5):
    gen = SpatioTemporalQAGenerator(
        prompts_dir=Path(__file__).resolve().parent / "prompts",
        scene_graphs_dir=scene_graphs_dir,
        instance_annotations_dir=instance_annotations_dir,
    )

    # Discover scene tokens from scene graph dirs
    sg_dir = Path(scene_graphs_dir)
    scene_tokens = [
        d.name for d in sorted(sg_dir.iterdir())
        if d.is_dir() and (d / "scene_graph.json").exists()
    ][:limit]

    if not scene_tokens:
        print("ERROR: No scene graph directories found.")
        return

    print(f"Testing {len(scene_tokens)} scenes...\n")
    print("=" * 80)

    category_hits = {cat: 0 for cat in gen.CATEGORIES}

    for scene_token in scene_tokens:
        print(f"\n{'='*80}")
        print(f"SCENE: {scene_token}")
        print(f"{'='*80}")

        # Load data
        scene_graph = gen.load_scene_graph(scene_token)
        if scene_graph is None:
            print("  ⚠ No scene graph found, skipping.")
            continue

        instance_annotations = gen.load_instance_annotations(scene_token)
        inst_ann = gen._preprocess_instance_annotations(instance_annotations)

        # Extract scene info (events + frames)
        scene_info = gen._extract_scene_info_from_dict(
            scene_graph, instance_annotations or {"annotations": []}
        )
        if scene_info is None:
            print("  ⚠ Could not extract scene info, skipping.")
            continue

        events = scene_info.get("events", [])
        frames = scene_info.get("frames", [])
        frames_map = gen._build_frames_map(frames)
        sorted_idxs = sorted(frames_map.keys())
        all_obj_ids = gen._collect_all_object_ids(frames)

        print(f"  Frames: {len(frames)}, Events: {len(events)}, Objects: {len(all_obj_ids)}")
        if events:
            print(f"  Sample events:")
            for ev in events[:3]:
                print(f"    - {ev['class']} {ev['activity'][:60]} (frames {ev['start_frame']}-{ev['end_frame']})")

        # Test each category builder
        builders = {
            "cat1a_sequential": gen._build_cat1a_sequential,
            "cat1b_in_between": gen._build_cat1b_in_between,
            "cat2a_snapshot": gen._build_cat2a_snapshot,
            "cat2b_intermediary": gen._build_cat2b_intermediary,
            "cat2c_comparative": gen._build_cat2c_comparative,
        }

        for cat_name, builder in builders.items():
            print(f"\n  --- {cat_name} ---")
            result = builder(events, frames_map, sorted_idxs, inst_ann, all_obj_ids)
            if result is None:
                print(f"  ✗ Returned None (insufficient data for this category)")
            else:
                category_hits[cat_name] += 1
                print(f"  ✓ SUCCESS")
                print(f"    Category:   {result['category_label']}")
                print(f"    Grounding:  {result['grounding_anchor'][:120]}...")
                print(f"    Target:     {result['target_query'][:120]}...")
                print(f"    Spatial:    {result['spatial_context'][:100]}...")
                print(f"    Temporal:   {result['temporal_context'][:100]}...")
                distractors = json.loads(result['distractor_candidates'])
                print(f"    Distractors ({len(distractors)}):")
                for d in distractors:
                    print(f"      - {d[:80]}")

                # Validate expected keys
                expected_keys = {
                    "category_label", "category_instructions", "scene_token",
                    "grounding_anchor", "target_query", "spatial_context",
                    "temporal_context", "distractor_candidates",
                }
                missing = expected_keys - set(result.keys())
                if missing:
                    print(f"    ⚠ MISSING KEYS: {missing}")

        # Test full pipeline (build_spatio_temporal_input)
        print(f"\n  --- Full Pipeline (build_spatio_temporal_input) ---")
        prompt_template = gen.load_prompts_from_disk()
        filled, chosen_cat = gen.build_spatio_temporal_input(scene_info, prompt_template, inst_ann)
        if filled:
            print(f"  ✓ Category chosen: {chosen_cat}")
            print(f"  ✓ Prompt length: {len(filled)} chars")
            print(f"  ✓ First 200 chars: {filled[:200]}...")
        else:
            print(f"  ✗ No prompt generated (all categories returned None)")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for cat, hits in category_hits.items():
        status = "✓" if hits > 0 else "✗"
        print(f"  {status} {cat}: {hits}/{len(scene_tokens)} scenes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_graphs_dir", type=str, required=True)
    parser.add_argument("--instance_annotations_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    run_test(args.scene_graphs_dir, args.instance_annotations_dir, args.limit)

"""
Batch annotation script for multiple scenes.

Processes multiple scenes in parallel or sequentially.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_annotator import VLMAnnotator
from nuscenes_dataloader import NuScenesLidarSegmentationLoader


def annotate_multiple_scenes(
    dataroot: str,
    version: str,
    output_dir: str,
    vllm_api: str,
    model: str,
    scene_indices: Optional[List[int]] = None,
    max_scenes: Optional[int] = None,
    max_frames_per_scene: Optional[int] = None,
    camera_preference: Optional[List[str]] = None
):
    """
    Annotate multiple scenes.
    
    Args:
        dataroot: Path to nuScenes dataset
        version: Dataset version
        output_dir: Output directory
        vllm_api: vLLM server URL
        model: VLM model name
        scene_indices: List of scene indices to process (None for all)
        max_scenes: Maximum number of scenes to process
        max_frames_per_scene: Maximum frames per scene
        camera_preference: Preferred camera order
    """
    # Initialize dataloader
    print("="*60)
    print("Batch nuScenes Scene Annotation")
    print("="*60)
    print(f"\nDataset: {dataroot}")
    print(f"Version: {version}")
    print(f"Output: {output_dir}")
    print(f"VLM Server: {vllm_api}")
    print(f"Model: {model}\n")
    
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=False
    )
    
    # Initialize annotator
    annotator = VLMAnnotator(
        api_base=vllm_api,
        model=model,
        dataloader=loader
    )
    
    # Get scenes to process
    all_scene_tokens = loader.get_scene_tokens()
    
    if scene_indices:
        scene_tokens = [all_scene_tokens[i] for i in scene_indices if i < len(all_scene_tokens)]
    else:
        scene_tokens = all_scene_tokens
    
    if max_scenes:
        scene_tokens = scene_tokens[:max_scenes]
    
    print(f"Processing {len(scene_tokens)} scenes out of {len(all_scene_tokens)} total\n")
    
    # Process each scene
    results_summary = []
    total_start_time = time.time()
    
    for idx, scene_token in enumerate(scene_tokens, 1):
        scene = loader.nusc.get('scene', scene_token)
        scene_name = scene['name']
        
        print(f"\n{'='*60}")
        print(f"Scene {idx}/{len(scene_tokens)}: {scene_name}")
        print(f"{'='*60}")
        
        scene_start_time = time.time()
        
        try:
            # Annotate scene
            results = annotator.annotate_scene(
                scene_token=scene_token,
                output_dir=output_dir,
                max_frames=max_frames_per_scene,
                camera_preference=camera_preference
            )
            
            scene_elapsed = time.time() - scene_start_time
            
            # Add to summary
            summary = {
                'scene_idx': idx - 1,
                'scene_token': scene_token,
                'scene_name': scene_name,
                'num_frames': results['num_frames'],
                'num_annotations': results['num_annotations'],
                'processing_time': scene_elapsed,
                'status': 'success'
            }
            results_summary.append(summary)
            
            print(f"\n✓ Scene completed in {scene_elapsed:.1f}s")
            print(f"  Frames: {results['num_frames']}")
            print(f"  Annotations: {results['num_annotations']}")
            
        except Exception as e:
            print(f"\n✗ Error processing scene {scene_name}: {e}")
            scene_elapsed = time.time() - scene_start_time
            
            summary = {
                'scene_idx': idx - 1,
                'scene_token': scene_token,
                'scene_name': scene_name,
                'error': str(e),
                'processing_time': scene_elapsed,
                'status': 'failed'
            }
            results_summary.append(summary)
            continue
    
    # Save batch summary
    total_elapsed = time.time() - total_start_time
    
    batch_summary = {
        'total_scenes': len(scene_tokens),
        'successful': sum(1 for s in results_summary if s['status'] == 'success'),
        'failed': sum(1 for s in results_summary if s['status'] == 'failed'),
        'total_frames': sum(s.get('num_frames', 0) for s in results_summary),
        'total_annotations': sum(s.get('num_annotations', 0) for s in results_summary),
        'total_time': total_elapsed,
        'scenes': results_summary
    }
    
    summary_file = Path(output_dir) / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*60)
    print("Batch Processing Summary")
    print("="*60)
    print(f"Total scenes processed: {len(scene_tokens)}")
    print(f"  Successful: {batch_summary['successful']}")
    print(f"  Failed: {batch_summary['failed']}")
    print(f"Total frames: {batch_summary['total_frames']}")
    print(f"Total annotations: {batch_summary['total_annotations']}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\nSummary saved to: {summary_file}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Batch annotate nuScenes scenes')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Dataset version')
    parser.add_argument('--output', type=str, default='outputs/batch_annotations', help='Output directory')
    parser.add_argument('--vllm-api', type=str, default='http://localhost:8001/v1', help='vLLM server URL')
    parser.add_argument('--model', type=str, default='llava-hf/llava-1.5-7b-hf', help='VLM model name')
    parser.add_argument('--scene-indices', type=int, nargs='+', help='Specific scene indices to process')
    parser.add_argument('--max-scenes', type=int, help='Maximum number of scenes to process')
    parser.add_argument('--max-frames', type=int, help='Maximum frames per scene')
    parser.add_argument('--cameras', type=str, nargs='+', 
                       default=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
                       help='Preferred camera order')
    
    args = parser.parse_args()
    
    annotate_multiple_scenes(
        dataroot=args.dataroot,
        version=args.version,
        output_dir=args.output,
        vllm_api=args.vllm_api,
        model=args.model,
        scene_indices=args.scene_indices,
        max_scenes=args.max_scenes,
        max_frames_per_scene=args.max_frames,
        camera_preference=args.cameras
    )


if __name__ == "__main__":
    main()


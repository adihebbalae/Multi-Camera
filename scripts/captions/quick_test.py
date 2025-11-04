"""
Quick test script to verify VLM annotation setup.

Tests the annotation pipeline on a single frame.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_annotator import VLMAnnotator
from nuscenes_dataloader import NuScenesLidarSegmentationLoader


def quick_test(dataroot: str, version: str = 'v1.0-mini'):
    """
    Quick test of annotation pipeline.
    
    Args:
        dataroot: Path to nuScenes dataset
        version: Dataset version
    """
    print("="*60)
    print("Quick Test: VLM Annotation Pipeline")
    print("="*60)
    
    # Initialize dataloader
    print("\n1. Initializing dataloader...")
    loader = NuScenesLidarSegmentationLoader(
        dataroot=dataroot,
        version=version,
        verbose=False
    )
    
    scene_tokens = loader.get_scene_tokens()
    print(f"   ✓ Found {len(scene_tokens)} scenes")
    
    # Get first frame
    print("\n2. Loading first frame...")
    scene_token = scene_tokens[0]
    sample_tokens = loader.get_samples_in_scene(scene_token)
    frame = loader.get_frame_data(sample_tokens[0])
    
    print(f"   ✓ Frame: {frame.sample_token}")
    print(f"   ✓ Objects: {len(frame.objects)}")
    
    # Show objects with camera visibility
    print("\n3. Objects with camera visibility:")
    visible_objects = [obj for obj in frame.objects if obj.visible_cameras]
    print(f"   ✓ {len(visible_objects)} objects visible in cameras")
    
    for i, obj in enumerate(visible_objects[:5], 1):
        class_name = obj.name.split('.')[-1]
        cameras = ', '.join(obj.visible_cameras)
        print(f"   {i}. {class_name:20s} -> [{cameras}]")
    
    if len(visible_objects) > 5:
        print(f"   ... and {len(visible_objects) - 5} more")
    
    # Check camera images
    print("\n4. Checking camera images...")
    for camera, token in list(frame.camera_tokens.items())[:3]:
        cam_data = loader.nusc.get('sample_data', token)
        img_path = loader.dataroot / cam_data['filename']
        exists = img_path.exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {camera:20s}: {img_path.name}")
    
    # Test VLM connection (optional)
    print("\n5. Testing VLM connection...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")
        
        # Try to list models
        models = client.models.list()
        print(f"   ✓ VLM server is running")
        print(f"   ✓ Available models: {len(models.data)}")
        for model in models.data:
            print(f"     - {model.id}")
    except Exception as e:
        print(f"   ✗ VLM server not accessible: {e}")
        print(f"   → Start vLLM server before annotation")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"✓ Dataloader: OK")
    print(f"✓ Frame loading: OK")
    print(f"✓ Camera visibility: OK ({len(visible_objects)} objects)")
    print(f"✓ Camera images: Check paths above")
    print(f"  VLM server: Check connection above")
    
    print("\n" + "="*60)
    print("Ready for Annotation!")
    print("="*60)
    print("\nNext steps:")
    print("1. Ensure vLLM server is running")
    print("2. Run: python vlm_annotator.py --dataroot {} --max-frames 1".format(dataroot))
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test annotation setup')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='Dataset version')
    
    args = parser.parse_args()
    
    quick_test(args.dataroot, args.version)


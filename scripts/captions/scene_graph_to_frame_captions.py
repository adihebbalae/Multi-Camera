"""
Scene Graph to Frame Captions Generator

This module uses GPT-4o-mini to generate detailed frame-by-frame captions
from scene graph data, including object activities and relationships.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    sys.exit(1)

from scenegraph.nuscenes_dataloader import NuScenesLidarSegmentationLoader
class FrameCaptionGenerator:
    """
    Generates detailed frame captions from scene graph data using GPT-4o-mini.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-5-mini",
        api_base: str = None,
        max_workers: int = 5,
        temperature: float = 0.7,
        scene_graphs_dir: str = "outputs/scene_graphs",
        instance_annotations_dir: str = "outputs/instance_annotations"
    ):
        """
        Initialize the frame caption generator.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model name (default: gpt-4o-mini)
            api_base: Optional custom API base URL for vLLM or other endpoints
            max_workers: Maximum number of parallel workers for batch processing
            temperature: Temperature for generation (0.0-1.0)
            scene_graphs_dir: Directory containing scene graph JSON files
            instance_annotations_dir: Directory containing instance annotation JSON files
        """
        self.model = model
        self.temperature = temperature
        self.max_workers = max_workers
        self.scene_graphs_dir = Path(scene_graphs_dir)
        self.instance_annotations_dir = Path(instance_annotations_dir)
        
        # Initialize OpenAI client
        if api_base:
            self.client = OpenAI(api_key=api_key or "EMPTY", base_url=api_base)
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Storage for instance activities
        self.instance_activities = {}
    
    def load_scene_data(self, scene_token: str) -> Dict[str, Any]:
        """
        Load scene graph and instance annotations based on scene token.
        
        Args:
            scene_token: Scene token identifier
            
        Returns:
            Scene graph data dictionary
        """
        # Load scene graph
        scene_graph_path = self.scene_graphs_dir / scene_token / "scene_graph.json"
        if not scene_graph_path.exists():
            raise FileNotFoundError(f"Scene graph not found: {scene_graph_path}")
        
        with open(scene_graph_path, 'r') as f:
            scene_graph_data = json.load(f)
        
        # Load instance annotations
        instance_annotations_path = self.instance_annotations_dir / f"{scene_token}_instance_annotations.json"
        if instance_annotations_path.exists():
            with open(instance_annotations_path, 'r') as f:
                instance_data = json.load(f)
                
            # Create mapping from instance_token to activity
            self.instance_activities = {
                ann['instance_token']: {
                    'activity': ann.get('activity', ''),
                    'description': ann.get('description', ''),
                    'object_class': ann.get('object_class', '')
                }
                for ann in instance_data.get('annotations', [])
            }
            print(f"Loaded {len(self.instance_activities)} instance annotations")
        else:
            print(f"Warning: Instance annotations not found: {instance_annotations_path}")
            self.instance_activities = {}
        
        return scene_graph_data
    
    def _format_object_info(self, obj: Dict[str, Any]) -> str:
        """
        Format object information for the prompt, including activity if available.
        
        Args:
            obj: Object dictionary from scene graph
            
        Returns:
            Formatted object string
        """
        obj_class = obj['object_class'].split('.')[-1]  # Get last part (e.g., 'car' from 'vehicle.car')
        object_id = obj.get('object_id', '')
        
        # Format position
        pos = obj['position']
        pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        
        # Format velocity if available
        velocity = obj.get('velocity')
        if velocity and velocity is not None:
            speed = (velocity[0]**2 + velocity[1]**2)**0.5
            vel_str = f", speed: {speed:.1f} m/s"
        else:
            vel_str = ", stationary"
        
        # Format attributes
        attrs = obj.get('attributes', [])
        attr_str = ""
        if attrs:
            # Clean up attribute names
            clean_attrs = [attr.split('.')[-1] for attr in attrs]
            attr_str = f", {', '.join(clean_attrs)}"
        
        # Add activity if available
        activity_str = ""
        if object_id in self.instance_activities:
            activity = self.instance_activities[object_id].get('activity', '')
            if activity:
                activity_str = f"\n  Activity: {activity}"
            description = self.instance_activities[object_id].get('description', '')
            if description:
                description_str = f"\n  Description: {description}"
        
        return f"- {obj_class} at position {pos_str}{vel_str}{attr_str}{activity_str}{description_str}"
    
    def _format_relationships(self, relationships: List[Dict[str, Any]], 
                             objects: List[Dict[str, Any]]) -> str:
        """
        Format relationship information for the prompt.
        
        Args:
            relationships: List of relationship dictionaries
            objects: List of object dictionaries
            
        Returns:
            Formatted relationships string
        """
        if not relationships:
            return "No significant spatial relationships detected."
        
        # Create object ID to class mapping
        obj_map = {obj['object_id']: obj['object_class'].split('.')[-1] 
                   for obj in objects}
        
        rel_strings = []
        for rel in relationships:
            source = obj_map.get(rel['source_id'], 'unknown')
            target = obj_map.get(rel['target_id'], 'unknown')
            rel_type = rel['relationship_type']
            distance = rel.get('distance')
            
            if distance is not None:
                rel_str = f"- {source} is {rel_type} {target} (distance: {distance:.1f}m)"
            else:
                rel_str = f"- {source} is {rel_type} {target}"
            
            rel_strings.append(rel_str)
        
        return "\n".join(rel_strings)
    
    def _create_frame_prompt(self, frame_data: Dict[str, Any]) -> str:
        """
        Create a detailed prompt for frame caption generation.
        
        Args:
            frame_data: Frame data dictionary from scene graph
            
        Returns:
            Formatted prompt string
        """
        frame_idx = frame_data['frame_idx']
        objects = frame_data['objects']
        relationships = frame_data.get('relationships', [])
        
        # Format objects
        object_info = "\n".join([self._format_object_info(obj) for obj in objects])
        
        # Format relationships
        rel_info = self._format_relationships(relationships, objects)
        
        # Count objects by category
        obj_counts = {}
        for obj in objects:
            category = obj['object_class'].split('.')[0]  # Get top-level category
            obj_counts[category] = obj_counts.get(category, 0) + 1
        
        count_str = ", ".join([f"{count} {cat}" for cat, count in obj_counts.items()])
        
        prompt = f"""You are an expert at describing driving scenes. Given the following scene information from frame {frame_idx}, generate a detailed, natural language caption that describes what is happening in this frame.

Scene Statistics:
- Total objects: {len(objects)} ({count_str})
- Relationships: {len(relationships)}

Objects in the scene (with activities where available):
{object_info}

Spatial Relationships:
{rel_info}

Generate a comprehensive, natural-sounding caption that:
1. Describes the overall scene composition and context
2. Highlights all object activities, descriptions and movements (using the activity information provided)
3. Mentions all important spatial relationships between objects
4. Uses natural language without listing technical details
5. Integrates all the specific object activities and descriptions into the narrative flow
6. Don't make up any information for instance names of companies that is not provided in the scene graph or instance annotations

Caption:"""
        # print(prompt)
        return prompt
    
    def generate_frame_caption(
        self, 
        frame_data: Dict[str, Any],
        retry_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Generate caption for a single frame.
        
        Args:
            frame_data: Frame data dictionary from scene graph
            retry_on_error: Whether to retry on API errors
            
        Returns:
            Dictionary containing frame info and generated caption
        """
        frame_idx = frame_data['frame_idx']
        
        try:
            # Create prompt
            prompt = self._create_frame_prompt(frame_data)
            
            # Call GPT-4o-mini
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at describing autonomous driving scenes clearly and accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=300
            )
            
            caption = response.choices[0].message.content.strip()
            # print("Frame Caption: ", caption)
            return {
                'frame_idx': frame_idx,
                'sample_token': frame_data.get('sample_token', ''),
                'timestamp': frame_data.get('timestamp', 0),
                'caption': caption,
                'num_objects': len(frame_data['objects']),
                'num_relationships': len(frame_data.get('relationships', [])),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error generating caption for frame {frame_idx}: {e}")
            return {
                'frame_idx': frame_idx,
                'sample_token': frame_data.get('sample_token', ''),
                'timestamp': frame_data.get('timestamp', 0),
                'caption': None,
                'error': str(e),
                'status': 'error'
            }
    
    def generate_captions_for_scene(
        self,
        scene_graph_data: Dict[str, Any],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate captions for all frames in a scene.
        
        Args:
            scene_graph_data: Complete scene graph data
            parallel: Whether to process frames in parallel
            
        Returns:
            List of frame caption dictionaries
        """
        frames = scene_graph_data['frames']
        
        if parallel and self.max_workers > 1:
            # Process frames in parallel
            captions = []
            futures = []
            
            for frame_data in frames:
                future = self.executor.submit(
                    self.generate_frame_caption,
                    frame_data
                )
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Generating captions"):
                captions.append(future.result())
            
            # Sort by frame index
            captions.sort(key=lambda x: x['frame_idx'])
        else:
            # Process frames sequentially
            captions = []
            for frame_data in tqdm(frames, desc="Generating captions"):
                caption = self.generate_frame_caption(frame_data)
                captions.append(caption)
        
        return captions
    
    def save_captions(
        self,
        captions: List[Dict[str, Any]],
        output_path: str,
        scene_token: str = ''
    ):
        """
        Save generated captions to JSON file.
        
        Args:
            captions: List of caption dictionaries
            output_path: Path to save the captions JSON
            scene_token: Scene token identifier
        """
        output_data = {
            'scene_token': scene_token,
            'model': self.model,
            'temperature': self.temperature,
            'num_frames': len(captions),
            'captions': captions
        }
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Print statistics
        success_count = sum(1 for c in captions if c['status'] == 'success')
        error_count = len(captions) - success_count
        print(f"\nSaved {success_count} captions to {output_path}")
        if error_count > 0:
            print(f"Warning: {error_count} frames failed to generate captions")
    
    def generate_summary_statistics(self, captions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for the captions.
        
        Args:
            captions: List of caption dictionaries
            
        Returns:
            Dictionary of statistics
        """
        total_frames = len(captions)
        successful = [c for c in captions if c['status'] == 'success']
        
        if not successful:
            return {
                'total_frames': total_frames,
                'successful_captions': 0,
                'failed_captions': total_frames,
                'avg_caption_length': 0
            }
        
        avg_length = sum(len(c['caption']) for c in successful) / len(successful)
        avg_objects = sum(c['num_objects'] for c in successful) / len(successful)
        avg_relationships = sum(c['num_relationships'] for c in successful) / len(successful)
        
        return {
            'total_frames': total_frames,
            'successful_captions': len(successful),
            'failed_captions': total_frames - len(successful),
            'avg_caption_length': avg_length,
            'avg_objects_per_frame': avg_objects,
            'avg_relationships_per_frame': avg_relationships
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate frame captions from scene graph data using GPT-4o-mini'
    )
    
    parser.add_argument(
        '--scene-token',
        type=str,
        required=False,
        help='Scene token identifier (e.g., 0c601ff2bf004fccafec366b08bf29e2)'
    )
    
    parser.add_argument(
        '--num_scenes',
        type=int,
        default=1,
        help='Number of scenes to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the frame captions JSON (defaults to outputs/captions/{scene_token}_captions.json)'
    )
    
    parser.add_argument(
        '--scene-graphs-dir',
        type=str,
        default='outputs/scene_graphs',
        help='Directory containing scene graph JSON files (default: outputs/scene_graphs)'
    )
    
    parser.add_argument(
        '--instance-annotations-dir',
        type=str,
        default='outputs/instance_annotations',
        help='Directory containing instance annotation JSON files (default: outputs/instance_annotations)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (defaults to OPENAI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--api-base',
        type=str,
        default=None,
        help='Custom API base URL (for vLLM or other endpoints)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5-mini',
        help='Model name (default: gpt-5-mini)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for generation (default: 0.7)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of parallel workers (default: 5)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing'
    )
    
    parser.add_argument(
        '--dataroot',
        type=str,
        default='/nas/standard_datasets/nuscenes',
        help='Path to nuScenes dataset (default: data/nuscenes)'
    )
    parser.add_argument( 
        '--version',
        type=str,
        default='v1.0-trainval',
        help='Version of nuScenes dataset (default: v1.0-trainval)'
    )
    return parser.parse_args()

def caption_scenes(args, generator, scene_token=None):
    """Caption scenes."""

    # Load scene data (scene graph + instance annotations)

    if scene_token is not None:
        args.scene_token = scene_token
        args.output = f'outputs/captions/{args.scene_token}_captions.json'

    # Load the captions if they exist
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            captions = json.load(f)
        print(f"Loaded {len(captions)} captions from {args.output}")
        all_success = True
        for caption in captions['captions']:
            if caption['status'] != 'success':
                all_success = False
                break
        if all_success:
            print(f"All captions already generated for {args.scene_token}")
            return
        else:
            print(f"Some captions already generated for {args.scene_token}, continuing...")

    print(f"\nLoading scene data for token: {args.scene_token}")
    try:
        scene_graph_data = generator.load_scene_data(args.scene_token)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    scene_token = scene_graph_data.get('scene_token', args.scene_token)
    num_frames = len(scene_graph_data['frames'])
    print(f"Loaded scene graph with {num_frames} frames")
    
    # Generate captions
    print("\nGenerating frame captions with activity information...")
    captions = generator.generate_captions_for_scene(
        scene_graph_data,
        parallel=not args.no_parallel
    )
    
    # Save captions
    generator.save_captions(captions, args.output, args.scene_token)

    # Print statistics
    stats = generator.generate_summary_statistics(captions)
    print("\n=== Caption Generation Statistics ===")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Successful captions: {stats['successful_captions']}")
    print(f"Failed captions: {stats['failed_captions']}")
    if stats['successful_captions'] > 0:
        print(f"Average caption length: {stats['avg_caption_length']:.1f} characters")
        print(f"Average objects per frame: {stats['avg_objects_per_frame']:.1f}")
        print(f"Average relationships per frame: {stats['avg_relationships_per_frame']:.1f}")
    
    print("\n✅ Caption generation complete!")
    
def main():
    """Main execution function."""
    args = parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        os.makedirs('outputs/captions', exist_ok=True)
        args.output = f'outputs/captions/{args.scene_token}_captions.json'
    
    # Initialize caption generator
    print(f"Initializing caption generator with model: {args.model}")
    print(f"Scene graphs directory: {args.scene_graphs_dir}")
    print(f"Instance annotations directory: {args.instance_annotations_dir}")
    loader = NuScenesLidarSegmentationLoader(
        dataroot=args.dataroot,
        version=args.version,
        verbose=True
    )
    generator = FrameCaptionGenerator(
            api_key=args.api_key,
            model=args.model,
            api_base=args.api_base,
            max_workers=args.max_workers,
            temperature=args.temperature,
            scene_graphs_dir=args.scene_graphs_dir,
            instance_annotations_dir=args.instance_annotations_dir
    )
    if args.num_scenes > 1 or args.scene_token is None:
        scene_tokens = loader.get_scene_tokens()
        for i in tqdm(range(args.num_scenes), desc="Processing scenes"):
            scene_token = scene_tokens[i]
            print(f"Processing scene {i} of {args.num_scenes}: {scene_token}")
            caption_scenes(args, generator, scene_token)
    else:
        caption_scenes(args, generator)
        
       


if __name__ == "__main__":
    main()


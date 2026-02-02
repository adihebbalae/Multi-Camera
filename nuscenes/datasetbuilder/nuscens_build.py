"""
NuScenes Dataset Builder for Counting and Spatial Relationship Questions

This script extracts relevant information from nuScenes scene graphs to generate:
1. Counting questions: "How many X objects are in the video?"
2. Spatial relationship questions (MCQ): directional relationships between two objects.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import openai
from waymo.scenegraph.nuscenes_dataloader import NuScenesLidarSegmentationLoader
import argparse

class QASample:
    def __init__(self, scene_token: str, question_type: str, question: str, answer: str, metadata: Dict[str, Any]):
        self.scene_token = scene_token
        self.question_type = question_type
        self.question = question
        self.answer = answer
        self.metadata = metadata
    
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scene_token': self.scene_token,
            'question_type': self.question_type,
            'question': self.question,
            'answer': self.answer,
            'metadata': self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> 'QASample':
        return QASample(
            scene_token=data['scene_token'],
            question_type=data['question_type'],
            question=data['question'],
            answer=data['answer'],
            metadata=data['metadata']
        )

class QAGenerator:
    """
    Base QA generator with generic utilities for prompts, captions, and GPT calls.
    Specialized QA generators (e.g., SpatialQAGenerator) should extend this.
    """
    def __init__(self, prompts_dir: Optional[Path] = None, captions_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or (Path(__file__).parent / "prompts")
        self.captions_dir = captions_dir or Path("/home/hg22723/projects/Multi-Camera/outputs/captions")
        self.scene_graphs_dir = scene_graphs_dir or Path("/home/hg22723/projects/Multi-Camera/outputs/scene_graphs")
        self.instance_annotations = instance_annotations_dir or Path("/home/hg22723/projects/Multi-Camera/outputs/instance_annotations")
    
    def gpt(self, prompt: str, api_key: Optional[str] = None, *, model: str = "gpt-4",
            temperature: float = 0.7, max_tokens: int = 500) -> str:
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    def format_prompt_input(self, scene_info: Dict[str, Any], counting_prompt: str) -> str:
        
        raise NotImplementedError("This method should be implemented by the subclass")

    def load_prompts_from_disk(self) -> Tuple[str, str]:
        
        raise NotImplementedError("This method should be implemented by the subclass")


    def load_captions_for_scene(self, scene_token: str) -> Optional[Dict[int, str]]:
        captions_file = self.captions_dir / f"{scene_token}_captions.json"
        if not captions_file.exists():
            return None
        data = json.loads(captions_file.read_text())
        captions = {}
        for item in data.get('captions', []):
            captions[item.get('frame_idx')] = item.get('caption', '')
        return captions

    def load_scene_graph(self, scene_token: str) -> Dict[str, Any]:
        scene_graph_file = self.scene_graphs_dir / f"{scene_token}/scene_graph.json"
        if not scene_graph_file.exists():
            return None
        with open(scene_graph_file, 'r') as f:
            scene_graph = json.load(f)
        return scene_graph
    
    def load_instance_annotations(self, scene_token: str) -> Dict[str, Any]:
        instance_annotations_file = self.instance_annotations_dir / f"{scene_token}_instance_annotations.json"
        if not instance_annotations_file.exists():
            return None
        with open(instance_annotations_file, 'r') as f:
            instance_annotations = json.load(f)
        return instance_annotations
    
    # Returns the A sample
    def construct_qa_sample(self, scene_token: str, question_type: str, question: str, answer: str, metadata: Dict[str, Any]) -> QASample:
        return QASample(scene_token=scene_token, question_type=question_type, question=question, answer=answer, metadata=metadata)


class CountingQAGenerator(QAGenerator):
    """
    Counting QA generator: builds counting questions using object counts
    contained in the scene graph and a short scene description.
    """
    def format_prompt_input(self, scene_info: Dict[str, Any], counting_prompt: str) -> str:
        object_count_str = "\n".join([f"- {cls}: {count}" for cls, count in scene_info['object_counts'].items()])
        return counting_prompt.format(
            scene_token=scene_info['scene_token'],
            num_frames=scene_info['num_frames'],
            object_counts=object_count_str
        )
        
    def load_prompts_from_disk(self) -> Tuple[str, str]:
        prompts_dir = self.prompts_dir
        counting_prompt_path = prompts_dir / "counting_questions.txt"
        if not counting_prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt files not found in {prompts_dir}. "
                "Please ensure counting_questions.txt exists."
            )
        return counting_prompt_path.read_text()
    
    def generate_for_scene(self, scene_token: str, api_key: Optional[str] = None) -> QASample:
        scene_info = self.load_scene_graph(scene_token)
        counting_prompt = self.load_prompts_from_disk()
        counting_input = self.format_prompt_input(scene_info, counting_prompt)
        counting_question = self.gpt(counting_input, api_key=api_key)
        return self.construct_qa_sample(scene_token, "counting", counting_question, scene_info['object_counts'], scene_info['num_frames'])
   
    
class SpatialQAGenerator(QAGenerator):
    """
    TODO: Fix accordingly Spatial QA generator: builds MCQ questions using directional relationships
    contained in the scene graph and a short scene description.
    """
    def _select_candidate(self, directional: List[Dict[str, Any]], frames_meta: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidate: Optional[Dict[str, Any]] = None
        for rel in directional:
            frame_idx = rel['frame_idx']
            frame_objects = frames_meta.get(frame_idx, {}).get('objects', [])
            num_source_class_in_frame = sum(1 for o in frame_objects if o['class'] == rel['source']['class'])
            if num_source_class_in_frame == 1:
                candidate = rel
                break
        if candidate is None and directional:
            candidate = directional[0]
        return candidate
    
    
    def format_prompt_input(self, scene_info: Dict[str, Any], spatial_prompt: str) -> str:
        

        object_count_str = "\n".join([f"- {cls}: {count}" for cls, count in scene_info['object_counts'].items()])
        return spatial_prompt.format(
            scene_token=scene_info['scene_token'],
            num_frames=scene_info['num_frames'],
            object_counts=object_count_str
        )

    def load_prompts_from_disk(self) -> Tuple[str, str]:
        
        prompts_dir = self.prompts_dir
        spatial_prompt_path = prompts_dir / "spatial_questions.txt"
        if not counting_prompt_path.exists() or not spatial_prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt files not found in {prompts_dir}. "
                "Please ensure counting_questions.txt and spatial_questions.txt exist."
            )
        return counting_prompt_path.read_text(), spatial_prompt_path.read_text()

    def build_spatial_input(self, scene_info: Dict[str, Any], spatial_prompt: str) -> str:
        directional = scene_info.get('directional_relationships', [])
        frames_meta = {f['frame_idx']: f for f in scene_info.get('frames', [])}
        candidate = self._select_candidate(directional, frames_meta)

        captions_map = _load_scene_captions(scene_info['scene_token']) or {}
        frame_idx_for_prompt = candidate['frame_idx'] if candidate else 0
        scene_description = captions_map.get(frame_idx_for_prompt, '')[:600]

        rels_same_frame = [r for r in directional if r['frame_idx'] == frame_idx_for_prompt][:6]
        rel_lines = []
        for r in rels_same_frame:
            rel_lines.append(
                f"- Frame {r['frame_idx']}: {r['source']['class']} is {r['type'].replace('_', ' ')} {r['target']['class']}"
            )
        directional_relationships_str = "\n".join(rel_lines) if rel_lines else "- None"

        return spatial_prompt.format(
            scene_token=scene_info['scene_token'],
            frame_idx=frame_idx_for_prompt,
            scene_description=scene_description if scene_description else "A driving scene with various road users.",
            directional_relationships=directional_relationships_str,
            object_counts="\n".join([f"- {cls}: {count}" for cls, count in scene_info['object_counts'].items()])
        )

    def generate_for_scene(self, scene_info: Dict[str, Any], counting_prompt: str,
                           spatial_prompt: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        counting_input = self.format_counting_input(scene_info, counting_prompt)
        spatial_input = self.build_spatial_input(scene_info, spatial_prompt)
        counting_question = self.gpt(counting_input, api_key=api_key)
        spatial_question = self.gpt(spatial_input, api_key=api_key)
        return {
            'scene_token': scene_info['scene_token'],
            'counting_question': counting_question,
            'spatial_question': spatial_question,
            'metadata': {
                'num_frames': scene_info['num_frames'],
                'object_counts': scene_info['object_counts'],
                'num_directional_relationships': len(scene_info.get('directional_relationships') or [])
            }
        }
    
def extract_scene_info(scene_graph: str) -> Dict[str, Any]:
    """
    Extract relevant information from a scene graph for question generation.
    
    Returns:
        Dict containing:
        - object_counts: Dict of object_class -> count
        - object_instances: Dict of object_class -> list of unique object_ids
        - spatial_relationships: List of tuples (obj1, obj2, relationship)
        - scene_token: The scene identifier
    """
    print(f"Processing {scene_graph_path}...")
    
    with open(scene_graph_path, 'r') as f:
        # Read file in chunks to handle large files
        first_line = f.readline()
        scene_data = json.loads(first_line + f.read())
    
    scene_token = scene_data['scene_token']
    frames = scene_data.get('frames', [])
    
    # Track unique objects across all frames
    unique_objects = defaultdict(set)  # object_class -> set of object_ids
    object_info = {}  # object_id -> {class, positions, frames_seen, which_cameras}
    
    # Process each frame
    for frame in frames:
        objects = frame.get('objects', [])
        
        for obj in objects:
            obj_id = obj['object_id']
            obj_class = obj['object_class']
            position = obj['position']
            
            # Track unique objects
            unique_objects[obj_class].add(obj_id)
            
            # Track object information
            if obj_id not in object_info:
                object_info[obj_id] = {
                    'class': obj_class,
                    'positions': [],
                    'frames_seen': 0,
                    'which_cameras': []
                }
            
            object_info[obj_id]['positions'].append(position)
            object_info[obj_id]['frames_seen'] += 1
            object_info[obj_id]['which_cameras'].append(obj['visible_cameras'])
            
    
    # Calculate average positions
    for obj_id, info in object_info.items():
        positions = np.array(info['positions'])
        info['avg_position'] = positions.mean(axis=0).tolist()
    
    # Count objects per class
    object_counts = {obj_class: len(ids) for obj_class, ids in unique_objects.items()}
    
    # Collect directional relationships directly from scene graph frames
    directional_types = {"in_front", "behind", "left", "right"}
    directional_relationships: List[Dict[str, Any]] = []

    for frame in frames:
        frame_idx = frame.get('frame_idx')
        objects = frame.get('objects', [])
        id_to_class = {o['object_id']: o['object_class'] for o in objects}

        for rel in frame.get('relationships', []) or []:
            rel_type = rel.get('relationship_type')
            if rel_type in directional_types:
                source_id = rel.get('source_id')
                target_id = rel.get('target_id')
                directional_relationships.append({
                    'frame_idx': frame_idx,
                    'type': rel_type,
                    'distance': rel.get('distance'),
                    'source': {
                        'id': source_id,
                        'class': id_to_class.get(source_id, 'unknown')
                    },
                    'target': {
                        'id': target_id,
                        'class': id_to_class.get(target_id, 'unknown')
                    }
                })
    
    return {
        'scene_token': scene_token,
        'num_frames': len(frames),
        'object_counts': object_counts,
        'object_instances': {k: list(v) for k, v in unique_objects.items()},
        'directional_relationships': directional_relationships,
        'total_unique_objects': sum(len(ids) for ids in unique_objects.values()),
        'frames': [{'frame_idx': f.get('frame_idx'),
                    'objects': [{'id': o['object_id'], 'class': o['object_class']} for o in f.get('objects', [])]}
                   for f in frames]
    }





# def process_all_scenes(scene_graphs_dir: str, output_dir: str, limit: int = None):
#     """
#     Process all scene graphs and extract information.
    
#     Args:
#         scene_graphs_dir: Directory containing scene graph subdirectories
#         output_dir: Directory to save extracted information
#         limit: Maximum number of scenes to process (None for all)
#     """
#     scene_graphs_path = Path(scene_graphs_dir)
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     # Find all scene graph files
#     scene_graph_files = list(scene_graphs_path.glob("*/scene_graph.json"))
    
#     if limit:
#         scene_graph_files = scene_graph_files[:limit]
    
#     print(f"Found {len(scene_graph_files)} scene graphs to process")
    
#     all_scene_info = []
    
#     for sg_file in scene_graph_files:
#         try:
#             scene_info = extract_scene_info(str(sg_file))
#             all_scene_info.append(scene_info)
#         except Exception as e:
#             print(f"Error processing {sg_file}: {e}")
#             continue
    
#     # Save extracted information
#     output_file = output_path / "extracted_scene_info.json"
#     with open(output_file, 'w') as f:
#         json.dump(all_scene_info, f, indent=2)
    
#     print(f"\nExtracted information saved to {output_file}")
#     print(f"Processed {len(all_scene_info)} scenes successfully")
    
#     # Print summary statistics
#     print("\n=== Summary Statistics ===")
#     total_object_counts = defaultdict(int)
#     for scene in all_scene_info:
#         for obj_class, count in scene['object_counts'].items():
#             total_object_counts[obj_class] += count
    
#     print("\nTotal object counts across all scenes:")
#     for obj_class, count in sorted(total_object_counts.items(), key=lambda x: x[1], reverse=True):
#         print(f"  {obj_class}: {count}")
    
#     print(f"\nTotal directional relationships found: {sum(len(s['directional_relationships']) for s in all_scene_info)}")
    
#     return all_scene_info



def process_questions(qa_generator: QAGenerator, scene_tokens: List[str], api_key: Optional[str] = None) -> List[QASample]:
    qa_samples = []
    for scene_token in scene_tokens:
        qa_sample = qa_generator.generate_for_scene(scene_token, api_key=api_key)
        qa_samples.append(qa_sample)
    return qa_samples

def parse_args():
    parser = argparse.ArgumentParser(description='Process scene graphs and extract information.')
    parser.add_argument('--prompts_dir', type=str, default="/home/hg22723/projects/Multi-Camera/datasetbuilder/prompts", help='Directory containing prompts')
    parser.add_argument('--scene_graphs_dir', type=str, default="/home/hg22723/projects/Multi-Camera/outputs/scene_graphs", help='Directory containing scene graph subdirectories')
    parser.add_argument('--output_dir', type=str, default="/home/hg22723/projects/Multi-Camera/outputs", help='Directory to save extracted information')
    parser.add_argument('--limit', type=int, default=250, help='Maximum number of scenes to process')
    parser.add_argument('--question_type', type=str, default="counting", help='Question type: counting, spatial, temporal, event_ordering, causality, perception, summarization, which_camera')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key')
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Configuration
    SCENE_GRAPHS_DIR = "/home/hg22723/projects/Multi-Camera/outputs/scene_graphs"
    OUTPUT_DIR = "/home/hg22723/projects/Multi-Camera/datasetbuilder/outputs"
    
    args = parse_args()
    if args.api_key is None:
        args.api_key = os.getenv('OPENAI_API_KEY')
        
    
    if args.question_type == "counting":
        qa_generator = CountingQAGenerator(prompts_dir=args.prompts_dir, scene_graphs_dir=args.scene_graphs_dir)
    elif args.question_type == "spatial":
        qa_generator = SpatialQAGenerator(prompts_dir=args.prompts_dir, scene_graphs_dir=args.scene_graphs_dir)
    else:
        raise NotImplementedError(f"Question type {args.question_type} not implemented")
    
    # Extract information from scene graphs
    print("=" * 80)
    print("NuScenes Dataset Builder - Extracting Scene Information")
    print("=" * 80)
    
    nuscenes_dataloader = NuScenesLidarSegmentationLoader(dataroot=args.dataroot, version=args.version)
    scene_tokens = nuscenes_dataloader.get_scene_tokens()
    qa_samples = process_questions(qa_generator, scene_tokens, api_key=args.api_key)
    
    print("\n" + "=" * 80)
    print("Extraction complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the extracted information in datasetbuilder/outputs/extracted_scene_info.json")
    print("2. Create prompt templates in datasetbuilder/prompts/")
    print("3. Run question generation with: python -c 'from datasetbuilder.nuscens_build import generate_questions; generate_questions()'")


if __name__ == "__main__":
    main()



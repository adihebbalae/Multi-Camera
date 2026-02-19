"""
NuScenes Dataset Builder for Counting and Spatial Relationship Questions

This script extracts relevant information from nuScenes scene graphs to generate:
1. Counting questions: "How many X objects are in the video?"
2. Spatial relationship questions (MCQ): directional relationships between two objects.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import random
load_dotenv()

from waymo.scenegraph.nuscenes_dataloader import NuScenesLidarSegmentationLoader


class QASample:
    def __init__(self, scene_token: str, question_type: str, question: str, answer: str, metadata: Dict[str, Any]):
        self.scene_token = scene_token
        self.question_type = question_type
        self.question = question
        self.answer = answer
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_token": self.scene_token,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata,
        }

    def from_dict(self, data: Dict[str, Any]) -> "QASample":
        return QASample(
            scene_token=data["scene_token"],
            question_type=data["question_type"],
            question=data["question"],
            answer=data["answer"],
            metadata=data["metadata"],
        )


class QAGenerator:
    """
    Base QA generator with generic utilities for prompts, captions, and GPT calls.
    Specialized QA generators (e.g., SpatialQAGenerator) should extend this.
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        captions_dir: Optional[Path] = None,
        scene_graphs_dir: Optional[Path] = None,
        instance_annotations_dir: Optional[Path] = None,
        gpt_logs_dir: Optional[Path] = None,
    ):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else (Path(__file__).parent / "prompts")
        self.captions_dir = (
            Path(captions_dir) if captions_dir else Path("/home/hg22723/projects/Multi-Camera/outputs/captions")
        )
        self.scene_graphs_dir = (
            Path(scene_graphs_dir)
            if scene_graphs_dir
            else Path("/home/hg22723/projects/Multi-Camera/outputs/scene_graphs")
        )
        self.instance_annotations_dir = (
            Path(instance_annotations_dir)
            if instance_annotations_dir
            else Path("/home/hg22723/projects/Multi-Camera/outputs/instance_annotations")
        )
        self.gpt_logs_dir = Path(gpt_logs_dir) if gpt_logs_dir else None

    def _sanitize_log_filename(self, scene_token: str) -> str:
        """Create a filesystem-safe filename from scene_token."""
        safe = re.sub(r"[^\w\-.]", "_", str(scene_token))
        return (safe[:128] + "_") if len(safe) > 128 else safe

    def _log_gpt_call(
        self,
        input_text: str,
        output_text: str,
        model: str,
        temperature: float,
        max_tokens: int,
        scene_token: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Write GPT conversation to log file under gpt_logs_dir."""
        if self.gpt_logs_dir is None:
            return
        self.gpt_logs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._sanitize_log_filename(scene_token or "unknown")
        log_file = self.gpt_logs_dir / f"{safe_name}.json"
        log_entry = {
            "input": input_text,
            "output": output_text,
            "metadata": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "scene_token": scene_token,
                **metadata,
            },
        }
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)

    def gpt(
        self,
        prompt: str,
        api_key: Optional[str] = None,
        *,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 500,
        **log_metadata: Any,
    ) -> str:
        key = api_key or os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content.strip()

        if self.gpt_logs_dir is not None:
            self._log_gpt_call(
                input_text=prompt,
                output_text=result,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **log_metadata,
            )
        return result

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
        for item in data.get("captions", []):
            captions[item.get("frame_idx")] = item.get("caption", "")
        return captions

    def load_scene_graph(self, scene_token: str) -> Optional[Dict[str, Any]]:
        scene_graph_file = self.scene_graphs_dir / f"{scene_token}/scene_graph.json"
        if not scene_graph_file.exists():
            return None
        with open(scene_graph_file, "r") as f:
            scene_graph = json.load(f)
        return self._extract_scene_info_from_dict(scene_graph)

    def _extract_scene_info_from_dict(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw scene graph dict into format with object_counts, num_frames, etc."""
        scene_token = scene_data.get("scene_token", "")
        frames = scene_data.get("frames", [])

        unique_objects = defaultdict(set)
        for frame in frames:
            for obj in frame.get("objects", []):
                obj_id = obj.get("object_id") or obj.get("annotation_token", "")
                obj_class = obj.get("object_class", "unknown")
                if obj_id:
                    unique_objects[obj_class].add(obj_id)

        object_counts = {obj_class: len(ids) for obj_class, ids in unique_objects.items()}
        directional_types = {"in_front", "behind", "left", "right"}
        directional_relationships: List[Dict[str, Any]] = []

        for frame in frames:
            frame_idx = frame.get("frame_idx")
            objects = frame.get("objects", [])
            id_to_class = {
                (o.get("object_id") or o.get("annotation_token", "")): o.get("object_class", "unknown") for o in objects
            }
            for rel in frame.get("relationships", []) or []:
                rel_type = rel.get("relationship_type")
                if rel_type in directional_types:
                    source_id = rel.get("source_id", "")
                    target_id = rel.get("target_id", "")
                    directional_relationships.append(
                        {
                            "frame_idx": frame_idx,
                            "type": rel_type,
                            "distance": rel.get("distance"),
                            "source": {"id": source_id, "class": id_to_class.get(source_id, "unknown")},
                            "target": {"id": target_id, "class": id_to_class.get(target_id, "unknown")},
                        }
                    )

        return {
            "scene_token": scene_token,
            "num_frames": len(frames),
            "object_counts": object_counts,
            "directional_relationships": directional_relationships,
            "frames": [{"frame_idx": f.get("frame_idx"), "objects": f.get("objects", [])} for f in frames],
        }

    def load_instance_annotations(self, scene_token: str) -> Dict[str, Any]:
        instance_annotations_file = self.instance_annotations_dir / f"{scene_token}_instance_annotations.json"
        if not instance_annotations_file.exists():
            return None
        with open(instance_annotations_file, "r") as f:
            instance_annotations = json.load(f)
        return instance_annotations

    # Returns the A sample
    def construct_qa_sample(
        self, scene_token: str, question_type: str, question: str, answer: str, metadata: Dict[str, Any]
    ) -> QASample:
        return QASample(
            scene_token=scene_token, question_type=question_type, question=question, answer=answer, metadata=metadata
        )


class CountingQAGenerator(QAGenerator):
    """
    Counting QA generator: builds counting questions using object counts
    contained in the scene graph and a short scene description.
    """

    def format_prompt_input(self, scene_info: Dict[str, Any], counting_prompt: str) -> str:
        object_count_str = "\n".join([f"- {cls}: {count}" for cls, count in scene_info["object_counts"].items()])
        return counting_prompt.format(
            scene_token=scene_info["scene_token"], num_frames=scene_info["num_frames"], object_counts=object_count_str
        )

    def load_prompts_from_disk(self) -> Tuple[str, str]:
        prompts_dir = self.prompts_dir
        counting_prompt_path = prompts_dir / "counting_questions.txt"
        if not counting_prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt files not found in {prompts_dir}. " "Please ensure counting_questions.txt exists."
            )
        return counting_prompt_path.read_text()

    def generate_for_scene(self, scene_token: str, api_key: Optional[str] = None) -> Optional[QASample]:
        scene_info = self.load_scene_graph(scene_token)
        if scene_info is None:
            return None
        counting_prompt = self.load_prompts_from_disk()
        counting_input = self.format_prompt_input(scene_info, counting_prompt)
        counting_question = self.gpt(
            counting_input, api_key=api_key, scene_token=scene_token, question_type="counting"
        )
        return self.construct_qa_sample(
            scene_token,
            "counting",
            counting_question,
            scene_info["object_counts"],
            metadata={"num_frames": scene_info["num_frames"], "object_counts": scene_info["object_counts"]},
        )


class SpatialQAGenerator(QAGenerator):
    """
    TODO: Fix accordingly Spatial QA generator: builds MCQ questions using directional relationships
    contained in the scene graph and a short scene description.
    """

    def _select_candidate(
        self, directional: List[Dict[str, Any]], frames_meta: Dict[int, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        candidate: Optional[Dict[str, Any]] = None
        for rel in directional:
            frame_idx = rel["frame_idx"]
            frame_objects = frames_meta.get(frame_idx, {}).get("objects", [])
            obj_class = lambda o: o.get("object_class") or o.get("class", "")
            num_source_class_in_frame = sum(1 for o in frame_objects if obj_class(o) == rel["source"]["class"])
            if num_source_class_in_frame == 1:
                candidate = rel
                break
        if candidate is None and directional:
            candidate = directional[0]
        return candidate

    def format_prompt_input(self, scene_info: Dict[str, Any], spatial_prompt: str) -> str:

        object_count_str = "\n".join([f"- {cls}: {count}" for cls, count in scene_info["object_counts"].items()])
        return spatial_prompt.format(
            scene_token=scene_info["scene_token"], num_frames=scene_info["num_frames"], object_counts=object_count_str
        )

    def load_prompts_from_disk(self) -> str:
        prompts_dir = self.prompts_dir
        spatial_prompt_path = prompts_dir / "spatial_questions.txt"
        if not spatial_prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found in {prompts_dir}. " "Please ensure spatial_questions.txt exists."
            )
        return spatial_prompt_path.read_text()

    def build_spatial_input(self, scene_info: Dict[str, Any], spatial_prompt: str) -> str:
        directional = scene_info.get("directional_relationships", [])
        frames_meta = {f["frame_idx"]: f for f in scene_info.get("frames", [])}
        candidate = self._select_candidate(directional, frames_meta)

        captions_map = self.load_captions_for_scene(scene_info["scene_token"]) or {}
        frame_idx_for_prompt = candidate["frame_idx"] if candidate else 0
        scene_description = captions_map.get(frame_idx_for_prompt, "")[:600]

        rels_same_frame = [r for r in directional if r["frame_idx"] == frame_idx_for_prompt][:6]
        rel_lines = []
        for r in rels_same_frame:
            rel_lines.append(
                f"- Frame {r['frame_idx']}: {r['source']['class']} is {r['type'].replace('_', ' ')} {r['target']['class']}"
            )
        directional_relationships_str = "\n".join(rel_lines) if rel_lines else "- None"
        object_counts_str = "\n".join([f"- {cls}: {count}" for cls, count in scene_info["object_counts"].items()])
        scene_desc = scene_description if scene_description else "A driving scene with various road users."

        # Use explicit substitution so JSON examples in the prompt (with { }) are not interpreted by .format()
        return (
            spatial_prompt.replace("{scene_token}", scene_info["scene_token"])
            .replace("{frame_idx}", str(frame_idx_for_prompt))
            .replace("{scene_description}", scene_desc)
            .replace("{directional_relationships}", directional_relationships_str)
            .replace("{object_counts}", object_counts_str)
        )

    def generate_for_scene(self, scene_token: str, api_key: Optional[str] = None) -> Optional[QASample]:
        scene_info = self.load_scene_graph(scene_token)
        if scene_info is None:
            return None
        spatial_prompt = self.load_prompts_from_disk()
        spatial_input = self.build_spatial_input(scene_info, spatial_prompt)
        raw_response = self.gpt(
            spatial_input, api_key=api_key, scene_token=scene_token, question_type="spatial"
        )
        # Parse JSON from response (may be wrapped in markdown code blocks)
        response_text = raw_response.strip()
        if "```" in response_text:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = {"question": raw_response[:500], "options": {}, "correct_option": "", "rationale": raw_response}
        question = parsed.get("question", "")
        answer = {
            "options": parsed.get("options", {}),
            "correct_option": parsed.get("correct_option", ""),
            "rationale": parsed.get("rationale", ""),
        }
        metadata = {
            "num_frames": scene_info["num_frames"],
            "object_counts": scene_info["object_counts"],
            "num_directional_relationships": len(scene_info.get("directional_relationships") or []),
        }
        return self.construct_qa_sample(
            scene_token=scene_token,
            question_type="spatial",
            question=question,
            answer=answer,
            metadata=metadata,
        )


# Default summarization question (fixed for every scene)
SUMMARIZATION_QUESTION = "Provide a comprehensive summary of the ego-actor's interactions across all views."


class SummarizationQAGenerator(QAGenerator):
    """
    Summarization QA generator: produces a fixed question with an answer that condenses
    the temporal scene graph into a 2-3 sentence summary via OpenAI.
    """

    DEFAULT_QUESTION = SUMMARIZATION_QUESTION

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        captions_dir: Optional[Path] = None,
        scene_graphs_dir: Optional[Path] = None,
        instance_annotations_dir: Optional[Path] = None,
        gpt_logs_dir: Optional[Path] = None,
        max_frames_for_summary: int = 15,
    ):
        super().__init__(
            prompts_dir=prompts_dir,
            captions_dir=captions_dir,
            scene_graphs_dir=scene_graphs_dir,
            instance_annotations_dir=instance_annotations_dir,
            gpt_logs_dir=gpt_logs_dir,
        )
        self.max_frames_for_summary = max_frames_for_summary

    def load_raw_scene_graph(self, scene_token: str) -> Optional[Dict[str, Any]]:
        """Load raw scene graph JSON (full frames, objects, relationships)."""
        scene_graph_file = self.scene_graphs_dir / f"{scene_token}/scene_graph.json"
        if not scene_graph_file.exists():
            return None
        with open(scene_graph_file, "r") as f:
            return json.load(f)

    def _build_temporal_scene_graph(
        self,
        scene_token: str,
        raw_scene: Dict[str, Any],
        captions: Optional[Dict[int, str]],
        instance_annotations: Optional[Dict[str, Any]],
    ) -> str:
        """
        Condense raw scene graph + captions into a textual temporal representation
        for summarization.
        """
        frames = raw_scene.get("frames", [])
        if not frames:
            return "No frame data available."

        # Map instance_token -> {activity, description} from annotations
        inst_activities: Dict[str, Dict[str, str]] = {}
        if instance_annotations:
            for ann in instance_annotations.get("annotations", []):
                tok = ann.get("instance_token", "")
                if tok:
                    inst_activities[tok] = {
                        "activity": ann.get("activity", ""),
                        "description": ann.get("description", ""),
                        "class": ann.get("object_class", ""),
                    }

        # Sample frames evenly across the scene
        n_frames = len(frames)
        if n_frames <= self.max_frames_for_summary:
            sampled_indices = list(range(n_frames))
        else:
            step = max(1, n_frames // self.max_frames_for_summary)
            sampled_indices = list(range(0, n_frames, step))[: self.max_frames_for_summary]
            if sampled_indices[-1] != n_frames - 1:
                sampled_indices.append(n_frames - 1)

        lines: List[str] = []
        lines.append(f"Scene: {scene_token}")
        lines.append(f"Total frames: {n_frames}")
        lines.append("")

        directional_types = {"in_front", "behind", "left", "right", "near"}

        for idx in sampled_indices:
            if idx >= len(frames):
                continue
            frame = frames[idx]
            frame_idx = frame.get("frame_idx", idx)
            objects = frame.get("objects", [])
            relationships = frame.get("relationships", []) or []

            lines.append(f"--- Frame {frame_idx} ---")

            # Use caption if available (richest description)
            if captions and frame_idx in captions:
                cap = captions[frame_idx]
                if cap:
                    lines.append(f"Caption: {cap[:400]}{'...' if len(cap) > 400 else ''}")
            else:
                # Fall back to object list + relationships
                obj_classes = [o.get("object_class", "unknown") for o in objects]
                obj_counts: Dict[str, int] = defaultdict(int)
                for c in obj_classes:
                    short = c.split(".")[-1] if "." in c else c
                    obj_counts[short] += 1
                count_str = ", ".join(f"{v} {k}" for k, v in sorted(obj_counts.items()))
                lines.append(f"Objects: {count_str}")

                # Add activities for objects if available
                for obj in objects[:8]:  # Limit to avoid verbosity
                    oid = obj.get("object_id") or obj.get("annotation_token", "")
                    oclass = (obj.get("object_class") or "unknown").split(".")[-1]
                    act = inst_activities.get(oid, {})
                    act_str = act.get("activity", "")
                    desc_str = act.get("description", "")
                    if act_str or desc_str:
                        extra = f" [{act_str}; {desc_str}]" if act_str and desc_str else f" [{act_str or desc_str}]"
                        lines.append(f"  - {oclass}: {extra}")

            # Key spatial relationships (sample a few)
            dir_rels = [r for r in relationships if r.get("relationship_type") in directional_types]
            id_to_class = {
                (o.get("object_id") or o.get("annotation_token", "")): (
                    o.get("object_class", "unknown")
                ).split(".")[-1]
                for o in objects
            }
            for rel in dir_rels[:5]:
                src = id_to_class.get(rel.get("source_id", ""), "?")
                tgt = id_to_class.get(rel.get("target_id", ""), "?")
                rtype = rel.get("relationship_type", "").replace("_", " ")
                dist = rel.get("distance")
                dstr = f" ({dist:.1f}m)" if dist is not None else ""
                lines.append(f"  Relation: {src} is {rtype} {tgt}{dstr}")

            lines.append("")

        return "\n".join(lines).strip()

    def load_prompts_from_disk(self) -> str:
        prompts_dir = self.prompts_dir
        path = prompts_dir / "summarization_questions.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found in {prompts_dir}. "
                "Please ensure summarization_questions.txt exists."
            )
        return path.read_text()

    def generate_for_scene(self, scene_token: str, api_key: Optional[str] = None) -> Optional[QASample]:
        raw_scene = self.load_raw_scene_graph(scene_token)
        if raw_scene is None:
            return None

        captions = self.load_captions_for_scene(scene_token)
        instance_annotations = self.load_instance_annotations(scene_token)

        temporal_graph = self._build_temporal_scene_graph(
            scene_token, raw_scene, captions, instance_annotations
        )

        prompt_template = self.load_prompts_from_disk()
        prompt = prompt_template.replace("{temporal_scene_graph}", temporal_graph)

        summary = self.gpt(
            prompt,
            api_key=api_key,
            model="gpt-4o",
            temperature=0.5,
            max_tokens=300,
            scene_token=scene_token,
            question_type="summarization",
        )

        scene_info = self._extract_scene_info_from_dict(raw_scene)
        metadata = {
            "num_frames": scene_info["num_frames"],
            "object_counts": scene_info.get("object_counts", {}),
            "used_captions": captions is not None and len(captions) > 0,
        }

        return self.construct_qa_sample(
            scene_token=scene_token,
            question_type="summarization",
            question=self.DEFAULT_QUESTION,
            answer=summary,
            metadata=metadata,
        )

class TemporalQAGenerator(QAGenerator):
    """
    Temporal QA generator: Identifies two distinct events in the video (Grounding vs Target)
    and classifies their relationship (Before, After, During) to generate reasoning questions.
    """

    def load_prompts_from_disk(self) -> str:
        # Assumes the prompt provided in your query is saved here
        prompts_dir = self.prompts_dir
        temporal_prompt_path = prompts_dir / "temporal.txt"
        if not temporal_prompt_path.exists():
            # Fallback for demonstration if file doesn't exist
            return "" 
        return temporal_prompt_path.read_text()

    def preprocess_instance_annotations(self, instance_annotations: Dict[str, Any]) -> Dict[str, Any]:
        processed_annotations = {}
        for annotation in instance_annotations.get("annotations", []):
            processed_annotations[annotation["instance_token"]] = {
                "activity": annotation["activity"],
                "description": annotation["description"]
            }
        return processed_annotations
    
    def _extract_scene_info_from_dict(self, scene_data: Dict[str, Any], instance_annotations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses raw scene data into:
        1. Frame-by-frame object lookups
        2. Consolidated 'Events' (intervals of consistent activity)
        """
        frames = scene_data.get("frames", [])
        scene_token = scene_data.get("scene_token", "")
        
        # 1. Basic Metadata
        unique_objects = defaultdict(set)
        directional_relationships = []
        
        # 2. Track activities per object across frames to build intervals
        # Structure: { obj_id: [ (frame_idx, activity, description, obj_class), ... ] }
        raw_timelines = defaultdict(list)
        processed_annotations = self.preprocess_instance_annotations(instance_annotations)
        for frame in frames:
            frame_idx = frame.get("frame_idx")
            
            # Map ID to Class for this frame
            id_to_class = {}
            
            for obj in frame.get("objects", []):
                obj_id = obj.get("object_id") or obj.get("annotation_token", "")
                obj_class = obj.get("object_class", "unknown")
                
                #activity = obj.get("activity", "moving") # default to moving if missing
                #desc = obj.get("description", "")
                activity = processed_annotations.get(obj_id, {}).get("activity", "moving")
                desc = processed_annotations.get(obj_id, {}).get("description", "")
                
                if obj_id:
                    unique_objects[obj_class].add(obj_id)
                    id_to_class[obj_id] = obj_class
                    raw_timelines[obj_id].append({
                        "frame_idx": frame_idx,
                        "activity": activity,
                        "description": desc,
                        "class": obj_class
                    })

            # Extract Directional Relationships (for Spatio-Temporal use later)
            for rel in frame.get("relationships", []) or []:
                if rel.get("relationship_type") in {"in_front", "behind", "left", "right"}:
                    directional_relationships.append({
                        "frame_idx": frame_idx,
                        "type": rel.get("relationship_type"),
                        "source": {"id": rel["source_id"], "class": id_to_class.get(rel["source_id"], "unknown")},
                        "target": {"id": rel["target_id"], "class": id_to_class.get(rel["target_id"], "unknown")},
                    })

        # 3. Consolidate Raw Timelines into "Events"
        # Event = {obj_id, class, activity, start_frame, end_frame, description}
        consolidated_events = self._consolidate_events(raw_timelines)

        return {
            "scene_token": scene_token,
            "num_frames": len(frames),
            "object_counts": {k: len(v) for k, v in unique_objects.items()},
            "directional_relationships": directional_relationships,
            "events": consolidated_events, 
            "frames": frames # Keep raw frames for spatial lookups
        }

    def _consolidate_events(self, raw_timelines: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """
        Merges consecutive frames with the same activity into a single Event interval.
        """
        events = []
        for obj_id, timeline in raw_timelines.items():
            if not timeline: continue
            
            # Sort by frame index just in case
            timeline.sort(key=lambda x: x["frame_idx"])
            
            current_event = None
            
            for step in timeline:
                if current_event is None:
                    current_event = {
                        "obj_id": obj_id,
                        "class": step["class"],
                        "activity": step["activity"],
                        "description": step["description"],
                        "start_frame": step["frame_idx"],
                        "end_frame": step["frame_idx"]
                    }
                else:
                    # If activity is same and frames are consecutive (or close enough), extend event
                    if step["activity"] == current_event["activity"] and (step["frame_idx"] - current_event["end_frame"] <= 2):
                        current_event["end_frame"] = step["frame_idx"]
                        # specific description might update, keep longest or last
                        if len(step["description"]) > len(current_event["description"]):
                            current_event["description"] = step["description"]
                    else:
                        # Close old event, start new one
                        events.append(current_event)
                        current_event = {
                            "obj_id": obj_id,
                            "class": step["class"],
                            "activity": step["activity"],
                            "description": step["description"],
                            "start_frame": step["frame_idx"],
                            "end_frame": step["frame_idx"]
                        }
            if current_event:
                events.append(current_event)
        
        # Filter out very short events (noise)
        return [e for e in events if (e["end_frame"] - e["start_frame"]) > 2]
    
    def _select_temporal_pair(
            self, 
            events: List[Dict[str, Any]], 
            target_relationship: str = "Before"
        ) -> Tuple[Optional[Dict], Optional[Dict], str]:
            """
            Selects a Grounding Event and a Target Event that satisfy the specific
            target_relationship (Before, After, During, In-Between).
            """
            if len(events) < 2:
                return None, None, target_relationship

            # Shuffle events to ensure variety in generation
            shuffled_events = list(events)
            random.shuffle(shuffled_events)

            # Logic for 'In-Between' (Requires 3 events: Grounding Start, Grounding End, Target Middle)
            # Note: The prompt usually expects 2 grounding events for "In-Between", but here we structure it
            # as Grounding (Range) -> Target (Inside).
            if target_relationship == "In-Between":
                # We need a Grounding Start (e1) and Grounding End (e2) with a Target (e3) between them.
                # However, standard prompts often take 1 Grounding and 1 Target. 
                # If your prompt supports 3 distinct events, the logic changes.
                # Assuming standard Grounding/Target pair where Target is 'In-Between' two implicitly related events
                # is complex. 
                # instead, let's strictly look for: Grounding Event A (early) and Grounding Event B (late)
                # and return them such that the Target Event is the one "In-Between".
                
                # Alternative Interpretation based on your prompt: 
                # Grounding: "Event A and Event B" 
                # Target: "Event C"
                # For this helper, we will try to return a Grounding Event that represents the "Range"
                # or simply find a Target that is between two others and formatting it externally.
                
                # SIMPLIFIED STRATEGY for strict pair return:
                # We return e1 (Grounding) and e2 (Target) where e2 is "During" e1, 
                # OR we switch to a specialized logic if you want the "A... [Target] ... B" format.
                # Below implements "Target occurs between End of A and Start of B".
                
                for i in range(len(shuffled_events)):
                    for j in range(len(shuffled_events)):
                        if i == j: continue
                        e1 = shuffled_events[i] # Grounding Start
                        e2 = shuffled_events[j] # Grounding End
                        
                        if e1["end_frame"] < e2["start_frame"]:
                            # Find a target (e_target) that fits in the gap
                            gap_start = e1["end_frame"]
                            gap_end = e2["start_frame"]
                            
                            for e_target in shuffled_events:
                                if e_target["obj_id"] in [e1["obj_id"], e2["obj_id"]]:
                                    continue
                                    
                                # Check if target is substantially inside the gap
                                if e_target["start_frame"] >= gap_start and e_target["end_frame"] <= gap_end:
                                    # Found a valid triplet!
                                    # To fit the Tuple[Dict, Dict, str] return signature:
                                    # We combine e1 and e2 into a single "Grounding Context" dict
                                    combined_grounding = {
                                        "description": [e1['description'], e2['description']],
                                        "class": [e1['class'], e2['class']],
                                        "activity": [e1['activity'], e2['activity']],
                                        "start_frame": e1["start_frame"],
                                        "end_frame": e2["end_frame"]
                                    }
                                    return combined_grounding, e_target, "In-Between"

            # Standard Pair Logic for Before, After, During
            for i in range(len(shuffled_events)):
                for j in range(len(shuffled_events)):
                    if i == j: continue
                    
                    e1 = shuffled_events[i] # Grounding
                    e2 = shuffled_events[j] # Target
                    
                    if e1["obj_id"] == e2["obj_id"]:
                        continue 

                    # Check specific relationship requirements
                    if target_relationship == "Before":
                        # Question: What happened to Target BEFORE Grounding? 
                        # Logic: Target Ends < Grounding Starts
                        if e2["end_frame"] < e1["start_frame"]:
                            return e1, e2, "Before"

                    elif target_relationship == "After":
                        # Question: What happened to Target AFTER Grounding?
                        # Logic: Target Starts > Grounding Ends
                        if e2["start_frame"] > e1["end_frame"]:
                            return e1, e2, "After"

                    elif target_relationship == "During":
                        # Question: What happened to Target WHILE Grounding was happening?
                        start = max(e1["start_frame"], e2["start_frame"])
                        end = min(e1["end_frame"], e2["end_frame"])
                        overlap = max(0, end - start)
                        min_len = min(e1["end_frame"] - e1["start_frame"], e2["end_frame"] - e2["start_frame"])
                        
                        if min_len > 0 and (overlap / min_len) > 0.5:
                            return e1, e2, "During"

            # If strict search failed, fallback to relaxation or default
            # (Optional: Recursively call with different relationship or return None)
            return None, None, target_relationship

    def build_temporal_input(self, scene_info: Dict[str, Any], prompt_template: str) -> str:
        events = scene_info.get("events", [])
        all_relationships = ["Before", "After", "During", "In-Between"]
        grounding = None
        target = None
        while grounding is None or target is None:
            target_relationship = random.choice(all_relationships)
            grounding, target, rel = self._select_temporal_pair(events, target_relationship)
            if grounding is not None and target is not None:
                break
            target_relationship = random.choice(all_relationships)

        
        print(grounding, target, rel)


        # Format descriptions for the prompt
        # e.g., "The white van (id: 123) is turning left"
        if isinstance(grounding['description'], list):
            grounding_str = f"Event A: The {grounding['class'][0]} {grounding['activity'][0]} ({grounding['description'][0]} Event B: The {grounding['class'][1]} {grounding['activity'][1]} ({grounding['description'][1]})"
        else:
            grounding_str = f"The {grounding['class']} {grounding['activity']} ({grounding['description']})"

        target_str = f"The {target['class']} {target['activity']} ({target['description']})"

        # Fill the template slots defined in your prompt: {grounding_input}, {target_input}, {rel_type}
        return prompt_template.format(
            grounding_input=grounding_str,
            target_input=target_str,
            rel_type=rel
        ), {"grounding": grounding, "target": target, "rel": rel}
    
    def load_scene_graph(self, scene_token: str) -> Optional[Dict[str, Any]]:
        scene_graph_file = self.scene_graphs_dir / f"{scene_token}/scene_graph.json"
        if not scene_graph_file.exists():
            return None
        with open(scene_graph_file, "r") as f:
            scene_graph = json.load(f)
        return scene_graph
    
    def generate_for_scene(self, scene_token: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        # Pre-process scene info to get events
        scene_graph = self.load_scene_graph(scene_token)
        instance_annotations = self.load_instance_annotations(scene_token)
        scene_info = self._extract_scene_info_from_dict(scene_graph, instance_annotations)
        if scene_info is None:
            return None
        temporal_prompt = self.load_prompts_from_disk()
        prompt_input, temporal_info = self.build_temporal_input(scene_info, temporal_prompt)

        if not prompt_input:
            return {"error": "Not enough events for temporal QA"}


        response = self.gpt(prompt_input, api_key=api_key)

        return self.construct_qa_sample(
            scene_token,
            "temporal",
            response,
            None,
            metadata=temporal_info,
        )


class EventOrderingQAGenerator(TemporalQAGenerator):
    """
    Event Ordering QA: Selects N events and asks the model to generate a question
    asking the user to sort them chronologically.
    """
    def load_prompts_from_disk(self) -> str:
        ordering_prompt_file = self.prompts_dir / "ordering.txt"
        with open(ordering_prompt_file, "r") as f:
            return f.read()
    
    def build_ordering_input(self, scene_info: Dict[str, Any], prompt_template: str, length_of_chain: int = 4, buffer_frames: int = 5) -> str:
        events = scene_info.get("events", [])
        
        # Sort by start time for sequential processing
        sorted_events = sorted(events, key=lambda x: x["start_frame"])            

        def find_chain(require_unique_classes: bool, target_length: int):
            """
            Tries to find a chain of 4 events.
            If require_unique_classes is True: All 4 events must have different 'class'.
            If False: No single class can appear more than 2 times in the chain.
            """
            
            for start_idx in range(len(sorted_events)):
                current_chain = [sorted_events[start_idx]]
                current_classes = [sorted_events[start_idx]["class"]]
                last_end = sorted_events[start_idx]["end_frame"]
                
                # Scan remaining events to fill the chain
                for i in range(start_idx + 1, len(sorted_events)):
                    candidate = sorted_events[i]
                    cand_class = candidate["class"]
                    
                    # 1. Check Temporal Spacing upto an overlap of buffer_frames
                    if candidate["start_frame"] < last_end - buffer_frames:
                        continue
                    
                    # Must ensure that the candidate event ends after the last event in the chain
                    if candidate["end_frame"] < last_end:
                        continue
                        
                    # 2. Check Object Uniqueness Constraints
                    if candidate["obj_id"] in [e["obj_id"] for e in current_chain]:
                        continue # Always require distinct object IDs

                    # 3. Check Class/Type Constraints
                    if require_unique_classes:
                        # Strict: Must be a type we haven't seen yet
                        if cand_class in current_classes:
                            continue
                    else:
                        # Relaxed: Allow repeats, but max 2 of any specific type
                        if current_classes.count(cand_class) >= 2:
                            continue

                    # Add to chain
                    current_chain.append(candidate)
                    current_classes.append(cand_class)
                    last_end = candidate["end_frame"]
                    
                    if len(current_chain) == target_length:
                        return current_chain
            
            return None
        
        # Attempt 1: Strict Diversity (4 distinct classes)
        selected_chain = find_chain(require_unique_classes=True, target_length=length_of_chain)
        
        # Attempt 2: Relaxed Diversity (Max 2 of same type)
        if not selected_chain:
            selected_chain = find_chain(require_unique_classes=False, target_length=length_of_chain)
        
        # If we still don't have a chain, try again with a shorter chain
        if not selected_chain:
            selected_chain = find_chain(require_unique_classes=True, target_length=length_of_chain - 1)

        # If we still don't have a chain, try again with a shorter chain
        if not selected_chain:
            selected_chain = find_chain(require_unique_classes=False, target_length=length_of_chain - 1)
        
        # If we still don't have a chain, return an empty string
        if not selected_chain:
            return "", None

        # Randomize presentation order for the Question
        # The LLM needs to figure out the correct order based on descriptions/context
        presentation_order = list(selected_chain)

        events_desc = []
        for i, e in enumerate(presentation_order):
            # Using Letters A, B, C, D for options
            events_desc.append(f"Event {chr(65+i)}: A {e['class']} is {e['activity']} ({e['description']})")
        
        events_block = "\n".join(events_desc)
        
        return prompt_template.format(events_list=events_block), events_block

    def generate_for_scene(self, scene_token: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        
        scene_graph = self.load_scene_graph(scene_token)
        instance_annotations = self.load_instance_annotations(scene_token)
        scene_info = self._extract_scene_info_from_dict(scene_graph, instance_annotations)
        if scene_info is None:
            return None
        ordering_prompt = self.load_prompts_from_disk()
        
        prompt_input = None
        events_block = None
        length_of_chain = 4
        buffer_frames = 5
        while prompt_input is None or events_block is None:

            prompt_input, events_block = self.build_ordering_input(scene_info, ordering_prompt, length_of_chain=length_of_chain, buffer_frames=buffer_frames)
            length_of_chain = random.randint(3, 5)
            buffer_frames = random.randint(3, 5)
        if not prompt_input or not events_block:
            return {"error": "Not enough events for ordering QA"}
            
        response = self.gpt(prompt_input, api_key=api_key)
        return self.construct_qa_sample(
            scene_token,
            "event_ordering",
            response,
            None,
            metadata=events_block,
        )


class SpatioTemporalQAGenerator(TemporalQAGenerator):
    """
    Spatio-Temporal QA: Generates questions that mix spatial grounding (left/right/next to)
    with temporal logic (before/after).
    """

    def _select_event_grounded_spatial_query(self, events, frames_map):
        """
        Type 1: Time -> Space
        Logic: Pick an Event. Look at the frame where it starts. Pick a spatial relationship 
        visible at that frame.
        Question: "At the start of [Event], what was [Spatial Rel]?"
        """
        if not events: return None
        
        # 1. Pick an Anchor Event (The "Time")
        anchor_event = random.choice(events)
        time_idx = anchor_event["start_frame"]
        
        # 2. Get frame data for that time
        frame_data = frames_map.get(time_idx)
        if not frame_data: return None

        # 3. Find valid spatial relationships in this frame
        # We want relationships like: "A is Left of B"
        rels = [r for r in frame_data.get("relationships", []) 
                if r["relationship_type"] in ["left", "right", "in_front", "behind", "next_to"]]
        
        if not rels: return None
        
        target_rel = random.choice(rels)
        
        # 4. Format for Prompt
        return {
            "mode": "Type 1: Event-Grounded Spatial Query (Time -> Space)",
            "context_info": f"Anchor Event: A {anchor_event['class']} is {anchor_event['activity']} ({anchor_event['description']}).",
            "target_info": f"Spatial Anchor Object: {target_rel['source_class']} (ID: {target_rel['source_id']}).",
            "relationship_query": f"Question Goal: Identify the object that is '{target_rel['relationship_type']}' the Spatial Anchor at the time of the Anchor Event. (Correct Answer: {target_rel['target_class']})"
        }

    def _select_spatially_grounded_temporal_query(self, events, frames_map):
        """
        Type 2: Space -> Time
        Logic: Pick an object defined by a spatial relationship (e.g., "Car next to bus").
        Then find an event involving that object that happens *after* that spatial context.
        Question: "What did the [Object next to bus] do afterwards?"
        """
        # 1. Find candidates: Objects that have a spatial relationship AND an event later
        candidates = []
        
        # Iterate through events to find a "Target Event" first
        for target_event in events:
            # We want to describe this actor based on where they were *before* this event
            # Let's look at a frame shortly before the event starts (or at start)
            lookback_frame = max(0, target_event["start_frame"] - 5)
            frame_data = frames_map.get(lookback_frame)
            
            if not frame_data: continue
            
            # Does the actor (target_event['obj_id']) have a spatial descriptor here?
            # We look for: [Actor] is [Relation] [Anchor]
            actor_rels = [r for r in frame_data.get("relationships", []) 
                          if r["source_id"] == target_event["obj_id"]]
            
            if actor_rels:
                # Found a spatial descriptor for this actor!
                candidates.append((target_event, actor_rels[0]))

        if not candidates: return None
        
        # 2. Pick a random candidate
        target_event, spatial_rel = random.choice(candidates)
        
        # 3. Format for Prompt
        # Describe the object purely spatially
        spatial_desc = f"The {spatial_rel['source_class']} located {spatial_rel['relationship_type'].replace('_', ' ')} the {spatial_rel['target_class']}"
        
        return {
            "mode": "Type 2: Spatially-Grounded Temporal Query (Space -> Time)",
            "context_info": f"Spatially Defined Actor: {spatial_desc}.",
            "target_info": f"Target Activity: {target_event['activity']} ({target_event['description']}).",
            "relationship_query": "Question Goal: Ask what this Spatially Defined Actor did immediately after/during the spatial context. (Relationship: Sequence/Action)"
        }

    def build_spatiotemporal_input(self, scene_info: Dict[str, Any], prompt_template: str) -> str:
        events = scene_info.get("events", [])
        
        # Pre-process frames for O(1) lookup
        frames_map = {f["frame_idx"]: self._process_frame_relationships(f) for f in scene_info.get("frames", [])}
        
        if not events or not frames_map: return ""

        # Randomly choose between Type 1 and Type 2 logic
        # You can adjust weights if you want one type more often
        if random.random() < 0.5:
            data = self._select_event_grounded_spatial_query(events, frames_map)
        else:
            data = self._select_spatially_grounded_temporal_query(events, frames_map)
            
        # Fallback if selection failed (e.g. no relationships found for Type 2)
        if not data:
            data = self._select_event_grounded_spatial_query(events, frames_map)
        
        if not data:
            return ""

        return prompt_template.format(
            logic_mode=data["mode"],
            context_info=data["context_info"],
            target_info=data["target_info"],
            relationship_query=data["relationship_query"]
        )
    
    def _process_frame_relationships(self, frame):
        """Helper to map IDs to Classes in relationship data"""
        id_to_class = {o["object_id"]: o["object_class"] for o in frame.get("objects", [])}
        processed_rels = []
        for r in frame.get("relationships", []):
            # Filter for relevant spatial prepositions
            if r["relationship_type"] in ["in_front", "behind", "left", "right", "next_to"]:
                processed_rels.append({
                    "relationship_type": r["relationship_type"],
                    "source_id": r["source_id"],
                    "source_class": id_to_class.get(r["source_id"], "unknown"),
                    "target_id": r["target_id"],
                    "target_class": id_to_class.get(r["target_id"], "unknown")
                })
        return {"relationships": processed_rels}

    def generate_for_scene(self, scene_info: Dict[str, Any], spatiotemporal_prompt: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        processed_info = self._extract_scene_info_from_dict(scene_info)
        prompt_input = self.build_spatiotemporal_input(processed_info, spatiotemporal_prompt)
        
        if not prompt_input:
            return {"error": "Could not generate spatio-temporal scenario (lack of spatial rels)"}
            
        response = self.gpt(prompt_input, api_key=api_key)
        return {
            "scene_token": scene_info["scene_token"],
            "question_data": response,
            "type": "spatio_temporal"
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
    print(f"Processing {scene_graph}...")

    with open(scene_graph, "r") as f:
        # Read file in chunks to handle large files
        first_line = f.readline()
        scene_data = json.loads(first_line + f.read())

    scene_token = scene_data["scene_token"]
    frames = scene_data.get("frames", [])

    # Track unique objects across all frames
    unique_objects = defaultdict(set)  # object_class -> set of object_ids
    object_info = {}  # object_id -> {class, positions, frames_seen, which_cameras}

    # Process each frame
    for frame in frames:
        objects = frame.get("objects", [])

        for obj in objects:
            obj_id = obj["object_id"]
            obj_class = obj["object_class"]
            position = obj["position"]

            # Track unique objects
            unique_objects[obj_class].add(obj_id)

            # Track object information
            if obj_id not in object_info:
                object_info[obj_id] = {"class": obj_class, "positions": [], "frames_seen": 0, "which_cameras": []}

            object_info[obj_id]["positions"].append(position)
            object_info[obj_id]["frames_seen"] += 1
            object_info[obj_id]["which_cameras"].append(obj["visible_cameras"])

    # Calculate average positions
    for obj_id, info in object_info.items():
        positions = np.array(info["positions"])
        info["avg_position"] = positions.mean(axis=0).tolist()

    # Count objects per class
    object_counts = {obj_class: len(ids) for obj_class, ids in unique_objects.items()}

    # Collect directional relationships directly from scene graph frames
    directional_types = {"in_front", "behind", "left", "right"}
    directional_relationships: List[Dict[str, Any]] = []

    for frame in frames:
        frame_idx = frame.get("frame_idx")
        objects = frame.get("objects", [])
        id_to_class = {o["object_id"]: o["object_class"] for o in objects}

        for rel in frame.get("relationships", []) or []:
            rel_type = rel.get("relationship_type")
            if rel_type in directional_types:
                source_id = rel.get("source_id")
                target_id = rel.get("target_id")
                directional_relationships.append(
                    {
                        "frame_idx": frame_idx,
                        "type": rel_type,
                        "distance": rel.get("distance"),
                        "source": {"id": source_id, "class": id_to_class.get(source_id, "unknown")},
                        "target": {"id": target_id, "class": id_to_class.get(target_id, "unknown")},
                    }
                )

    return {
        "scene_token": scene_token,
        "num_frames": len(frames),
        "object_counts": object_counts,
        "object_instances": {k: list(v) for k, v in unique_objects.items()},
        "directional_relationships": directional_relationships,
        "total_unique_objects": sum(len(ids) for ids in unique_objects.values()),
        "frames": [
            {
                "frame_idx": f.get("frame_idx"),
                "objects": [{"id": o["object_id"], "class": o["object_class"]} for o in f.get("objects", [])],
            }
            for f in frames
        ],
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


def _save_sample_incremental(output_file: Path, sample: QASample, existing: List[Dict[str, Any]]) -> None:
    """Append a single sample to the output file, keeping valid JSON."""
    sample_dict = sample.to_dict() if hasattr(sample, "to_dict") else sample
    existing.append(sample_dict)
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def process_questions(
    qa_generator: QAGenerator,
    scene_tokens: List[str],
    api_key: Optional[str] = None,
    *,
    verbose: bool = True,
    output_file: Optional[Path] = None,
) -> List[QASample]:
    qa_samples = []
    saved_data: List[Dict[str, Any]] = []
    total = len(scene_tokens)
    for idx, scene_token in enumerate(scene_tokens):
        if verbose:
            print(f"\n[{idx + 1}/{total}] Processing scene {scene_token[:16]}...")
        qa_sample = qa_generator.generate_for_scene(scene_token, api_key=api_key)
        if qa_sample is None:
            if verbose:
                print(f"  Skipped (no scene graph found)")
            continue
        qa_samples.append(qa_sample)
        if output_file is not None:
            _save_sample_incremental(output_file, qa_sample, saved_data)
        if verbose:
            q_preview = (qa_sample.question[:80] + "...") if len(qa_sample.question) > 80 else qa_sample.question
            ans = qa_sample.answer
            ans_str = json.dumps(ans) if isinstance(ans, dict) else str(ans)
            ans_preview = (ans_str[:60] + "...") if len(ans_str) > 60 else ans_str
            print(f"  Q: {q_preview}")
            print(f"  A: {ans_preview}")
    return qa_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Process scene graphs and extract information.")
    parser.add_argument(
        "--dataroot", type=str, default="/nas/standard_datasets/nuscenes", help="Path to nuScenes dataset root"
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-trainval", help="nuScenes dataset version (e.g. v1.0-trainval, v1.0-mini)"
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="./nuscenes/datasetbuilder/prompts",
        help="Directory containing prompts",
    )
    parser.add_argument(
        "--scene_graphs_dir",
        type=str,
        default="/home/hg22723/projects/Multi-Camera/outputs/scene_graphs",
        help="Directory containing scene graph subdirectories",
    )
    parser.add_argument(
        "--instance_annotations_dir",
        type=str,
        default="/home/hg22723/projects/Multi-Camera/outputs/instance_annotations",
        help="Directory containing instance annotations subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/nas/neurosymbolic/multi-cam-dataset/nuscenes",
        help="Directory to save extracted information",
    )
    parser.add_argument("--limit", type=int, default=250, help="Maximum number of scenes to process")
    parser.add_argument(
        "--question_type",
        type=str,
        default="summarization",
        help="Question type: counting, spatial, temporal, event_ordering, causality, perception, summarization, which_camera",
    )
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-scene progress and question output")
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    if args.api_key is None:
        args.api_key = os.getenv("OPENAI_API_KEY")

    output_dir = Path(args.output_dir)
    gpt_logs_dir = output_dir / "gpt-logs" / args.question_type
    gen_kwargs = {
        "prompts_dir": args.prompts_dir,
        "scene_graphs_dir": args.scene_graphs_dir,
        "gpt_logs_dir": gpt_logs_dir,
    }

    if args.question_type == "counting":
        qa_generator = CountingQAGenerator(**gen_kwargs)
    elif args.question_type == "spatial":
        qa_generator = SpatialQAGenerator(prompts_dir=args.prompts_dir, scene_graphs_dir=args.scene_graphs_dir)
    elif args.question_type == "temporal":
        qa_generator = TemporalQAGenerator(prompts_dir=args.prompts_dir, scene_graphs_dir=args.scene_graphs_dir, instance_annotations_dir=args.instance_annotations_dir)
    elif args.question_type == "event_ordering":
        qa_generator = EventOrderingQAGenerator(prompts_dir=args.prompts_dir, scene_graphs_dir=args.scene_graphs_dir, instance_annotations_dir=args.instance_annotations_dir)
    elif args.question_type == "spatio_temporal":
        qa_generator = SpatialQAGenerator(**gen_kwargs)
    elif args.question_type == "summarization":
        qa_generator = SummarizationQAGenerator(**gen_kwargs)
    else:
        raise NotImplementedError(f"Question type {args.question_type} not implemented")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NuScenes Dataset Builder - Extracting Scene Information")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Question type: {args.question_type}")
    print(f"GPT logs: {gpt_logs_dir}")
    print(f"Limit: {args.limit} scenes")
    print("=" * 80)

    nuscenes_dataloader = NuScenesLidarSegmentationLoader(dataroot=args.dataroot, version=args.version)
    scene_tokens = nuscenes_dataloader.get_scene_tokens()
    scene_tokens = scene_tokens[: args.limit]
    print(f"Processing {len(scene_tokens)} scenes...")
    output_file = output_dir / f"qa_samples_{args.question_type}.json"
    print(f"Saving incrementally to {output_file}")

    qa_samples = process_questions(
        qa_generator,
        scene_tokens,
        api_key=args.api_key,
        verbose=not args.quiet,
        output_file=output_file,
    )

    print("\n" + "=" * 80)
    print("Extraction complete!")
    print("=" * 80)
    print(f"Saved {len(qa_samples)} QA samples to {output_file}")
    print(f"  Skipped: {len(scene_tokens) - len(qa_samples)} scenes (no scene graph)")


if __name__ == "__main__":
    main()

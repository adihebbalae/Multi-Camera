"""
V7 build_scene_graph.py — Step 2: Entity-based scene graph with IoU matching + aliases.

Builds an entity-based scene graph from parsed events + geom.yml bounding boxes.
Each entity = (camera_id, actor_id) with time span, keyframe bboxes, events, and
human-readable alias for question text.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict, field

from .parse_annotations import Event, find_clips_for_slot, DEFAULT_FRAMERATE
from .utils.yaml_stream import get_actor_keyframe_bboxes, get_actor_frame_range
from .utils.krtd import load_camera_model, CameraModel, INDOOR_CAMERAS
from .utils.iou import compute_iou
from .activity_hierarchy import humanize_activity


# 2% of 1920×1080 = 2,073,600 × 0.02 ≈ 41,472 px² (~200×208 px minimum)
# For close-up cameras (G419, G420, G421 etc.) where people fill the frame
MIN_BBOX_AREA = 41472
# 0.1% of 1920×1080 ≈ 2,073 px² (~46×46 px minimum)
# For wide-field outdoor KRTD cameras (G336, G328, G638 etc.) where people are small
MIN_BBOX_AREA_KRTD = 2048

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Entity:
    """An entity (person/vehicle) tracked on one camera."""
    entity_id: str           # "{camera_id}_actor_{actor_id}"
    camera_id: str
    actor_id: int
    entity_type: str         # "person" or "vehicle"
    first_frame: int
    last_frame: int
    first_sec: float
    last_sec: float
    keyframe_bboxes: Dict[int, List[int]]  # {frame: [x1,y1,x2,y2]}
    events: List[str]        # event_ids this entity participates in
    alias: str = ""          # Human-readable label (set after construction)

    def make_alias(self, event_list: list = None) -> str:
        """
        Generate a human-readable entity alias for question text.
        
        Uses the entity's primary activity + actor ID + camera + timestamp.
        """
        # Find this entity's primary activity from event list
        primary_activity = None
        mid_sec = None
        if event_list:
            for evt in event_list:
                if evt.camera_id == self.camera_id:
                    for actor in evt.actors:
                        if actor["actor_id"] == self.actor_id:
                            primary_activity = evt.activity
                            mid_sec = round((evt.start_sec + evt.end_sec) / 2)
                            break
                    if primary_activity:
                        break
        
        # Build the alias
        if primary_activity:
            short_act = humanize_activity(primary_activity)
            # Use entity's own time span if we didn't find event timing
            t = mid_sec if mid_sec is not None else int(self.first_sec)
            
            # For small actor IDs, include them; for hash IDs, skip
            if isinstance(self.actor_id, int) and self.actor_id < 10000:
                return f"Person {short_act} (#{self.actor_id} on {self.camera_id} @ {t}s)"
            else:
                return f"Person {short_act} (on {self.camera_id} @ {t}s)"
        
        # Fallback: no activity
        if isinstance(self.actor_id, int) and self.actor_id < 10000:
            return f"Actor #{self.actor_id} on {self.camera_id}"
        
        return f"Person on {self.camera_id}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alias"] = self.alias
        return d


@dataclass
class CameraNode:
    """Camera metadata for scene graph."""
    camera_id: str
    is_indoor: bool
    has_krtd: bool
    position_enu: Optional[Tuple[float, float, float]]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class SceneGraph:
    """Complete scene graph for one slot."""
    slot: str
    cameras: Dict[str, CameraNode]
    entities: Dict[str, Entity]       # {entity_id: Entity}
    events: List[Event]
    events_by_camera: Dict[str, List[Event]]

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "cameras": {k: v.to_dict() for k, v in self.cameras.items()},
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "events": [e.to_dict() for e in self.events],
            "events_by_camera": {
                k: [e.to_dict() for e in v]
                for k, v in self.events_by_camera.items()
            },
        }


# ============================================================================
# Scene Graph Builder
# ============================================================================

def build_scene_graph(slot: str, events: List[Event], 
                      verbose: bool = False) -> SceneGraph:
    """
    Build an entity-based scene graph from parsed events.
    
    Steps:
        1. Build camera nodes with KRTD info
        2. Extract entities from events + geom.yml bboxes
        3. Link events to entities
    
    Args:
        slot: Slot name
        events: Parsed Event objects from parse_annotations
        verbose: Print progress
    
    Returns:
        SceneGraph with entities and events
    """
    if verbose:
        print(f"Building scene graph for {slot}")

    # Collect unique cameras
    camera_ids = sorted(set(e.camera_id for e in events))
    
    # 1. Build camera nodes
    cameras: Dict[str, CameraNode] = {}
    for cam_id in camera_ids:
        model = load_camera_model(cam_id)
        is_indoor = cam_id in INDOOR_CAMERAS
        cameras[cam_id] = CameraNode(
            camera_id=cam_id,
            is_indoor=is_indoor,
            has_krtd=model is not None,
            position_enu=tuple(model.camera_center.tolist()) if model else None,
        )

    # 2. Extract entities: collect unique (camera, actor) pairs from events
    # Also collect actor_ids per camera for geom.yml lookup
    entity_actor_ids: Dict[str, Set[int]] = defaultdict(set)  # cam -> actor_ids
    entity_types: Dict[str, Dict[int, str]] = defaultdict(dict)  # cam -> {aid: type}
    entity_events: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    for evt in events:
        for actor in evt.actors:
            aid = actor["actor_id"]
            entity_actor_ids[evt.camera_id].add(aid)
            entity_types[evt.camera_id][aid] = actor.get("entity_type", "unknown")
            entity_events[evt.camera_id][aid].append(evt.event_id)

    # 3. Try to get keyframe bboxes from geom.yml (stream-parsed)
    clips = find_clips_for_slot(slot)
    clip_by_camera = {c["camera_id"]: c for c in clips}
    
    entity_bboxes: Dict[str, Dict[int, Dict[int, List[int]]]] = {}  # cam -> {aid: {frame: bbox}}
    entity_frame_ranges: Dict[str, Dict[int, tuple]] = {}  # cam -> {aid: (first, last)}
    
    for cam_id, actor_ids in entity_actor_ids.items():
        if cam_id not in clip_by_camera:
            continue
        clip = clip_by_camera[cam_id]
        geom_path = Path(clip["activities_file"]).with_name(
            Path(clip["activities_file"]).name.replace(".activities.yml", ".geom.yml")
        )
        if geom_path.exists():
            try:
                # V7: Load ALL frames (sample_every=1) for complete trajectories
                bboxes = get_actor_keyframe_bboxes(geom_path, actor_ids, sample_every=1)
                entity_bboxes[cam_id] = bboxes
                
                # Also get frame ranges
                ranges = get_actor_frame_range(geom_path)
                entity_frame_ranges[cam_id] = ranges
                
                if verbose:
                    print(f"  {cam_id}: streamed geom.yml — {len(bboxes)} actors with bboxes")
            except Exception as e:
                if verbose:
                    print(f"  {cam_id}: geom.yml parse error: {e}")

    # 4. Build entity objects
    entities: Dict[str, Entity] = {}
    framerate = DEFAULT_FRAMERATE
    
    for cam_id in sorted(entity_actor_ids.keys()):
        actor_ids = sorted(entity_actor_ids[cam_id])
        cam_ranges = entity_frame_ranges.get(cam_id, {})
        cam_bboxes = entity_bboxes.get(cam_id, {})
        
        for aid in actor_ids:
            entity_id = f"{cam_id}_actor_{aid}"
            
            # Frame range from geom.yml if available, else from events
            if aid in cam_ranges:
                first_frame, last_frame = cam_ranges[aid]
            else:
                # Estimate from events
                actor_events = [e for e in events 
                               if e.camera_id == cam_id 
                               and any(a["actor_id"] == aid for a in e.actors)]
                if actor_events:
                    first_frame = min(e.start_frame for e in actor_events)
                    last_frame = max(e.end_frame for e in actor_events)
                else:
                    first_frame, last_frame = 0, 0
            
            entity = Entity(
                entity_id=entity_id,
                camera_id=cam_id,
                actor_id=aid,
                entity_type=entity_types.get(cam_id, {}).get(aid, "unknown"),
                first_frame=first_frame,
                last_frame=last_frame,
                first_sec=round(first_frame / framerate, 2),
                last_sec=round(last_frame / framerate, 2),
                keyframe_bboxes=cam_bboxes.get(aid, {}),
                events=entity_events.get(cam_id, {}).get(aid, []),
            )

            # Filter out entities whose bounding boxes are too small.
            # Wide-field KRTD cameras have a much lower threshold since people
            # appear small but are still valid for 3D projection.
            if entity.keyframe_bboxes:
                areas = [(bb[2]-bb[0]) * (bb[3]-bb[1]) for bb in entity.keyframe_bboxes.values()]
                median_area = sorted(areas)[len(areas)//2]
                is_krtd_cam = cameras[cam_id].has_krtd
                threshold = MIN_BBOX_AREA_KRTD if is_krtd_cam else MIN_BBOX_AREA
                if median_area < threshold:
                    if verbose:
                        print(f"    Skipping {entity_id}: median bbox area {median_area} < {threshold}")
                    continue

            entities[entity_id] = entity

    # Group events by camera
    events_by_camera: Dict[str, List[Event]] = defaultdict(list)
    for evt in events:
        events_by_camera[evt.camera_id].append(evt)

    # 5. Generate entity aliases for human-readable question text
    for eid, entity in entities.items():
        entity.alias = entity.make_alias(events)

    sg = SceneGraph(
        slot=slot,
        cameras=cameras,
        entities=entities,
        events=events,
        events_by_camera=dict(events_by_camera),
    )

    if verbose:
        print(f"  Total: {len(entities)} entities, {len(events)} events, "
              f"{len(cameras)} cameras")

    return sg

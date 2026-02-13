# Multi-Camera Repository Reference Guide

> **Repository**: [UTAustin-SwarmLab/Multi-Camera](https://github.com/UTAustin-SwarmLab/Multi-Camera)  
> **Last accessed**: February 6, 2026  
> **Purpose**: Extract key algorithmic patterns, prompting strategies, and architectural components for MEVA multi-camera QA work

## Key Architectural Patterns

### 1. QA Generator Hierarchy

The repository uses a clean inheritance pattern for different question types:

```python
class QAGenerator:  # Base class
    def __init__(self, prompts_dir, scene_graphs_dir, gpt_logs_dir, ...):
        self.prompts_dir = Path(prompts_dir)
        self.gpt_logs_dir = Path(gpt_logs_dir)
    
    def gpt(self, prompt, api_key, model="gpt-4", temperature=0.7, max_tokens=500):
        """Standardized GPT call with automatic logging"""
        # GPT call + automatic logging to gpt_logs_dir
    
    def construct_qa_sample(self, scene_token, question_type, question, answer, metadata):
        return QASample(scene_token, question_type, question, answer, metadata)

class TemporalQAGenerator(QAGenerator):
    def generate_for_scene(self, scene_token, api_key):
        # Load scene graph
        # Extract events from timeline
        # Select temporal pairs (grounding + target events)
        # Generate Q&A via GPT
```

**Pattern for MEVA**: Our `generate_qa_v3_multicam.py` follows a similar pattern with category-specific generators inheriting from a base class.

### 2. Event Consolidation Algorithm

**Key insight**: Raw frame-by-frame annotations are consolidated into temporal "events" with consistent activities:

```python
def _consolidate_events(self, raw_timelines: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Merges consecutive frames with the same activity into a single Event interval.
    Logic:
    - If activity changes OR frame gap > 2: close current event, start new one
    - Extend event description if longer description found
    - Filter out very short events (< 3 frames) as noise
    """
    events = []
    for obj_id, timeline in raw_timelines.items():
        timeline.sort(key=lambda x: x["frame_idx"])
        current_event = None
        for step in timeline:
            if current_event is None:
                current_event = {
                    "obj_id": obj_id, "class": step["class"], 
                    "activity": step["activity"], "description": step["description"],
                    "start_frame": step["frame_idx"], "end_frame": step["frame_idx"]
                }
            else:
                # Extend if same activity and consecutive frames
                if (step["activity"] == current_event["activity"] and 
                    step["frame_idx"] - current_event["end_frame"] <= 2):
                    current_event["end_frame"] = step["frame_idx"]
                    # Keep longest description
                    if len(step["description"]) > len(current_event["description"]):
                        current_event["description"] = step["description"]
                else:
                    # Close old event, start new
                    events.append(current_event)
                    current_event = { ... }  # New event
        if current_event:
            events.append(current_event)
    
    # Filter noise: events must span > 2 frames
    return [e for e in events if (e["end_frame"] - e["start_frame"]) > 2]
```

**Application to MEVA**: Our `extract_logic_tuples.py` already does similar consolidation in `parse_activities_json()` when building LogicTuple intervals.

### 3. Temporal Relationship Selection

**Advanced algorithm** for selecting meaningful temporal pairs:

```python
def _select_temporal_pair(self, events, target_relationship="Before"):
    """
    Selects Grounding Event and Target Event for specific relationships:
    - Before: Target.end < Grounding.start
    - After: Target.start > Grounding.end  
    - During: Overlap > 50% of shorter event
    - In-Between: Target occurs between end of Event A and start of Event B
    """
    shuffled_events = list(events)
    random.shuffle(shuffled_events)  # Ensure variety
    
    if target_relationship == "In-Between":
        # Complex logic: find triplet where Target fits between two others
        for i in range(len(shuffled_events)):
            for j in range(len(shuffled_events)):
                e1, e2 = shuffled_events[i], shuffled_events[j]
                if e1["end_frame"] < e2["start_frame"]:  # Valid gap
                    gap_start, gap_end = e1["end_frame"], e2["start_frame"]
                    for e_target in shuffled_events:
                        if (e_target["start_frame"] >= gap_start and 
                            e_target["end_frame"] <= gap_end):
                            # Found valid triplet: combine e1+e2 as grounding
                            combined_grounding = {
                                "description": [e1['description'], e2['description']],
                                "class": [e1['class'], e2['class']],
                                "start_frame": e1["start_frame"],
                                "end_frame": e2["end_frame"]
                            }
                            return combined_grounding, e_target, "In-Between"
    
    # Standard pairwise logic for Before/After/During
    for e1, e2 in combinations(shuffled_events, 2):
        if e1["obj_id"] == e2["obj_id"]: continue  # Different objects only
        
        if target_relationship == "Before":
            if e2["end_frame"] < e1["start_frame"]:
                return e1, e2, "Before"  # Target before Grounding
        elif target_relationship == "After":
            if e2["start_frame"] > e1["end_frame"]:
                return e1, e2, "After"   # Target after Grounding
        elif target_relationship == "During":
            overlap = max(0, min(e1["end_frame"], e2["end_frame"]) - 
                           max(e1["start_frame"], e2["start_frame"]))
            min_len = min(e1["end_frame"] - e1["start_frame"],
                         e2["end_frame"] - e2["start_frame"])
            if min_len > 0 and (overlap / min_len) > 0.5:
                return e1, e2, "During"  # Significant overlap
    
    return None, None, target_relationship  # No valid pair found
```

**Application to MEVA**: Our V3 generators could use this sophisticated relationship logic instead of simple time filtering.

### 4. Scene Graph Spatial Relationships

**Algorithm** for computing spatial relationships from 3D coordinates:

```python
def _compute_spatial_relationships(self, objects, distance_threshold=10.0):
    """
    Computes directional relationships between objects based on relative position.
    
    Logic:
    - near: distance < threshold
    - in_front/behind/left/right: based on angle from ego perspective
    """
    relationships = []
    for obj1, obj2 in combinations(objects, 2):
        pos1, pos2 = np.array(obj1["position"]), np.array(obj2["position"])
        distance = np.linalg.norm(pos2 - pos1)
        
        if distance < distance_threshold:
            relationships.append({
                "source_id": obj1["object_id"], "target_id": obj2["object_id"],
                "relationship_type": "near", "distance": distance
            })
            
            # Compute directional relationship  
            delta = pos2 - pos1  # Vector from obj1 to obj2
            angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi  # Degrees
            
            if -45 <= angle <= 45:
                rel_type = "in_front"
            elif 45 < angle <= 135:
                rel_type = "left"
            elif -135 <= angle < -45:
                rel_type = "right"
            else:  # 135 < angle or angle < -135
                rel_type = "behind"
                
            relationships.append({
                "source_id": obj1["object_id"], "target_id": obj2["object_id"],
                "relationship_type": rel_type, "distance": distance
            })
    
    return relationships
```

**Note**: Our MEVA dataset doesn't have 3D coordinates, but this pattern could be adapted for 2D bounding box relationships.

### 5. Spatio-Temporal QA Generation

**Two-mode approach** combining spatial and temporal reasoning:

```python
class SpatioTemporalQAGenerator(TemporalQAGenerator):
    def _select_event_grounded_spatial_query(self, events, frames_map):
        """
        Type 1: Time -> Space
        Pick an Event, look at spatial relationships at its start frame
        Q: "At the start of [Event], what was to the left of [Object]?"
        """
        anchor_event = random.choice(events)
        time_idx = anchor_event["start_frame"]
        frame_data = frames_map.get(time_idx)
        spatial_rels = [r for r in frame_data.get("relationships", []) 
                       if r["type"] in ["left", "right", "in_front", "behind"]]
        target_rel = random.choice(spatial_rels)
        return {
            "context_info": f"Anchor Event: A {anchor_event['class']} is {anchor_event['activity']}",
            "target_info": f"Spatial Query: What is {target_rel['type']} the {target_rel['source_class']}?",
            "answer": target_rel['target_class']
        }
    
    def _select_spatially_grounded_temporal_query(self, events, frames_map):
        """
        Type 2: Space -> Time  
        Pick object defined by spatial relationship, ask what it did next
        Q: "What did the [car next to the bus] do afterwards?"
        """
        for target_event in events:
            # Look for spatial descriptor before this event
            lookback_frame = max(0, target_event["start_frame"] - 5)
            frame_data = frames_map.get(lookback_frame)
            
            # Find spatial relationships involving this actor
            actor_rels = [r for r in frame_data.get("relationships", []) 
                         if r["source_id"] == target_event["obj_id"]]
            
            if actor_rels:
                spatial_rel = actor_rels[0]
                spatial_desc = f"The {spatial_rel['source_class']} located {spatial_rel['type']} the {spatial_rel['target_class']}"
                return {
                    "context_info": f"Spatially Defined Actor: {spatial_desc}",
                    "target_info": f"Target Activity: {target_event['activity']}",
                    "answer": target_event['description']
                }
        return None
```

**Application to MEVA**: Could enhance our `camera_transition` category with more sophisticated spatial reasoning if we extract relative positions from bounding boxes.

## Scene Graph Data Structures

### Core Data Models

```python
@dataclass
class SceneGraphNode:
    object_id: str           # Unique instance token
    object_class: str        # Object category  
    position: tuple          # 3D position
    size: tuple             # Object dimensions
    velocity: Optional[tuple]  # Object velocity
    attributes: List[str]    # Object attributes
    frame_idx: int          # Frame index
    timestamp: int          # Timestamp
    num_lidar_pts: int      # Number of lidar points
    visibility: int         # Visibility level (0-4)
    
    # VLM-annotated properties (filled later)
    activity: Optional[str] = None
    description: Optional[str] = None
    caption: Optional[str] = None

@dataclass  
class SceneGraphRelationship:
    source_id: str           # Source object instance token
    target_id: str          # Target object instance token
    relationship_type: str   # 'near', 'behind', 'in_front', 'left', 'right'
    distance: Optional[float]  # Distance in meters
    frame_idx: int          # Frame index
```

**MEVA Adaptation**: Our LogicTuple structure serves a similar role, representing object states and relationships over time.

## GPT Prompting Strategies

### 1. Temporal Reasoning Prompts

**Structure**: Grounding Event + Target Event + Relationship Type

```
You are an expert at understanding temporal relationships in multi-camera video sequences.

Given:
Grounding Event: {grounding_input}
Target Event: {target_input}  
Relationship: {rel_type}

Generate a multiple-choice question that tests understanding of the {rel_type} relationship between these events. Format as JSON:

{
  "question": "Based on the video, what happened to [Target] {rel_type} [Grounding Event]?",
  "options": {
    "A": "...",
    "B": "...", 
    "C": "...",
    "D": "..."
  },
  "correct_option": "A",
  "rationale": "..."
}

Guidelines:
- Use specific object classes and activities, not generic terms
- Include plausible distractors that test temporal reasoning  
- Ensure (correct) answer follows {rel_type} logic
- Keep question under 100 words
```

**Key insight**: Template-based prompts with explicit slot filling work better than open-ended generation.

### 2. Event Ordering Prompts

**Structure**: Present events in random order, ask for chronological sequence

```
Given these events from a multi-camera video:

{events_list}

Generate a question asking users to arrange these events in chronological order. Present the events in scrambled order within the question.

JSON format:
{
  "question": "Arrange these events in chronological order from earliest to latest: [scrambled list]",
  "options": {
    "A": "Event A, Event B, Event C, Event D",
    "B": "Event B, Event A, Event D, Event C", 
    "C": "...",
    "D": "..."
  },
  "correct_option": "C",
  "rationale": "Based on the temporal sequence..."
}
```

**Application**: Could enhance our ordering category with more sophisticated event chain logic.

### 3. Summarization Prompts

**Structure**: Temporal scene graph → Comprehensive summary

```
Generate a comprehensive summary of the ego-actor's interactions across all camera views.

Temporal Scene Graph:
{temporal_scene_graph}

Requirements:
- 2-3 sentences maximum
- Focus on key activities and interactions
- Mention spatial context when relevant  
- Use specific object classes, not generic terms
- Maintain temporal flow (before, during, after)

Example: "A black sedan approaches the intersection and stops behind a white bus. While waiting, a pedestrian crosses from left to right. After the pedestrian clears, the sedan turns right and exits the camera view."
```

**Note**: Multi-Camera uses GPT-4o with temperature=0.5, max_tokens=300 for summaries.

## Integration Strategies for MEVA

### 1. Import Key Algorithms

```python
# From Multi-Camera temporal reasoning
def extract_temporal_pairs(logic_tuples, relationship_type):
    """Adapt Multi-Camera's _select_temporal_pair for LogicTuples"""
    events = consolidate_logic_tuples_to_events(logic_tuples)
    return select_temporal_pair(events, relationship_type)

def consolidate_logic_tuples_to_events(logic_tuples):
    """Convert MEVA LogicTuples to Multi-Camera event format"""
    events = []
    for actor_id, tuples in group_by_actor(logic_tuples).items():
        for i, curr_tuple in enumerate(tuples):
            if i == 0 or curr_tuple.activity != tuples[i-1].activity:
                # Start new event
                event = {
                    "obj_id": actor_id,
                    "class": curr_tuple.actor_type,
                    "activity": curr_tuple.activity,
                    "start_frame": curr_tuple.start_frame,
                    "end_frame": curr_tuple.end_frame,
                    "description": curr_tuple.description
                }
                events.append(event)
            else:
                # Extend current event
                events[-1]["end_frame"] = curr_tuple.end_frame
    return events
```

### 2. Enhanced Prompt Templates

Store Multi-Camera prompt patterns in `/home/ah66742/scripts/prompts/`:

- `temporal_advanced.txt` - Multi-event relationship templates  
- `ordering_chains.txt` - Sequential event arrangement
- `spatiotemporal.txt` - Combined spatial/temporal reasoning
- `summarization.txt` - Scene synthesis prompts

### 3. Logging and Evaluation

Adopt Multi-Camera's GPT logging pattern:
```python
def log_gpt_call(self, input_text, output_text, model, temperature, scene_token, **metadata):
    """Log all GPT interactions for evaluation and debugging"""
    log_file = self.gpt_logs_dir / f"{scene_token}.json"
    log_entry = {
        "input": input_text, "output": output_text,
        "metadata": {"model": model, "temperature": temperature, "scene_token": scene_token, **metadata}
    }
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)
```

## Key Takeaways for MEVA

1. **Event Consolidation**: Multi-Camera's frame-to-event consolidation algorithm could improve our LogicTuple processing
2. **Temporal Relationship Logic**: Their sophisticated Before/After/During/In-Between selection is more nuanced than our current time-based filtering  
3. **Spatio-Temporal Integration**: Two-mode approach (Time→Space, Space→Time) could enhance camera_transition questions
4. **Prompt Engineering**: Template-based prompts with explicit slot filling produce more consistent results than open-ended generation
5. **Logging Infrastructure**: Comprehensive GPT call logging enables better evaluation and prompt iteration

## Repository Stats
- **Languages**: 68.9% Jupyter Notebook, 30.5% Python, 0.6% Shell
- **Contributors**: 3 (sahilshah379, Syzygianinfern0, harshg99)  
- **Last Update**: February 5, 2026 (very recent)
- **Size**: ~15-20 Python files, comprehensive prompt library

This reference captures the key algorithmic insights without needing to maintain the full repository locally.
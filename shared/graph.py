import os

@dataclass
class SceneGraphNode:
    annotation_token: str  # Annotation token
    object_id: str  # Unique object instance token
    object_class: str  # Object category name
    position: tuple  # 3D position
    size: tuple  # Object dimensions
    rotation: tuple  # Rotation quaternion (w, x, y, z)
    velocity: Optional[tuple]  # Object velocity
    attributes: List[str]  # Object attributes
    frame_idx: int  # Frame index
    timestamp: int  # Timestamp
    num_lidar_pts: int  # Number of lidar points
    visibility: int  # Visibility level
    sample_token: str  # Sample token for this frame
    visible_cameras: Optional[List[str]] = None  # Cameras where object is visible
    
    # VLM-annotated properties (to be filled later)
    activity: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Edge:
    def __init__(self, from_node, to_node):
        pass

class SceneGraph:
    def __init__(self):
        pass
    def build(self):
        pass



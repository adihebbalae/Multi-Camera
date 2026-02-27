#!/usr/bin/env python3
"""
scene_context.py — Camera-aware spatial context using KRTD + building PLYs.

Provides:
  - Named building proximity for entity 3D positions ("near the school entrance")
  - Cardinal direction from camera to entity ("to the north")
  - Semantic location labels ("in the parking lot", "by the gas station")
  - Cross-camera entity matching boost via 3D proximity

Uses:
  - KRTD camera models (ENU coordinate system) from utils/krtd.py
  - Building segmentation PLYs from /nas/mars/dataset/MEVA/model_segmentations/
  - trimesh for PLY loading and nearest-point queries

Coordinate system: ENU (East-North-Up) — shared across all outdoor cameras.
  x = East, y = North, z = Up (meters)
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import trimesh
except ImportError:
    trimesh = None  # Graceful degradation — scene context disabled

try:
    from .utils.krtd import load_camera_model, load_all_camera_models, CameraModel
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.krtd import load_camera_model, load_all_camera_models, CameraModel


# ============================================================================
# Constants
# ============================================================================

MODEL_SEG_DIR = Path("/nas/mars/dataset/MEVA/model_segmentations")

# Human-friendly building names for each PLY file
BUILDING_NAMES = {
    "school":      "the school",
    "building1":   "Building 1 (east annex)",
    "building2":   "Building 2 (southeast lab)",
    "building3":   "Building 3 (south wing)",
    "building4":   "Building 4 (west annex)",
    "building5":   "Building 5 (northwest housing)",
    "gas_station": "the gas station",
    "background":  None,  # "background" mesh = ground/trees, no label
}

# Short labels for question text (avoid verbosity)
BUILDING_SHORT_LABELS = {
    "school":      "the school",
    "building1":   "the east building",
    "building2":   "the southeast building",
    "building3":   "the south building",
    "building4":   "the west building",
    "building5":   "the northwest building",
    "gas_station": "the gas station",
}

# Site → which buildings are relevant (cameras only see nearby structures)
SITE_BUILDINGS = {
    "school":   ["school", "building1", "building2", "gas_station"],
    "admin":    ["building3", "building4", "building5"],
    "bus":      ["building4", "building5", "gas_station"],
    "hospital": ["building3", "building4"],
}

# Distance thresholds for proximity labels (meters)
# Note: Building centroids are center-of-mass, so even "adjacent" entities may
# be 20-30m from centroid. Use generous thresholds.
NEAR_THRESHOLD = 25.0       # "near X"
MODERATE_THRESHOLD = 60.0   # "in the area of X"
# Beyond moderate = "away from X" / use cardinal direction instead

# Cardinal direction names
_CARDINAL_NAMES = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest"
]


# ============================================================================
# Scene Geometry Cache
# ============================================================================

_scene_cache: Dict[str, 'SceneContext'] = {}


def get_scene_context(site: str) -> Optional['SceneContext']:
    """Get or create SceneContext for a site (cached)."""
    if trimesh is None:
        return None
    if site not in _scene_cache:
        try:
            _scene_cache[site] = SceneContext(site)
        except Exception:
            _scene_cache[site] = None
    return _scene_cache[site]


# ============================================================================
# SceneContext Class
# ============================================================================

class SceneContext:
    """
    Scene geometry for a MEVA site: buildings, cameras, spatial queries.
    
    Loads PLY meshes and precomputes centroids for fast nearest-building lookup.
    """
    
    def __init__(self, site: str):
        """
        Args:
            site: 'school', 'bus', 'admin', or 'hospital'
        """
        self.site = site
        self.buildings: Dict[str, trimesh.Trimesh] = {}
        self.centroids: Dict[str, np.ndarray] = {}
        self._load_buildings()
    
    def _load_buildings(self):
        """Load PLY meshes for buildings relevant to this site."""
        building_ids = SITE_BUILDINGS.get(self.site, list(BUILDING_NAMES.keys()))
        
        for bid in building_ids:
            ply_path = MODEL_SEG_DIR / f"{bid}.ply"
            if not ply_path.exists():
                continue
            try:
                mesh = trimesh.load(str(ply_path))
                self.buildings[bid] = mesh
                self.centroids[bid] = np.array(mesh.centroid)
            except Exception:
                continue
    
    def nearest_building(self, point_3d: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Find the nearest named building to a 3D point.
        
        Args:
            point_3d: [east, north, up] in ENU meters
            
        Returns:
            (building_id, distance_meters) or None if no buildings loaded.
        """
        if not self.buildings:
            return None
        
        point = np.array(point_3d[:2])  # Use 2D (east, north) for horizontal distance
        
        best_id = None
        best_dist = float('inf')
        
        for bid, mesh in self.buildings.items():
            if BUILDING_NAMES.get(bid) is None:
                continue  # Skip "background"
            
            # Use centroid distance as fast approximation
            centroid_2d = self.centroids[bid][:2]
            dist = np.linalg.norm(point - centroid_2d)
            
            if dist < best_dist:
                best_dist = dist
                best_id = bid
        
        if best_id is None:
            return None
        return (best_id, float(best_dist))
    
    def get_location_label(self, point_3d: np.ndarray) -> Optional[str]:
        """
        Generate a semantic location label for a 3D point.
        
        Returns labels like:
          "near the school"
          "in the parking lot area"
          "by the gas station"
          None if no meaningful label can be generated
        """
        result = self.nearest_building(point_3d)
        if result is None:
            return None
        
        bid, dist = result
        short_name = BUILDING_SHORT_LABELS.get(bid, bid)
        
        if dist < NEAR_THRESHOLD:
            return f"near {short_name}"
        elif dist < MODERATE_THRESHOLD:
            return f"in the area near {short_name}"
        else:
            return None  # Too far for meaningful label
    
    def get_cardinal_direction(self, from_point: np.ndarray,
                                to_point: np.ndarray) -> str:
        """
        Cardinal direction from one point to another.
        
        Args:
            from_point: [east, north, up] - observer position (e.g., camera)
            to_point:   [east, north, up] - target position (e.g., entity)
        
        Returns:
            One of: "north", "northeast", "east", "southeast",
                    "south", "southwest", "west", "northwest"
        """
        delta_east = to_point[0] - from_point[0]
        delta_north = to_point[1] - from_point[1]
        
        # atan2(east, north) gives angle from north, clockwise
        angle_rad = math.atan2(delta_east, delta_north)
        angle_deg = math.degrees(angle_rad) % 360
        
        # Quantize to 8 directions (each 45°, centered on 0°=North)
        idx = round(angle_deg / 45.0) % 8
        return _CARDINAL_NAMES[idx]
    
    def get_camera_relative_direction(self, camera_id: str,
                                       entity_point: np.ndarray) -> Optional[str]:
        """
        Cardinal direction from camera center to entity world position.
        
        Returns: "to the north", "to the southeast", etc.
                 None if camera model not available.
        """
        cam_model = load_camera_model(camera_id)
        if cam_model is None:
            return None
        
        cam_center = cam_model.camera_center
        direction = self.get_cardinal_direction(cam_center, entity_point)
        return f"to the {direction}"
    
    def annotate_entity(self, point_3d: np.ndarray,
                         camera_id: Optional[str] = None) -> Dict[str, str]:
        """
        Generate spatial context annotations for an entity at a 3D position.
        
        Returns dict with:
          location_label: "near the school" or None
          cardinal_from_camera: "to the north" or None
          nearest_building: "school" or None
          building_distance_m: float or None
        """
        annotations = {}
        
        # Location label
        annotations["location_label"] = self.get_location_label(point_3d)
        
        # Cardinal direction from camera
        if camera_id:
            annotations["cardinal_from_camera"] = \
                self.get_camera_relative_direction(camera_id, point_3d)
        
        # Nearest building ID + distance
        result = self.nearest_building(point_3d)
        if result:
            bid, dist = result
            annotations["nearest_building"] = bid
            annotations["building_distance_m"] = round(dist, 1)
        
        return annotations
    
    def entity_3d_distance(self, point_a: np.ndarray,
                            point_b: np.ndarray) -> float:
        """3D Euclidean distance between two entity world positions (meters)."""
        return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))
    
    def entity_2d_distance(self, point_a: np.ndarray,
                            point_b: np.ndarray) -> float:
        """2D horizontal distance (East-North plane) in meters."""
        return float(np.linalg.norm(
            np.array(point_a[:2]) - np.array(point_b[:2])
        ))


# ============================================================================
# Cross-Camera Entity Matching with 3D Proximity
# ============================================================================

def compute_3d_matching_score(
    cam_a: str, bbox_a: List[float],
    cam_b: str, bbox_b: List[float],
    time_gap_sec: float,
    description_overlap: float = 0.0,
) -> Optional[Dict]:
    """
    Compute cross-camera entity matching score using 3D + temporal + visual.
    
    Projects bbox footpoints to 3D world coords, checks proximity.
    
    Args:
        cam_a, cam_b: Camera IDs
        bbox_a, bbox_b: [x1, y1, x2, y2] bounding boxes
        time_gap_sec: Temporal gap between entity A (last frame) and B (first frame)
        description_overlap: Token overlap ratio of entity descriptions (0-1)
    
    Returns:
        {
            "spatial_score": float,    # 0-1, based on 3D proximity
            "temporal_score": float,   # 0-1, based on time gap
            "description_score": float,# 0-1, based on visual similarity
            "combined_score": float,   # weighted combination
            "distance_3d_m": float,    # 3D distance in meters
            "point_a": list,           # [e, n, u]
            "point_b": list,           # [e, n, u]
        }
        None if cameras don't have KRTD models or projection fails.
    """
    model_a = load_camera_model(cam_a)
    model_b = load_camera_model(cam_b)
    if model_a is None or model_b is None:
        return None
    
    # Project bbox footpoints to world
    point_a = model_a.bbox_foot_to_world(bbox_a)
    point_b = model_b.bbox_foot_to_world(bbox_b)
    if point_a is None or point_b is None:
        return None
    
    # 3D distance
    dist = float(np.linalg.norm(point_a - point_b))
    
    # Spatial score: closer = higher score
    # 0-5m → 1.0, 5-20m → linear decay, >20m → 0.0
    if dist < 5.0:
        spatial_score = 1.0
    elif dist < 20.0:
        spatial_score = max(0.0, 1.0 - (dist - 5.0) / 15.0)
    else:
        spatial_score = 0.0
    
    # Temporal score: smaller gap = higher score
    # 0-2s → 1.0, 2-10s → linear decay, >10s → 0.4 minimum
    if time_gap_sec < 2.0:
        temporal_score = 1.0
    elif time_gap_sec < 10.0:
        temporal_score = max(0.4, 1.0 - (time_gap_sec - 2.0) / 8.0 * 0.6)
    else:
        temporal_score = 0.4
    
    # Description score is passed in directly
    desc_score = max(0.0, min(1.0, description_overlap))
    
    # Combined: 0.4 * temporal + 0.3 * spatial + 0.3 * description
    combined = 0.4 * temporal_score + 0.3 * spatial_score + 0.3 * desc_score
    
    return {
        "spatial_score": round(spatial_score, 3),
        "temporal_score": round(temporal_score, 3),
        "description_score": round(desc_score, 3),
        "combined_score": round(combined, 3),
        "distance_3d_m": round(dist, 2),
        "point_a": point_a.tolist(),
        "point_b": point_b.tolist(),
    }


# ============================================================================
# Utility: Enrich Question Text with Spatial Context
# ============================================================================

def enrich_description_with_location(
    description: str,
    point_3d: Optional[np.ndarray],
    site: str,
) -> str:
    """
    Append spatial context to an entity description for question text.
    
    "a person in navy top and khaki pants opens a vehicle door"
    → "a person in navy top and khaki pants opens a vehicle door near the school parking lot"
    
    Only adds context if a meaningful location label exists.
    Does NOT reveal camera ID (important for perception questions).
    """
    if point_3d is None:
        return description
    
    ctx = get_scene_context(site)
    if ctx is None:
        return description
    
    label = ctx.get_location_label(point_3d)
    if label is None:
        return description
    
    # Avoid repeating location if already mentioned
    if label.lower() in description.lower():
        return description
    
    return f"{description} {label}"


# ============================================================================
# CLI for testing / debugging
# ============================================================================

def main():
    """Quick test: show scene context for the school site."""
    import argparse
    parser = argparse.ArgumentParser(description="Scene context test")
    parser.add_argument("--site", default="school", help="Site name")
    parser.add_argument("--camera", default="G339", help="Camera ID to test")
    args = parser.parse_args()
    
    if trimesh is None:
        print("ERROR: trimesh not installed. Run: pip install trimesh")
        return
    
    ctx = get_scene_context(args.site)
    if ctx is None:
        print(f"Failed to create scene context for {args.site}")
        return
    
    print(f"Scene Context: {args.site}")
    print(f"  Buildings loaded: {list(ctx.buildings.keys())}")
    for bid, centroid in ctx.centroids.items():
        print(f"  {bid:15s}: centroid=({centroid[0]:8.1f}, {centroid[1]:8.1f}, {centroid[2]:8.1f})")
    
    # Test with camera center
    cam = load_camera_model(args.camera)
    if cam:
        cc = cam.camera_center
        print(f"\n  Camera {args.camera} center: ({cc[0]:.1f}, {cc[1]:.1f}, {cc[2]:.1f})")
        ann = ctx.annotate_entity(cc, args.camera)
        print(f"  Annotations: {ann}")
        
        # Test a point slightly offset from camera
        test_point = cc + np.array([10, 0, 0])  # 10m east
        ann2 = ctx.annotate_entity(test_point, args.camera)
        label = ctx.get_location_label(test_point)
        print(f"\n  Test point (10m east of camera): ({test_point[0]:.1f}, {test_point[1]:.1f})")
        print(f"  Location label: {label}")
        print(f"  Annotations: {ann2}")


if __name__ == "__main__":
    main()

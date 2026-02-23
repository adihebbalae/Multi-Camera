"""
V6 utils/krtd.py — KRTD camera calibration parsing and 3D projection.

Ported from V4's CameraModel class with improvements for V6 pipeline.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KRTD_DIR = Path("/nas/mars/dataset/MEVA/meva-data-repo/metadata/camera-models/krtd")

# Indoor cameras (local coordinate frame, NOT in shared ENU)
INDOOR_CAMERAS = {"G299", "G330"}


class CameraModel:
    """
    KRTD camera calibration model.
    
    Projects 2D pixel coordinates to 3D world coordinates on the ground plane.
    Uses the ENU (East-North-Up) coordinate system shared across outdoor cameras.
    """
    
    def __init__(self, krtd_path: Path):
        self.path = krtd_path
        self.K: np.ndarray = None   # 3x3 intrinsic
        self.R: np.ndarray = None   # 3x3 rotation  
        self.T: np.ndarray = None   # 3x1 translation
        self.D: np.ndarray = None   # distortion coefficients
        self._parse(krtd_path)
    
    def _parse(self, path: Path):
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.K = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
        self.R = np.array([[float(x) for x in lines[i].split()] for i in range(3, 6)])
        self.T = np.array([float(x) for x in lines[6].split()])
        if len(lines) > 7:
            self.D = np.array([float(x) for x in lines[7].split()])
    
    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world (ENU) coordinates: C = -R^T * T"""
        return -self.R.T @ self.T
    
    def project_to_ground(self, u: float, v: float, ground_z: float = 0.0) -> Optional[np.ndarray]:
        """
        Back-project pixel (u, v) onto the ground plane (ENU z = ground_z).
        
        Returns [east, north, up] in meters, or None if ray is parallel to ground.
        """
        K_inv = np.linalg.inv(self.K)
        d_cam = K_inv @ np.array([u, v, 1.0])
        d_world = self.R.T @ d_cam
        C = self.camera_center
        
        if abs(d_world[2]) < 1e-10:
            return None  # ray parallel to ground
        
        t = (ground_z - C[2]) / d_world[2]
        if t < 0:
            return None  # behind camera
        
        return C + t * d_world
    
    def bbox_foot_to_world(self, bbox: List[float], ground_z: float = 0.0) -> Optional[np.ndarray]:
        """
        Project the bottom-center of a bounding box to world coordinates.
        
        bbox = [x1, y1, x2, y2] in pixel coordinates.
        Returns [east, north, up] or None.
        """
        x1, y1, x2, y2 = bbox
        foot_u = (x1 + x2) / 2.0  # horizontal center
        foot_v = max(y1, y2)       # bottom of bbox
        return self.project_to_ground(foot_u, foot_v, ground_z)


def load_camera_model(camera_id: str) -> Optional[CameraModel]:
    """Load KRTD calibration for a camera. Returns None if not available."""
    if camera_id in INDOOR_CAMERAS:
        return None
    krtd_files = list(KRTD_DIR.glob(f"*.{camera_id}.krtd"))
    if not krtd_files:
        return None
    try:
        return CameraModel(krtd_files[0])
    except Exception:
        return None


def load_all_camera_models(camera_ids: List[str]) -> Dict[str, CameraModel]:
    """Load KRTD models for a list of cameras. Skips indoor / unavailable."""
    models = {}
    for cam_id in camera_ids:
        model = load_camera_model(cam_id)
        if model is not None:
            models[cam_id] = model
    return models


def compute_entity_distance(model_a: CameraModel, bbox_a: List[float],
                             model_b: CameraModel, bbox_b: List[float]) -> Optional[float]:
    """
    Compute 3D distance between two entities given their bounding boxes and camera models.
    
    Returns distance in meters, or None if projection fails.
    """
    pos_a = model_a.bbox_foot_to_world(bbox_a)
    pos_b = model_b.bbox_foot_to_world(bbox_b)
    if pos_a is None or pos_b is None:
        return None
    return float(np.linalg.norm(pos_a - pos_b))


def classify_proximity(distance_m: float) -> str:
    """Classify distance into proximity buckets."""
    if distance_m <= 5.0:
        return "near"
    elif distance_m <= 15.0:
        return "moderate"
    else:
        return "far"

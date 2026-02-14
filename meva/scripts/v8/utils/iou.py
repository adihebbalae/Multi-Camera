"""
V6 utils/iou.py — IoU (Intersection over Union) utilities for bounding box matching.
"""

from typing import List, Optional


def compute_iou(bbox_a: List[int], bbox_b: List[int]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox_a: [x1, y1, x2, y2]
        bbox_b: [x1, y1, x2, y2]
    
    Returns:
        IoU value [0.0, 1.0]
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    
    if union <= 0:
        return 0.0
    
    return inter / union

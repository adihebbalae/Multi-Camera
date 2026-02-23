"""
V6 utils/yaml_stream.py — Memory-efficient geom.yml streaming parser.

Parses geom.yml files line-by-line using regex, yielding {id1, ts0, g0} dicts
without loading the entire file into memory. This avoids OOM on large files (50MB+).
"""

import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set


def stream_geom_records(path: Path) -> Generator[Dict, None, None]:
    """
    Stream-parse a geom.yml file line-by-line.
    Yields dicts: {"id1": int, "ts0": int, "g0": [x1, y1, x2, y2]}
    
    Handles TWO Kitware geom.yml formats:
      1. kitware format (unquoted keys, fixed field order):
         - { geom: {id1: NNN, id0: M, ts0: FFF, ts1: T.T, g0: X1 Y1 X2 Y2, ...} }
      2. kitware-meva-training format (Python dict notation, quoted keys, arbitrary order):
         - {'geom': {'g0': '886 418 929 504', 'id0': 1, 'id1': 10, 'ts0': 1475}}
    Also handles multi-line format (fallback).
    """
    # Format 1: kitware inline (unquoted, id1 before ts0 before g0)
    kitware_re = re.compile(
        r'id1:\s*(\d+).*?ts0:\s*(\d+).*?g0:\s*(\d+\s+\d+\s+\d+\s+\d+)'
    )
    
    # Format 2: kitware-meva-training (quoted keys, any order)
    # Extract fields individually since order varies
    training_id1_re = re.compile(r"'id1':\s*(\d+)")
    training_ts0_re = re.compile(r"'ts0':\s*(\d+)")
    training_g0_re = re.compile(r"'g0':\s*'(\d+\s+\d+\s+\d+\s+\d+)'")
    
    # Fallback: multi-line parsing
    current: Dict = {}
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try kitware inline format first (most common)
            m = kitware_re.search(line)
            if m:
                yield {
                    'id1': int(m.group(1)),
                    'ts0': int(m.group(2)),
                    'g0': [int(x) for x in m.group(3).split()],
                }
                continue
            
            # Try kitware-meva-training format (quoted keys, any field order)
            m_id1 = training_id1_re.search(line)
            m_ts0 = training_ts0_re.search(line)
            m_g0 = training_g0_re.search(line)
            if m_id1 and m_ts0 and m_g0:
                yield {
                    'id1': int(m_id1.group(1)),
                    'ts0': int(m_ts0.group(1)),
                    'g0': [int(x) for x in m_g0.group(1).split()],
                }
                continue
            
            # Fallback: multi-line format
            m = re.match(r"'?id1'?:\s*(\d+)", line)
            if m:
                current['id1'] = int(m.group(1))
                continue
            m = re.match(r"'?ts0'?:\s*(\d+)", line)
            if m:
                current['ts0'] = int(m.group(1))
                continue
            m = re.match(r"'?g0'?:\s*['\"]?(\d+\s+\d+\s+\d+\s+\d+)", line)
            if m:
                current['g0'] = [int(x) for x in m.group(1).split()]
                if 'id1' in current and 'ts0' in current:
                    yield dict(current)
                current = {}


def get_actor_keyframe_bboxes(path: Path, actor_ids: Optional[Set[int]] = None,
                               sample_every: int = 1) -> Dict[int, Dict[int, List[int]]]:
    """
    Extract bounding boxes for specific actors (or all actors).
    
    Args:
        path: Path to geom.yml
        actor_ids: Set of actor IDs to extract (None = all)
        sample_every: Sample every Nth frame (default 1 = all frames)
    
    Returns:
        {actor_id: {frame: [x1, y1, x2, y2], ...}, ...}
    """
    result: Dict[int, Dict[int, List[int]]] = {}
    for rec in stream_geom_records(path):
        aid = rec['id1']
        if actor_ids is not None and aid not in actor_ids:
            continue
        frame = rec['ts0']
        if frame % sample_every != 0:
            continue
        if aid not in result:
            result[aid] = {}
        result[aid][frame] = rec['g0']
    return result


def get_actor_frame_range(path: Path) -> Dict[int, tuple]:
    """
    Get (min_frame, max_frame) for each actor in a geom.yml.
    Memory-efficient: only tracks frame extremes.
    
    Returns:
        {actor_id: (first_frame, last_frame), ...}
    """
    ranges: Dict[int, list] = {}
    for rec in stream_geom_records(path):
        aid = rec['id1']
        frame = rec['ts0']
        if aid not in ranges:
            ranges[aid] = [frame, frame]
        else:
            if frame < ranges[aid][0]:
                ranges[aid][0] = frame
            if frame > ranges[aid][1]:
                ranges[aid][1] = frame
    return {aid: tuple(r) for aid, r in ranges.items()}


def get_bbox_at_frame(path: Path, actor_id: int, target_frame: int,
                      tolerance: int = 5) -> Optional[List[int]]:
    """
    Get a single bbox for a specific actor at (or near) a target frame.
    
    Args:
        path: Path to geom.yml
        actor_id: Actor to look for
        target_frame: Frame number to find
        tolerance: Accept frames within ±tolerance
    
    Returns:
        [x1, y1, x2, y2] or None
    """
    best = None
    best_dist = tolerance + 1
    for rec in stream_geom_records(path):
        if rec['id1'] != actor_id:
            continue
        dist = abs(rec['ts0'] - target_frame)
        if dist < best_dist:
            best_dist = dist
            best = rec['g0']
        if dist == 0:
            break
    return best

"""
V6 entity_resolution.py — Step 3: Cross-camera entity linking.

Uses MEVID ground truth person IDs + heuristic temporal handoff for
cross-camera entity resolution. Produces entity clusters where each
cluster represents the same real-world person across cameras.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

from .build_scene_graph import SceneGraph, Entity
from .utils.mevid import find_mevid_persons_for_slot


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CrossCameraLink:
    """A link between two entities on different cameras."""
    entity_a: str
    entity_b: str
    camera_a: str
    camera_b: str
    link_type: str          # "mevid_ground_truth" or "temporal_handoff"
    confidence: float       # 1.0 for MEVID, 0.0-1.0 for heuristic
    mevid_person_id: Optional[int] = None
    time_gap_sec: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EntityCluster:
    """A cluster of entities representing the same real-world person."""
    cluster_id: str
    entities: List[str]     # entity_ids
    cameras: List[str]      # camera_ids involved
    mevid_person_id: Optional[int] = None
    link_type: str = "heuristic"  # "mevid_ground_truth" or "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResolvedGraph:
    """Scene graph with resolved cross-camera entity links."""
    cross_camera_links: List[CrossCameraLink]
    entity_clusters: List[EntityCluster]
    mevid_persons_in_slot: int
    heuristic_link_count: int
    mevid_link_count: int

    def to_dict(self) -> dict:
        return {
            "cross_camera_links": [l.to_dict() for l in self.cross_camera_links],
            "entity_clusters": [c.to_dict() for c in self.entity_clusters],
            "mevid_persons_in_slot": self.mevid_persons_in_slot,
            "heuristic_link_count": self.heuristic_link_count,
            "mevid_link_count": self.mevid_link_count,
        }


# ============================================================================
# Union-Find for clustering
# ============================================================================

class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
    
    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
    
    def clusters(self) -> Dict[str, Set[str]]:
        groups: Dict[str, Set[str]] = defaultdict(set)
        for item in self.parent:
            groups[self.find(item)].add(item)
        return dict(groups)


# ============================================================================
# MEVID-Based Entity Resolution (Ground Truth)
# ============================================================================

def _resolve_mevid(sg: SceneGraph, verbose: bool = False) -> Tuple[List[CrossCameraLink], Dict[int, Set[str]]]:
    """
    Use MEVID person IDs to establish cross-camera entity links.
    
    Since we can't map MEVID person_ids to specific Kitware actor_ids without
    extracted tracklet images, we establish POTENTIAL cross-camera links:
    For each MEVID person appearing on 2+ cameras in this slot, we know that
    some entity on camera A is the same person as some entity on camera B.
    
    Returns:
        (links, mevid_persons_map)
    """
    slot_cameras = list(sg.cameras.keys())
    mevid_persons = find_mevid_persons_for_slot(sg.slot, slot_cameras)
    
    if verbose:
        print(f"  MEVID: {len(mevid_persons)} persons with 2+ cameras in slot")
        for pid, cams in sorted(mevid_persons.items())[:5]:
            print(f"    Person {pid}: cameras {sorted(cams)}")
    
    # We can't create specific entity-to-entity links without tracklet extraction
    # but we know which cameras share persons — this validates heuristic links
    return [], mevid_persons


# ============================================================================
# Heuristic Entity Resolution (Temporal Handoff)
# ============================================================================

# Entry/exit activities that suggest camera handoff
EXIT_ACTIVITIES = {
    "person_exits_scene_through_structure",
    "person_exits_vehicle",
}
ENTRY_ACTIVITIES = {
    "person_enters_scene_through_structure", 
    "person_enters_vehicle",
}


def _resolve_heuristic(sg: SceneGraph, verbose: bool = False) -> List[CrossCameraLink]:
    """
    Heuristic cross-camera entity linking via temporal handoff.
    
    Strategy: If a person-type entity's time span ENDS on camera A around time T,
    and another person-type entity's time span STARTS on camera B around time T,
    AND both have a small number of events (suggesting a brief appearance = handoff),
    link them as potentially the same person.
    
    Only considers entities that participate in at least 1 event (not background actors).
    """
    MAX_HANDOFF_GAP = 10.0   # seconds — tight for heuristic
    MIN_HANDOFF_GAP = 1.0    # seconds (avoid linking simultaneous entities)
    MAX_LINKS_PER_ENTITY = 2  # prevent one entity linking to everything
    
    links = []
    link_count: Dict[str, int] = {}  # entity_id -> # links created
    
    # Only consider entities that participate in events
    active_entities = []
    for eid, entity in sg.entities.items():
        if entity.entity_type != "person":
            continue
        if not entity.events:
            continue
        active_entities.append({
            "entity_id": eid,
            "camera_id": entity.camera_id,
            "first_sec": entity.first_sec,
            "last_sec": entity.last_sec,
        })
    
    # Sort by last_sec (when they leave)
    active_entities.sort(key=lambda x: x["last_sec"])
    
    # Index by first_sec for efficient lookup
    by_first = sorted(active_entities, key=lambda x: x["first_sec"])
    
    for ea in active_entities:
        if link_count.get(ea["entity_id"], 0) >= MAX_LINKS_PER_ENTITY:
            continue
        
        # Look for entities that START shortly after ea ENDS
        for eb in by_first:
            if ea["camera_id"] == eb["camera_id"]:
                continue
            
            gap = eb["first_sec"] - ea["last_sec"]
            
            if gap < MIN_HANDOFF_GAP:
                continue
            if gap > MAX_HANDOFF_GAP:
                break  # sorted, so no more matches
            
            if link_count.get(eb["entity_id"], 0) >= MAX_LINKS_PER_ENTITY:
                continue
            
            confidence = max(0.4, 1.0 - gap / MAX_HANDOFF_GAP)
            links.append(CrossCameraLink(
                entity_a=ea["entity_id"],
                entity_b=eb["entity_id"],
                camera_a=ea["camera_id"],
                camera_b=eb["camera_id"],
                link_type="temporal_handoff",
                confidence=round(confidence, 2),
                time_gap_sec=round(gap, 2),
            ))
            link_count[ea["entity_id"]] = link_count.get(ea["entity_id"], 0) + 1
            link_count[eb["entity_id"]] = link_count.get(eb["entity_id"], 0) + 1
    
    if verbose:
        print(f"  Heuristic: {len(links)} temporal handoff links "
              f"(from {len(active_entities)} active entities)")
    
    return links


# ============================================================================
# Combined Entity Resolution
# ============================================================================

def resolve_entities(sg: SceneGraph, verbose: bool = False) -> ResolvedGraph:
    """
    Run entity resolution: MEVID ground truth + heuristic temporal handoff.
    
    Args:
        sg: Scene graph from build_scene_graph
        verbose: Print progress
    
    Returns:
        ResolvedGraph with cross-camera links and entity clusters
    """
    if verbose:
        print("Resolving cross-camera entities...")
    
    # 1. MEVID ground truth (camera-level mapping)
    mevid_links, mevid_persons = _resolve_mevid(sg, verbose)
    
    # 2. Heuristic temporal handoff
    heuristic_links = _resolve_heuristic(sg, verbose)
    
    # 3. Combine links and build clusters using Union-Find
    all_links = mevid_links + heuristic_links
    
    uf = UnionFind()
    for link in all_links:
        if link.confidence >= 0.7:  # stricter threshold for clustering
            uf.union(link.entity_a, link.entity_b)
    
    # Also add all entities (even unlinked) to UnionFind
    for eid in sg.entities:
        uf.find(eid)
    
    # Build entity clusters (only multi-entity clusters)
    raw_clusters = uf.clusters()
    entity_clusters = []
    cluster_idx = 0
    for root, members in raw_clusters.items():
        if len(members) < 2:
            continue
        cameras = sorted(set(
            sg.entities[eid].camera_id for eid in members if eid in sg.entities
        ))
        if len(cameras) < 2:
            continue
        entity_clusters.append(EntityCluster(
            cluster_id=f"cluster_{cluster_idx}",
            entities=sorted(members),
            cameras=cameras,
            link_type="heuristic",
        ))
        cluster_idx += 1
    
    if verbose:
        print(f"  Clusters: {len(entity_clusters)} cross-camera entity clusters")
        for c in entity_clusters[:3]:
            print(f"    {c.cluster_id}: {c.entities} across {c.cameras}")
    
    return ResolvedGraph(
        cross_camera_links=all_links,
        entity_clusters=entity_clusters,
        mevid_persons_in_slot=len(mevid_persons),
        heuristic_link_count=len(heuristic_links),
        mevid_link_count=len(mevid_links),
    )

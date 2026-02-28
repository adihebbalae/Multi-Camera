#!/usr/bin/env python3
"""
VLM Entity Description Enrichment — Use InternVL2.5-8B to generate rich
natural-language descriptions of MEVA person entities.

Takes entity crops already extracted by extract_entity_descriptions.py,
sends them to a local vLLM server, and produces rich descriptions like:
  "a middle-aged man in a dark blue jacket and khaki pants, carrying a
   black backpack, walking briskly"

This is a POST-PROCESSING step that enriches existing SegFormer descriptions.
It does NOT replace SegFormer — it adds a `vlm_description` field alongside
the existing color-based `description`.

Prerequisites:
  - vLLM server running (launched automatically or manually)
  - Entity descriptions already extracted (SegFormer batch)
  - Crops cached or re-extractable from video

Usage:
    # Process a single slot
    python3 -m meva.scripts.v10.vlm_describe_entities --slot 2018-03-11.11-25.school -v

    # Batch process all slots with existing entity descriptions
    python3 -m meva.scripts.v10.vlm_describe_entities --batch -v

    # Use existing vLLM server
    python3 -m meva.scripts.v10.vlm_describe_entities --slot ... --api-url http://localhost:8001/v1

Cost: $0 (local model, no API calls)
GPU: 1x RTX A5000 24GB (InternVL2.5-8B ~16GB VRAM)
Time: ~1-2 sec per entity, ~3-5 min per slot
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# ============================================================================
# Paths
# ============================================================================

_REPO_DATA = Path(__file__).resolve().parent.parent.parent / "data"
ENTITY_DESC_DIR = Path(os.environ.get("MEVA_ENTITY_DESC_DIR") or "/nas/mars/dataset/MEVA/entity_descriptions")
VLM_DESC_DIR = Path(os.environ.get("MEVA_VLM_DESC_DIR") or "/nas/mars/dataset/MEVA/entity_descriptions/vlm")

# Reuse the crop extraction infrastructure
try:
    from .extract_entity_descriptions import parse_geom, extract_crops, find_slot_files
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extract_entity_descriptions import parse_geom, extract_crops, find_slot_files

# ============================================================================
# Constants
# ============================================================================

VLM_MODEL = "OpenGVLab/InternVL2_5-8B"
DEFAULT_PORT = 8001      # Avoid conflict with SegFormer on default port
DEFAULT_GPU = "1"        # GPU 1 (0 used by SegFormer re-extraction)
MAX_WORKERS = 8          # Parallel VLM requests
CROPS_FOR_VLM = 3        # Crops to send per entity (middle-of-track, best quality)
MIN_CROP_HEIGHT_VLM = 80 # Min height for meaningful VLM description
MAX_ENTITIES_PER_SLOT = 200  # Cap to avoid extremely long slots


# ============================================================================
# VLM Server Management
# ============================================================================

_vllm_proc = None


def start_vllm_server(gpu: str = DEFAULT_GPU, port: int = DEFAULT_PORT,
                      model: str = VLM_MODEL, verbose: bool = False) -> str:
    """Start a vLLM server if not already running. Returns API URL."""
    global _vllm_proc
    api_url = f"http://localhost:{port}/v1"

    # Check if server is already responsive
    if _check_server(api_url):
        if verbose:
            print(f"  vLLM server already running at {api_url}")
        return api_url

    if verbose:
        print(f"  Starting vLLM server on GPU {gpu}, port {port}...")
        print(f"  Model: {model}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    cmd = [
        "vllm", "serve", model,
        "--tensor-parallel-size", "1",
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
    ]

    log_path = Path.home() / "data" / "extraction_logs" / f"vllm_server_{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _vllm_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    if verbose:
        print(f"  Waiting for vLLM server to start (PID {_vllm_proc.pid})...")

    # Wait up to 120s for server to become healthy
    for i in range(120):
        time.sleep(1)
        if _check_server(api_url):
            if verbose:
                print(f"  vLLM server ready after {i+1}s")
            return api_url
        # Check if process died
        if _vllm_proc.poll() is not None:
            print(f"  ERROR: vLLM server died (exit code {_vllm_proc.returncode})")
            print(f"  Check log: {log_path}")
            sys.exit(1)

    print(f"  ERROR: vLLM server did not become ready in 120s")
    print(f"  Check log: {log_path}")
    sys.exit(1)


def _check_server(api_url: str) -> bool:
    """Check if vLLM server is healthy."""
    try:
        import urllib.request
        req = urllib.request.urlopen(f"{api_url}/models", timeout=3)
        return req.status == 200
    except Exception:
        return False


def stop_vllm_server():
    """Stop the vLLM server if we started it."""
    global _vllm_proc
    if _vllm_proc is not None:
        os.killpg(os.getpgid(_vllm_proc.pid), signal.SIGTERM)
        _vllm_proc.wait(timeout=10)
        _vllm_proc = None


# ============================================================================
# VLM Description Generation
# ============================================================================

def _encode_image(image: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG string."""
    ret, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode("utf-8")


def _build_prompt(num_images: int, segformer_desc: str = "") -> str:
    """Build the VLM prompt for person description."""
    context = ""
    if segformer_desc and segformer_desc != "a person":
        context = f"\nA computer vision system described this person as: \"{segformer_desc}\"\n"

    return f"""You are viewing {num_images} cropped image(s) of the SAME person from a surveillance camera at different moments.
{context}
Describe this person's appearance in ONE concise sentence (max 25 words). Focus on:
- Clothing colors and types (shirt, jacket, hoodie, pants, shorts, dress)
- Distinguishing features (hat, backpack, bag, glasses, beard, hair style/color)
- Apparent build (tall/short/heavyset if clearly visible)
- DO NOT mention actions, activities, or what the person is doing
- DO NOT mention image quality or camera angle
- Use natural casual language, e.g. "a tall man in a dark blue jacket and tan pants, carrying a gray backpack"

Respond with ONLY the description sentence, starting with "a" or "an". Nothing else."""


def describe_entity_vlm(
    crops: List[np.ndarray],
    api_url: str,
    model: str = VLM_MODEL,
    segformer_desc: str = "",
) -> Optional[str]:
    """
    Send entity crops to VLM and get a rich description.

    Args:
        crops: List of BGR crop images (1-3)
        api_url: vLLM server URL
        model: Model name
        segformer_desc: Existing SegFormer description for context

    Returns:
        Description string or None on failure
    """
    try:
        from openai import OpenAI
    except ImportError:
        # Fallback to raw HTTP
        return _describe_entity_vlm_http(crops, api_url, model, segformer_desc)

    client = OpenAI(api_key="EMPTY", base_url=api_url)

    # Select best crops (largest, middle-of-track already selected by extractor)
    use_crops = crops[:CROPS_FOR_VLM]

    # Build content
    prompt = _build_prompt(len(use_crops), segformer_desc)
    content = [{"type": "text", "text": prompt}]

    for crop in use_crops:
        encoded = _encode_image(crop)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
        })

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()

        # Clean up: ensure it starts with "a " or "an "
        result = result.strip('"\'')
        if not result.lower().startswith(("a ", "an ")):
            result = "a " + result

        # Truncate if too long
        if len(result) > 200:
            result = result[:197] + "..."

        return result

    except Exception as e:
        return None


def _describe_entity_vlm_http(
    crops: List[np.ndarray],
    api_url: str,
    model: str,
    segformer_desc: str = "",
) -> Optional[str]:
    """Fallback HTTP-based VLM call if openai package not available."""
    import urllib.request
    import json as _json

    use_crops = crops[:CROPS_FOR_VLM]
    prompt = _build_prompt(len(use_crops), segformer_desc)

    content = [{"type": "text", "text": prompt}]
    for crop in use_crops:
        encoded = _encode_image(crop)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "temperature": 0.1,
    }

    try:
        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = _json.loads(resp.read())
        text = result["choices"][0]["message"]["content"].strip().strip('"\'')
        if not text.lower().startswith(("a ", "an ")):
            text = "a " + text
        return text[:200]
    except Exception:
        return None


# ============================================================================
# Slot Processing
# ============================================================================

def process_slot_vlm(
    slot: str,
    api_url: str,
    model: str = VLM_MODEL,
    verbose: bool = False,
    max_entities: int = MAX_ENTITIES_PER_SLOT,
    max_workers: int = MAX_WORKERS,
) -> Dict:
    """
    Enrich entity descriptions for a slot with VLM captioning.

    Reads existing SegFormer descriptions, extracts crops for entities
    that are large enough, queries VLM for rich descriptions, and saves
    the result alongside the SegFormer data.

    Returns stats dict.
    """
    t0 = time.time()

    # Load existing SegFormer descriptions
    segformer_path = ENTITY_DESC_DIR / f"{slot}.json"
    if not segformer_path.exists():
        if verbose:
            print(f"  No SegFormer descriptions for {slot}, skipping")
        return {"error": "no_segformer_data", "slot": slot}

    with open(segformer_path) as f:
        segformer_data = json.load(f)

    actors = segformer_data.get("actors", {})
    if not actors:
        return {"error": "no_actors", "slot": slot}

    # Find entities worth sending to VLM (large enough crops)
    vlm_candidates = []
    for eid, edata in actors.items():
        h = edata.get("avg_crop_height", 0)
        if h >= MIN_CROP_HEIGHT_VLM:
            vlm_candidates.append((eid, edata))

    if not vlm_candidates:
        if verbose:
            print(f"  No entities large enough for VLM in {slot}")
        return {"slot": slot, "total": len(actors), "vlm_described": 0}

    # Cap entities
    if len(vlm_candidates) > max_entities:
        vlm_candidates = vlm_candidates[:max_entities]

    if verbose:
        print(f"  {len(vlm_candidates)}/{len(actors)} entities qualify for VLM")

    # Extract crops from video
    files = find_slot_files(slot)
    if not files:
        return {"error": "no_video_files", "slot": slot}

    # Group candidates by camera
    cam_entities = defaultdict(list)
    for eid, edata in vlm_candidates:
        cam_entities[edata["camera"]].append((eid, edata))

    # Extract crops per camera
    all_crops = {}  # eid → [crop, ...]
    for cam, entities in cam_entities.items():
        # Find matching camera file
        cam_file = None
        for cf in files:
            if cf["camera"] == cam:
                cam_file = cf
                break
        if cam_file is None or cam_file.get("video_path") is None:
            continue

        # Parse geom for this camera
        geom_actors = parse_geom(cam_file["geom_path"])

        # Filter to only our candidates
        target_actors = {}
        for eid, edata in entities:
            aid = edata.get("actor_id")
            if aid is not None and aid in geom_actors:
                target_actors[aid] = geom_actors[aid]

        if not target_actors:
            continue

        # Extract crops (reuse existing function, but with VLM-specific min height)
        cam_crops = extract_crops(
            cam_file["video_path"],
            target_actors,
            max_crops=CROPS_FOR_VLM,
            min_h=MIN_CROP_HEIGHT_VLM,
            min_w=48,
        )

        # Map back to entity IDs
        for eid, edata in entities:
            aid = edata.get("actor_id")
            if aid in cam_crops and cam_crops[aid]:
                all_crops[eid] = cam_crops[aid]

    if verbose:
        print(f"  Extracted crops for {len(all_crops)} entities")

    if not all_crops:
        return {"slot": slot, "total": len(actors), "vlm_described": 0}

    # Query VLM in parallel
    vlm_descriptions = {}
    failed = 0

    def _vlm_worker(eid_crops):
        eid, crops = eid_crops
        segformer_desc = actors[eid].get("description", "")
        desc = describe_entity_vlm(crops, api_url, model, segformer_desc)
        return eid, desc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_vlm_worker, (eid, crops)): eid
            for eid, crops in all_crops.items()
        }
        for future in as_completed(futures):
            eid = futures[future]
            try:
                eid_result, desc = future.result()
                if desc:
                    vlm_descriptions[eid_result] = desc
                else:
                    failed += 1
            except Exception:
                failed += 1

    if verbose:
        print(f"  VLM: {len(vlm_descriptions)} described, {failed} failed")
        # Show a few examples
        for eid, desc in list(vlm_descriptions.items())[:3]:
            old = actors[eid].get("description", "")
            print(f"    {eid}:")
            print(f"      SegFormer: {old}")
            print(f"      VLM:      {desc}")

    # Save VLM descriptions alongside SegFormer data
    VLM_DESC_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "slot": slot,
        "model": model,
        "total_actors": len(actors),
        "vlm_described": len(vlm_descriptions),
        "vlm_failed": failed,
        "processing_time_sec": round(time.time() - t0, 1),
        "descriptions": vlm_descriptions,
    }

    output_path = VLM_DESC_DIR / f"{slot}.vlm.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    if verbose:
        print(f"  Saved → {output_path.name} ({time.time() - t0:.1f}s)")

    return {
        "slot": slot,
        "total": len(actors),
        "vlm_described": len(vlm_descriptions),
        "vlm_failed": failed,
        "time_sec": round(time.time() - t0, 1),
    }


# ============================================================================
# Batch Processing
# ============================================================================

def process_batch(
    api_url: str,
    model: str = VLM_MODEL,
    verbose: bool = False,
    force: bool = False,
    max_workers: int = MAX_WORKERS,
) -> Dict:
    """Process all slots that have SegFormer descriptions."""
    existing = sorted(ENTITY_DESC_DIR.glob("*.json"))
    slots = [f.stem for f in existing if not f.stem.startswith(".")]

    if verbose:
        print(f"Found {len(slots)} slots with SegFormer descriptions")

    results = []
    total_described = 0
    total_failed = 0

    for i, slot in enumerate(slots, 1):
        # Skip if VLM already done (unless force)
        vlm_path = VLM_DESC_DIR / f"{slot}.vlm.json"
        if vlm_path.exists() and not force:
            if verbose:
                print(f"[{i:3d}/{len(slots)}] {slot}: already done, skipping")
            continue

        if verbose:
            print(f"\n[{i:3d}/{len(slots)}] {slot}")

        result = process_slot_vlm(
            slot, api_url, model,
            verbose=verbose,
            max_workers=max_workers,
        )
        results.append(result)
        total_described += result.get("vlm_described", 0)
        total_failed += result.get("vlm_failed", 0)

    return {
        "slots_processed": len(results),
        "total_described": total_described,
        "total_failed": total_failed,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enrich MEVA entity descriptions with VLM captioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slot", type=str, help="Process a single slot")
    parser.add_argument("--batch", action="store_true", help="Process all slots")
    parser.add_argument("--api-url", type=str, default=None,
        help="vLLM server URL (default: auto-start on GPU 1)")
    parser.add_argument("--gpu", type=str, default=DEFAULT_GPU,
        help=f"GPU to use for vLLM server (default: {DEFAULT_GPU})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
        help=f"Port for vLLM server (default: {DEFAULT_PORT})")
    parser.add_argument("--model", type=str, default=VLM_MODEL,
        help=f"VLM model name (default: {VLM_MODEL})")
    parser.add_argument("--force", action="store_true",
        help="Force re-description even if VLM output exists")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
        help=f"Max parallel VLM requests (default: {MAX_WORKERS})")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.slot and not args.batch:
        parser.error("Specify --slot or --batch")

    # Start or connect to vLLM server
    api_url = args.api_url
    we_started_server = False
    if api_url is None:
        api_url = start_vllm_server(
            gpu=args.gpu, port=args.port,
            model=args.model, verbose=args.verbose
        )
        we_started_server = True

    try:
        if args.slot:
            result = process_slot_vlm(
                args.slot, api_url, args.model,
                verbose=args.verbose,
                max_workers=args.max_workers,
            )
            print(f"\nResult: {json.dumps(result, indent=2)}")

        elif args.batch:
            result = process_batch(
                api_url, args.model,
                verbose=args.verbose,
                force=args.force,
                max_workers=args.max_workers,
            )
            print(f"\nBatch Result: {json.dumps(result, indent=2)}")

    finally:
        if we_started_server:
            if args.verbose:
                print("Stopping vLLM server...")
            stop_vllm_server()


if __name__ == "__main__":
    main()

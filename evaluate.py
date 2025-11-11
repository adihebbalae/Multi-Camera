from orbit.target_identification.target_identification import *
from orbit.nsvs.model_checker.frame_validator import *
from orbit.nsvs.video.read_video import *
from orbit.datamanager.egoexo4d import *
from orbit.nsvs.vlm.obj import *
from orbit.nsvs.nsvs import *
from orbit.puls.puls import *

import json
import os
import re

def exec_puls(entry): # Step 1
    output = PULS(entry["question"])

    entry["puls"] = {}
    entry["puls"]["proposition"] = output["proposition"]
    entry["puls"]["specification"] = output["specification"]
    entry["puls"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_target_identification(entry): # Step 2
    output = identify_target(
        entry["question"],
        entry["candidates"],
        entry["puls"]["specification"],
        entry["puls"]["conversation_history"]
    )

    entry["target_identification"] = {}
    entry["target_identification"]["frame_window"] = output["frame_window"]
    entry["target_identification"]["explanation"] = output["explanation"]
    entry["target_identification"]["conversation_history"] = os.path.join(os.getcwd(), output["saved_path"])

def exec_nsvs(entry, sample_rate, device, model_name): # Step 3
    multi_video_data = []
    for video_path in entry["video_paths"]:
        reader = Mp4Reader(path=video_path, sample_rate=sample_rate)
        multi_video_data.append(reader.read_video())

    fps = set([video_data["video_info"]["fps"] for video_data in multi_video_data])
    frame_count = set([video_data["video_info"]["frame_count"] for video_data in multi_video_data])
    num_images = set([len(video_data["images"]) for video_data in multi_video_data])

    entry["metadata"] = {}
    try:
        if len(fps) == 1 and len(frame_count) == 1 and len(num_images) == 1:
            entry["metadata"]["fps"] = fps.pop()
            entry["metadata"]["frame_count"] = frame_count.pop()
        else:
            errors = [
                f"Different FPS values found: {fps}" if len(fps) != 1 else None,
                f"Different frame counts found: {frame_count}" if len(frame_count) != 1 else None,
                f"Different number of images found: {num_images}" if len(num_images) != 1 else None
            ]
            raise ValueError(" ; ".join(filter(None, errors)))

        output, indices = run_nsvs(
            multi_video_data,
            entry["video_paths"],
            entry["puls"]["proposition"],
            entry["puls"]["specification"],
            device=device,
            model_name=model_name,
        )
    except Exception as e:
        entry["metadata"]["error"] = repr(e)
        print(repr(e))
        output = {-1: {}}
        indices = []
    
    entry["nsvs"] = {}
    entry["nsvs"]["output"] = output
    entry["nsvs"]["indices"] = [list(idx) for idx in indices]

def exec_merge(entry): # Step 4
    inner = entry["target_identification"]["frame_window"].strip()[1:-1]
    parts = inner.split(',')
    result = []
    for part in parts:
        part = part.strip()
        match = re.search(r'([+-])\s*(\d+)', part)
        if match:
            sign, num = match.groups()
            result.append(int(sign + num))
        else:
            result.append(0)

    if entry["nsvs"]["output"] != {-1: {}}:
        min_frame_nsvs = min(entry["nsvs"]["output"].keys())
        max_frame_nsvs = max(entry["nsvs"]["output"].keys())

        start_offset_frames = result[0] * entry["metadata"]["fps"]
        end_offset_frames = result[1] * entry["metadata"]["fps"]
        all_camera_ids = [f"cam{i}" for i in range(len(entry["video_paths"]))]

        frames_of_interest = {}
        if start_offset_frames < 0:
            start_ext = max(0, int(min_frame_nsvs + start_offset_frames))
            for frame_num in range(start_ext, min_frame_nsvs):
                frames_of_interest[frame_num] = all_camera_ids
        for k, v in entry["nsvs"]["output"].items():
            frames_of_interest[k] = v
        if end_offset_frames > 0:
            end_ext = min(entry["metadata"]["frame_count"] - 1, int(max_frame_nsvs + end_offset_frames))
            for frame_num in range(max_frame_nsvs + 1, end_ext + 1):
                frames_of_interest[frame_num] = all_camera_ids
        
        entry["frames_of_interest"] = frames_of_interest
    else:
        entry["frames_of_interest"] = {-1: {}}

def run_orbit(output_dir, device_number, current_split, total_splits):
    loader = EgoExo4D()
    data = loader.load_data()
    
    output = []

    starting = (len(data) * (current_split-1)) // total_splits
    ending = (len(data) * current_split) // total_splits
    for i in range(starting, ending):
        print("\n" + "*"*50 + f" {i}/{len(data)-1} " + "*"*50)
        entry = data[i]
        exec_puls(entry)
        exec_target_identification(entry)
        exec_nsvs(entry, sample_rate=1, device=device_number, model_name="OpenGVLab/InternVL3_5-14B")
        exec_merge(entry)
        output.append(entry)

    with open(output_dir, "w") as f:
        json.dump(output, f, indent=4)

def postprocess(output_dir):
    loader = EgoExo4D()
    loader.postprocess_data(output_dir)

def main():
    # current_split = 3
    # total_splits = 3
    # device_number = current_split
    # output_dir = f"/nas/mars/experiment_result/orbit/2_full_output/ego_exo4d_{current_split}.json"
    # run_orbit(output_dir, device_number, current_split, total_splits)

    orbit_dir = f"/nas/mars/experiment_result/orbit/2_full_output/ego_exo4d.json"
    postprocess(orbit_dir)

if __name__ == "__main__":
    main()


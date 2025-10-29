import glob
import json
import os
import re

DATASET_PATH = "/nas/mars/dataset/Ego-Exo4D"
NUM_PER_CATEGORY = 6

def get_unique_video_types():
    videos_path = os.path.join(DATASET_PATH, "takes")
    all_entries = os.listdir(videos_path)
    dir_names = [d for d in all_entries if os.path.isdir(os.path.join(videos_path, d))]
    category_names = [re.split(r'[_]?\d+', d)[0] for d in dir_names]
    unique_categories = sorted(list(set(category_names)))
    return unique_categories

def read_narrations_map():
    annotations = {}
    file_names = ["atomic_descriptions_train.json", "atomic_descriptions_val.json"]
    for file_name in file_names:
        with open(os.path.join(DATASET_PATH, "annotations", file_name), "r") as f:
            annotations.update(json.load(f)["annotations"])
    return annotations

def task_uid_map():
    with open(os.path.join(DATASET_PATH, "takes.json"), "r") as f:
        takes = json.load(f)
    name_to_uid = {}
    for take in takes:
        name_to_uid[take["take_name"]] = take["take_uid"]
    return name_to_uid

def get_uids_with_atomic_descriptions():
    with open(os.path.join(DATASET_PATH, "annotations", "splits.json"), "r") as f:
        splits = json.load(f)["take_uid_to_benchmark"]
    ret = set()
    for k, v in splits.items():
        if "atomic_action_descriptions" in v:
            ret.add(k)
    return ret

if __name__ == "__main__":
    unique_types = get_unique_video_types()
    all_videos = sorted([d for d in os.listdir(os.path.join(DATASET_PATH, "takes"))])

    narrations_map = read_narrations_map()

    name_to_uid = task_uid_map()

    atomic_description_uids = get_uids_with_atomic_descriptions()

    main_dict = {}
    # 1. iterate through video types
    for video_type in unique_types:
        videos_in_category = [v for v in all_videos if v.startswith(video_type)]
        total_added = 0
        # 2. iterate through all videos in that type
        for video in videos_in_category:
            # 3. convert video name to uid
            uid = name_to_uid[video]
            # 4. check if uid has atomic descriptions
            if uid in atomic_description_uids:
                # 5. add to main dict if one caption exists
                if uid in narrations_map and len(narrations_map[uid]) == 1:
                    total_added += 1
                    mp4_files = glob.glob(os.path.join(DATASET_PATH, "takes", video, "frame_aligned_videos", "downscaled", "448", "cam*.mp4"))
                    narrations = narrations_map[uid][0]["descriptions"]
                    modified_narrations = []
                    for narration in narrations:
                        if "subject" not in narration or narration["subject"] == None:
                            continue
                        modified_text = re.sub(r'\b' + re.escape(narration["subject"]) + r'\b', 'Subject', narration["text"])
                        modified_narrations.append((modified_text, narration["timestamp"]))
                    main_dict[video] = {
                        "video_paths": sorted(mp4_files),
                        "narrations": modified_narrations
                    }
            # 6. break once amount per category criteria reached
            if total_added >= NUM_PER_CATEGORY:
                break

    print(json.dumps(main_dict, indent=4))

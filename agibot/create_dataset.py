import json
import glob
import os
from tqdm import tqdm


DATASET_PATH = "/nas/mars/dataset/AgiBotWorld"

def read_dataset():
    task_info_dir = os.path.join(DATASET_PATH, "task_info")
    json_files = glob.glob(os.path.join(task_info_dir, "*.json"))
    task_numbers = []
    for filepath in json_files:
        filename = os.path.basename(filepath)
        task_num = filename.replace("task_", "").replace(".json", "")
        task_numbers.append(int(task_num))

    data = []
    for task_num in tqdm(sorted(task_numbers), desc="Tasks"):
        filepath = os.path.join(task_info_dir, f"task_{task_num}.json")
        with open(filepath, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        episodes_array = []
        for episode in tqdm(task_data, desc=f"  Task {task_num} episodes", leave=False):
            videos_dir = os.path.join(DATASET_PATH, "observations", str(task_num), str(episode["episode_id"]), "videos")
            video_paths = glob.glob(os.path.join(videos_dir, "*"))

            if video_paths:
                episodes_array.append(
                    {
                        "id": episode["episode_id"],
                        "annotations": episode["label_info"]["action_config"],
                        "paths": video_paths,
                    }
                )

        if episodes_array:
            data.append({
                "task": task_num,
                "task_name": task_data[0]["task_name"],
                "scene_descriptions": task_data[0]["init_scene_text"],
                "episodes": episodes_array,
            })

    return data

def main():
    data = read_dataset()
    output_path = "compiled.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()


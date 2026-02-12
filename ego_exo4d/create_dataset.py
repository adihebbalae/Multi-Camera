import json
import glob
import os


DATASET_PATH = "/nas/mars/dataset/Ego-Exo4D"

def read_dataset():
    with open(os.path.join(DATASET_PATH, "takes.json"), "r") as f:
        takes = json.load(f)
    data = {}
    with open(os.path.join(DATASET_PATH, "annotations", "splits.json"), "r") as f:
        splits_data = json.load(f)
    with open(os.path.join(DATASET_PATH, "metadata.json"), "r") as f:
        metadata_data = json.load(f)

    with open(os.path.join(DATASET_PATH, "annotations", "keystep_train.json"), "r") as f:
        keystep_train = json.load(f)
    with open(os.path.join(DATASET_PATH, "annotations", "keystep_val.json"), "r") as f:
        keystep_val = json.load(f)
    keystep_annotations_data = keystep_train["annotations"] | keystep_val["annotations"]
    keystep_taxonomy_data = keystep_train["taxonomy"] | keystep_val["taxonomy"]

    with open(os.path.join(DATASET_PATH, "annotations", "atomic_descriptions_train.json"), "r") as f:
        atomic_descriptions_train = json.load(f)
    with open(os.path.join(DATASET_PATH, "annotations", "atomic_descriptions_val.json"), "r") as f:
        atomic_descriptions_val = json.load(f)
    atomic_descriptions_data = atomic_descriptions_train["annotations"] | atomic_descriptions_val["annotations"]

    def get_parent_hierarchy(node_id, taxonomy_for_scenario):
        node_info = taxonomy_for_scenario.get(str(node_id))
        if not node_info:
            return None
        current_node = {
            "node_id": node_info["id"],
            "node_name": node_info["name"],
        }
        if node_info["parent_id"] is not None:
            current_node["parent"] = get_parent_hierarchy(node_info["parent_id"], taxonomy_for_scenario)
        else:
            current_node["parent"] = None
        return current_node

    for take in takes:
        if take["validated"] and take["is_narrated"]:
            # organize keystep
            keystep_take = keystep_annotations_data.get(take["take_uid"], {})
            keystep_array = []
            if keystep_take:
                segments = keystep_take["segments"]
                keystep_array = []
                for s in segments:
                    if str(s["step_id"]) in keystep_taxonomy_data[keystep_take["scenario"]]:
                        keystep_array.append(
                            {
                                "start_time": s["start_time"],
                                "end_time": s["end_time"],
                                "category": keystep_take["scenario"],
                                "is_essential": s["is_essential"],
                                "description": s["step_description"],
                                "node": get_parent_hierarchy(s["step_id"], keystep_taxonomy_data[keystep_take["scenario"]]),
                            }
                        )


            # organize atomic descriptions
            atomic_descriptions_take = atomic_descriptions_data.get(take["take_uid"], [])
            atomic_descriptions_array = []
            if atomic_descriptions_take:
                for video_set in atomic_descriptions_take:
                    to_add = []
                    descriptions_list = video_set.get("descriptions", [])
                    for description in descriptions_list:
                        if not description["unsure"]:
                            to_add.append({
                                "timestamp": description["timestamp"],
                                "text": description["text"],
                                "subject": description["subject"],
                                "best_camera": (description.get("best_exo") or {}).get("cam_id")
                            })
                    atomic_descriptions_array.append(to_add)

            data[take["take_name"]] = {
                "take_name": take["take_name"],
                "take_uid": take["take_uid"],
                "task_id": metadata_data["tasks"][str(take["task_id"])],
                "best_camera": take["best_exo"],
                "video_files": glob.glob(os.path.join(DATASET_PATH, "takes", take["take_name"], "frame_aligned_videos", "downscaled", "448", "cam*.mp4"))
                + glob.glob(os.path.join(DATASET_PATH, "takes", take["take_name"], "frame_aligned_videos", "downscaled", "448", "gp*.mp4")),
                "benchmarks": splits_data["take_uid_to_benchmark"].get(take["take_uid"], []),
                "objects": [(obj["name"], obj["object_uid"]) for obj in take["objects"]],
                "annotations": atomic_descriptions_array,
                "keystep_annotations": keystep_array

            }
    return data


def main():
    data = read_dataset()
    print(json.dumps(data, indent=4))

    # read_relations()

def read_relations():
    with open(os.path.join(DATASET_PATH, "annotations", "relations_train.json"), "r") as f:
        relations_data = json.load(f)

    print(json.dumps(relations_data["annotations"][list(relations_data["annotations"].keys())[0]], indent=4))

if __name__ == "__main__":
    main()

'''
X = used
Y = not needed

[ ] captures.json
[X] metadata.json
[Y] participants.json -> info on each participant
[Y] physical_setting.json -> info on where it was recorded
[X] takes.json
[X] visual_objects.json
[X] annotations/atomic_descriptions.json
[ ] annotations/expert_commentary.json
[X] annotations/keystep.json
[Y] annotations/procedure_understanding.json -> repeat of keystep
[Y] annotations/proficiency_demonstration.json -> skill level
[Y] annotations/proficiency_demonstrator.json -> skill level
[ ] annotations/relations.json
[X] annotations/splits.json
[ ] takes/*/*.mp4 -> need ego
'''

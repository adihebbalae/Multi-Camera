from orbit.datamanager.manager import Manager

from collections import defaultdict
from tqdm import tqdm
import hashlib
import shutil
import json
import copy
import os


class EgoExo4D(Manager):
    def __init__(self):
        self.compile_position = False
        self.compile_full = True

        self._dataset_path = "/nas/mars/dataset/Ego-Exo4D"
        self._question_path = "/nas/mars/experiment_result/orbit/1_dataset_json/ego_exo4d_dataset.json"
        self._cropped_output_video_path = "/nas/mars/experiment_result/orbit/3_cropped_videos/Ego-Exo4D"
        
    def load_data(self):
        with open(self._question_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        ret = []
        for key in dataset:
            ret.append({
                "question": dataset[key]["question"],
                "candidates": dataset[key]["candidates"],
                "correct_answer": dataset[key]["correct_answer"],
                "video_paths": dataset[key]["video_paths"],
                "video_id": key
            })
        return ret


    def postprocess_data(self, output_dir):
        with open(output_dir, "r") as f:
            data = json.load(f)
        os.makedirs(self._cropped_output_video_path, exist_ok=True)

        for entry in tqdm(data, desc="Processing videos"):
            save_path = os.path.join(self._cropped_output_video_path, f"{entry['video_id']}.mp4")
            self.crop_video(entry, save_path)



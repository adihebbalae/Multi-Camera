from abc import ABC, abstractmethod
import subprocess
import json
import cv2
import numpy as np
import os

class Manager(ABC):
    @abstractmethod
    def load_data(self) -> list:
        pass
    
    @abstractmethod
    def postprocess_data(self, output_dir):
        pass

    def crop_video(self, entry, save_path):
        if entry.get("nsvs", {}).get("output") == [-1] or len(entry["video_paths"]) == 0:
            return

        caps = {}
        video_paths = {}
        for path in entry["video_paths"]:
            cam_name = os.path.basename(path).split('.')[0]
            video_paths[cam_name] = path
            caps[cam_name] = cv2.VideoCapture(path)

        first_cap = list(caps.values())[0]
        width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = first_cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        sorted_frame_nums = sorted([int(f) for f in entry["frames_of_interest"].keys()])

        for frame_num in sorted_frame_nums:
            cams_for_frame = entry["frames_of_interest"][str(frame_num)]
            
            frames_to_stitch = []
            labels = []

            for cam_name in sorted(cams_for_frame):
                if cam_name in caps:
                    cap = caps[cam_name]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ok, frame = cap.read()
                    if ok:
                        frames_to_stitch.append(frame)
                        labels.append(cam_name)

            if not frames_to_stitch:
                continue
            
            num_frames_to_stitch = len(frames_to_stitch)

            if num_frames_to_stitch == 1:
                rows, cols = 1, 1
            elif num_frames_to_stitch == 2:
                rows, cols = 2, 1
            else:
                rows = 2
                cols = (num_frames_to_stitch + 1) // 2

            new_width = width // cols
            new_height = height // rows

            resized_frames = []
            for frame in frames_to_stitch:
                resized_frames.append(cv2.resize(frame, (new_width, new_height)))

            labeled_frames = []
            for frame, label in zip(resized_frames, labels):
                labeled_frame = frame.copy()
                cv2.putText(labeled_frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                labeled_frames.append(labeled_frame)

            num_missing = rows * cols - num_frames_to_stitch
            for _ in range(num_missing):
                labeled_frames.append(np.zeros((new_height, new_width, 3), dtype=np.uint8))

            grid_rows = []
            for i in range(rows):
                start_index = i * cols
                end_index = start_index + cols
                grid_rows.append(cv2.hconcat(labeled_frames[start_index:end_index]))
            
            stitched_frame = cv2.vconcat(grid_rows)

            stitched_h, stitched_w, _ = stitched_frame.shape
            if stitched_h != height or stitched_w != width:
                 stitched_frame = cv2.resize(stitched_frame, (width, height))

            writer.write(stitched_frame)

        for cap in caps.values():
            cap.release()
        writer.release()



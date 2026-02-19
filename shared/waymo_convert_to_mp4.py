import pandas as pd
import sys
import glob
import os
import random
import cv2
from PIL import Image
import io
import numpy as np
from tqdm import tqdm

CAMERA_NAME_MAP = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

def get_scene_ids_and_counts():
    """
    Gets all unique scene IDs from camera_image subdirectories and their counts per folder.
    """
    scene_ids = set()
    folder_counts = {}
    parent_dirs = ["training", "testing", "testing_location", "validation"]

    for parent_dir in parent_dirs:
        file_pattern = f"/nas/standard_datasets/waymo/{parent_dir}/camera_image/*.parquet"
        file_paths = glob.glob(file_pattern)
        
        folder_scene_ids = set()
        for path in file_paths:
            filename = os.path.basename(path)
            if '_' in filename:
                try:
                    scene_id = filename.split('_')[0]
                    if scene_id.isdigit():
                        scene_ids.add(scene_id)
                        folder_scene_ids.add(scene_id)
                except IndexError:
                    pass # ignore files that don't match the pattern
        
        if folder_scene_ids:
            folder_counts[parent_dir] = len(folder_scene_ids)

    return sorted(list(scene_ids)), folder_counts

def save_scene_as_mp4(scene_id):
    """
    For a given scene, this function finds all corresponding parquet files
    in camera_image directories, reads them, and saves each camera's video as an MP4 file.
    """
    parent_dirs = ["training", "testing", "testing_location", "validation"]
    found_files = False
    
    VIDEO_DIR = "/nas/standard_datasets/waymo/created_videos/"
    OUTPUT_DIR = os.path.join(VIDEO_DIR, scene_id)
    FPS = 10

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for parent_dir in parent_dirs:
        file_pattern = f"/nas/standard_datasets/waymo/{parent_dir}/camera_image/{scene_id}*.parquet"
        file_paths = glob.glob(file_pattern)

        if file_paths:
            found_files = True
            # print(f"\nCreating videos for scene ID: {scene_id} from {parent_dir}/camera_image")
            
            camera_frames = {name: [] for name in CAMERA_NAME_MAP.values()}

            for file_path in file_paths:
                try:
                    df = pd.read_parquet(file_path)
                    for _, row in df.iterrows():
                        camera_name_int = row["key.camera_name"]
                        if camera_name_int in CAMERA_NAME_MAP:
                            camera_name = CAMERA_NAME_MAP[camera_name_int]
                            timestamp = row["key.frame_timestamp_micros"]
                            image_data = row["[CameraImageComponent].image"]

                            try:
                                image = Image.open(io.BytesIO(image_data))
                                frame = np.array(image)
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                camera_frames[camera_name].append((timestamp, frame))
                            except Exception as e:
                                pass # print(f"  - Error decoding image: {e}")
                except Exception as e:
                    pass # print(f"  - Error processing file: {e}")

            for camera_name, frames in camera_frames.items():
                if not frames:
                    continue

                frames.sort(key=lambda x: x[0])
                height, width, _ = frames[0][1].shape
                
                output_path = os.path.join(OUTPUT_DIR, f"{camera_name.lower()}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

                for _, frame in frames:
                    video_writer.write(frame)

                video_writer.release()
                # print(f"  - Saved video to: {output_path}")

            break # Found and processed, so break

    if not found_files:
        pass # print(f"No camera_image files found for scene ID: {scene_id}")

if __name__ == "__main__":
    try:
        import pandas
        import cv2
        from PIL import Image
        from tqdm import tqdm
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install the required libraries by running:")
        print("pip install pandas pyarrow opencv-python-headless Pillow tqdm")
    else:
        scene_ids, folder_counts = get_scene_ids_and_counts()
        if not scene_ids:
            print("No scene IDs found.")
        else:
            print(f"Found {len(scene_ids)} possible scene IDs across the following folders:")
            for folder, count in folder_counts.items():
                print(f"- {folder}: {count} scenes")
            print()
            
            for scene_id in tqdm(scene_ids, desc="Processing scenes"):
                save_scene_as_mp4(scene_id)

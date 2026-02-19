from httpx import get
from openai import OpenAI
import numpy as np
import base64
import cv2
import json
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor


NUM_SAMPLES = 48
NUM_WORKERS = 4

class VLLMClient:
    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8002/v1",
        model="OpenGVLab/InternVL3_5-14B",
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def _encode_frame(self, frame):
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def multiple_choice(self, frames_by_cam: dict, question: str, candidates: list[str]) -> str:
        user_content = []
        if len(frames_by_cam) == 1:
            user_content.append(
                {
                    "type": "text",
                    "text": "The following is the sequence of images",
                }
            )
            frames = list(frames_by_cam.values())[0]
            encoded_images = [self._encode_frame(frame) for frame in frames]
            for encoded in encoded_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
        else:
            user_content.append(
                {
                    "type": "text",
                    "text": "The following is the sequence of images from multiple cameras",
                }
            )
            for cam_name, frames in frames_by_cam.items():
                user_content.append({"type": "text", "text": f"Camera {cam_name}:"})
                encoded_images = [self._encode_frame(frame) for frame in frames]
                for encoded in encoded_images:
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        }
                    )

        parsing_rule = "You must only return the letter of the answer choice, and nothing else. Do not include any other symbols, information, text, or justification in your answer. For example, if the correct answer is 'a) ...', you must only return 'a'."
        prompt = f"{question}\n"
        for candidate in candidates:
            prompt += f"{candidate}\n"
        prompt += f"\n[PARSING RULE]: {parsing_rule}"
        user_content.append({"type": "text", "text": prompt})

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            max_tokens=1,
            temperature=0.0,
        )
        
        return chat_response.choices[0].message.content.lower().strip()

class VLLMClientMultiprocessing(VLLMClient):
    def __init__(
        self,
        max_workers=NUM_WORKERS,
    ):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def multiple_choice_batch(self, batch_args):
        futures = [
            self.executor.submit(self.multiple_choice, *args) for args in batch_args
        ]
        
        results = []
        for future in tqdm.tqdm(futures, desc="Processing batch"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing a task: {e}")
                results.append(None)
                
        return results

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return frame_count

def load_video_frames_exp1(video_paths, num_frames):
    num_videos = len(video_paths)
    frame_count = get_video_frame_count(video_paths[0])
    total_frames = frame_count * num_videos
    if total_frames < num_frames:
        frame_indices_global = np.arange(total_frames)
    else:
        frame_indices_global = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames_to_load = {i: {} for i in range(num_videos)}
    for g_idx in frame_indices_global:
        video_idx = g_idx % num_videos
        frame_idx = g_idx // num_videos
        frames_to_load[video_idx][frame_idx] = g_idx

    temp_images = {}
    for video_idx, local_frames in frames_to_load.items():
        if not local_frames:
            continue
        
        cap = cv2.VideoCapture(video_paths[video_idx])
        for local_idx in sorted(local_frames.keys()):
            g_idx = local_frames[local_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                temp_images[g_idx] = frame_rgb
        cap.release()

    sorted_g_indices = sorted(temp_images.keys())
    images = [temp_images[i] for i in sorted_g_indices]
    
    return images

def load_video_frames_exp2(video_paths, num_frames):
    frame_count = get_video_frame_count(video_paths[0])
    if frame_count < num_frames:
        frame_indices = np.arange(frame_count)
    else:
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames_by_cam = {}
    for video_path in video_paths:
        cam_name = os.path.basename(video_path).split(".")[0]
        frames_by_cam[cam_name] = []
        cap = cv2.VideoCapture(video_path)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_by_cam[cam_name].append(frame_rgb)
        cap.release()
    
    return frames_by_cam

def stitch_frames(frames_to_stitch, labels):
    labeled_frames = []
    for frame, label in zip(frames_to_stitch, labels):
        labeled_frame = frame.copy()
        cv2.putText(labeled_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        labeled_frames.append(labeled_frame)

    stitched_frame = cv2.hconcat(labeled_frames)
    return stitched_frame

def load_video_frames_exp3(video_paths, num_frames):
    frame_count = get_video_frame_count(video_paths[0])
    if frame_count < num_frames:
        frame_indices = np.arange(frame_count)
    else:
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    stitched_images = []
    caps = [cv2.VideoCapture(p) for p in video_paths]
    for idx in frame_indices:
        frames_to_stitch = []
        labels = []
        for i, cap in enumerate(caps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_to_stitch.append(frame_rgb)
                labels.append(os.path.basename(video_paths[i]).split(".")[0])
        
        if len(frames_to_stitch) == len(video_paths):
            try:
                stitched_frame = stitch_frames(frames_to_stitch, labels)
                stitched_images.append(stitched_frame)
            except cv2.error as e:
                for cap in caps:
                    cap.release()
                return []

    for cap in caps:
        cap.release()

    return stitched_images

def load_video_frames_exp4(video_paths, num_frames):
    aria_video_path = None
    for path in video_paths:
        if "aria" in os.path.basename(path).lower():
            aria_video_path = path
            break
    if aria_video_path is None:
        return []

    frame_count = get_video_frame_count(aria_video_path)
    if frame_count < num_frames:
        frame_indices = np.arange(frame_count)
    else:
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    images = []
    cap = cv2.VideoCapture(aria_video_path)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            images.append(frame_rgb)
    cap.release()

    return images

def run_experiment(exp_num: int, dataset, vllm_client):
    print("*"*50 + f" Experiment {exp_num} " + "*"*50)

    results = {}
    all_entries = {}
    batch_args_all_calls = []

    for key in tqdm.tqdm(list(dataset.keys()), desc=f"Preparing"):
        entry = dataset[key]

        if not entry["video_paths"] or len(entry["video_paths"]) == 0:
            continue
        frame_counts = [get_video_frame_count(p) for p in entry["video_paths"]]
        if not frame_counts or not all(fc == frame_counts[0] for fc in frame_counts) or frame_counts[0] == 0:
            continue

        if exp_num == 1:
            frames = load_video_frames_exp1(entry["video_paths"], num_frames=NUM_SAMPLES)
        elif exp_num == 2:
            frames = load_video_frames_exp2(entry["video_paths"], num_frames=int(NUM_SAMPLES/len(entry["video_paths"])))
        elif exp_num == 3:
            frames = load_video_frames_exp3(entry["video_paths"], num_frames=int(NUM_SAMPLES/len(entry["video_paths"])))
        elif exp_num == 4:
            frames = load_video_frames_exp4(entry["video_paths"], num_frames=NUM_SAMPLES)
        else:
            return
        if not frames:
            print(f"Could not load frames from {entry['video_paths']}")
            continue
        
        question = entry["question"]
        candidates = entry["candidates"]
        if exp_num == 2:
            batch_args_all_calls.append((frames, question, candidates))
        else:
            batch_args_all_calls.append(({"main": frames}, question, candidates))
        all_entries[key] = entry

    predicted_answers_all_calls = vllm_client.multiple_choice_batch(batch_args_all_calls)
    total_correct = 0
    for i, key in enumerate(all_entries):
        entry = all_entries[key]
        predicted_answer = predicted_answers_all_calls[i]
        correct_answer = entry["correct_answer"]
        is_correct = 1 if predicted_answer == correct_answer else 0
        total_correct += is_correct
        results[key] = {
            "question": entry["question"],
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        }

    with open(f"exp{exp_num}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    accuracy = total_correct / len(all_entries) if all_entries else 0
    print(f"Accuracy: {accuracy:.2%}")

def main():
    dataset_path = "/nas/mars/experiment_result/orbit/1_dataset_json/ego_exo4d_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    vllm_client = VLLMClientMultiprocessing()

    # Experiment 1: uniformly sampling 12 frames from an interleaved timeline of all available camera views
    # Experiment 2: uniformly sampling 12 frames from each camera view separately and providing all camera views to the model
    # Experiment 3: uniformly sampling 12 frames from each camera view separately, stitching the frames from different views side-by-side for each timestamp, and providing the stitched frames to the model
    # Experiment 4: uniformly sampling 12 frames from only the ARIA camera view

    experiments = [1, 2, 3, 4]
    for n in experiments:
        run_experiment(
            exp_num=n,
            dataset=dataset,
            vllm_client=vllm_client
        )

if __name__ == "__main__":
    main()

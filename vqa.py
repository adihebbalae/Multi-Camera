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

def load_video_frames(video_path, num_frames):
    frame_count = get_video_frame_count(video_path)
    if frame_count < num_frames:
        frame_indices = np.arange(frame_count)
    else:
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    images = []
    cap = cv2.VideoCapture(video_path)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            images.append(frame_rgb)
    cap.release()
    return images

def run_experiment(dataset, vllm_client):
    results = {}
    all_entries = {}
    batch_args_all_calls = []

    for key in tqdm.tqdm(list(dataset.keys()), desc=f"Preparing"):
        entry = dataset[key]
        video_path = os.path.join("/nas/mars/experiment_result/orbit/3_cropped_videos/Ego-Exo4D", f"{key}.mp4")
        if not os.path.exists(video_path):
            continue
        
        frames = load_video_frames(video_path, num_frames=NUM_SAMPLES)
        if not frames:
            # print(f"Could not load frames from {video_path}")
            continue
        
        question = entry["question"]
        candidates = entry["candidates"]
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

    with open(f"/nas/mars/experiment_result/orbit/4_vqa_results/ego_exo4d_results.json", "w") as f:
        json.dump(results, f, indent=4)

    accuracy = total_correct / len(all_entries) if all_entries else 0
    print(f"Accuracy: {accuracy:.2%}")

def main():
    dataset_path = "/nas/mars/experiment_result/orbit/1_dataset_json/ego_exo4d_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    vllm_client = VLLMClientMultiprocessing()
    run_experiment(
        dataset=dataset,
        vllm_client=vllm_client
    )

if __name__ == "__main__":
    main()

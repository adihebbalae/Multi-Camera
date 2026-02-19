from openai import OpenAI
import numpy as np
import base64
import cv2
import json
import os
import tqdm
import random
import math
from rouge_score import rouge_scorer

NUM_SAMPLES = 16
CATEGORY_QUESTIONS = 250

class VLLMClient:
    def __init__(
        self,
        api_key="EMPTY",
        api_base="http://localhost:8006/v1",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        # model="OpenGVLab/InternVL2-8B",
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        print(f"Using model: {self.model}")

    def _encode_frame(self, frame):
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def multiple_choice(self, frames_data, question: str, candidates: list[str], strategy_type: str = "uniform") -> str:
        """
        Generate multiple choice answer from frames.

        Args:
            frames_data: Either Dict[str, List] for uniform strategy or List for stitched strategy
            question: The question to answer
            candidates: List of answer choices
            strategy_type: "uniform" or "stitched"

        Returns:
            Predicted answer letter
        """
        user_content = []

        if strategy_type == "stitched":
            # frames_data is a list of stitched grid images
            user_content.append(
                {
                    "type": "text",
                    "text": "The following is a sequence of multi-camera grid images",
                }
            )
            encoded_images = [self._encode_frame(frame) for frame in frames_data]
            for encoded in encoded_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            pass
        else:
            # strategy_type == "uniform": frames_data is dict of frames by camera
            frames_by_cam = frames_data
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

    def summarize(self, frames_data, question: str, strategy_type: str = "uniform") -> str:
        """
        Generate a free-form text answer (for summarization tasks).

        Args:
            frames_data: Either Dict[str, List] for uniform strategy or List for stitched strategy
            question: The summarization question
            strategy_type: "uniform" or "stitched"

        Returns:
            Generated summary text
        """
        user_content = []

        if strategy_type == "stitched":
            user_content.append({"type": "text", "text": "The following is a sequence of multi-camera grid images"})
            encoded_images = [self._encode_frame(frame) for frame in frames_data]
            for encoded in encoded_images:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        else:
            frames_by_cam = frames_data
            if len(frames_by_cam) == 1:
                user_content.append({"type": "text", "text": "The following is the sequence of images"})
                frames = list(frames_by_cam.values())[0]
                encoded_images = [self._encode_frame(frame) for frame in frames]
                for encoded in encoded_images:
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
            else:
                user_content.append({"type": "text", "text": "The following is the sequence of images from multiple cameras"})
                for cam_name, frames in frames_by_cam.items():
                    user_content.append({"type": "text", "text": f"Camera {cam_name}:"})
                    encoded_images = [self._encode_frame(frame) for frame in frames]
                    for encoded in encoded_images:
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})

        user_content.append({"type": "text", "text": question})

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1024,
            temperature=0.7,
        )
        return chat_response.choices[0].message.content.strip()


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


def create_camera_grid(frames_dict, labels_dict):
    """
    Create a grid image from frames at the same timestamp with camera labels.

    Args:
        frames_dict: Dict mapping camera names to frames (single frame per camera)
        labels_dict: Dict mapping camera names to display labels

    Returns:
        Single stitched grid image with labels
    """
    num_cameras = len(frames_dict)
    if num_cameras == 0:
        return None

    # Calculate grid dimensions
    grid_cols = math.ceil(math.sqrt(num_cameras))
    grid_rows = math.ceil(num_cameras / grid_cols)

    # Get frames and labels in consistent order
    camera_names = sorted(frames_dict.keys())
    frames = [frames_dict[cam] for cam in camera_names]

    # Resize all frames to common dimensions (use first frame as reference)
    if not frames:
        return None

    ref_height, ref_width = frames[0].shape[:2]
    resized_frames = []
    for frame in frames:
        if frame.shape[:2] != (ref_height, ref_width):
            resized = cv2.resize(frame, (ref_width, ref_height))
        else:
            resized = frame.copy()
        resized_frames.append(resized)

    # Add labels to each frame
    labeled_frames = []
    for cam_name, frame in zip(camera_names, resized_frames):
        labeled_frame = frame.copy()
        label = labels_dict.get(cam_name, cam_name)

        # Add text label to top-left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            labeled_frame,
            (5, 5),
            (15 + text_width, 15 + text_height),
            bg_color,
            -1
        )

        # Draw text
        cv2.putText(
            labeled_frame,
            label,
            (10, 10 + text_height),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

        labeled_frames.append(labeled_frame)

    # Create grid canvas
    grid_height = grid_rows * ref_height
    grid_width = grid_cols * ref_width
    grid_canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Place frames in grid
    for idx, frame in enumerate(labeled_frames):
        row = idx // grid_cols
        col = idx % grid_cols
        y_start = row * ref_height
        y_end = (row + 1) * ref_height
        x_start = col * ref_width
        x_end = (col + 1) * ref_width
        grid_canvas[y_start:y_end, x_start:x_end] = frame

    return grid_canvas


def uniform_sampling_strategy(video_paths, num_samples):
    """
    Sample frames uniformly and independently from each camera.

    Args:
        video_paths: List of video file paths
        num_samples: Number of frames to sample from each video

    Returns:
        Dict[str, List[np.ndarray]] - frames organized by camera
    """
    frames_by_cam = {}
    for video_path in video_paths:
        if not os.path.exists(video_path):
            continue

        cam_name = os.path.splitext(os.path.basename(video_path))[0]
        frames = load_video_frames(video_path, num_frames=num_samples)
        if frames:
            frames_by_cam[cam_name] = frames

    return frames_by_cam


def stitched_frames_sampling_strategy(video_paths, num_samples):
    """
    Sample frames at synchronized timestamps and stitch into labeled grid images.

    Args:
        video_paths: List of video file paths
        num_samples: Number of synchronized time points to sample

    Returns:
        List[np.ndarray] - list of stitched grid images
    """
    if not video_paths:
        return []

    # First, determine frame indices for synchronized sampling
    # Get minimum frame count across all videos
    video_frame_counts = {}
    valid_paths = []

    for video_path in video_paths:
        if not os.path.exists(video_path):
            continue
        frame_count = get_video_frame_count(video_path)
        if frame_count > 0:
            video_frame_counts[video_path] = frame_count
            valid_paths.append(video_path)

    if not valid_paths:
        return []

    # Use minimum frame count to ensure all videos can provide frames
    min_frame_count = min(video_frame_counts.values())

    # Calculate synchronized frame indices
    if min_frame_count < num_samples:
        frame_indices = np.arange(min_frame_count)
    else:
        frame_indices = np.linspace(0, min_frame_count - 1, num_samples, dtype=int)

    # Load all frames for each video
    frames_by_video = {}
    for video_path in valid_paths:
        cam_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                # If frame read fails, use a black frame
                if frames:
                    black_frame = np.zeros_like(frames[0])
                    frames.append(black_frame)

        cap.release()
        if frames:
            frames_by_video[cam_name] = frames

    if not frames_by_video:
        return []

    # Create labels dict
    labels_dict = {cam_name: f"Camera {cam_name}" for cam_name in frames_by_video.keys()}

    # Stitch frames at each timestamp into grid images
    stitched_grids = []
    num_time_points = len(next(iter(frames_by_video.values())))

    for time_idx in range(num_time_points):
        # Collect frames at this timestamp from all cameras
        frames_at_time = {
            cam_name: frames[time_idx]
            for cam_name, frames in frames_by_video.items()
            if time_idx < len(frames)
        }

        # Create grid for this timestamp
        grid = create_camera_grid(frames_at_time, labels_dict)
        if grid is not None:
            stitched_grids.append(grid)

    return stitched_grids

def run_category_experiment(category_name, category_dataset, vllm_client, model_name, strategy="uniform"):
    """
    Run experiment for a single category.

    Args:
        category_name: Name of the question category
        category_dataset: Dataset for this category
        vllm_client: VLLMClient instance
        model_name: Name of the model being used
        strategy: Sampling strategy - "uniform" or "stitched"

    Returns:
        Dict with correct and total counts
    """
    results = {}
    correct = 0
    total = 0
    rouge_scores = []

    print(f"\nProcessing '{category_name}' category with {strategy} strategy...")
    for key in tqdm.tqdm(list(category_dataset.keys()), desc=f"{category_name}"):
        entry = category_dataset[key]

        if "video_paths" not in entry or not entry["video_paths"]:
            continue

        # Apply the selected sampling strategy
        if strategy == "uniform":
            frames_data = uniform_sampling_strategy(entry["video_paths"], NUM_SAMPLES)
            if not frames_data:
                continue
        elif strategy == "stitched":
            frames_data = stitched_frames_sampling_strategy(entry["video_paths"], NUM_SAMPLES)
            if not frames_data:
                continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        question = entry["question"]
        candidates = entry["candidates"]
        correct_answer = entry["correct_answer"]

        try:
            if candidates is None:
                # Summarization: free-form generation scored with ROUGE
                generated = vllm_client.summarize(frames_data, question, strategy_type=strategy)
                scores = compute_rouge(generated, correct_answer)
                rouge_scores.append(scores)
                total += 1
                results[key] = {
                    "question": question,
                    "generated_answer": generated,
                    "reference_answer": correct_answer,
                    "rouge_scores": scores,
                    "question_type": category_name,
                    "strategy": strategy,
                }
            else:
                # Multiple choice: scored by exact letter match
                predicted_answer = vllm_client.multiple_choice(
                    frames_data, question, candidates, strategy_type=strategy
                )
                is_correct = 1 if predicted_answer == correct_answer else 0
                correct += is_correct
                total += 1
                results[key] = {
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "question_type": category_name,
                    "strategy": strategy,
                }
        except Exception as e:
            print(f"Error processing {key}: {e}")
            continue

    # Save category results with strategy in filename
    model_name_clean = model_name.replace("/", "_")
    output_file = f"{strategy}_{model_name_clean}_{category_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    if rouge_scores:
        avg = {k: sum(s[k] for s in rouge_scores) / len(rouge_scores) for k in ("rouge1", "rouge2", "rougeL")}
        print(f"\n{category_name} avg ROUGE ({len(rouge_scores)} samples): "
              f"R-1={avg['rouge1']:.4f}  R-2={avg['rouge2']:.4f}  R-L={avg['rougeL']:.4f}")
    else:
        avg = None
        accuracy = correct / total if total > 0 else 0
        print(f"{category_name} accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Results saved to: {output_file}")

    return {"correct": correct, "total": total, "rouge_scores": rouge_scores}


def run_experiment(datasets_by_category, vllm_client, model_name, strategy="uniform"):
    """
    Run experiments for all categories, one at a time.

    Args:
        datasets_by_category: Dict of datasets organized by category
        vllm_client: VLLMClient instance
        model_name: Name of the model being used
        strategy: Sampling strategy - "uniform" or "stitched"
    """
    all_category_stats = {}

    # Process each category sequentially
    for category_name in sorted(datasets_by_category.keys()):
        category_dataset = datasets_by_category[category_name]
        stats = run_category_experiment(
            category_name, category_dataset, vllm_client, model_name, strategy=strategy
        )
        all_category_stats[category_name] = stats

    # Print final summary table
    print("\n" + "="*50)
    print(f"Final Results ({strategy} strategy):")
    print("="*50)

    mc_correct = 0
    mc_total = 0

    for category in sorted(all_category_stats.keys()):
        stats = all_category_stats[category]
        if stats["rouge_scores"]:
            rs = stats["rouge_scores"]
            avg = {k: sum(s[k] for s in rs) / len(rs) for k in ("rouge1", "rouge2", "rougeL")}
            print(f"{category:20s}: R-1={avg['rouge1']:.4f}  R-2={avg['rouge2']:.4f}  R-L={avg['rougeL']:.4f}  ({len(rs)} samples)")
        else:
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{category:20s}: {acc:.2%} ({stats['correct']}/{stats['total']})")
            mc_correct += stats["correct"]
            mc_total += stats["total"]

    print("="*50)
    if mc_total > 0:
        overall_accuracy = mc_correct / mc_total
        print(f"{'MC Overall':20s}: {overall_accuracy:.2%} ({mc_correct}/{mc_total})")
    print("="*50)

_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge(hypothesis: str, reference: str) -> dict:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scores = _rouge.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def load_datasets():
    dataset_dir = "/nas/neurosymbolic/multi-cam-dataset/ego-exo4d/"
    json_files = [
        "qa_camera.json",
        "qa_causal.json",
        "qa_event_ordering.json",
        "qa_temporal.json",
        "qa_summarization.json",
    ]

    datasets_by_category = {}
    for json_file in json_files:
        file_path = os.path.join(dataset_dir, json_file)
        print(f"Loading {json_file}...")
        with open(file_path, "r") as f:
            data = json.load(f)

        category_data = []
        for i, entry in enumerate(data):
            key = f"{entry['video_id']}_{entry['question_type']}_{i}"
            if "options" in entry:
                candidates = [f"{k}) {v}" for k, v in sorted(entry['options'].items())]
                correct_answer = entry["answer"].lower()
            else:
                candidates = None
                correct_answer = entry["answer"]

            category_data.append((key, {
                "question": entry["question"],
                "candidates": candidates,
                "correct_answer": correct_answer,
                "question_type": entry["question_type"],
                "video_id": entry["video_id"],
                "video_paths": entry.get("video_paths", [])
            }))

        question_type = data[0]["question_type"] if data else "unknown"
        datasets_by_category[question_type] = category_data

    # Sample questions per category and organize as separate datasets
    random.seed(42)  # Set seed for consistent sampling
    sampled_by_category = {}
    for category, category_data in datasets_by_category.items():
        limit = CATEGORY_QUESTIONS
        if len(category_data) > limit:
            sampled_data = random.sample(category_data, limit)
            print(f"Sampled {limit} questions from {len(category_data)} {category} questions")
        else:
            sampled_data = category_data
            print(f"Using all {len(category_data)} {category} questions")

        # Convert to dict for this category
        category_dict = {}
        for key, entry in sampled_data:
            category_dict[key] = entry
        sampled_by_category[category] = category_dict

    return sampled_by_category


def main():
    # Configuration: select sampling strategy
    # Options: "uniform" (default) or "stitched"
    STRATEGY = "uniform"

    datasets_by_category = load_datasets()

    # Calculate total questions
    total_questions = sum(len(dataset) for dataset in datasets_by_category.values())
    print(f"\nLoaded {total_questions} total questions across {len(datasets_by_category)} categories")
    print(f"Using sampling strategy: {STRATEGY}")

    vllm_client = VLLMClient()

    # Extract model name from the client
    model_name = vllm_client.model

    run_experiment(
        datasets_by_category=datasets_by_category,
        vllm_client=vllm_client,
        model_name=model_name,
        strategy=STRATEGY
    )

if __name__ == "__main__":
    main()

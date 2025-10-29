import os
import json
import cv2
import base64
from tqdm import tqdm
import tensorflow as tf
from openai import OpenAI


class VLLMCaptioner:
    """A client to generate captions using a vLLM server."""
    def __init__(self, api_base, model):
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model

    def _encode_image(self, frame):
        """Encodes a cv2 frame to a base64 string."""
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def get_caption(self, image, prompt):
        """Generates a caption for a single image using the vLLM server."""
        encoded_image = self._encode_image(image)

        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            }
        ]

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        return chat_response.choices[0].message.content.strip()


def main():
    """Main function to run the captioning process."""
    # --- Configuration ---
    video_dir = 'datasets/'
    output_dir = 'outputs'
    captions_filename = 'caption_output.json'
    prompt = 'Provide a detailed description of the image.'
    vllm_api_base = "http://localhost:8001/v1"
    vllm_model = "llava-hf/llava-1.5-7b-hf"
    # ---------------------

    tf.config.set_visible_devices([], 'GPU')
    captioner = VLLMCaptioner(api_base=vllm_api_base, model=vllm_model)
    os.makedirs(output_dir, exist_ok=True)

    video_paths_by_id = {}
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_id = os.path.basename(root)
                video_paths_by_id.setdefault(video_id, []).append(video_path)

    output_data = {}
    for video_id, video_paths in tqdm(video_paths_by_id.items(), desc="Processing video groups"):
        video_id_data = {}
        for video_path in video_paths:
            camera_name = os.path.splitext(os.path.basename(video_path))[0]
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            if fps == 0:
                print(f"Skipping video {video_path} due to 0 FPS.")
                continue

            for frame_index in range(0, frame_count, fps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    try:
                        caption = captioner.get_caption(frame, prompt)
                        video_id_data.setdefault(camera_name, {})[str(frame_index)] = caption
                    except Exception as e:
                        print(f"Error processing frame at index {frame_index} from video {video_path}: {e}")
            cap.release()

        # Save a JSON for the current index (video_id)
        idx_captions_path = os.path.join(output_dir, f'caption_output_{video_id}.json')
        with open(idx_captions_path, 'w') as f:
            json.dump({video_id: video_id_data}, f, indent=4)

        output_data[video_id] = video_id_data

    # Save the main JSON file
    captions_path = os.path.join(output_dir, captions_filename)
    with open(captions_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Finished captioning. Captions saved to {captions_path}")

if __name__ == "__main__":
    main()

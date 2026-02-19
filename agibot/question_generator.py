import json
import random
import os

from llm import LLM
from tqdm import tqdm

OUTPUT_DIRECTORY = "/nas/neurosymbolic/multi-cam-dataset/agibot/"
NUM_QUESTIONS = 2  # = number of episodes sampled per task; 1 question generated per episode


def _sample_two_distinct_episodes(episodes):
    """Return a pair (ep1, ep2) where no action_text appears in both episodes.
    Returns None if no valid pair is found after retries."""
    if len(episodes) < 2:
        return None
    indices = list(range(len(episodes)))
    for _ in range(50):
        i, j = random.sample(indices, 2)
        ep1, ep2 = episodes[i], episodes[j]
        texts1 = {ann["action_text"] for ann in ep1["annotations"]}
        texts2 = {ann["action_text"] for ann in ep2["annotations"]}
        if not (texts1 & texts2):
            return ep1, ep2
    return None


def temporal(data, num_questions=NUM_QUESTIONS):
    with open("prompts/temporal.txt", "r") as f:
        prompt_template = f.read()

    all_generated_questions = []

    for task in tqdm(data, desc="Generating temporal questions"):
        task_id = task["task"]
        task_name = task["task_name"]
        scene_desc = task["scene_descriptions"]
        episodes = task["episodes"]

        pair = _sample_two_distinct_episodes(episodes)
        if pair is None:
            continue

        for episode in pair:
            ep_id = episode["id"]
            annotations = episode["annotations"]
            if not annotations:
                continue

            found_suitable_events = False
            grounding_event = []
            target_event = None
            grounding_input_desc = ""
            target_input_desc = ""
            rel_type = "Before"

            for _ in range(20):
                rel_type = random.choice(["Before", "After", "In-Between"])

                if rel_type == "In-Between":
                    if len(annotations) < 3:
                        continue
                    event_choices = random.sample(annotations, 3)
                    event_choices.sort(key=lambda x: x["start_frame"])
                    event_start, event_middle, event_end = event_choices
                    grounding_input_desc = (
                        f"\n- Start: \"{event_start['action_text']}\"\n"
                        f"- End: \"{event_end['action_text']}\""
                    )
                    target_input_desc = f"\"{event_middle['action_text']}\""
                    grounding_event = [event_start, event_end]
                    target_event = event_middle
                    found_suitable_events = True
                    break

                else:  # Before or After
                    if len(annotations) < 2:
                        continue
                    event_choices = random.sample(annotations, 2)
                    event_choices.sort(key=lambda x: x["start_frame"])
                    event_earlier, event_later = event_choices
                    if rel_type == "Before":
                        grounding_input_desc = f"\"{event_later['action_text']}\""
                        target_input_desc = f"\"{event_earlier['action_text']}\""
                        grounding_event = [event_later]
                        target_event = event_earlier
                    else:  # After
                        grounding_input_desc = f"\"{event_earlier['action_text']}\""
                        target_input_desc = f"\"{event_later['action_text']}\""
                        grounding_event = [event_earlier]
                        target_event = event_later
                    found_suitable_events = True
                    break

            if not found_suitable_events:
                continue
            assert target_event is not None

            prompt = prompt_template.format(
                grounding_input=grounding_input_desc,
                target_input=target_input_desc,
                rel_type=f"\"{rel_type}\""
            )

            llm = LLM()
            response = llm.prompt(prompt)
            if not isinstance(response, str):
                continue
            if response.startswith("```json"):
                response = response[7:-4]

            try:
                qa_pair = json.loads(response)
            except json.JSONDecodeError:
                continue

            all_generated_questions.append({
                "task_id": task_id,
                "task_name": task_name,
                "episode_id": ep_id,
                "question_type": "temporal",
                **qa_pair,
                "video_paths": episode["paths"],
                "metadata": {
                    "scene_description": scene_desc,
                    "grounding": [
                        {
                            "activity": g["action_text"],
                            "skill": g["skill"],
                            "start_frame": g["start_frame"],
                            "end_frame": g["end_frame"],
                        } for g in grounding_event
                    ],
                    "target": {
                        "activity": target_event["action_text"],
                        "skill": target_event["skill"],
                        "start_frame": target_event["start_frame"],
                        "end_frame": target_event["end_frame"],
                    },
                    "rel": rel_type.lower()
                }
            })

    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_temporal.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_questions, f, indent=2)


def event_ordering(data, num_questions=NUM_QUESTIONS):
    with open("prompts/ordering.txt", "r") as f:
        prompt_template = f.read()

    all_generated_questions = []

    for task in tqdm(data, desc="Generating event ordering questions"):
        task_id = task["task"]
        task_name = task["task_name"]
        scene_desc = task["scene_descriptions"]
        episodes = task["episodes"]

        pair = _sample_two_distinct_episodes(episodes)
        if pair is None:
            continue

        for episode in pair:
            ep_id = episode["id"]
            annotations = episode["annotations"]
            if not annotations:
                continue

            found_suitable_events = False
            selected_events = []

            for _ in range(20):
                num_events_to_select = random.choice([3, 4])
                if len(annotations) < num_events_to_select:
                    continue
                event_choices = random.sample(annotations, num_events_to_select)
                event_choices.sort(key=lambda x: x["start_frame"])

                # Ensure unique action texts
                texts = [e["action_text"] for e in event_choices]
                if len(set(texts)) < num_events_to_select:
                    continue

                selected_events = event_choices
                found_suitable_events = True
                break

            if not found_suitable_events:
                continue

            events_list_for_prompt = "".join(
                f"{i + 1}. {ev['action_text']}\n" for i, ev in enumerate(selected_events)
            )

            prompt = prompt_template.format(events_list=events_list_for_prompt)

            llm = LLM()
            response = llm.prompt(prompt)
            if not isinstance(response, str):
                continue
            if response.startswith("```json"):
                response = response[7:-4]

            try:
                qa_pair = json.loads(response)
            except json.JSONDecodeError:
                continue

            all_generated_questions.append({
                "task_id": task_id,
                "task_name": task_name,
                "episode_id": ep_id,
                "question_type": "event_ordering",
                **qa_pair,
                "video_paths": episode["paths"],
                "metadata": {
                    "scene_description": scene_desc,
                    "ordered_events": [
                        {
                            "activity": ev["action_text"],
                            "skill": ev["skill"],
                            "start_frame": ev["start_frame"],
                            "end_frame": ev["end_frame"],
                        } for ev in selected_events
                    ]
                }
            })

    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_event_ordering.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_questions, f, indent=2)


def summarization(data, num_questions=NUM_QUESTIONS):
    with open("prompts/summarization.txt", "r") as f:
        prompt_template = f.read()

    all_generated_summaries = []

    for task in tqdm(data, desc="Generating summaries"):
        task_id = task["task"]
        task_name = task["task_name"]
        scene_desc = task["scene_descriptions"]
        episodes = task["episodes"]

        pair = _sample_two_distinct_episodes(episodes)
        if pair is None:
            continue

        for episode in pair:
            ep_id = episode["id"]
            annotations = episode["annotations"]
            if not annotations:
                continue

            annotations_list_for_prompt = "".join(
                f"- {ann['action_text']}\n" for ann in annotations
            )

            prompt = prompt_template.format(
                task_name=task_name,
                scene_description=scene_desc,
                annotations_list=annotations_list_for_prompt,
            )

            llm = LLM()
            response = llm.prompt(prompt)
            if not isinstance(response, str):
                continue
            if response.startswith("```json"):
                response = response[7:-4]

            try:
                summary_data = json.loads(response)
            except json.JSONDecodeError:
                continue

            all_generated_summaries.append({
                "task_id": task_id,
                "task_name": task_name,
                "episode_id": ep_id,
                "question_type": "summarization",
                "question": "Provide a comprehensive summary of the robot's actions in this episode.",
                "answer": summary_data.get("summary", ""),
                "video_paths": episode["paths"],
                "metadata": {
                    "scene_description": scene_desc,
                    "annotations": [
                        {
                            "activity": ann["action_text"],
                            "skill": ann["skill"],
                            "start_frame": ann["start_frame"],
                            "end_frame": ann["end_frame"],
                        } for ann in annotations
                    ]
                }
            })

    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_summarization.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_summaries, f, indent=2)


if __name__ == "__main__":
    with open("compiled.json", "r") as f:
        data = json.load(f)
    # temporal(data)
    # event_ordering(data)
    summarization(data)

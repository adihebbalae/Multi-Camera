import json
import random
import os
import re # Import regex module

from llm import LLM
from tqdm import tqdm

OUTPUT_DIRECTORY = "/nas/neurosymbolic/multi-cam-dataset/ego-exo4d/"


def temporal(data, num_questions_per_video=2):
    with open("prompts/temporal.txt", "r") as f:
        prompt_template = f.read()

    all_generated_questions = []

    for video_id, video_data in tqdm(data.items(), desc="Generating temporal questions for videos"):
        annotations = [ann for sublist in video_data["annotations"] for ann in sublist]
        if not annotations:
            continue

        # print(f"Generating questions for video: {video_id}")
        for _ in range(num_questions_per_video):
            found_suitable_events = False
            
            grounding_event = []
            target_event = None

            for _ in range(20): # Retry more times to find a suitable triplet or pair
                rel_type = random.choice(["Before", "After", "In-Between"])
                
                if rel_type == "In-Between":
                    if len(annotations) < 3:
                        continue

                    event_choices = random.sample(annotations, 3)
                    event_choices.sort(key=lambda x: x['timestamp'])

                    event_start = event_choices[0]
                    event_middle = event_choices[1]
                    event_end = event_choices[2]

                    if (event_middle['timestamp'] - event_start['timestamp'] > 1.0 and
                        event_end['timestamp'] - event_middle['timestamp'] > 1.0):
                        
                        grounding_input_desc = (
                            f"\n- Start: \"{event_start['text']}\"\n"
                            f"- End: \"{event_end['text']}\""
                        )
                        target_input_desc = f"\"{event_middle['text']}\""

                        grounding_event = [event_start, event_end]
                        target_event = event_middle
                        found_suitable_events = True
                        break

                else: # Before or After
                    if len(annotations) < 2:
                        continue

                    event_choices = random.sample(annotations, 2)
                    event_choices.sort(key=lambda x: x['timestamp'])

                    event_earlier = event_choices[0]
                    event_later = event_choices[1]

                    if (event_later['timestamp'] - event_earlier['timestamp']) > 1.0:
                        if rel_type == "Before":
                            grounding_input_desc = f"\"{event_later['text']}\""
                            target_input_desc = f"\"{event_earlier['text']}\""

                            grounding_event = [event_later]
                            target_event = event_earlier
                            found_suitable_events = True
                            break

                        elif rel_type == "After":
                            grounding_input_desc = f"\"{event_earlier['text']}\""
                            target_input_desc = f"\"{event_later['text']}\""

                            grounding_event = [event_earlier]
                            target_event = event_later
                            found_suitable_events = True
                            break
            
            if not found_suitable_events:
                continue

            prompt = prompt_template.format(
                grounding_input=grounding_input_desc,
                target_input=target_input_desc,
                rel_type=f"\"{rel_type}\""
            )

            llm = LLM()
            response = llm.prompt(prompt)
            if response.startswith("```json"):
                response = response[7:-4]
            
            try:
                qa_pair = json.loads(response)
            except json.JSONDecodeError as e:
                continue

            formatted_question = {
                "video_id": video_id,
                "question_type": "temporal",
                **qa_pair,
                "video_paths": video_data["video_files"],
                "metadata": {
                    "grounding": [
                        {
                            "activity": g['text'],
                            "start_timestamp": g['timestamp'],
                        } for g in grounding_event
                    ],
                    "target": {
                        "activity": target_event['text'],
                        "start_timestamp": target_event['timestamp'],
                    },
                    "rel": rel_type.lower()
                }
 
            }

            all_generated_questions.append(formatted_question)
            # print(json.dumps(qa_pair, indent=4))

    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_temporal.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_questions, f, indent=2)

def event_ordering(data, num_questions_per_video=2):
    with open("prompts/ordering.txt", "r") as f:
        prompt_template = f.read()

    all_generated_questions = []

    for video_id, video_data in tqdm(data.items(), desc="Generating event ordering questions for videos"):
        annotations = [ann for sublist in video_data["annotations"] for ann in sublist]
        if not annotations:
            continue

        for _ in range(num_questions_per_video):
            found_suitable_events = False
            selected_events = []

            for _ in range(20): # Retry to find suitable events
                num_events_to_select = random.choice([3, 4]) # Choose between 3 or 4 events
                if len(annotations) < num_events_to_select:
                    continue

                event_choices = random.sample(annotations, num_events_to_select)
                event_choices.sort(key=lambda x: x['timestamp'])

                # Check if events are distinct enough in time
                suitable_sequence = True
                for i in range(len(event_choices) - 1):
                    if (event_choices[i+1]['timestamp'] - event_choices[i]['timestamp']) < 1.0: # Minimum 1 second difference
                        suitable_sequence = False
                        break
                
                if suitable_sequence:
                    selected_events = event_choices
                    found_suitable_events = True
                    break
            
            if not found_suitable_events:
                continue

            events_list_for_prompt = ""
            for i, event in enumerate(selected_events):
                events_list_for_prompt += f"{i+1}. {event['text']}\n"
            
            prompt = prompt_template.format(events_list=events_list_for_prompt)

            llm = LLM()
            response = llm.prompt(prompt)
            if response.startswith("```json"):
                response = response[7:-4]
            
            try:
                qa_pair = json.loads(response)
            except json.JSONDecodeError as e:
                continue

            formatted_question = {
                "video_id": video_id,
                "question_type": "event_ordering",
                **qa_pair,
                "video_paths": video_data["video_files"],
                "metadata": {
                    "ordered_events": [
                        {
                            "activity": ev['text'],
                            "start_timestamp": ev['timestamp'],
                        } for ev in selected_events
                    ]
                }
            }
            all_generated_questions.append(formatted_question)
    
    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_event_ordering.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_questions, f, indent=2)

def causal(data, num_questions_per_video=2):
    with open("prompts/causal.txt", "r") as f:
        prompt_template = f.read()

    all_generated_questions = []

    for video_id, video_data in tqdm(data.items(), desc="Generating causal questions for videos"):
        keystep_annotations = video_data.get("keystep_annotations")
        if not keystep_annotations:
            continue

        num_to_generate = min(num_questions_per_video, len(keystep_annotations))
        selected_keysteps = random.sample(keystep_annotations, num_to_generate)

        for keystep in selected_keysteps:
            parent_hierarchy = []
            curr_node = keystep["node"]
            while curr_node:
                parent_hierarchy.append(curr_node["node_name"])
                curr_node = curr_node["parent"]
            parent_hierarchy_str = ""
            for name in parent_hierarchy:
                parent_hierarchy_str += f"- {name}\n"

            prompt = prompt_template.format(
                description=keystep["description"],
                parent_hierarchy=parent_hierarchy_str
            )
            llm = LLM()
            response = llm.prompt(prompt)
            if response.startswith("```json"):
                response = response[7:-4]
            
            try:
                qa_pair = json.loads(response)
            except json.JSONDecodeError as e:
                continue

            formatted_question = {
                "video_id": video_id,
                "question_type": "causal",
                **qa_pair,
                "video_paths": video_data["video_files"],
                "metadata": {
                    "keystep_description": keystep["description"],
                    "keystep_start_time": keystep["start_time"],
                    "keystep_end_time": keystep["end_time"],
                    "keystep_category": keystep["category"],
                    "node_hierarchy": keystep["node"]
                }
            }
            all_generated_questions.append(formatted_question)

    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_causal.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_questions, f, indent=2)

def camera(data, num_questions_per_video=2):
    with open("prompts/camera.txt", "r") as f:
        prompt_template = f.read()

    all_generated_questions = []

    def _format_camera_name(cam_id_raw):
        if cam_id_raw.startswith("cam"):
            return f"camera {int(cam_id_raw[3:])}"
        elif cam_id_raw.startswith("gp"):
            return f"camera {int(cam_id_raw[2:])}"

    for video_id, video_data in tqdm(data.items(), desc="Generating camera questions for videos"):
        annotations = [ann for sublist in video_data["annotations"] for ann in sublist]
        annotations.sort(key=lambda x: x['timestamp'])

        camera_ids_set = set()
        for path in video_data["video_files"]:
            cam_id_raw = os.path.basename(path).split('.')[0]
            camera_ids_set.add(_format_camera_name(cam_id_raw))

        all_camera_options = sorted(list(camera_ids_set), key=lambda x: int(x.split()[-1]))

        if not all_camera_options:
            continue

        # Pre-compute eligibility flags
        scene_eligible = (
            "best_camera" in video_data and video_data["best_camera"] is not None
            and len([c for c in all_camera_options if c != _format_camera_name(video_data["best_camera"])]) >= 3
        )
        suitable_annotations = [ann for ann in annotations if "best_camera" in ann and ann["best_camera"] is not None]
        remaining_annotations = list(suitable_annotations)

        scene_used = False

        for _ in range(num_questions_per_video):
            # Build the eligible subtype pool for this iteration
            current_eligible = []

            if scene_eligible and not scene_used:
                current_eligible.append("best_camera_scene")

            cam_groups = {}
            if remaining_annotations:
                current_eligible.append("best_camera_single_event")
                for ann in remaining_annotations:
                    cam_groups.setdefault(ann["best_camera"], []).append(ann)
                if any(len(anns) >= 2 for anns in cam_groups.values()):
                    current_eligible.append("best_camera_multiple_events")

            if not current_eligible:
                break

            chosen = random.choice(current_eligible)

            # --- best_camera_scene ---
            if chosen == "best_camera_scene":
                scene_used = True
                best_camera_scene = _format_camera_name(video_data["best_camera"])
                annotations_list = "".join(f"- {ann['text']}\n" for ann in annotations)

                prompt = prompt_template.format(
                    question_type="Best camera angle for the scene",
                    description=f"Annotations:\n{annotations_list}",
                    best_camera=best_camera_scene,
                    all_cameras=", ".join(all_camera_options)
                )

                llm = LLM()
                response = llm.prompt(prompt)
                if response.startswith("```json"):
                    response = response[7:-4]

                try:
                    qa_pair = json.loads(response)
                except json.JSONDecodeError:
                    continue

                all_generated_questions.append({
                    "video_id": video_id,
                    "question_type": "camera",
                    **qa_pair,
                    "video_paths": video_data["video_files"],
                    "metadata": {
                        "camera_question_subtype": "best_camera_scene",
                        "best_camera_scene": video_data["best_camera"]
                    }
                })

            # --- best_camera_multiple_events ---
            elif chosen == "best_camera_multiple_events":
                eligible_cams = [cam for cam, anns in cam_groups.items() if len(anns) >= 2]
                cam_key = random.choice(eligible_cams)
                selected_pair = random.sample(cam_groups[cam_key], 2)
                selected_pair.sort(key=lambda x: x['timestamp'])
                event1, event2 = selected_pair

                remaining_annotations = [ann for ann in remaining_annotations if ann not in selected_pair]

                best_camera_event = _format_camera_name(cam_key)
                if len([c for c in all_camera_options if c != best_camera_event]) < 3:
                    continue

                prompt = prompt_template.format(
                    question_type="Best camera angle for multiple events",
                    description=(
                        f"Event 1: {event1['text']} (timestamp={event1['timestamp']}); "
                        f"Event 2: {event2['text']} (timestamp={event2['timestamp']})"
                    ),
                    best_camera=best_camera_event,
                    all_cameras=", ".join(all_camera_options)
                )

                llm = LLM()
                response = llm.prompt(prompt)
                if response.startswith("```json"):
                    response = response[7:-4]

                try:
                    qa_pair = json.loads(response)
                except json.JSONDecodeError:
                    continue

                all_generated_questions.append({
                    "video_id": video_id,
                    "question_type": "camera",
                    **qa_pair,
                    "video_paths": video_data["video_files"],
                    "metadata": {
                        "camera_question_subtype": "best_camera_multiple_events",
                        "event1_description": event1["text"],
                        "event1_timestamp": event1["timestamp"],
                        "event2_description": event2["text"],
                        "event2_timestamp": event2["timestamp"],
                        "best_camera_event": cam_key
                    }
                })

            # --- best_camera_single_event ---
            else:
                selected_annotation = random.choice(remaining_annotations)
                remaining_annotations.remove(selected_annotation)

                best_camera_event = _format_camera_name(selected_annotation["best_camera"])
                if len([c for c in all_camera_options if c != best_camera_event]) < 3:
                    continue

                prompt = prompt_template.format(
                    question_type="Best camera angle for a specific event",
                    description=f"{selected_annotation['text']} (timestamp={selected_annotation['timestamp']})",
                    best_camera=best_camera_event,
                    all_cameras=", ".join(all_camera_options)
                )

                llm = LLM()
                response = llm.prompt(prompt)
                if response.startswith("```json"):
                    response = response[7:-4]

                try:
                    qa_pair = json.loads(response)
                except json.JSONDecodeError:
                    continue

                all_generated_questions.append({
                    "video_id": video_id,
                    "question_type": "camera",
                    **qa_pair,
                    "video_paths": video_data["video_files"],
                    "metadata": {
                        "camera_question_subtype": "best_camera_single_event",
                        "event_description": selected_annotation["text"],
                        "event_timestamp": selected_annotation["timestamp"],
                        "best_camera_event": selected_annotation["best_camera"]
                    }
                })

    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_camera.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_questions, f, indent=2)

def summarization(data, num_summaries_per_video=1):
    with open("prompts/summarization.txt", "r") as f:
        prompt_template = f.read()

    all_generated_summaries = []

    for video_id, video_data in tqdm(data.items(), desc="Generating summaries for videos"):
        annotations = [ann for sublist in video_data["annotations"] for ann in sublist]
        annotations.sort(key=lambda x: x['timestamp']) # Ensure annotations are in temporal order
        
        if not annotations:
            continue

        # Annotations
        annotations_list_for_prompt = ""
        for ann in annotations:
            annotations_list_for_prompt += f"- {ann['text']}\n"
        
        # Objects
        objects_list_for_prompt = ""
        processed_objects = []
        if "objects" in video_data and video_data["objects"]:
            for obj_item in video_data["objects"]:
                obj_name_raw = obj_item[0]
                obj_name_processed = '_'.join(obj_name_raw.split('_')[:-1]).replace('_', ' ') if obj_name_raw.split('_')[-1].isdigit() else obj_name_raw.replace('_', ' ')
                processed_objects.append(obj_name_processed)
                objects_list_for_prompt += f"- {obj_name_processed}\n"

        for _ in range(num_summaries_per_video):
            prompt = prompt_template.format(
                annotations_list=annotations_list_for_prompt,
                objects_list=objects_list_for_prompt if objects_list_for_prompt else "None"
            )

            llm = LLM()
            response = llm.prompt(prompt)
            if response.startswith("```json"):
                response = response[7:-4]
            
            try:
                summary_data = json.loads(response)
            except json.JSONDecodeError as e:
                continue

            formatted_summary = {
                "video_id": video_id,
                "question_type": "summarization",
                "question": "Provide a very comprehensive, well thought-out summary of the ego-actor's interactions across all views. Make it a couple sentences and around 500-1000 characters",
                "answer": summary_data["summary"],
                "video_paths": video_data["video_files"],
                "metadata": {
                    "annotations": [
                        {
                            "activity": ann['text'],
                            "start_timestamp": ann['timestamp'],
                        } for ann in annotations
                    ],
                    "objects": processed_objects
                }
            }
            all_generated_summaries.append(formatted_summary)
    
    output_file_path = os.path.join(OUTPUT_DIRECTORY, "qa_summarization.json")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as f:
        json.dump(all_generated_summaries, f, indent=2)

if __name__ == "__main__":
    with open("compiled.json", "r") as f:
        data = json.load(f)
    # temporal(data)
    # event_ordering(data)
    # summarization(data)
    # camera(data)
    causal(data)

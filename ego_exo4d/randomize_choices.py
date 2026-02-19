import json
import os
import random
import glob

OUTPUT_DIRECTORY = "/nas/neurosymbolic/multi-cam-dataset/ego-exo4d/"


def randomize():
    json_files = sorted(glob.glob(os.path.join(OUTPUT_DIRECTORY, "qa_*.json")))

    for filepath in json_files:
        with open(filepath) as f:
            records = json.load(f)

        changed = 0
        for rec in records:
            if "options" not in rec or "answer" not in rec:
                continue

            options = rec["options"]
            correct_value = options[rec["answer"]]

            keys = list(options.keys())
            values = list(options.values())
            random.shuffle(values)

            new_options = dict(zip(keys, values))
            rec["options"] = new_options
            rec["answer"] = next(k for k, v in new_options.items() if v == correct_value)
            changed += 1

        with open(filepath, "w") as f:
            json.dump(records, f, indent=2)

        print(f"Randomized {changed}/{len(records)} records in {os.path.basename(filepath)}")


if __name__ == "__main__":
    randomize()

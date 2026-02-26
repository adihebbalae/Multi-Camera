# MEVA QA Pipeline — Quick Start

## Prerequisites

- Access to `/nas/mars/dataset/MEVA/` (shared on the `/nas` machine and remote cluster)
- Python 3.10+
- `OPENAI_API_KEY` set in your environment (required for Step 2 — naturalization only)

### Install dependencies
```bash
pip install pyyaml numpy opencv-python openai
```
Or use the shared venv:
```bash
source /home/ah66742/venv/bin/activate
```

---

## Running the Pipeline

All commands must be run from the **`meva/` directory** inside this repo:

```bash
cd /path/to/repo/meva
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OPENAI_API_KEY=sk-...          # only needed for Step 2
export MEVA_OUTPUT_DIR=~/data         # where QA JSON + logs are saved (default: ~/data)
```

### Step 1 — Raw QA generation (free, ~5 seconds per slot)

```bash
python3 -m scripts.v10.run_pipeline --slot "2018-03-15.15-00.school" -v --seed 42
```

Output: `$MEVA_OUTPUT_DIR/qa_pairs/2018-03-15.15-00.school/2018-03-15.15-00.school.final.raw.json`

### Step 2 — Naturalize with GPT (costs tokens, ~$0.002/slot with gpt-4o-mini)

```bash
python3 -m scripts.v10.naturalize \
  --input $MEVA_OUTPUT_DIR/qa_pairs/2018-03-15.15-00.school/2018-03-15.15-00.school.final.raw.json \
  -v --yes
```

Output: `...2018-03-15.15-00.school.final.naturalized.json`

Use `--preprocess-only` to run deterministic text cleanup without any GPT call (free):
```bash
python3 -m scripts.v10.naturalize --input <raw.json> --preprocess-only
```

### Step 3 — Export to multi-cam-dataset format

```bash
python3 -m scripts.v10.export_to_multicam_format --slot "2018-03-15.15-00.school"
```

Output: `/nas/neurosymbolic/multi-cam-dataset/meva/qa_pairs/2018-03-15.15-00.school.json`

### All-in-one

Pass a slot as argument (or edit `SLOT=` inside the script):

```bash
bash run.sh "2018-03-15.15-00.school"
```

---

## What's in `meva/data/`?

These files are checked into the repo and are automatically found by the scripts:

| File | Contents |
|------|----------|
| `data/canonical_slots.json` | 929 annotated slots with camera lists |
| `data/slot_index.json` | Clip-level annotation index |
| `data/person_database_yolo.json` | MEVID person descriptions (YOLO+GPT) |
| `data/person_database.json` | MEVID person descriptions (original) |
| `data/mevid_supported_slots.json` | Slots with MEVID re-ID coverage |
| `data/geom_slot_index.json` | Geom-file index for entity descriptions |

---

## Slot Name Format

Slots follow the pattern: `YYYY-MM-DD.HH-MM.site`

Example: `2018-03-15.15-00.school`

Sites: `school`, `admin`, `bus`, `hospital`

To list all available annotated slots:
```bash
python3 -m scripts.v10.run_pipeline --list-slots
```

---

## Output Layout

```
$MEVA_OUTPUT_DIR/
  qa_pairs/
    2018-03-15.15-00.school/
      2018-03-15.15-00.school.final.raw.json          ← Step 1 output
      2018-03-15.15-00.school.final.naturalized.json  ← Step 2 output
      validation_videos/                              ← optional render step
  gpt_logs/
    2018-03-15.15-00.school/
      naturalize_gpt-4o-mini.json
  entity_descriptions/
    2018-03-15.15-00.school.json                      ← auto-generated on first run
```

---

## Common Issues

**`canonical slot and slot index is required`**  
→ You ran `python -m meva.scripts.v10.run_pipeline` from the repo root. There is no `meva/__init__.py`, so that path doesn't work.  
→ **Fix**: `cd meva/ && export PYTHONPATH=$PYTHONPATH:$(pwd)` first, then use `python3 -m scripts.v10.run_pipeline`.

**`No events found for slot ...`**  
→ Check the slot exists: `python3 -m scripts.v10.run_pipeline --list-slots | grep <slot>`  
→ Slot format must be `YYYY-MM-DD.HH-MM.site` (no seconds).

**`OPENAI_API_KEY not set`**  
→ Only needed for Step 2. Steps 1 and 3 are free.

**Output goes to wrong directory**  
→ Set `export MEVA_OUTPUT_DIR=/your/home/data` before running.

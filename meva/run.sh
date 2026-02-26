#!/usr/bin/env bash
# run.sh — Full V10 QA pipeline for one MEVA slot.
#
# SETUP (one-time):
#   1. cd into this directory: cd /path/to/repo/meva
#   2. Set your API key: export OPENAI_API_KEY=sk-...
#   3. (Optional) Set output dir: export MEVA_OUTPUT_DIR=~/data
#      Defaults to ~/data if not set. QA JSON + logs go there.
#   4. Run: bash run.sh
#
# DATASET (shared, read-only):
#   /nas/mars/dataset/MEVA/  — videos, annotations, camera models
#
# REQUIREMENTS:
#   pip install pyyaml numpy opencv-python openai
#   (or: source /home/ah66742/venv/bin/activate)

set -e

SLOT="${1:-2018-03-11.11-25.school}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/data}"

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=== Step 1: Raw QA generation (slot: $SLOT) ==="
python3 -m scripts.v10.run_pipeline \
  --slot "$SLOT" \
  -v \
  --seed 42

RAW_JSON="$OUTPUT_DIR/qa_pairs/$SLOT.final.raw.json"

echo ""
echo "=== Step 2: Naturalization (GPT — requires OPENAI_API_KEY) ==="
python3 -m scripts.v10.naturalize \
  --input "$RAW_JSON" \
  -v --yes

echo ""
echo "=== Step 3: Export to multi-cam-dataset format ==="
python3 -m scripts.v10.export_to_multicam_format \
  --slot "$SLOT"

echo ""
echo "Done. Output in $OUTPUT_DIR/qa_pairs/$SLOT/"

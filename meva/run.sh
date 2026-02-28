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
#   (or: source your-venv/bin/activate)

# set -e


OUTPUT_DIR="/nas/neurosymbolic/multi-cam-dataset/meva/data"
mkdir -p "$OUTPUT_DIR/qa_pairs/raw"
ENV_FILE="/home/ah66742/.env"  # this is where OPENAI_API_KEY is
export MEVA_OUTPUT_DIR="/nas/neurosymbolic/multi-cam-dataset/meva/data/"


# Parse args: extract --n flag and positional slot
NATURALIZE=false
SLOT=""
for arg in "$@"; do
  if [[ "$arg" == "--n" ]]; then
    NATURALIZE=true
  elif [[ "$arg" != --* ]]; then
    SLOT="$arg"
  fi
done
SLOT="${SLOT:-2018-03-09.10-15.school}"

# export PYTHONPATH=$PYTHONPATH:$(pwd)

# echo "=== Step 1: Raw QA generation (slot: $SLOT) ==="
python3 -m scripts.v10.run_pipeline \
  --slot "$SLOT" \
  -v \
  --seed 42


RAW_JSON="$OUTPUT_DIR/qa_pairs/raw/$SLOT.raw.json"

# echo ""
# echo "=== Step 2: Naturalization (GPT — requires OPENAI_API_KEY) ==="
if $NATURALIZE; then
  # Load .env for OPENAI_API_KEY if not already set
  if [[ -z "$OPENAI_API_KEY" && -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
  fi
  python3 -m scripts.v10.naturalize \
    --input "$RAW_JSON" \
    --output "$OUTPUT_DIR/qa_pairs/$SLOT.naturalized.json" \
    -v --yes
fi

# echo ""
# echo "=== Step 3: Export to multi-cam-dataset format ==="
# python3 -m scripts.v10.export_to_multicam_format \
#   --slot "$SLOT"

# echo ""
# echo "Done. Output in $OUTPUT_DIR/qa_pairs/$SLOT/"

#!/usr/bin/env bash
# run.sh — Full V10 QA pipeline for one MEVA slot.
#
# USAGE:
#   bash run.sh <slot>           # raw QA only (free, no API key needed)
#   bash run.sh <slot> --n       # raw QA + GPT naturalization
#
# SETUP (one-time):
#   1. cd into the repo: cd /path/to/repo/meva
#   2. Activate your venv: source /path/to/venv/bin/activate
#   3. For naturalization (--n only): export OPENAI_API_KEY=sk-...
#      Or put OPENAI_API_KEY=sk-... in ~/.env  (loaded automatically)
#   4. (Optional) Override output dir:
#        export MEVA_OUTPUT_DIR=/your/output/dir
#        Default: /nas/neurosymbolic/multi-cam-dataset/meva/data
#
# DATASET (shared NAS, read-only):
#   /nas/mars/dataset/MEVA/  — videos, annotations, camera models
#
# REQUIREMENTS:
#   pip install pyyaml numpy opencv-python openai

set -e

# ---------------------------------------------------------------------------
# Output directory — override with MEVA_OUTPUT_DIR env var
# ---------------------------------------------------------------------------
OUTPUT_DIR="${MEVA_OUTPUT_DIR:-/nas/neurosymbolic/multi-cam-dataset/meva/data}"
mkdir -p "$OUTPUT_DIR/qa_pairs/raw" || {
  echo "ERROR: Cannot create output directory: $OUTPUT_DIR/qa_pairs/raw"
  echo "       Set MEVA_OUTPUT_DIR to a writable path, e.g.:"
  echo "         export MEVA_OUTPUT_DIR=\$HOME/meva_output"
  exit 1
}
export MEVA_OUTPUT_DIR="$OUTPUT_DIR"

# .env file for OPENAI_API_KEY — looked up in caller's home dir (not hardcoded)
ENV_FILE="${ENV_FILE:-$HOME/.env}"


# ---------------------------------------------------------------------------
# Parse args: extract --n flag and positional slot
# ---------------------------------------------------------------------------
NATURALIZE=false
SLOT=""
for arg in "$@"; do
  if [[ "$arg" == "--n" ]]; then
    NATURALIZE=true
  elif [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    echo "Usage: bash run.sh <slot> [--n]"
    echo "  <slot>   e.g. 2018-03-11.11-25.school"
    echo "  --n      also run GPT naturalization (requires OPENAI_API_KEY)"
    echo ""
    echo "Available sites: admin, bus, hospital, school"
    echo "List slots:  python3 -m scripts.v10.run_pipeline --list-slots"
    exit 0
  elif [[ "$arg" != --* ]]; then
    SLOT="$arg"
  fi
done

if [[ -z "$SLOT" ]]; then
  echo "ERROR: No slot specified."
  echo "Usage: bash run.sh <slot> [--n]"
  echo "       bash run.sh 2018-03-11.11-25.school"
  echo "       bash run.sh 2018-03-11.11-25.school --n"
  echo ""
  echo "List all available slots:"
  echo "  python3 -m scripts.v10.run_pipeline --list-slots"
  exit 1
fi


# ---------------------------------------------------------------------------
# Step 1: Raw QA generation
# ---------------------------------------------------------------------------
echo "=== Raw QA: slot=$SLOT ==="
python3 -m scripts.v10.run_pipeline \
  --slot "$SLOT" \
  -v \
  --seed 42

RAW_JSON="$OUTPUT_DIR/qa_pairs/raw/$SLOT.raw.json"

# ---------------------------------------------------------------------------
# Step 2: Naturalization (GPT — only if --n flag passed)
# ---------------------------------------------------------------------------
if $NATURALIZE; then
  # Load .env for OPENAI_API_KEY if not already set in environment
  if [[ -z "$OPENAI_API_KEY" && -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
  fi

  if [[ -z "$OPENAI_API_KEY" ]]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY is not set. Naturalization requires it."
    echo "  Option 1: export OPENAI_API_KEY=sk-..."
    echo "  Option 2: put OPENAI_API_KEY=sk-... in $ENV_FILE"
    exit 1
  fi

  echo ""
  echo "=== Naturalization (gpt-4o-mini) ==="
  python3 -m scripts.v10.naturalize \
    --input "$RAW_JSON" \
    --output "$OUTPUT_DIR/qa_pairs/$SLOT.naturalized.json" \
    -v --yes

  echo ""
  echo "Done. Output:"
  echo "  Raw:         $RAW_JSON"
  echo "  Naturalized: $OUTPUT_DIR/qa_pairs/$SLOT.naturalized.json"
else
  echo ""
  echo "Done. Output: $RAW_JSON"
  echo "(Run with --n to also naturalize with GPT)"
fi

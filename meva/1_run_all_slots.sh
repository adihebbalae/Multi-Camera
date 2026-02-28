#!/usr/bin/env bash

LIST_FILE="data/slot_list_from_slot_index.txt"
OUTPUT_DIR="/nas/neurosymbolic/multi-cam-dataset/meva/data"
RUN_FIRST_TEN="true"

mkdir -p "$OUTPUT_DIR/qa_pairs/raw"
export MEVA_OUTPUT_DIR="/nas/neurosymbolic/multi-cam-dataset/meva/data_all_slots/"

TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_DIR="${MEVA_OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/${TIMESTAMP}_1_run_all_slots.log") 2>&1

count=0
while IFS= read -r slot; do
  if [[ -z "$slot" ]]; then
    continue
  fi
  if [[ "$RUN_FIRST_TEN" == "true" && "$count" -ge 10 ]]; then
    break
  fi
  python -m scripts.v10.run_pipeline \
    --slot "$slot" \
    -v \
    --seed 42
  count=$((count + 1))
done < "$LIST_FILE"

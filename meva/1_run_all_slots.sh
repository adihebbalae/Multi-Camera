#!/usr/bin/env bash

LIST_FILE="data/slot_list_from_slot_index.txt"
OUTPUT_DIR="/nas/neurosymbolic/multi-cam-dataset/meva/data"
RUN_FIRST_TEN="true"

mkdir -p "$OUTPUT_DIR/qa_pairs/raw"
export MEVA_OUTPUT_DIR="/nas/neurosymbolic/multi-cam-dataset/meva/data_all_slots/"

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

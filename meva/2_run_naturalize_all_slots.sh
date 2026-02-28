#!/usr/bin/env bash

LIST_FILE="data/slot_list_from_slot_index.txt"
RUN_FIRST_TEN="true"
MODEL="gpt-5.2"
TEMPERATURE="0.3"
VERBOSE="true"

export MEVA_OUTPUT_DIR="/nas/neurosymbolic/multi-cam-dataset/meva/data_all_slots/"
mkdir -p "$MEVA_OUTPUT_DIR/qa_pairs/raw"

TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_DIR="${MEVA_OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/${TIMESTAMP}_2_run_naturalize_all_slots.log") 2>&1

count=0
while IFS= read -r slot; do
  if [[ -z "$slot" ]]; then
    continue
  fi
  if [[ "$RUN_FIRST_TEN" == "true" && "$count" -ge 10 ]]; then
    break
  fi

  input_path="$MEVA_OUTPUT_DIR/qa_pairs/raw/${slot}.raw.json"
  if [[ ! -f "$input_path" ]]; then
    echo "Skipping (missing raw): $input_path"
    continue
  fi

  args=(--input "$input_path" --model "$MODEL" --temperature "$TEMPERATURE" --yes)
  if [[ "$VERBOSE" == "true" ]]; then
    args+=("-v")
  fi
  python -m scripts.v10.naturalize "${args[@]}"

  count=$((count + 1))
done < "$LIST_FILE"

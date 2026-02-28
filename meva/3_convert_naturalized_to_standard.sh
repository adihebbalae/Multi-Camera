#!/usr/bin/env bash

INPUT_DIR="/home/ss99569/code/multi-cam/Multi-Camera/datasets/multi-cam-dataset/meva/data_all_slots/qa_pairs/raw"
OUTPUT_DIR="/home/ss99569/code/multi-cam/Multi-Camera/datasets/multi-cam-dataset/meva/"

TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/${TIMESTAMP}_3_convert_naturalized_to_standard.log") 2>&1

python /home/ss99569/code/multi-cam/Multi-Camera/meva/scripts/convert_naturalized_to_standard.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

#!/usr/bin/env bash

INPUT_DIR="/home/ss99569/code/multi-cam/Multi-Camera/datasets/multi-cam-dataset/meva/data_all_slots/qa_pairs/raw"
OUTPUT_DIR="/home/ss99569/code/multi-cam/Multi-Camera/datasets/multi-cam-dataset/meva/"

python /home/ss99569/code/multi-cam/Multi-Camera/meva/scripts/convert_naturalized_to_standard.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR"

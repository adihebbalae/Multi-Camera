#!/usr/bin/env bash

# Add current directory to path
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -m scripts.v10.run_pipeline \
  --slot "2018-03-11.11-25.school" \
  -v \
  --seed 42 \

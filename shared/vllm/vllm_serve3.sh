#!/bin/bash

MODEL="OpenGVLab/InternVL3_5-14B"
# MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="3"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=1 # Assumes CUDA_VISIBLE_DEVICES is set
PORT=8003
vllm serve $MODEL \
    --tensor-parallel-size $NUM_GPUS \
    --port $PORT \
    --trust-remote-code

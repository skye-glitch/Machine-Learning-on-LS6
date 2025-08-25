#!/bin/bash
LOCAL_RANK=$PMI_RANK

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
if [[ -z "${NODEFILE}" ]]; then
    RANKS=$NODEFILE
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

export MAIN_RANK
export RANKS
export NNODES
export LOCAL_RANK
export GPUS_PER_NODE=3


if [ "$LOCAL_RANK" -eq 0 ]; then
    python generate_config.py
fi

PRELOAD="export CUDA_HOME=/opt/apps/cuda/12.8/;"
PRELOAD+="source .venv/bin/activate;  "

CMD="accelerate launch \
       --config_file multi_config.yaml \
       --machine_rank=$LOCAL_RANK \
       --main_process_ip=$MAIN_RANK --main_process_port=1234 --rdzv_backend static \
       --num_processes $((NNODES * GPUS_PER_NODE)) \
       ./finetune.py "

FULL_CMD="$PRELOAD $CMD"
echo "Training command: $FULL_CMD"

eval $FULL_CMD

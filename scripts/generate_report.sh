#!/bin/bash
set -euo pipefail

MODEL=$2

DATASET=$1 # (ResearchRubrics, RigorousBench, ResearcherBench)

if [[ "$DATASET" == "RigorousBench" ]]; then
    SUBSET=100
else
    SUBSET=-1
fi

BATCH_SIZE=${5:-10}

TURN=${3:-1}
TYPE=${4:-init}

# Create logs directory if it doesn't exist
mkdir -p ./logs

python call_dra.py \
    --round $TURN \
    --type $TYPE \
    --dataset $DATASET \
    --model $MODEL \
    --num_questions $SUBSET \
    --batch_size $BATCH_SIZE | tee ./logs/gen_${MODEL}_${TYPE}_${DATASET}.log
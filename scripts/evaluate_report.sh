#!/bin/bash
set -euo pipefail

REPORT_FILE=$1
DATASET_FILE=$2

EVAL_MODE=${3:-all}
NUM_QUESTIONS=${4:--1}

# Create logs directory if it doesn't exist
mkdir -p ./logs

python evaluate_report.py \
    --report_file "$REPORT_FILE" \
    --dataset_file "$DATASET_FILE" \
    --eval_mode "$EVAL_MODE" \
    --num_questions "$NUM_QUESTIONS" | tee ./logs/eval_$(basename "$REPORT_FILE" .jsonl).log
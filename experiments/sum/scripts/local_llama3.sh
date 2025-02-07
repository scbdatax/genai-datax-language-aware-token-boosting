#!/bin/bash

source ../../../.env

LANG_CODES=("en" "ru" "zh" "ja" "fr" "th" "hi" "ko" "ar")

for lang_code in "${LANG_CODES[@]}"; do
    # without LATB
    python ../gen_response.py \
        --lang_code "$lang_code" \
        --model_path "$LLAMA3_PATH" \
        --data_path "$DATA_PATH" \
        --output_path "../out/local_llama3" \
        --use_latb false \
        --subsample_data 1000 \
        --save_freq 1 \
        --batch_size 5 \
        --max_new_tokens 1000 \
        --temperatue 1.0 \
        --top_p 1.0 \
        --do_sample true \
        --num_return_sequences 1
done

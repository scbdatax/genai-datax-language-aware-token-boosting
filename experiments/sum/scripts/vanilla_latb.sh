#!/bin/bash

source ../../../.env

LANG_CODES=("en" "ru" "zh" "ja" "fr" "th" "hi" "ko" "ar")
ALPHA=5

for lang_code in "${LANG_CODES[@]}"; do
    # with latb
    python ../gen_response.py \
        --lang_code "$lang_code" \
        --model_path "$LLAMA3_PATH" \
        --data_path "$DATA_PATH" \
        --output_path "../out/vanilla_latb" \
        --use_latb true \
        --adaptive false \
        --token_lang_id_method "unicode" \
        --latb_value $ALPHA \
        --subsample_data 1000 \
        --save_freq 1 \
        --batch_size 5 \
        --max_new_tokens 1000 \
        --temperatue 1.0 \
        --top_p 1.0 \
        --do_sample true \
        --num_return_sequences 1
done

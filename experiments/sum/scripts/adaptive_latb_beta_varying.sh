#!/bin/bash

source ../../../.env

LANG_CODES=("en" "ru" "zh" "ja" "fr" "th" "hi" "ko" "ar")
ALPHA=1000
BETA_LIST=(0.3 0.5 0.9)

for BETA in "${BETA_LIST[@]}"; do
    if [ ! -d ./out/adaptive_beta_$BETA/ ]; then
        mkdir -p ../out/adaptive_beta_$BETA/;
    fi
    for lang_code in "${LANG_CODES[@]}"; do
        python ../gen_response.py \
            --lang_code "$lang_code" \
            --model_path "$LLAMA3_PATH" \
            --data_path "$DATA_PATH" \
            --output_path "../out/adaptive_beta_$BETA/" \
            --use_latb true \
            --adaptive true \
            --thresh_diff $BETA \
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
done
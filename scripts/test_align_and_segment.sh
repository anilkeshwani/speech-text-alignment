#!/usr/bin/env bash

if [ ! -f align_and_segment.py ]; then
    echo "Run script from the fairseq/ directory"
    exit 1
fi

PROJECT_ROOT_DIR=".." python ./examples/mms/data_prep/align_and_segment.py \
    --audio "${PROJECT_ROOT_DIR}/data/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav" \
    --text_filepath "${PROJECT_ROOT_DIR}/data/transcript_word_per_line.txt" \
    --lang "en-US" \
    --outdir "${PROJECT_ROOT_DIR}/fairseq_outdir" \
    --uroman "uroman/bin"

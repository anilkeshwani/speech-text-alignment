#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT_DIR="${SPEECH_TEXT_ALIGN_ROOT_DIR:-${HOME}/speech-text-alignment}"
SARDALIGN_DIR="${PROJECT_ROOT_DIR}/sardalign"
JSONL_PATH="/media/scratch/anilkeshwani/towerspeech/LJSpeech-1.1/metadata.jsonl"
LANG='eng'
OUTPUT_DIR="${PROJECT_ROOT_DIR}/tests/output/segment_tokens/"
UROMAN_BINARY="${PROJECT_ROOT_DIR}/submodules/uroman/bin"

rm -rf /home/anilkeshwani/speech-text-alignment/tests/output/segment_tokens

(
    cd "$SARDALIGN_DIR"
    python ./segment_tokens.py \
        --jsonl "$JSONL_PATH" \
        --lang "$LANG" \
        --outdir "$OUTPUT_DIR" \
        --uroman "$UROMAN_BINARY" \
        --transcript-stem-suffix \
        --sample 10
)

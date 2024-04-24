#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT_DIR="${SPEECH_TEXT_ALIGN_ROOT_DIR:-${HOME}/speech-text-alignment}"
SARDALIGN_DIR="${PROJECT_ROOT_DIR}/sardalign"
AUDIO_PATH="${PROJECT_ROOT_DIR}/tests/data/LJ001-0001_SR_16000.wav"
OUTPUT_DIR="${PROJECT_ROOT_DIR}/tests/output/$(basename "${AUDIO_PATH%.*}")"

(
    cd "$SARDALIGN_DIR"
    python align_and_segment.py \
        --audio "$AUDIO_PATH" \
        --text_filepath "${PROJECT_ROOT_DIR}/tests/data/LJ001-0001.txt" \
        --lang 'eng' \
        --outdir "$OUTPUT_DIR" \
        --uroman "${PROJECT_ROOT_DIR}/submodules/uroman/bin"
)

#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT_DIR="/Users/anilkeshwani/Desktop/speech-text-alignment"

cd "${PROJECT_ROOT_DIR}/sardalign"

python align_and_segment.py \
    --audio "${PROJECT_ROOT_DIR}/tests/data/LJ001-0001.wav" \
    --text_filepath "${PROJECT_ROOT_DIR}/tests/data/LJ001-0001.txt" \
    --lang 'eng' \
    --outdir "../tests_output" \
    --uroman "${PROJECT_ROOT_DIR}/submodules/uroman/bin"

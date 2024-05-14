#!/usr/bin/env bash

set -euo pipefail

HAFH='/mnt/scratch-artemis/anilkeshwani' # your "${HOME}" away from "${HOME}"

segment_tokens_py_executable="${HAFH}/speech-text-alignment/sardalign/segment_tokens.py"
jsonl="${HAFH}/data/MLS/mls_english/dev/transcripts.jsonl"
audio_dir="${HAFH}/data/MLS/mls_english/dev/audio"
lang='eng'
output_dir="${HAFH}/tmp/MLS/mls_english/dev/audio_segmented"

python "${segment_tokens_py_executable}" \
    --jsonl "$jsonl" \
    --audio-dir "$audio_dir" \
    --out-dir "$output_dir" \
    --lang "$lang" \
    --sample 10

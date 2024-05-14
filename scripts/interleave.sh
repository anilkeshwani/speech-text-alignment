#!/usr/bin/env bash

set -euo pipefail

HAFH='/mnt/scratch-artemis/anilkeshwani' # your "${HOME}" away from "${HOME}"

interleave_py_executable="${HAFH}/speech-text-alignment/sardalign/interleave.py"
jsonl="${HAFH}/data/MLS/mls_english/dev/transcripts.jsonl"
audio_dir="${HAFH}/data/MLS/mls_english/dev/audio"
lang='eng'
output_dir="${HAFH}/tmp/MLS/mls_english/dev/audio_segmented"

# HuBERT
ckpt_path='/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt'
layer='6'

# K-means
km_path="${HOME}/tmp/hubert_kmeans_test/kmeans_model.joblib"

python "${interleave_py_executable}" \
    --jsonl "$jsonl" \
    --audio-dir "$audio_dir" \
    --out-dir "$output_dir" \
    --lang "$lang" \
    --ckpt-path "$ckpt_path" \
    --layer "$layer" \
    --km-path "$km_path" \
    --sample 10

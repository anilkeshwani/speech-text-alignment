#!/usr/bin/env bash

set -euo pipefail

HAFH='/mnt/scratch-artemis/anilkeshwani' # your "${HOME}" away from "${HOME}"

dump_hubert_feature_py_executable="${HAFH}/speech-text-alignment/sardalign/dump_hubert_feature.py"
jsonl="${HAFH}/data/MLS/mls_english/dev/transcripts.jsonl"
audio_dir="${HAFH}/data/MLS/mls_english/dev/audio"
ckpt_path='/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt'
layer='6'
feat_dir="${HAFH}/tmp/hubert_features_test/"

python "${dump_hubert_feature_py_executable}" \
    --jsonl "${jsonl}" \
    --audio-dir "${audio_dir}" \
    --ckpt-path "${ckpt_path}" \
    --layer "${layer}" \
    --feat-dir "${feat_dir}"

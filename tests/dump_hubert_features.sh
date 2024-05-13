#!/usr/bin/env bash

dump_hubert_feature_py_executable="${HOME}/speech-text-alignment/sardalign/dump_hubert_feature.py"
tsv_dir='/media/scratch/anilkeshwani/data/LJSpeech-1.1/test'
split='metadata'
ckpt_path='/media/scratch/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt'
layer='6'
nshard='1'
rank='0'
feat_dir="${HOME}/tmp/hubert_features_test/"

python "${dump_hubert_feature_py_executable}" \
    "${tsv_dir}" "${split}" \
    "${ckpt_path}" "${layer}" \
    "${nshard}" "${rank}" \
    "${feat_dir}"

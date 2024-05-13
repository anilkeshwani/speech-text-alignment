#!/usr/bin/env bash

learn_kmeans_py_executable="${HOME}/speech-text-alignment/sardalign/learn_kmeans.py"
split='metadata'
nshard='1'
feat_dir="${HOME}/tmp/hubert_features_test/"
n_cluster=500
km_path="${HOME}/tmp/hubert_kmeans_test/kmeans_model.joblib"

python "${learn_kmeans_py_executable}" \
    "${feat_dir}" \
    "${split}" \
    "${nshard}" \
    "${km_path}" \
    "${n_cluster}" \
    --percent -1 # use all data

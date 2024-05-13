#!/usr/bin/env bash

dump_km_label_py_executable="${HOME}/speech-text-alignment/sardalign/dump_km_label.py"
split='metadata'
rank='0'
nshard='1'
feat_dir="${HOME}/tmp/hubert_features_test/"
km_path="${HOME}/tmp/hubert_kmeans_test/kmeans_model.joblib"
lab_dir="${HOME}/tmp/hubert_kmeans_labels/"

python "${dump_km_label_py_executable}" "${feat_dir}" "${split}" "${km_path}" "${nshard}" "${rank}" "${lab_dir}"

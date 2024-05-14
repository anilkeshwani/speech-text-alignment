#!/usr/bin/env python

from pathlib import Path
from pprint import pprint

import sox
from sardalign.utils import ljspeech_id_to_path, read_jsonl
from tqdm import tqdm


k_means_labels_path: Path = Path("/home/anilkeshwani/tmp/hubert_kmeans_labels/metadata_0_1.km")
ljspeech_jsonl: Path = Path("/media/scratch/anilkeshwani/data/LJSpeech-1.1/metadata.jsonl")
ljspeech_wavs_dir: Path = Path("/media/scratch/anilkeshwani/data/LJSpeech-1.1/wavs_16000")

ljspeech_data = read_jsonl(ljspeech_jsonl)
n_samples_s: list[int] = []
for s in tqdm(ljspeech_data):  # surprisingly slow; single-threaded...
    n_samples_s.append(sox.file_info.num_samples(ljspeech_id_to_path(s["ID"], ljspeech_wavs_dir)))
k_means_labels_s = [line.strip() for line in k_means_labels_path.read_text().splitlines()]
n_dsus_s = [len(k_means_labels.split()) for k_means_labels in k_means_labels_s]

if not len(n_dsus_s) == len(n_samples_s):
    raise ValueError("Number of samples not equal across jsonl and k-means labels files")

ds_ratio_s = []

for n_samples, n_dsus in zip(n_samples_s, n_dsus_s):
    ds_ratio_s.append(n_samples / n_dsus)

pprint(f"{ds_ratio_s = }")
pprint(f"{set(ds_ratio_s) = }")

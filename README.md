# Speech-Text Alignment

Scripts to align speech audio with their text transcriptions in time. 

In addition to this README, broader [documentation can be found in docs/](/docs/).

## Setup

### Clone Repository

```bash
git clone git@github.com:anilkeshwani/speech-text-alignment.git && 
    cd speech-text-alignment &&
    git submodule update --init --recursive --progress
```

### Set Up Environment

Ensure the necessary binary requirements are installed:

```bash
apt install sox ffmpeg
```

Install the package and with it all dependencies including useful dependencies for development; specified via "dev" option to `pip install`.

```bash
conda create -n sardalign python=3.10.6 -y &&
    conda activate sardalign &&
    pip install -e .["dev"]
```

> Note: We do not install the _dataclasses_ library as per the [fairseq MMS README](https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/examples/mms/data_prep/README.md) it ships out of the box with Python 3.10.6.

<details>
  <summary>Note: When running on Artemis / Poseidon, ensure support for CUDA is provided.</summary>
  
  At the time of writing, NVIDIA / CUDA drivers were:
  - NVIDIA-SMI: 525.89.02
  - Driver Version: 525.89.02
  - CUDA Version: 12.0
  
</details>

## Featurization with HuBERT

```bash
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
```

## Performing Alignment

Example call:

```bash
HAFH='/mnt/scratch-artemis/anilkeshwani' # $HOME away from $HOME; allows flexible relative paths

.scripts/segment_tokens.py \
    --jsonl "${HAFH}/data/MLS/mls_english/train/head_transcripts_stratified_sample_2702009.jsonl" \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio" \
    --out-dir "${HAFH}/tmp/MLS/mls_english/dev/audio_segmented" \
    --lang 'eng' \
    --sample 10
```

## Generating Interleaved Datasets

```
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
```

## Scripts

### Resample an Audio File (single file; Python)

```
usage: resample_audio_file.py [-h] -f FILE [-o OUTPUT] -t TARGET_SAMPLE_RATE

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Path to input audio file
  -o OUTPUT, --output OUTPUT
                        Path to output resampled audio file
  -t TARGET_SAMPLE_RATE, --target-sample-rate TARGET_SAMPLE_RATE
                        Target sampling rate
```

### Resample Audio Files with Parallel (multiple files; GNU Parallel and FFmpeg CLI)

Place all input files into a directory (`IN_DIR`). Set the arguments:

```bash
IN_DIR='/mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs'
OUT_DIR='/mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs_16000_7'
TARGET_SR=16000
N_JOBS=32
```

Run:

```bash
./resample_audio_files_ffmpeg_parallel.sh
```

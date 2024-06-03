# Scripts

## Featurization with HuBERT

```bash
HAFH='/mnt/scratch-artemis/anilkeshwani' # $HOME away from $HOME

jsonl="${HAFH}/data/MLS/mls_english/dev/transcripts.jsonl"
audio_dir="${HAFH}/data/MLS/mls_english/dev/audio"
ckpt_path='/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt'
layer=6
feat_dir="${HAFH}/tmp/hubert_features_test/"

./scripts/dump_hubert_feature.py \
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

./scripts/segment_tokens.py \
    --jsonl "${HAFH}/data/MLS/mls_english/train/transcripts_stratified_sample_2702009_uroman_existing_files_only.jsonl" \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio" \
    --out-dir "${HAFH}/tmp/MLS/mls_english/train/audio_segmented" \
    --lang 'eng' \
    --head 10
```

## Generating Interleaved Datasets

**WIP call - testing**:

```bash
rm -r /mnt/scratch-artemis/anilkeshwani/tmp/MLS/mls_english/train/audio_segmented

HAFH='/mnt/scratch-artemis/anilkeshwani' # $HOME away from $HOME; allows flexible relative paths

./scripts/generate_interleaved_dataset.py \
    --jsonl "${HAFH}/data/MLS/mls_english/train/transcripts_stratified_sample_2702009_uroman.jsonl" \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio" \
    --out-dir "${HAFH}/tmp/MLS/mls_english/train/audio_segmented" \
    --lang 'eng' \
    --head 10
```


```bash
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

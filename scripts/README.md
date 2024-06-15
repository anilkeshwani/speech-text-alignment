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

## Generating Datasets with Alignments and HuBERT Speech Tokens

Alignment and encoding of audio into HuBERT speech tokens is performed in a single script to reduce overhead. 

```bash
HAFH='/mnt/scratch-artemis/anilkeshwani' && cd "${HAFH}/speech-text-alignment"

./scripts/align_and_hubert_encode.py \
    --jsonl "${HAFH}/data/MLS/mls_english/train/transcripts_stratified_sample_2702009_uroman_shards/transcripts_stratified_sample_2702009_uroman_shard_5.jsonl" \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio"          \
    --lang 'eng' \
    --hubert-ckpt-path '/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt' \
    --layer 22 \
    --km-ckpt-path '/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000'
```

The `--head ${num_lines}` option can be passed to run a test using only the top `num_lines` lines. 

### Generated Interleaved Speech-Text Datasets

```bash
HAFH='/mnt/scratch-artemis/anilkeshwani'

./scripts/generate_interleaved_data.py \
    "${HAFH}/tmp/MLS/mls_english/train/head_transcripts_stratified_sample_2702009_uroman_shard_0_aligned_hubert.jsonl"
```

### MLS EDA

```bash
HAFH='/mnt/scratch-artemis/anilkeshwani'
./scripts/mls/eda.py \
    /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts_stratified_sample_2702009.jsonl \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio" \
    --hist-dir "${HAFH}/speech-text-alignment/docs/assets/"
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

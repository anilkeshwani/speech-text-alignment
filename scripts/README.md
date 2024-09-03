# Core Scripts

Usage instructions and example CLI calls for scripts.

All scripts can be called with a `--help` option

Note: For convenience, on the research lab's server, I have a _home away from home_:

```bash
export HAFH='/mnt/scratch-artemis/anilkeshwani' # $HOME away from $HOME; allows flexible relative paths
```

## Text Preprocessing of Raw Datasets: Tokenization, Normalization and Uromanization

```bash
./scripts/uromanize.py \
    --text-key 'normalized_text' \
    '/mnt/scratch-artemis/anilkeshwani/data/voxpopuli_hf/VoxPopuli.jsonl'
```

This script parallelizes work over processes and assigns the number of available CPUs - 1 to perform preprocessing.

## Generating Datasets with Alignments and HuBERT Speech Tokens

Alignment and encoding of audio into HuBERT speech tokens is performed in a single script to reduce overhead e.g. I/O.

These alignments are at the word or token level.

```bash
./scripts/align_and_hubert_encode.py \
    --jsonl '/mnt/scratch-artemis/anilkeshwani/data/voxpopuli_hf/VoxPopuli_uroman.jsonl' \
    --hubert-ckpt-path '/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt' \
    --layer 22 \
    --km-ckpt-path '/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000'
```

Example for **MLS**, which uses IDs its JSON lines manifest (filelist) not paths.

```bash
./scripts/align_and_hubert_encode.py \
    --jsonl "/mnt/scratch-artemis/anilkeshwani/tmp/mls_english_train_ss_head_200_uroman.jsonl" \
    --ids-not-paths \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio" \
    --lang 'eng' \
    --hubert-ckpt-path '/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt' \
    --layer 22 \
    --km-ckpt-path '/mnt/scratch-artemis/kshitij/clustering/kmeans_model/3datsets_combined_kmeans_5000'
```

The `--head ${num_lines}` option can be passed to run a test using only the top `num_lines` lines.

### Interleaving Speech and Text Data

```bash
HAFH='/mnt/scratch-artemis/anilkeshwani'

./scripts/interleave.py \
    "${HAFH}/tmp/MLS/mls_english/train/head_transcripts_stratified_sample_2702009_uroman_shard_0_aligned_hubert.jsonl"
```

### Extending SentencePiece Tokenizer and Model

```bash
./extend_tinyllama_sentencepiece_dsus.py \
    /mnt/scratch-artemis/anilkeshwani/models/base-hf/TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574/ \
    --output-dir /mnt/scratch-artemis/anilkeshwani/models/base-hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000 \
    --n-dsus 5000
```

### Extending Hugging Face Tokenizer and Model

From a local directory:

```bash
./extend_tinyllama_hf_tokenizer_dsus.py.py \
    --pretrained-model-name-or-path '/mnt/scratch-artemis/anilkeshwani/models/base-hf/TinyLlama-1.1B-intermediate-step-1431k-3T/snapshots/036fa4651240b9a1487f709833b9e4b96b4c1574/' \
    --n-dsus 5000 \
    --output-dir '/mnt/scratch-artemis/anilkeshwani/models/base-hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-5000'
```

Note: When using Hugging Face and referencing a local directory path, this must be the innermost path - i.e. referencing the model and individual revision within the `snapshots` directory (usually containing symlinks pointing to relative paths in an adjacent `blobs` directory).

From a Hugging Face repo name:

```bash
./extend_tinyllama_hf_tokenizer_dsus.py.py \
    --pretrained-model-name-or-path 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' \
    --n-dsus 5000 \
    --output-dir '/mnt/scratch-artemis/anilkeshwani/models/base-hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-5000-reponame'
```

# Auxiliary Scripts

## HuBERT Featurization

```bash
./scripts/dump_hubert_feature.py \
    --jsonl "${HAFH}/data/MLS/mls_english/dev/transcripts.jsonl" \
    --audio-dir "${HAFH}/data/MLS/mls_english/dev/audio" \
    --ckpt-path '/mnt/scratch-artemis/kshitij/clustering/feature_extraction/model/hubert_large_ll60k.pt' \
    --layer 6 \
    --feat-dir "${HAFH}/tmp/hubert_features_test/"
```

## Segmenting Speech Audio by Tokens via Speech-Text Alignment

These alignments are at the word or token level.

```bash
./scripts/segment_tokens.py \
    --jsonl "${HAFH}/data/MLS/mls_english/train/transcripts_stratified_sample_2702009_uroman_existing_files_only.jsonl" \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio" \
    --out-dir "${HAFH}/tmp/MLS/mls_english/train/audio_segmented" \
    --lang 'eng' \
    --head 10
```

## Resample an Audio File

Note that this can be done more simply for multiple files via FFmpeg using the Bash snippet below.

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

# Snippets

## Resample Audio Files in parallel with FFmpeg and GNU Parallel

Replace the input and output directories (in this case `/mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs` and `/mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs_16000_7`) with appropriate paths.

```bash
find '/mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs' \
    -type f \
    -name "*.wav" \
    | parallel \
    -j 32 \
    "ffmpeg -i {} -ar 16000 /mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/wavs_16000_7/{/}"
```

See parallel --help for more details.

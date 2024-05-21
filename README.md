# Speech-Text Alignment

Scripts to align speech audio with their text transcriptions in time. 

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

## Performing Alignment

Example call:

```bash
HAFH='/mnt/scratch-artemis/anilkeshwani' # $HOME away from $HOME; allows flexible relative paths

cd "${HAFH}/speech-text-alignment/sardalign" # enter the package directory

python segment_tokens.py \
    --jsonl "${HAFH}/data/MLS/mls_english/dev/transcripts.jsonl" \
    --audio-dir "${HAFH}/data/MLS/mls_english/dev/audio" \
    --out-dir "${HAFH}/tmp/MLS/mls_english/dev/audio_segmented" \
    --lang 'eng' \
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

#### Benchmarks with different numbers of processes

Benchmarks on Artemis via

```bash
time ./resample_audio_files_ffmpeg_parallel.sh
```

|N_JOBS|real     |user     |sys      |
|------|---------|---------|---------|
|100   |0m39.185s|8m30.834s|6m27.732s|
|64    |0m38.822s|8m31.884s|6m28.664s|
|32    |0m39.157s|8m32.408s|6m26.920s|
|16    |0m52.070s|8m16.316s|6m6.157s |
|2     |6m25.868s|8m25.961s|5m28.323s|

### LJSpeech Scripts

#### Convert Metadata CSV to JSONL - LJSpeech

Converts the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) metadata file from pipe-separated CSV to JSON Lines format retaining all fields:

- ID
- transcript
- normalized transcript
- [optional]: additional `"lang"` field containing the ISO `"en-US"` code

Usage:

```bash
./ljspeech_csv_to_jsonl.py \
    --csv /mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/metadata.csv \
    --jsonl /mnt/scratch-artemis/anilkeshwani/towerspeech/LJSpeech-1.1/metadata.jsonl \
    --add-lang-code
```

#### Create Fairseq-style HuBERT metadata.tsv - LJSpeech

```bash
./jsonl_to_hubert_tsv.py \
    --in-jsonl-path /mnt/scratch-artemis/anilkeshwani/data/LJSpeech-1.1/metadata.jsonl \
    --out-tsv-path /mnt/scratch-artemis/anilkeshwani/data/LJSpeech-1.1/test/metadata.tsv \
    --ljspeech-wavs-dir /media/scratch/anilkeshwani/data/LJSpeech-1.1/wavs_16000
```

### MLS Scripts

#### Convert transcripts.txt to JSON lines 

See `./scripts/mls/transcripts_to_jsonl.py --help`.

Example usage:

```bash
./scripts/mls/transcripts_to_jsonl.py \
    --audio-dir "/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/audio/" \
    --transcripts "/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/transcripts.txt" \
    --output-jsonl "/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/transcripts.jsonl"
```

## To Implement

- Support for k-means clustering via [FAISS clustering](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization) - Motivation: speed. Not a priority if the k-means clustering is a bottlenecked due to computational speed. See also [_Implementing K-Means clustering with FAISS_](https://www.kdnuggets.com/2021/01/k-means-faster-lower-error-scikit-learn.html) and [FAISS k-means snippet](/snippets/faiss_kmeans.py).

## Benchmarking

### Dump HuBERT Features

Dumping HuBERT features for the _dev_ set of English MLS _as is_ (15.75 hrs; split into 3,807 audio files) took ~2:40 on a single GPU on Artemis (NVIDIA RTX A6000; 48GB VRAM) with utilisation well under 100%.

Output of `time ./tests/dump_hubert_features.sh` as of commit [ec1ccd2](https://github.com/anilkeshwani/speech-text-alignment/tree/ec1ccd22c41ba776bbcb76f5cb339a371e48fdce).

```
real    2m37.302s
user    2m21.711s
sys     0m17.245s
```


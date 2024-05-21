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
apt install sox
```

<details>
  <summary>Instal binaries on OSX with Homebrew</summary>
  
  ```bash
  brew install ffmpeg sox
  ```

</details>

> Note: We do not install the _dataclasses_ library as per the [fairseq MMS README](https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/examples/mms/data_prep/README.md) it ships out of the box with Python 3.10.6.

```bash
conda create -n sardalign python=3.10.6 -y &&
    conda activate sardalign &&
    pip install -e .["dev"]
```

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

## To Implement

- Support for k-means clustering via [FAISS clustering](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization) - Motivation: speed. Not a priority if the k-means clustering is a bottlenecked due to computational speed. See also [_Implementing K-Means clustering with FAISS_](https://www.kdnuggets.com/2021/01/k-means-faster-lower-error-scikit-learn.html) and [FAISS k-means snippet](/snippets/faiss_kmeans.py).

## Benchmarking

### Dump HuBERT Features

Dumping HuBERT features for the _dev_ set of English MLS _as is_ (15.75 hrs; split into 3,807 audio files) took ~2:40 on a single GPU on Artemis (NVIDIA RTX A6000; 48GB VRAM) with utilisation well under 100%.

Output of `time ./tests/dump_hubert_features.sh` as of [ec1ccd22c41ba776bbcb76f5cb339a371e48fdce](https://github.com/anilkeshwani/speech-text-alignment/tree/ec1ccd22c41ba776bbcb76f5cb339a371e48fdce).

```
real    2m37.302s
user    2m21.711s
sys     0m17.245s
```

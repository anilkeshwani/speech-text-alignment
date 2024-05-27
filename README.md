# Speech-Text Alignment

Scripts to align speech audio with their text transcriptions in time. 

In addition to this README, more granular [documentation can be found in docs/](/docs/).

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

#### Convert transcripts.txt to JSON lines - MLS

```bash
./scripts/mls/transcripts_to_jsonl.py \
    --audio-dir "/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/audio/" \
    --transcripts "/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts.txt" \
    --head 20000
```

Can optionally pass a `--output-jsonl` to specify where the output will be saved. If not explicitly passed, the JSON lines will be the the same as the txt file with a the suffix amended; additionally with the file name amended to reflect if the `--head` was taken. 

#### Stratified Sample of Transcripts JSON Lines - MLS

```bash
./scripts/mls/stratified_sample.py \
    --transcripts-jsonl /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts.jsonl \
    --sample 0.25
    --shuffle
```

#### Compute Speaker Distribution

```bash
./compute_speaker_distribution_jsonl.py \
    /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts_stratified_sample_0.25.jsonl
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

### uRomanization: Perl w IO vs Python w/o IO

Took $N=1000$ stratified sample of MLS train transcripts (JSON lines) and ran [_snippets/uromanization.py_ as of commit a42b72e](https://github.com/anilkeshwani/speech-text-alignment/blob/a42b72ee6b220df8fc3977feaaa8e49497c60b62/snippets/uromanization.py):

```bash
./snippets/uromanization.py \
    --jsonl /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts_stratified_sample_1000.jsonl \
    --out-dir ../tmp/uroman_benchmarking/ \
    --lang eng \
    --uroman-path ./submodules/uroman/bin/ \
    --uroman-data-dir ./submodules/uroman/data/
```

- Perl implementation with significant IO took ~10:28 to complete
- Python implementation with no IO took ~00:03 to complete
- Speedup of ~210x
- Outputs from each run (written to disk as JSON lines files) containing the uromanized versions of the input token lists were diffed and were found to be identical:

```bash
diff -s uromanized_perl.jsonl uromanized_python.jsonl
# Files uromanized_perl.jsonl and uromanized_python.jsonl are identical
```

Sample of the output:

```json
{"ID": "9923_9259_000000", "transcript": "in fact agreed as it was intended they should on this one point to wit that mr incoul was the most devoted of husbands and such apparently he was if maida had any lingering doubts as to the real reason of their return to paris little by little they faded", "uroman_tokens": ["i n", "f a c t", "a g r e e d", "a s", "i t", "w a s", "i n t e n d e d", "t h e y", "s h o u l d", "o n", "t h i s", "o n e", "p o i n t", "t o", "w i t", "t h a t", "m r", "i n c o u l", "w a s", "t h e", "m o s t", "d e v o t e d", "o f", "h u s b a n d s", "a n d", "s u c h", "a p p a r e n t l y", "h e", "w a s", "i f", "m a i d a", "h a d", "a n y", "l i n g e r i n g", "d o u b t s", "a s", "t o", "t h e", "r e a l", "r e a s o n", "o f", "t h e i r", "r e t u r n", "t o", "p a r i s", "l i t t l e", "b y", "l i t t l e", "t h e y", "f a d e d"]}
{"ID": "11636_11870_000000", "transcript": "the four travelers walked with ease through the trees until they came to the farther edge of the wood then to their surprise they found before them a high wall which seemed to be made of white china", "uroman_tokens": ["t h e", "f o u r", "t r a v e l e r s", "w a l k e d", "w i t h", "e a s e", "t h r o u g h", "t h e", "t r e e s", "u n t i l", "t h e y", "c a m e", "t o", "t h e", "f a r t h e r", "e d g e", "o f", "t h e", "w o o d", "t h e n", "t o", "t h e i r", "s u r p r i s e", "t h e y", "f o u n d", "b e f o r e", "t h e m", "a", "h i g h", "w a l l", "w h i c h", "s e e m e d", "t o", "b e", "m a d e", "o f", "w h i t e", "c h i n a"]}
{"ID": "3780_3018_000000", "transcript": "her soul rose in revolt and it grew hourly more difficult for her to renounce this pleasure she must pawn her dress the only decent dress she had left", "uroman_tokens": ["h e r", "s o u l", "r o s e", "i n", "r e v o l t", "a n d", "i t", "g r e w", "h o u r l y", "m o r e", "d i f f i c u l t", "f o r", "h e r", "t o", "r e n o u n c e", "t h i s", "p l e a s u r e", "s h e", "m u s t", "p a w n", "h e r", "d r e s s", "t h e", "o n l y", "d e c e n t", "d r e s s", "s h e", "h a d", "l e f t"]}
{"ID": "9732_11612_000000", "transcript": "and a mystery at a houseparty well whoever may stand proven as the mother of invention curiosity you know just as well as i do is the father of a great many very sprightly little adventures", "uroman_tokens": ["a n d", "a", "m y s t e r y", "a t", "a", "h o u s e p a r t y", "w e l l", "w h o e v e r", "m a y", "s t a n d", "p r o v e n", "a s", "t h e", "m o t h e r", "o f", "i n v e n t i o n", "c u r i o s i t y", "y o u", "k n o w", "j u s t", "a s", "w e l l", "a s", "i", "d o", "i s", "t h e", "f a t h e r", "o f", "a", "g r e a t", "m a n y", "v e r y", "s p r i g h t l y", "l i t t l e", "a d v e n t u r e s"]}
{"ID": "11783_11916_000000", "transcript": "lift up thine eyes unto the hills a pure and fragrant breath is wafted from their purple tops the heaven-sent breath of faith lift up thine eyes unto the hills beyond their shadowy slope", "uroman_tokens": ["l i f t", "u p", "t h i n e", "e y e s", "u n t o", "t h e", "h i l l s", "a", "p u r e", "a n d", "f r a g r a n t", "b r e a t h", "i s", "w a f t e d", "f r o m", "t h e i r", "p u r p l e", "t o p s", "t h e", "h e a v e n s e n t", "b r e a t h", "o f", "f a i t h", "l i f t", "u p", "t h i n e", "e y e s", "u n t o", "t h e", "h i l l s", "b e y o n d", "t h e i r", "s h a d o w y", "s l o p e"]}
{"ID": "3582_12556_000000", "transcript": "perhaps i shall a good deal gratify my own vanity indeed i scarce ever heard or saw the introductory words without vanity i may say etc but some vain thing immediately followed", "uroman_tokens": ["p e r h a p s", "i", "s h a l l", "a", "g o o d", "d e a l", "g r a t i f y", "m y", "o w n", "v a n i t y", "i n d e e d", "i", "s c a r c e", "e v e r", "h e a r d", "o r", "s a w", "t h e", "i n t r o d u c t o r y", "w o r d s", "w i t h o u t", "v a n i t y", "i", "m a y", "s a y", "e t c", "b u t", "s o m e", "v a i n", "t h i n g", "i m m e d i a t e l y", "f o l l o w e d"]}
{"ID": "9337_5402_000000", "transcript": "now above where the tardiest color flares a moment yet one point of light now two now three are set to form the starry stairs and in her fire-fly crown queen night on velvet slippered feet comes softly down", "uroman_tokens": ["n o w", "a b o v e", "w h e r e", "t h e", "t a r d i e s t", "c o l o r", "f l a r e s", "a", "m o m e n t", "y e t", "o n e", "p o i n t", "o f", "l i g h t", "n o w", "t w o", "n o w", "t h r e e", "a r e", "s e t", "t o", "f o r m", "t h e", "s t a r r y", "s t a i r s", "a n d", "i n", "h e r", "f i r e f l y", "c r o w n", "q u e e n", "n i g h t", "o n", "v e l v e t", "s l i p p e r e d", "f e e t", "c o m e s", "s o f t l y", "d o w n"]}
{"ID": "12707_10961_000000", "transcript": "this will happen where the beach is very sloping as is usual where the sea is shallow for then the velocity of the low flat earth wave is such that it slips as it were from under the undulation in the fluid above", "uroman_tokens": ["t h i s", "w i l l", "h a p p e n", "w h e r e", "t h e", "b e a c h", "i s", "v e r y", "s l o p i n g", "a s", "i s", "u s u a l", "w h e r e", "t h e", "s e a", "i s", "s h a l l o w", "f o r", "t h e n", "t h e", "v e l o c i t y", "o f", "t h e", "l o w", "f l a t", "e a r t h", "w a v e", "i s", "s u c h", "t h a t", "i t", "s l i p s", "a s", "i t", "w e r e", "f r o m", "u n d e r", "t h e", "u n d u l a t i o n", "i n", "t h e", "f l u i d", "a b o v e"]}
{"ID": "7828_6974_000000", "transcript": "and leaping to her feet ran quickly to the door where she shot a wooden bolt into its socket thus securing them from interference from without then she returned to the center of the room and spoke rapidly to the englishman gesturing occasionally toward the body of the slain man", "uroman_tokens": ["a n d", "l e a p i n g", "t o", "h e r", "f e e t", "r a n", "q u i c k l y", "t o", "t h e", "d o o r", "w h e r e", "s h e", "s h o t", "a", "w o o d e n", "b o l t", "i n t o", "i t s", "s o c k e t", "t h u s", "s e c u r i n g", "t h e m", "f r o m", "i n t e r f e r e n c e", "f r o m", "w i t h o u t", "t h e n", "s h e", "r e t u r n e d", "t o", "t h e", "c e n t e r", "o f", "t h e", "r o o m", "a n d", "s p o k e", "r a p i d l y", "t o", "t h e", "e n g l i s h m a n", "g e s t u r i n g", "o c c a s i o n a l l y", "t o w a r d", "t h e", "b o d y", "o f", "t h e", "s l a i n", "m a n"]}
{"ID": "11369_11135_000000", "transcript": "which the lictor remarked you are now on the road to death and not a single cash can you carry away with you repair this bridge and benefit the public", "uroman_tokens": ["w h i c h", "t h e", "l i c t o r", "r e m a r k e d", "y o u", "a r e", "n o w", "o n", "t h e", "r o a d", "t o", "d e a t h", "a n d", "n o t", "a", "s i n g l e", "c a s h", "c a n", "y o u", "c a r r y", "a w a y", "w i t h", "y o u", "r e p a i r", "t h i s", "b r i d g e", "a n d", "b e n e f i t", "t h e", "p u b l i c"]}
```

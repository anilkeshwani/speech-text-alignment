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

## Data Preparation: MLS

The [Multilingual LibriSpeech (MLS) dataset](https://arxiv.org/pdf/2012.03411) can be [downloaded from OpenSLR](https://www.openslr.org/94/) in subsets by language. 

The English subset is available as the original _flac_ audio files (2.4TB) or compressed _opus_ files (651GB). The GNU zipped tape-archived original flacs can be downloaded (in ~30 hours at ~25MB/s) with `wget` via:

```bash
wget https://dl.fbaipublicfiles.com/mls/mls_english.tar.gz # takes ~30 hours at ~25MB/s download speed
```

Since the archive is 2.4TB, a manifest of files contained can be obtained via the following command, which takes 3+ hours to complete:

```bash
tar --list -f mls_english.tar.gz > archive_list.txt # run inside a tmux session; takes 3+ hours to complete
```

Audio files (flacs) in MLS are organised into subdirectories for 
1. language
2. split
3. (audio)
4. speaker ID
5. book ID 

in the above order. 

Audio filenames are of the format `f"{speaker_id}_{book_id}_audio_id.flac"`.

For example:

```
mls_english/dev/audio/1982/1551/1982_1551_000037.flac
```

This results in a file containing 5,524,525 lines - this includes entries for directories. 

```bash
wc archive_list.txt
#   5524525   5524525 315287688 archive_list.txt
```

We can filter for files, which should result in 5,503,864 (i.e. we drop 20,661 lines)
```bash
grep "\." archive_list.txt > archive_list_files.txt
wc archive_list_files.txt
#  5503864   5503864 314564409 archive_list_files.txt
```

...and subsequently count the number of files of each type to validate the data:

```bash
./scripts/mls/count_file_types.py "/mnt/scratch-artemis/anilkeshwani/data/MLS/archive_list_files.txt"
# {'flac': 5503857, 'txt': 7}
```

We can separate out the audio (flac) files:

```bash
grep "\.flac" archive_list.txt > archive_list_flac_files.txt
wc archive_list_flac_files.txt #   5503857   5503857 314564195 archive_list_flac_files.txt
```

We can separate out the text files, which contain metadata and file manifests and unpack these:

```bash
grep "\.txt" archive_list.txt > archive_list_text_files.txt
wc archive_list_text_files.txt # 7   7 214 archive_list_text_files.txt
tar -xvf mls_english.tar.gz --files-from archive_list_text_files.txt
# mls_english/metainfo.txt
# mls_english/dev/transcripts.txt
# mls_english/dev/segments.txt
# mls_english/test/segments.txt
# mls_english/test/transcripts.txt
# mls_english/train/transcripts.txt
# mls_english/train/segments.txt
```

The metainfo.txt file shows metadata about the whole dataset and specifically the following fields:

- speaker
- gender
- partition
- minutes
- book id
- title
- chapter

```
 SPEAKER   |   GENDER   | PARTITION  |  MINUTES   |  BOOK ID   |             TITLE              |            CHAPTER            
  10232    |     M      |   secret   |   17.148   |   10057    | Expression of the Emotions in Man and Animals | Ch. II: General Principles of Expression, continued
   9508    |     F      |   secret   |   9.347    |   10105    | Stephen: A Soldier of the Cross | Good Tidings Out of the Desert
   9508    |     F      |   secret   |   8.123    |   12959    |         Vanished Hand          | CHAPTER II - WHAT WAS WRITTEN 
  10375    |     M      |   secret   |   10.803   |   10173    | Dutch Fairy Tales for Young Folks |   SANTA KLAAS AND BLACK PETE  
  10375    |     M      |   secret   |   6.764    |   10244    | Grimm's Fairy Tales - Retold in One-Syllable Words |          Hans in Luck         
  10655    |     M      |   secret   |   17.841   |   10173    | Dutch Fairy Tales for Young Folks | THE FARM THAT RAN AWAY AND CAME BACK
  10454    |     M      |   secret   |   1.782    |   10203    |             Verses             | The Cradle Tomb in Westminster Abbey
  10454    |     M      |   secret   |   2.316    |   10203    |             Verses             |          Commissioned         
  10454    |     M      |   secret   |   2.362    |   10335    | Grand'ther Baldwin's Thanksgiving, with Other Ballads and Poems |         Friar Anselmo         
```

MLS dataset splits ({train, dev, test}) are split into subdirectories, each containing transcripts.txt and segments.txt files. 

Each transcripts.txt contains:
- file identifier
- transcript

```
4800_10003_000000       oh my dear you must see him he expects you she answered almost gayly the procession of three moved down the long room towards a door phyllis's hand guiding the wheel-chair
4800_10003_000001       it was quite as much fun well almost as much hearing her as it would have been to play all of the contented and otherwise elderly people who inhabited the boarding-house with phyllis
4800_10003_000002       the man stole out and shut the door softly phyllis herself rose and went toward the window and busied herself in braiding up her hair there was almost silence in the room for a few minutes
4800_10003_000003       has it said phyllis it was like mrs harrington that careful planning of even where she should be put is mr harrington in his day-room now
4800_10003_000004       and she insisted that the pink paper stay on the electric lights after about a week of this phyllis suddenly remembered that she had not been selfish at all yet
4800_10003_000005       surprise i-i'm glad you like it said his wife shyly still backing away of course he'd like it said mrs de guenther's kind staccato voice behind him kiss your husband and tell him he's welcome home phyllis child
4800_10003_000006       you have everything that could be asked even to a certain cheerfulness of outlook which poor angela naturally lacks in a measure but-but what about me asked phyllis braithwaite a little piteously in answer to all this
4800_10003_000007       i've bought myself lots of things she defended herself most of this is really for me and-i can't help being good to him it's only common humanity
4800_10003_000008       his little crumpled black muzzle on the pillow close to allan's contented sleeping face she felt as if she wanted to cry the pathetic lack of interests which made the coming of a new little dog such an event
4800_10003_000009       she wondered afterwards how she could have spoken with that hard serenity how she could have gone steadily on with story after story poem after poem till allan's grip on her hands relaxed and he fell into a heavy tired sleep
```

Each segments.txt contains:

- file identifier
- URL to access file (as mp3)
- timestamp for segment start in seconds
- timestamp for segment end in seconds

```
4800_10003_000000       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_05_widdemer_64kb.mp3      401.76  417.57
4800_10003_000001       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_03_widdemer_64kb.mp3      238.58  252.82
4800_10003_000002       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_07_widdemer_64kb.mp3      1160.28 1174.41
4800_10003_000003       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_07_widdemer_64kb.mp3      599.02  612.41
4800_10003_000004       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_08_widdemer_64kb.mp3      363.34  376.76
4800_10003_000005       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_10_widdemer_64kb.mp3      993.58  1013.33
4800_10003_000006       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_04_widdemer_64kb.mp3      224.67  243.66
4800_10003_000007       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_09_widdemer_64kb.mp3      568.02  580.43
4800_10003_000008       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_12_widdemer_64kb.mp3      269.04  285.39
4800_10003_000009       http://www.archive.org/download/rose_garden_husband_1508_librivox/rose_garden_husband_14_widdemer_64kb.mp3      240.72  258.77
```

We can check that the number of files in e.g. the train split's transcripts file matches the total number in the file list we created with `tar` as follows:

```bash
wc -l mls_english/train/transcripts.txt
# 10808037
```

10,808,037
5,496,281

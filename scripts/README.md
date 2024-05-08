# Scripts

## Resample an Audio File (single file; Python)

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

## Resample Audio Files with Parallel (multiple files; GNU Parallel and FFmpeg CLI)

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

### Benchmarks with different numbers of processes

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

## LJSpeech: Convert Metadata CSV to JSONL

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


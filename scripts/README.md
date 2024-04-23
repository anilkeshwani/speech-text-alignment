# Scripts

## Resample an Audio File

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


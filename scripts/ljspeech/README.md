# LJSpeech Scripts

## Convert Metadata CSV to JSONL - LJSpeech

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

## Create Fairseq-style HuBERT metadata.tsv - LJSpeech

```bash
./jsonl_to_hubert_tsv.py \
    --in-jsonl-path /mnt/scratch-artemis/anilkeshwani/data/LJSpeech-1.1/metadata.jsonl \
    --out-tsv-path /mnt/scratch-artemis/anilkeshwani/data/LJSpeech-1.1/test/metadata.tsv \
    --ljspeech-wavs-dir /media/scratch/anilkeshwani/data/LJSpeech-1.1/wavs_16000
```

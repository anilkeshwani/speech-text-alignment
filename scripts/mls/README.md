# MLS Scripts

## Convert transcripts.txt to JSON lines - MLS

```bash
./scripts/mls/transcripts_to_jsonl.py \
    --audio-dir "/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/audio/" \
    --transcripts "/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts.txt" \
    --head 20000
```

Can optionally pass a `--output-jsonl` to specify where the output will be saved. If not explicitly passed, the JSON lines will be the the same as the txt file with a the suffix amended; additionally with the file name amended to reflect if the `--head` was taken.

## Stratified Sample of Transcripts JSON Lines - MLS

```bash
./scripts/mls/stratified_sample.py \
    --transcripts-jsonl /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts.jsonl \
    --sample 0.25
    --shuffle
```

## Compute Speaker Distribution

```bash
./compute_speaker_distribution_jsonl.py \
    /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts_stratified_sample_0.25.jsonl
```

### MLS EDA

```bash
./scripts/mls/eda.py \
    /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts_stratified_sample_2702009.jsonl \
    --audio-dir "${HAFH}/data/MLS/mls_english/train/audio" \
    --hist-dir "${HAFH}/speech-text-alignment/docs/assets/"
```

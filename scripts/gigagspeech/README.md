# GigaSpeech Scripts

Prefer Hugging Face over raw GigaSpeech.

## GigaSpeech Direct from Hugging Face datasets

Subsections here relate to processes if you download GigaSpeech via Hugging Face datasets, [datasets/speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech), i.e. by filling out the HF request form and running `load_dataset("speechcolab/gigaspeech", "l", trust_remote_code=True)` alternatively specifying the `"xl"` subset. Note: Be sure to have librosa, soundfile and any other dependencies specified if going this route.

HF conveniently also performs all preprocessing, in particular:
- segmenting audio into short audio clips
- conversion has already been performed from Opus to WAV format

### Preprocess Hugging Face metadata

Preprocessing of the GigaSpeech metadata is necessary to:
- replace GigaSpeech punctuation tokens with their natural language punctuation marks
- Truecasing of the raw text provided in the GigaSpeech metadata
- removal of any segments containing garbage utterances - as indicated by GigaSpeech's garbage utterance tags

This tasks are performed by _scripts/gigagspeech/preprocess_hf_gigaspeech.py_.

**See `./preprocess_hf_gigaspeech.py --help` for usage details.**

Note: At the time of writing, truecaser did not provide deterministic outputs. Running the script

## GigaSpeech Direct from SpeechColab

Subsections here relate to processes if you download GigaSpeech directly from the publishers via [SpeechColab/GigaSpeech](https://github.com/SpeechColab/GigaSpeech), i.e. by filling out the request form in the README, cloning the repo, placing a secret in `GigaSpeech/SAFEBOX` (received via email after request approval) and downloading the dataset via `bash GigaSpeech/utils/download_gigaspeech.sh`.

### Segment GigaSpeech Audios and Convert GigaSpeech.json to JSON lines

Raw GigaSpeech comes as long-form audio (e.g. minutes) in [Opus format](https://en.wikipedia.org/wiki/Opus_(audio_format)) and needs segmenting into chunks each with duration of several seconds. This script performs:
- segmentation based on the `segments` dictionary provided in the metadata file for each (long, unprocessed) audio file
- conversion to WAV format
- writing of a JSON lines metadata file

```bash
./segment_gigaspeech.py \
    --json '/mnt/scratch-artemis/anilkeshwani/data/GigaSpeech/data/GigaSpeech.json' \
    --data-dir '/mnt/scratch-artemis/anilkeshwani/data/GigaSpeech/data' \
    --output-dir '/mnt/scratch-artemis/anilkeshwani/tmp/gigapeech-processed-test'  \
    --head 3
```

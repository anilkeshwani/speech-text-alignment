# Snippets

Snippets and throwaway scripts kept for reference, reproducibility or as reminders.

## Join Alignment Data to MLS SpeechTokenizer Dataset

Joins the time alignment columns from `anilkeshwani/mls-hubert_large_ll60k-layer_22` onto the MLS dataset encoded as SpeechTokenizer RVQ layer/level 0 (RVQ_0) tokens, which are the "semantic" ones.

All the arguments are at the top of the snippet.

Follow this script by uploading via the HF CLI by running something like:

```bash
huggingface-cli upload anilkeshwani/mls-speechtokenizer-RVQ-0-aligned ./train train --repo-type dataset
```

## Check SentencePiece Tokenizer - Prints SentencePiece length

```bash
./scripts/check_sentencepiece_tokenizer.py \
    'models/base-hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-5000/tokenizer.model'
```

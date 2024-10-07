# Evaluation

## ASR

```bash
./scripts/evaluation/generation.py \
    "/mnt/scratch-artemis/anilkeshwani/experiments/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000/hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000-MLS-iter_0010500" \
    --test-jsonl "/tmp/dummy.jsonl" \
    --text-key "text" \
    --output-jsonl '/tmp/dummy_out.jsonl' \
    --prompt-template 'basic'
```

```bash
./scripts/evaluation/generation.py \
    "/mnt/scratch-artemis/anilkeshwani/experiments/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000/hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000-MLS-iter_0010500" \
    --test-jsonl "/tmp/dummy.jsonl" \
    --text-key "text" \
    --output-jsonl '/tmp/dummy_out.jsonl' \
    --prompt-template 'repeat_this_modality_switch'
```

```bash
./scripts/evaluation/generation.py \
    "/mnt/scratch-artemis/anilkeshwani/experiments/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000/hf/TinyLlama-1.1B-intermediate-step-1431k-3T-extended-sentencepiece-5000-MLS-iter_0010500" \
    --test-jsonl "/mnt/scratch-artemis/anilkeshwani/data/voxpopuli_hf/VoxPopuli_uroman_aligned_hubert.jsonl" \
    --text-key "text" \
    --output-jsonl '' \
    --prompt-template 'basic'
```

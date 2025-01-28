# Tower Speech

- **add_new_vocab_spm.py**: Add new tokens to the SentencePiece model directly. Script taken directly from [sentencepiece/python/add_new_vocab.ipynb](https://github.com/google/sentencepiece/blob/b0a15434fa991c0202b6d05e61c6e578138fac3d/python/add_new_vocab.ipynb)
- **check_tokenizer_lens.py**: Check length of tokenizers - original and extended after running _add_new_vocab_spm.py_
- **download_hf_tinyllama.py**: Download the TinyLlama model and tokenizer with the vanilla HF snippet. Uses default Hugging Face [cache directory](https://github.com/huggingface/huggingface_hub/blob/e58b5cc61cbc8dec52b4335e872db5f44249d06b/src/huggingface_hub/constants.py#L115) (symlink this directory)
- **extend_vocab_tinyllama.py**: Add new tokens via the Hugging Face tokenizer wrapper of the Llama SentencePiece tokenizer

## Training Pipeline

### Preprocessing

Arguments taken directly from hard-coded Bash script at [preprocess_data.sh](https://github.com/deep-spin/multilinguality_megatron/blob/0563f7e9254c1f06859958606f9452d6112efc24/preprocess_data.sh) at project root directory. 

```bash
python tools/preprocess_data.py \
    --input='/mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/train/transcripts_stratified_sample_2702009_uroman_aligned_hubert_interleaved.jsonl' \
    --output_prefix='/mnt/scratch-artemis/anilkeshwani/tmp/tower/data-preproc-test' \
    --tokenizer_type=SentencePieceTokenizer \
    --vocab_file='/mnt/scratch-artemis/anilkeshwani/huggingface/tower/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/tokenizer_extended.model' \
    --chunk_size=32 \
    --workers=16 \
    --no_new_tokens \
    --append_eod
```

### Model Extension

Run model extension script provided by Ben Peters from the _tools/vocab-extension_ directory of the Megatron codebase:

```bash
./extend_model.py \
    --model-path 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' \
    --out-dir '/mnt/scratch-artemis/anilkeshwani/models/tmp' \
    --new-tokenizer '/mnt/scratch-artemis/anilkeshwani/models/models--TinyLlama--TinyLlama-1.1B-intermediate-step-1431k-3T/tokenizer_extended.model' \
    --init-strategy 'mean' \
    --pad-multiple 64
```

### Checkpoint Conversion

> Note: On verifying the directory produced in the above _Model Extension_ step contained all the expected outputs, the directory was renamed from _tmp_ to the argument passed to `--out` below.

```bash
rm -r '/mnt/scratch-artemis/anilkeshwani/models/tmp' &&
python weights_conversion/hf_to_megatron.py llama2 \
    --size=1 \
    --out='/mnt/scratch-artemis/anilkeshwani/models/tmp' \
    --cache-dir='/mnt/scratch-artemis/anilkeshwani/models/TinyLlamaBaseExtendedInitialRun' \
    --model-path='/mnt/scratch-artemis/anilkeshwani/models/TinyLlamaBaseExtendedInitialRun'
```

### Model Sharding

```bash
python tools/checkpoint_util.py \    
    --target_tensor_parallel_size 2 \    
    --target_pipeline_parallel_size 1 \    
    --load_dir '/mnt/scratch-artemis/anilkeshwani/models/TinyLlamaBaseExtendedMegatronInitial' \    
    --save_dir '/mnt/scratch-artemis/anilkeshwani/models/TinyLlamaBaseExtendedMegatronInitialSharded/' \    
    --model_type llama2 \    
    --true_vocab_size 37056 \    
    --bf16
```

### Training

See amendments to continue_pretraining.sh. 

### Model Conversion via deploy.sh

```bash
bash deploy.sh -p '/mnt/scratch-artemis/anilkeshwani/models/TinyLlamaBaseExtendedMegatronCheckpoints/' \
    -v 37056 \
    -t 'llama' \
    -m 'InitialRunInterleaved' \
    -f '/mnt/scratch-artemis/anilkeshwani/models/TinyLlamaBaseExtendedInitialRun/tokenizer.model'
```

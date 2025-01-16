---
dataset_info:
- config_name: all
  features:
  - name: file
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: text
    dtype: string
  - name: speaker_id
    dtype: int64
  - name: chapter_id
    dtype: int64
  - name: id
    dtype: string
  splits:
  - name: test.clean
    num_bytes: 368417843
    num_examples: 2620
  - name: test.other
    num_bytes: 353195642
    num_examples: 2939
  download_size: 675421827
  dataset_size: 721613485
- config_name: clean
  features:
  - name: file
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: text
    dtype: string
  - name: speaker_id
    dtype: int64
  - name: chapter_id
    dtype: int64
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 368417843
    num_examples: 2620
  download_size: 346663984
  dataset_size: 368417843
- config_name: other
  features:
  - name: file
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: text
    dtype: string
  - name: speaker_id
    dtype: int64
  - name: chapter_id
    dtype: int64
  - name: id
    dtype: string
  splits:
  - name: test
    num_bytes: 353195642
    num_examples: 2939
  download_size: 328757843
  dataset_size: 353195642
---

Like [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) but the test set only.

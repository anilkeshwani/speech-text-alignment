# Codebase Audit Report — `speech-text-alignment`

## Overview

This project implements a speech-to-text alignment pipeline using CTC forced alignment (MMS/Wav2Vec2), HuBERT feature extraction, k-means quantization (DSUs), and multimodal text-speech interleaving. The core library lives in `sardalign/`, with data-pipeline scripts in `scripts/`. The findings below are grouped by topic.

---

## 1. Critical Bugs: Broken Return-Value Unpacking for `get_alignments`

**Files:** `scripts/auxiliary/align_and_segment.py:34`, `scripts/auxiliary/segment_tokens.py:85`

The function `get_alignments` returns **three** values — `(segments, stride_ms, waveform)`:

```python
# sardalign/align.py:65
def get_alignments(...) -> tuple[list[Segment], float, Tensor]:
    ...
    return segments, stride_ms, waveform
```

However, two calling scripts unpack only **two** values, which will raise `ValueError: too many values to unpack`:

```python
# align_and_segment.py:34 – has device but wrong unpacking
segments, stride = get_alignments(args.audio_filepath, tokens, model, dictionary, args.use_star, device)

# segment_tokens.py:85 – also missing the required `device` argument (TypeError) AND wrong unpacking
segments, stride = get_alignments(audio_path, tokens, model, dictionary, args.use_star)
```

`scripts/align_and_hubert_encode.py:151–153` shows the correct calling convention:
```python
segments, stride_ms, wave = get_alignments(
    audio_path, uroman_tokens, mms_aligner_model, mms_aligner_dict, args.use_star, device
)
```

**Impact:** Both `align_and_segment.py` and `segment_tokens.py` crash immediately at runtime every time they reach the alignment step. `segment_tokens.py` will also crash earlier (before alignment) with a `TypeError` due to the missing `device` argument.

---

## 2. Critical Bug: `unflatten_list` Incorrect Slice in Tutorial

**File:** `tutorial/align.py:74`

```python
def unflatten_list(flattened_list: list[Any], lengths: list[int]) -> list[list[Any]]:
    ...
    for length in lengths:
        nested.append(flattened_list[i : i + 1])  # BUG: should be i + length
        i += length
```

`i + 1` is hardcoded instead of `i + length`. Every reconstructed "word span" list will contain exactly one element, discarding all but the first character span per word. The validation check at the top (`len(flattened_list) != sum(lengths)`) would still pass if `sum(lengths) > 0`, masking the error silently. The downstream `segment_audio_by_spans` call will produce incorrect per-word segments — the end of each word segment will be pegged to the first character frame only.

---

## 3. Critical Bug: `generate_emissions` Ignores `check_sampling_rate` Parameter

**File:** `sardalign/align.py:33`

```python
def generate_emissions(
    model, audio_file, device, check_sampling_rate: int | None = SAMPLING_FREQ
) -> ...:
    waveform, sr = torchaudio.load(audio_file)
    if check_sampling_rate and (sr != SAMPLING_FREQ):          # uses constant, not parameter
        raise RuntimeError(f"Expected sampling rate {check_sampling_rate:,}, ...")
```

The guard condition always compares against the module-level constant `SAMPLING_FREQ` (16,000 Hz), regardless of what value was passed as `check_sampling_rate`. The parameter is only referenced in the error message string, never in the actual comparison. Passing a different expected rate (e.g. `check_sampling_rate=8000`) silently has no effect on the check performed.

---

## 4. Critical Bug: `uromanize.py` Calls `Path` Object as a Function

**File:** `scripts/uromanize.py:81`

```python
if not args.output_jsonl.parent.exists():
    args.output_jsonl.parent(parents=True, exist_ok=True)   # TypeError
```

`args.output_jsonl.parent` is a `pathlib.Path`. Calling it as `parent(parents=True, ...)` invokes `Path.__call__` which does not exist — this raises `TypeError: 'PosixPath' object is not callable`. The correct call is `.mkdir(parents=True, exist_ok=True)`.

---

## 5. Medium Bug: `use_star` Causes Mismatched Array Lengths in Output

**File:** `scripts/align_and_hubert_encode.py:129–163`

When `--use-star` is passed, the star token is prepended to `tokens_s`, `norm_tokens_s`, and `uroman_tokens_s` in the script's local variables, but the underlying `sample` dict read from the input JSONL is **not modified** — it still holds the original (star-less) `TOKENIZED_KEY` list. Since `spans` is computed from the star-prepended uroman tokens, `span_times` contains **n+1** entries, while `sample[TOKENIZED_KEY]` has **n** entries:

```python
# sample[TOKENIZED_KEY] has n elements (unchanged)
sample |= {
    ALIGNMENT_START_TIME_KEY: [span[0] for span in span_times],  # n+1 entries
    ALIGNMENT_END_TIME_KEY:   [span[1] for span in span_times],  # n+1 entries
}
```

The length guard `if len(tokens) != len(spans)` does not catch this because it compares the local (star-prepended) `tokens` variable, not `sample[TOKENIZED_KEY]`. Any downstream code that assumes `len(TOKENIZED_KEY) == len(ALIGNMENT_START_TIME_KEY)` will be off-by-one.

---

## 6. Medium Bug: `get_spans` Over-Strict Trailing-Blank Assertion

**File:** `sardalign/utils/align.py:158–161`

```python
for seg_idx, seg in enumerate(segments):
    if tokens_idx == len(tokens):
        assert seg_idx == len(segments) - 1    # fails if >1 trailing blank
        assert seg.label == "<blank>"
        continue
```

Once all tokens are consumed, the assertion `seg_idx == len(segments) - 1` requires that at most **one** trailing blank segment follows the last token. CTC decoders can produce multiple consecutive blanks at the end of a sequence, so this assertion will spuriously raise `AssertionError` for valid alignments where the final frames are all blank.

---

## 7. Medium Bug: `shard_jsonl` Filename Padding Breaks When `n_shards` Is Not Given

**File:** `sardalign/utils/__init__.py:133`

```python
shard_jsonl = shard_dir / jsonl.with_stem(
    f"{jsonl.stem}_shard_{i:0{len(str(n_shards))}}"
).name
```

When the caller specifies `shard_size` rather than `n_shards`, `n_shards` remains `None`. `str(None) == "None"`, so `len(str(n_shards)) == 4` — shard filenames are zero-padded to four digits regardless of the actual number of shards. More subtly, if the actual number of shards exceeds 9999, the padding will be insufficient and sorting will break. The padding should be computed from `len(shards)` (the actual number of shards produced) rather than from `n_shards`.

---

## 8. Medium Bug: `dsu2pua` Error Message Computes Subtraction Instead of Range

**File:** `sardalign/utils/__init__.py:28`

```python
def dsu2pua(idx_dsu: int) -> str:
    dsu_ord = PUA_PL_START + idx_dsu
    if dsu_ord > PUA_PL_END:
        raise RuntimeError(
            f"DSU ordinal out of PUA range: {idx_dsu}. PUA range: {PUA_PL_START - PUA_PL_END:,}"
        )
```

`PUA_PL_START - PUA_PL_END` evaluates to `983_040 - 1_048_573 = -65_533` — a negative number — displayed in the error message as the "PUA range". The sibling function `pua2dsu` does this correctly: `{PUA_PL_START:,} - {PUA_PL_END:,}`.

---

## 9. Medium Bug: `librispeech/preprocess_ls.py` File Handle Not Managed with `with`

**File:** `scripts/librispeech/preprocess_ls.py:38`

```python
f_jsonl = output_jsonl.open("x")
n_samples = 0
for trans_txt in trans_txts:
    with open(trans_txt, "r") as f_trans:
        for line in f_trans:
            ...
f_jsonl.close()
```

If any exception occurs during processing (e.g. a malformed transcript line), `f_jsonl.close()` is never called, leaving the file handle open. The entire block should be wrapped in `with output_jsonl.open("x") as f_jsonl:`.

---

## 10. Minor: Duplicate Entries in `special_isos_uroman`

**File:** `sardalign/utils/align.py:20–24`

```python
special_isos_uroman = (
    "ara", "bel", "bul", "deu", "ell", "eng", "fas", "grc", "ell",
    "eng", "heb", ...
)
```

Both `"ell"` (Greek) and `"eng"` (English) appear twice. This is harmless for the `in` membership test, but is clearly a copy-paste error and misleads anyone reading the list to understand which languages receive special treatment.

---

## 11. Minor: `norm_config.py` Shadows `exclamation_mark` and Has a Typo

**File:** `sardalign/norm_config.py:7, 93, 204`

```python
exclamation_mark = "!"          # line 7 — ASCII, correctly used in basic_punc
...
exclamation_mark = r"\u055C"    # line 93 — Armenian mark, silently overwrites above
```

The Armenian `exclamation_mark` variable shadows the ASCII one. `basic_punc` is already built at line 16 before the shadowing, so there is no runtime effect, but any code added after line 93 that references `exclamation_mark` expecting `"!"` will silently receive the Armenian Unicode escape instead.

Additionally:
```python
shared_mappping = {   # line 204 — triple-p typo
```
The name `shared_mappping` has three `p`s and is used consistently throughout the file, so it does not break anything, but it is a clear typo.

---

## 12. Minor: `join_alignment_mls_mimi.py` Has Wrong `ArgumentParser` Description

**File:** `scripts/mls/join_alignment_mls_mimi.py:40`

```python
parser = ArgumentParser(description="Join MLS time alignment with SpeechTokenizer RVQ tokens.")
```

This file handles **Mimi** tokens, not SpeechTokenizer tokens. The description was copy-pasted from `join_alignment_mls_speechtokenizer.py` without being updated.

---

## 13. Minor: Root Logger Used Instead of Module Logger

**Files:** `sardalign/learn_kmeans.py:65, 125`, `sardalign/dump_km_label.py:77`

Several call sites use the root `logging` logger directly instead of the module-level `LOGGER`:

```python
# learn_kmeans.py:65
logging.info(f"loaded feature with dimension {feat.shape}")  # root logger

# learn_kmeans.py:125 and dump_km_label.py:77 (both __main__ blocks)
logging.info(str(args))  # root logger
```

These messages will appear with different formatting than the rest of the module's log output and bypass any per-module log level filtering.

---

## Summary Table

| # | Severity | File | Line | Issue |
|---|----------|------|------|-------|
| 1 | **Critical** | `scripts/auxiliary/align_and_segment.py`, `segment_tokens.py` | 34, 85 | `get_alignments` return unpacked to 2 values (returns 3); `segment_tokens.py` also missing `device` arg |
| 2 | **Critical** | `tutorial/align.py` | 74 | `unflatten_list`: `[i : i + 1]` should be `[i : i + length]` |
| 3 | **Critical** | `sardalign/align.py` | 33 | `generate_emissions` checks `sr != SAMPLING_FREQ` not `sr != check_sampling_rate` |
| 4 | **Critical** | `scripts/uromanize.py` | 81 | `parent(...)` calls Path as function — must be `.parent.mkdir(...)` |
| 5 | **Medium** | `scripts/align_and_hubert_encode.py` | 129–163 | `use_star` causes alignment arrays to have n+1 entries vs n tokens in output |
| 6 | **Medium** | `sardalign/utils/align.py` | 158–161 | `get_spans` assertion fails on >1 trailing blank segment |
| 7 | **Medium** | `sardalign/utils/__init__.py` | 133 | `shard_jsonl` padding uses `len(str(None))=4` when `n_shards` not given |
| 8 | **Medium** | `sardalign/utils/__init__.py` | 28 | `dsu2pua` error message shows `PUA_PL_START - PUA_PL_END` (negative) not a range |
| 9 | **Medium** | `scripts/librispeech/preprocess_ls.py` | 38 | Output file opened without `with`, unclosed on exception |
| 10 | Minor | `sardalign/utils/align.py` | 20–24 | `special_isos_uroman` contains duplicate `"ell"`, `"eng"` entries |
| 11 | Minor | `sardalign/norm_config.py` | 7, 93, 204 | `exclamation_mark` silently shadowed; `shared_mappping` typo |
| 12 | Minor | `scripts/mls/join_alignment_mls_mimi.py` | 40 | Parser description says "SpeechTokenizer" — should say "Mimi" |
| 13 | Minor | `sardalign/learn_kmeans.py`, `dump_km_label.py` | 65, 125, 77 | Root `logging.info` used instead of module `LOGGER` |

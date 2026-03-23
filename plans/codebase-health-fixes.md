# Codebase Health Fixes Plan

## Status Tracker

Each fix below has a status: `TODO`, `IN PROGRESS`, or `DONE`.
Update this file after each iteration to communicate progress between loop iterations.

---

## Fix 1: Hardcoded Absolute Paths in Library Constants
**Status: TODO**
**Priority: High | Effort: Low**

### Problem
`sardalign/constants/__init__.py:14-17` hardcodes machine-specific paths for the CTC alignment model and dictionary. `sardalign/constants/voxpopuli.py:5` also has a hardcoded mount path.

### Fix
- Replace hardcoded paths with `os.environ.get()` calls using the current values as defaults.
- Use env var names: `CTC_ALIGNMENT_MODEL_PATH`, `CTC_ALIGNMENT_DICT_PATH`, `VOXPOPULI_LOCAL_DISK_DIR`.
- This preserves backward compatibility for the existing machine while making the package portable.

### Files to modify
- `sardalign/constants/__init__.py`
- `sardalign/constants/voxpopuli.py`

---

## Fix 2: Shell Command Injection + Unsafe `torch.load`
**Status: TODO**
**Priority: Medium-High | Effort: Low**

### Problem
- `sardalign/utils/align.py:47` uses a shell command call with string interpolation via `os.system` — this is a shell injection risk.
- `sardalign/utils/align.py:105` calls `torch.load()` without `weights_only=True`.

### Fix
- Replace the shell call with `subprocess.run()` using explicit argument lists and shell=False.
- Add `weights_only=True` to the `torch.load()` call.

### Files to modify
- `sardalign/utils/align.py`

---

## Fix 3: Live Bugs (`uromanize.py` and `tutorial/align.py`)
**Status: TODO**
**Priority: Medium | Effort: Low**

### Problem
**a)** `scripts/uromanize.py:81` — `.parent` called as method instead of property:
```python
args.output_jsonl.parent(parents=True, exist_ok=True)  # BUG: should be .parent.mkdir(...)
```

**b)** `tutorial/align.py:74` — off-by-one in `unflatten_list`:
```python
nested.append(flattened_list[i : i + 1])  # BUG: should be i : i + length
```

### Fix
- `uromanize.py:81`: Change to `args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)`
- `tutorial/align.py:74`: Change to `flattened_list[i : i + length]`

### Files to modify
- `scripts/uromanize.py`
- `tutorial/align.py`

---

## Fix 4: Over-Pinned Python Version + Inconsistent Dependencies
**Status: TODO**
**Priority: Medium | Effort: Low**

### Problem
- `pyproject.toml` pins `requires-python = "==3.10.6"` (exact patch version).
- `tutorial/requirements.txt` has incompatible `torch==2.7.1` with `torchaudio==2.2.2`.

### Fix
- Change `requires-python` to `">=3.10"`.
- Fix `tutorial/requirements.txt` to use compatible torch/torchaudio versions (both 2.7.x).

### Files to modify
- `pyproject.toml`
- `tutorial/requirements.txt`

---

## Fix 5: Automated Test Suite
**Status: TODO**
**Priority: High | Effort: Medium**

### Problem
No pytest tests exist. The `tests/` directory contains only shell scripts with hardcoded paths.

### Fix
Create a minimal but meaningful pytest test suite covering the pure-function core logic:

1. **`tests/test_utils.py`** — Tests for:
   - `dsu2pua` / `pua2dsu` (round-trip, boundary, error cases)
   - `count_lines` (normal file, missing trailing newline)
   - `get_integer_sample_size` (int and float inputs)
   - `parse_arg_int_or_float`
   - `write_jsonl` / `read_jsonl` (round-trip)
   - `shard_jsonl`

2. **`tests/test_align_utils.py`** — Tests for:
   - `merge_repeats` (simple path, all blanks, single token)
   - `time_to_frame`
   - `get_span_times`
   - `times_to_hubert_idxs`
   - `normalize_uroman`

3. **`tests/test_text_normalization.py`** — Tests for:
   - `text_normalize` with default config (`*`)
   - Language-specific configs (e.g. `ara`, `mon`)
   - Edge cases: empty string, all-digits, brackets

4. **`tests/test_interleave.py`** — Tests for:
   - `get_span_idxs_binomial` (deterministic with seed, boundary conditions)

5. **`tests/conftest.py`** — Shared fixtures (tmp directories, sample JSONL data).

6. **`pyproject.toml`** — Add `[tool.pytest.ini_options]` section and `pytest` to dev dependencies.

### Files to create
- `tests/conftest.py`
- `tests/test_utils.py`
- `tests/test_align_utils.py`
- `tests/test_text_normalization.py`
- `tests/test_interleave.py`

### Files to modify
- `pyproject.toml` (add pytest config and dependency)

---

## Iteration Log

| Iteration | Date | Fixes Completed | Notes |
|-----------|------|-----------------|-------|
| — | — | — | Not yet started |
